"""Certified equilibrium sensitivities on VMEX against exact-solution derivatives (integer family).

Observable: the Eulerian field B at fixed Cartesian points (an 80-point cloud and a cell-resolved
whole-volume quadrature of {s <= s_max}), and the volume mean of |B|.  Closure is NCURR=0
(iota prescribed), which is exact for this family: iota = -2 for every parameter, so iota is an
input here and is not a response observable.  Parameters are the dimensionless family parameters
(a, b, c, delta); c is a nonzero response, delta (the outer label) is the null response.

Per (case, NS) the record holds, for every parameter:

  root        host fsq/iterations and the true |F| (preconditioned and raw) at the returned state;
  tangent     the implicit tangent with the library's certificate and an independent matrix-free
              residual |A dz + F_P t| / |F_P t| of the raw operator;
  jvp         dB at the points from the tangent (forward mode, exact AD input tangent P_a);
  transpose   <w, J v> against <J^T w, v> with J^T from the certified block pullback;
  fd ladder   centred differences of reconverged solves for several h, the branch residual
              |A d_h + F_P q_h| / |F_P q_h|, endpoint residuals in the base and their own operator,
              and the drift of the entries the linearisation holds frozen;
  exact       error of the JVP and of each FD against the exact fixed-x derivative.

Usage: python sensitivity_vmex.py --case integer_axisymmetric --ns 33 65
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import vmex
from vmex.core import implicit
from vmex.core.extender import _cartesian_derivative, _invert_coordinates
from vmex.core.virtual_casing import _state_field_spectra

from analytic import ROOT, cases, field as exact_field
import derivative_map as dm

CHUNK = 512


def pad(x, n):
    x = jnp.asarray(x, dtype=float)
    return jnp.concatenate([x, jnp.zeros(n-x.size)])


def make_pmap(template, mpol, ntor, nfp):
    def pmap(p):
        rbc, zbs = dm.vmex_boundary(p, mpol, ntor, nfp)
        am, ai, phiedge = dm.profiles(p)
        return replace(template, rbc=rbc, zbs=zbs, am=pad(am, template.am.size),
                       ai=pad(ai, template.ai.size), phiedge=jnp.asarray(phiedge, dtype=float))
    return pmap


def tree_norm(tree):
    return float(jnp.sqrt(sum(jnp.vdot(v, v).real for v in jax.tree.leaves(tree))))


def tree_dot(a, b):
    return float(sum(jnp.vdot(x, y).real for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))))


def zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def axpy(a, x, y):
    return jax.tree.map(lambda u, v: a*u+v, x, y)


def chunks(n, size=CHUNK):
    return [(i, min(i+size, n)) for i in range(0, n, size)]


class Observer:
    """B at fixed Cartesian points through the continuous interior field, in fixed-size chunks."""

    def __init__(self, inp, cfg, xyz, seed_flux):
        self.inp, self.cfg, self.n = inp, cfg, len(xyz)
        self.xyz = jnp.asarray(xyz)
        self.seed = jnp.asarray(seed_flux)
        self.size = min(CHUNK, self.n)
        self.spans = chunks(self.n, self.size)

        def field(state, params, x, seed):
            spectra = _state_field_spectra(inp, state, implicit.runtime_from_params(params, cfg))
            w, valid = _invert_coordinates(spectra, x, newton_iterations=20, initial_flux=seed)
            return _cartesian_derivative(spectra, 0, w, valid)
        self.f = jax.jit(field)

    def piece(self, span):
        lo, hi = span
        idx = np.minimum(np.arange(lo, lo+self.size), self.n-1)
        return self.xyz[idx], self.seed[idx], hi-lo

    def B(self, state, params):
        out = [np.asarray(self.f(state, params, *self.piece(s)[:2]))[:s[1]-s[0]] for s in self.spans]
        return np.concatenate(out)

    def jvp(self, state, params, dstate, dparams):
        def one(s, p, x, seed):
            return jax.jvp(lambda a, b: self.f(a, b, x, seed), (s, p), (dstate, dparams))[1]
        j = jax.jit(one)
        out = [np.asarray(j(state, params, *self.piece(s)[:2]))[:s[1]-s[0]] for s in self.spans]
        return np.concatenate(out)

    def vjp(self, state, params, w):
        """Cotangents (state, params) of sum(w * B), accumulated over chunks (w zero on invalid points)."""
        gs, gp = zeros_like(state), zeros_like(params)

        def one(s, p, x, seed, wc):
            return jax.vjp(lambda a, b: self.f(a, b, x, seed), s, p)[1](wc)
        j = jax.jit(one)
        for span in self.spans:
            x, seed, m = self.piece(span)
            wc = np.zeros((self.size, 3))
            wc[:m] = w[span[0]:span[1]]
            cs, cp = j(state, params, x, seed, jnp.asarray(wc))
            gs, gp = axpy(1., gs, cs), axpy(1., gp, cp)
        return gs, gp


def seed_flux_of(inp, state, runtime, xyz):
    out = [np.asarray(vmex.VmecInteriorField.from_state(inp, state, runtime=runtime)
                      .flux_coordinates(jnp.asarray(xyz[i:j]))) for i, j in chunks(len(xyz), 2048)]
    return np.concatenate(out)


def host_status(cfg):
    hit = implicit._LAST_SOLVE.get(cfg)
    if hit is None:
        return {}
    r = hit[1]
    return {"fsqr": float(r.fsqr), "fsqz": float(r.fsqz), "fsql": float(r.fsql),
            "iterations": int(r.iterations), "converged": bool(r.converged)}


def operators(cfg, state, mask):
    P = implicit._dof_projector(cfg, mask)
    return P, {f: implicit.residual_fn(cfg, state, mask, f) for f in ("preconditioned", "raw")}


def root_norms(cfg, params, state, mask):
    P, F = operators(cfg, state, mask)
    z = P(state)
    return {f: tree_norm(F[f](z, params)) for f in F}


def linear_residual(cfg, params, state, mask, dstate, dparams):
    """|A dz + F_P t| / |F_P t| of the raw operator, with A = dF/dz, F_P = dF/dp at the root."""
    P, F = operators(cfg, state, mask)
    z = P(state)
    out = {}
    for f in ("raw", "preconditioned"):
        zero = zeros_like(z)
        rhs = jax.jvp(F[f], (z, params), (zero, dparams))[1]
        res = jax.jvp(F[f], (z, params), (P(dstate), dparams))[1]
        out[f] = {"relative_residual": tree_norm(res)/max(tree_norm(rhs), 1e-300),
                  "rhs_norm": tree_norm(rhs)}
    return out


def frozen_drift(cfg, mask, dstate):
    """Norm of the part of a state difference that is neither evolved nor on the edge row."""
    P = implicit._dof_projector(cfg, mask)
    edge = implicit._edge_mask(cfg)
    rest = jax.tree.map(lambda d, p, e: (d-p)*(1-e), dstate, P(dstate), edge)
    return tree_norm(rest)


def dense_control(cfg, params0, state0, mask, tangents, dstates, gs, gp, library_reverse, names, block=128):
    """Independent dense linear algebra at small size (axisymmetric, elementwise dof mask only).

    Builds the active-dof Jacobian A of the raw residual by forward mode and by reverse mode,
    solves A dz = -F_P t and A^T l = g_z densely, and compares with the library tangent and with
    the library pullback (J^T w).  ``gs, gp`` are the cotangents of sum(w * B) on the state and on
    the parameters at fixed state; ``library_reverse`` is the library's (J^T w)_k.
    """
    P, F = operators(cfg, state0, mask)
    z0 = P(state0)
    flat, unravel = ravel_pytree(z0)
    mask_flat = np.asarray(ravel_pytree(mask)[0])
    idx = np.flatnonzero(mask_flat > 0)
    n = len(idx)

    def act(v):
        full = jnp.zeros_like(flat).at[idx].set(v)
        return ravel_pytree(F["raw"](unravel(full), params0))[0][idx]
    v0 = flat[idx]
    jf = jax.jit(jax.vmap(lambda e: jax.jvp(act, (v0,), (e,))[1]))
    cols = [np.asarray(jf(jnp.eye(n)[i:i+block])) for i in range(0, n, block)]
    A = np.concatenate(cols).T  # columns of the Jacobian
    _, pullback = jax.vjp(act, v0)
    jr = jax.jit(jax.vmap(lambda e: pullback(e)[0]))
    rows = [np.asarray(jr(jnp.eye(n)[i:i+block])) for i in range(0, n, block)]
    A_rev = np.concatenate(rows)  # rows of the Jacobian
    sv = np.linalg.svd(A, compute_uv=False)
    edge = implicit._edge_mask(cfg)
    frozen = jax.lax.stop_gradient(state0)
    g_z = np.asarray(ravel_pytree(P(gs))[0])[idx]
    ell = np.linalg.solve(A.T, g_z)
    out = {"active_dofs": int(n), "sigma_min": float(sv[-1]), "sigma_max": float(sv[0]),
           "condition": float(sv[0]/sv[-1]),
           "forward_vs_reverse_jacobian_rel": float(np.linalg.norm(A-A_rev)/np.linalg.norm(A)),
           "tangent_rel_error": {}, "reverse_rel_error": {}}
    for k, name in enumerate(names):
        zero = zeros_like(z0)
        fp = ravel_pytree(jax.jvp(F["raw"], (z0, params0), (zero, tangents[k]))[1])[0]
        fp = np.asarray(fp)[idx]
        dz = -np.linalg.solve(A, fp)
        lib = np.asarray(ravel_pytree(P(dstates[k]))[0])[idx]
        out["tangent_rel_error"][name] = float(np.linalg.norm(dz-lib)/np.linalg.norm(lib))
        dx = jax.jvp(lambda prm: implicit._assemble(z0, implicit.runtime_from_params(prm, cfg), frozen, P, edge),
                     (params0,), (tangents[k],))[1]
        direct = tree_dot(gp, tangents[k])+tree_dot(gs, dx)
        dense_rev = direct-float(ell@fp)
        out["reverse_rel_error"][name] = float(abs(dense_rev-library_reverse[k])/max(abs(dense_rev), abs(library_reverse[k]), 1e-300))
    return out


def prepare(case_name, ns, ftol, niter, tcon0, seed_mode, refine_tol, adjoint_tol):
    case = cases()[case_name]
    inp = vmex.VmecInput.from_file(ROOT/f"inputs/input.{case_name}_iota")
    changes = dict(ns_array=np.asarray([ns]), ftol_array=np.asarray([ftol]), niter_array=np.asarray([niter]))
    if tcon0 is not None:
        changes["tcon0"] = float(tcon0)
    inp = replace(inp, **changes)
    cfg = implicit.make_config(inp, ns=ns, ftol=ftol, max_iterations=niter, refine_tol=refine_tol,
                               adjoint_tol=adjoint_tol, hot_restart=True)
    if seed_mode == "exact":
        from project_integer_vmex import project
        state, _ = project(inp, case, ns)
        implicit._HOT_CACHE[cfg] = jax.tree.map(jnp.asarray, state)
    return case, inp, cfg


def run(args, case_name, ns, out):
    names = list(args.params)
    case, inp, cfg = prepare(case_name, ns, args.ftol, args.niter, args.tcon0, args.seed,
                             args.refine_tol, args.adjoint_tol)
    mpol, ntor, nfp = int(inp.mpol), int(inp.ntor), int(inp.nfp)
    template = implicit.params_from_input(inp)
    pmap = make_pmap(template, mpol, ntor, nfp)
    p0 = jnp.asarray(case.parameters)
    params0 = pmap(p0)
    timing = {}

    cloud_xyz, cloud_coords = dm.point_cloud(case)
    vol_xyz, vol_w, vol_coords = dm.volume_rule(case, ns=args.vol[0], ntheta=args.vol[1],
                                                nphi=args.vol[2], s_max=args.s_max)
    sets = {"cloud": (cloud_xyz, None), "volume": (vol_xyz, vol_w)}
    if args.vol_fine:  # composite radial rule that resolves the near-axis shells; JVP only, no FD
        fine_xyz, fine_w, fine_coords = dm.volume_rule(case, ns=args.vol_fine[0], ntheta=args.vol_fine[1],
                                                       nphi=args.vol_fine[2], s_max=args.s_max, panels=True)
        sets["volume_fine"] = (fine_xyz, fine_w)
    fd_sets = [k for k in sets if k != "volume_fine"]
    exact = {k: dm.exact_tangents(case, xyz, names) for k, (xyz, _) in sets.items()}
    exact_B = {k: np.asarray(jax.vmap(lambda x: exact_field(case, x)[0])(jnp.asarray(xyz)))
               for k, (xyz, _) in sets.items()}

    t = time.perf_counter()
    state0, mask = jax.block_until_ready(implicit.solve_implicit_with_aux(params0, cfg))
    timing["base_solve_s"] = time.perf_counter()-t
    host = host_status(cfg)
    runtime = implicit.runtime_from_params(params0, cfg)
    obs = {}
    for k, (xyz, _) in sets.items():
        obs[k] = Observer(inp, cfg, xyz, seed_flux_of(inp, state0, runtime, xyz))
    B0 = {k: obs[k].B(state0, params0) for k in sets}

    record = {"case": case_name, "ns": ns, "mpol": mpol, "ntor": ntor, "nfp": nfp, "tcon0": float(inp.tcon0),
              "seed": args.seed, "ftol": args.ftol, "refine_tol": args.refine_tol,
              "adjoint_tol": args.adjoint_tol, "s_max": args.s_max, "volume_rule": list(args.vol),
              "points": {k: len(sets[k][0]) for k in sets}, "volume_weight_sum": float(vol_w.sum()),
              "host_solve": host, "root": root_norms(cfg, params0, state0, mask),
              "base_field_error": {k: dm.weighted_rms(B0[k]-exact_B[k], sets[k][1])/dm.weighted_rms(exact_B[k], sets[k][1])
                                   for k in sets},
              "params": {}}
    arrays = {"cloud_xyz": cloud_xyz, "cloud_coords_s_theta_phi": cloud_coords,
              "volume_xyz": vol_xyz, "volume_weights": vol_w, "volume_coords_s_theta_phi": vol_coords}
    if args.vol_fine:
        arrays.update(volume_fine_xyz=fine_xyz, volume_fine_weights=fine_w, volume_fine_coords_s_theta_phi=fine_coords)
    for k in sets:
        arrays[f"B0_{k}"], arrays[f"exact_B_{k}"] = B0[k], exact_B[k]

    # --- tangents with the exact AD input tangent P_a
    tangents = [jax.jvp(pmap, (p0,), (dm.unit(n),))[1] for n in names]
    batch = jax.tree.map(lambda *v: jnp.stack(v), *tangents)
    t = time.perf_counter()
    dstate_batch, report = implicit.implicit_state_tangent_multi_rhs(params0, cfg, state0, mask, batch)
    dstate_batch = jax.block_until_ready(dstate_batch)
    timing["tangent_solve_s_all_rhs"] = time.perf_counter()-t
    dstates = [jax.tree.map(lambda v, k=k: v[k], dstate_batch) for k in range(len(names))]
    record["tangent_report"] = {k: np.asarray(getattr(report, k)).tolist()
                                for k in ("residual_norm", "tolerance", "iterations", "converged")}
    t = time.perf_counter()
    J = {k: np.stack([obs[k].jvp(state0, params0, dstates[j], tangents[j]) for j in range(len(names))], -1)
         for k in sets}
    timing["field_jvp_s"] = time.perf_counter()-t

    # --- transpose consistency: <w, J e_j> against (J^T w)_j through the certified block pullback
    t = time.perf_counter()
    rng = np.random.default_rng(20261023)
    transposes, first = {}, {}
    for k in fd_sets:
        finite = np.isfinite(J[k]).all(axis=(1, 2))
        rows = []
        for trial in range(3):
            w = rng.standard_normal((len(finite), 3))*finite[:, None]
            if sets[k][1] is not None:
                w = w*np.sqrt(sets[k][1]/sets[k][1].mean())[:, None]
            gs, gp = obs[k].vjp(state0, params0, w)
            pull = implicit.implicit_state_pullback_multi_rhs(
                params0, cfg, state0, mask, jax.tree.map(lambda v: v[None], gs), solver="block")
            total = axpy(1., jax.tree.map(lambda v: v[0], pull), gp)
            rev = np.array([tree_dot(total, tangents[j]) for j in range(len(names))])
            fwd = np.array([float(np.sum(w[finite]*np.nan_to_num(J[k][finite][:, :, j]))) for j in range(len(names))])
            bound = np.array([np.linalg.norm(w)*np.linalg.norm(np.nan_to_num(J[k][finite][:, :, j]))
                              for j in range(len(names))])
            if trial == 0:
                first[k] = (gs, gp, rev)
            rows.append({"forward": fwd.tolist(), "reverse": rev.tolist(),
                         "mismatch_over_cauchy_bound": (abs(fwd-rev)/bound).tolist(),
                         "mismatch_relative": (abs(fwd-rev)/np.maximum(abs(fwd), abs(rev))).tolist()})
        transposes[k] = {"trials": rows, "names": names,
                         "max_mismatch_over_bound": float(max(max(r["mismatch_over_cauchy_bound"]) for r in rows)),
                         "max_mismatch_relative": float(max(max(r["mismatch_relative"]) for r in rows))}
    record["transpose"] = transposes
    timing["transpose_s"] = time.perf_counter()-t
    if args.dense:
        gs, gp, rev = first["cloud"]
        t_dense = time.perf_counter()
        record["dense_control"] = dense_control(cfg, params0, state0, mask, tangents, dstates, gs, gp, rev, names)
        timing["dense_s"] = time.perf_counter()-t_dense

    # --- per-parameter scores, linear residuals, FD ladder
    for j, n in enumerate(names):
        r = {"jvp": {}, "linear": linear_residual(cfg, params0, state0, mask, dstates[j], tangents[j]),
             "fd": []}
        for k in sets:
            ex = exact[k][:, :, j]
            r["jvp"][k] = dm.scores(J[k][:, :, j], ex, n if n == "delta" else "c", case, sets[k][1])
            r["jvp"][k]["gate_pass"] = dm.gate(n if n == "delta" else "c", r["jvp"][k])
            arrays[f"jvp_{k}_{n}"], arrays[f"exact_{k}_{n}"] = J[k][:, :, j], ex
        for h in args.steps:
            hh = h*float(p0[dm.PARAMETERS[n]])
            u = dm.unit(n)
            t = time.perf_counter()
            ends, Bpm = [], {k: [] for k in fd_sets}
            for sgn in (1, -1):
                pp = pmap(p0+sgn*hh*u)
                st, _ = implicit.solve_implicit_with_aux(pp, cfg)
                st = jax.block_until_ready(st)
                ends.append((pp, st, host_status(cfg)))
                for k in fd_sets:
                    Bpm[k].append(obs[k].B(st, pp))
            (pp_p, st_p, hs_p), (pp_m, st_m, hs_m) = ends
            qh = jax.tree.map(lambda a, b: (a-b)/(2*hh), pp_p, pp_m)
            dh = jax.tree.map(lambda a, b: (a-b)/(2*hh), st_p, st_m)
            branch = linear_residual(cfg, params0, state0, mask, dh, qh)
            P, Fbase = operators(cfg, state0, mask)
            endpoint = []
            for pp, st, hs in ends:
                _, Fown = operators(cfg, st, mask)
                z = P(st)
                endpoint.append({"base_operator": tree_norm(Fbase["preconditioned"](z, pp)),
                                 "own_operator": tree_norm(Fown["preconditioned"](z, pp)),
                                 "host": hs})
            augmented = {}
            if ntor > 0 and args.augment:
                # state tangent plus the drift of the entries the linearisation holds frozen: if the FD-vs-JVP
                # gap is that released m=1 pair, this tangent reproduces the reconverged-solve FD
                P_, _ = operators(cfg, state0, mask)
                edge_ = implicit._edge_mask(cfg)
                rest = jax.tree.map(lambda dd, pp, ee: (dd-pp)*(1-ee), dh, P_(dh), edge_)
                aug_state = axpy(1., dstates[j], rest)
                for k in fd_sets:
                    Ja = obs[k].jvp(state0, params0, aug_state, tangents[j])
                    fd_k = (Bpm[k][0]-Bpm[k][1])/(2*hh)
                    augmented[k] = {**dm.scores(Ja, exact[k][:, :, j], n if n == "delta" else "c", case, sets[k][1]),
                                    "minus_fd_rms_T": dm.weighted_rms(Ja-fd_k, sets[k][1])}
            row = {"relative_step": h, "absolute_step": hh, "branch_residual": branch, "augmented_jvp": augmented,
                   "frozen_drift_norm": frozen_drift(cfg, mask, dh), "endpoints": endpoint,
                   "dstate_branch_minus_tangent_rel": tree_norm(jax.tree.map(jnp.subtract, dh, dstates[j]))
                   / max(tree_norm(dstates[j]), 1e-300), "scores": {}}
            for k in fd_sets:
                fd = (Bpm[k][0]-Bpm[k][1])/(2*hh)
                arrays[f"fd_{k}_{n}_h{h:g}"] = fd
                row["scores"][k] = {**dm.scores(fd, exact[k][:, :, j], n if n == "delta" else "c", case, sets[k][1]),
                                    "minus_jvp_rms_T": dm.weighted_rms(fd-J[k][:, :, j], sets[k][1])}
                row["scores"][k]["gate_pass"] = dm.gate(n if n == "delta" else "c", row["scores"][k])
            row["two_solves_s"] = time.perf_counter()-t
            r["fd"].append(row)
        record["params"][n] = r

    # --- volume mean of |B|: scalar observable, exact Eulerian derivative over the fixed region
    w = vol_w
    absB = lambda B: np.linalg.norm(B, axis=-1)
    exact_mean_B = float(np.sum(w*absB(exact_B["volume"]))/np.sum(w))
    d_exact, d_jvp = {}, {}
    for j, n in enumerate(names):
        ex_d = np.sum(w*np.sum(exact_B["volume"]*exact["volume"][:, :, j], -1)/absB(exact_B["volume"]))/np.sum(w)
        jv_d = np.sum(w*np.sum(B0["volume"]*J["volume"][:, :, j], -1)/absB(B0["volume"]))/np.sum(w)
        d_exact[n], d_jvp[n] = float(ex_d), float(jv_d)
    record["mean_absB"] = {"exact": exact_mean_B, "vmex": float(np.sum(w*absB(B0["volume"]))/np.sum(w)),
                           "d_exact": d_exact, "d_jvp": d_jvp}
    record["timing"] = timing
    np.savez_compressed(out/f"{case_name}_ns{ns}{args.tag}.npz", **arrays)
    return record


def summary(rec):
    s = {"ns": rec["ns"], "root": rec["root"]}
    for n, v in rec["params"].items():
        s[n] = {"jvp_cloud": v["jvp"]["cloud"], "jvp_volume": v["jvp"]["volume"],
                "fd_last_cloud": v["fd"][-1]["scores"]["cloud"]}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="integer_axisymmetric")
    ap.add_argument("--ns", type=int, nargs="+", default=[33, 65])
    ap.add_argument("--params", nargs="+", default=["c", "delta"])
    ap.add_argument("--steps", type=float, nargs="*", default=[1e-3, 1e-4, 1e-5])
    ap.add_argument("--ftol", type=float, default=1e-14)
    ap.add_argument("--niter", type=int, default=40000)
    ap.add_argument("--tcon0", type=float, default=None)
    ap.add_argument("--seed", choices=("cold", "exact"), default="cold")
    ap.add_argument("--refine-tol", type=float, default=1e-11)
    ap.add_argument("--adjoint-tol", type=float, default=1e-10)
    ap.add_argument("--vol", type=int, nargs=3, default=(16, 24, 2), metavar=("NS", "NTHETA", "NPHI"))
    ap.add_argument("--vol-fine", type=int, nargs=3, default=None, metavar=("NODES_PER_PANEL", "NTHETA", "NPHI"))
    ap.add_argument("--s-max", type=float, default=0.9)
    ap.add_argument("--dense", action="store_true", help="dense control (axisymmetric, small NS only)")
    ap.add_argument("--augment", action="store_true", help="3-D: add the frozen-entry drift to the tangent (diagnostic)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=ROOT/"results/sensitivity")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src = Path(vmex.__file__).resolve().parents[1]
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for ns in args.ns:
        rec = run(args, args.case, ns, args.out)
        rec.update(schema=1, tag=args.tag, vmex_commit=commit, vmex_version=vmex.__version__, jax=jax.__version__,
                   devices=[str(d) for d in jax.devices()],
                   env={k: os.environ.get(k) for k in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
                   command=" ".join(["python", "sensitivity_vmex.py"]+[f"{k}" for k in os.sys.argv[1:]]),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   map_sha256=hashlib.sha256(Path(dm.__file__).read_bytes()).hexdigest())
        path = args.out/f"{args.case}_ns{ns}{args.tag}.json"
        path.write_text(json.dumps(rec, indent=1, allow_nan=True)+"\n")
        print(json.dumps(summary(rec)), flush=True)


if __name__ == "__main__":
    main()
