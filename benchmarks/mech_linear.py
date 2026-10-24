"""Stage 6: linear attribution of the solved-state error to the pieces of the discrete residual.

The VMEC solve finds a root of G(x) = F_MHD(x) + F_con(x) (scaled force, frozen preconditioner cache) on the
spectral state x.  The exact projected state x* is not a root: G(x*) = F_MHD(x*) + F_con(x*).  To first order the
root sits at  x* + delta, delta = -J^{-1} G(x*),  J = dG/dx at x* (dense, by forward-mode AD, no pseudo-inverse of an
unlabelled null space: dofs/equations that are structurally empty are removed and the rest is solved with LU;
sigma_min/sigma_max reported).  Because the solve is linear in the right-hand side, delta splits exactly:

    delta = delta_con + delta_mhd,   delta_mhd(axis shells) + delta_mhd(bulk shells),

and each piece can be scored (B, J per radial cell) like a solved state.  Predictions that this module checks against
the nonlinear solves of ``mech_solve`` (same runtime, same cache, same baseline):

* default constraint: x* + delta against the solved state (error, layer, state distance);
* ``exact`` baseline: F_con(x*) = 0 so delta = delta_mhd only (J with the exact baseline);
* ``tcon0 = 0``: delta = delta_mhd with J of the unconstrained operator.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech_core as M  # noqa: E402
import mech_solve as MS  # noqa: E402

ROOT = M.ROOT
CHANNELS_SYM = ("R_cos", "Z_sin", "L_sin")
CHANNELS_ALL = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def pack(state, channels):
    return jnp.concatenate([jnp.asarray(getattr(state, c)).ravel() for c in channels])


def unpack(vec, like, channels):
    from vmex.core.solver import SpectralState
    ns, mn = np.asarray(like.R_cos).shape
    parts = {c: vec[i*ns*mn:(i+1)*ns*mn].reshape(ns, mn) for i, c in enumerate(channels)}
    return SpectralState(**{c: parts.get(c, jnp.zeros((ns, mn))) for c in CHANNELS_ALL})


def make_G(seed, rt, cache, channels, fsqz_previous=1.0, iteration=100):
    """x-vector -> force-vector at the frozen cache (the map whose roots the solver finds)."""
    def G(x):
        force = M._scaled_lane(unpack(x, seed, channels), cache, rt, jnp.asarray(float(fsqz_previous)),
                               jnp.asarray(iteration+1))
        return pack(force, channels)
    return G


def dense_jacobian(G, x0, chunk=48):
    n = x0.shape[0]
    jv = jax.jit(lambda vs: jax.vmap(lambda v: jax.jvp(G, (x0,), (v,))[1])(vs))
    cols = []
    eye = jnp.eye(n)
    for i in range(0, n, chunk):
        block = eye[i:i+chunk]
        if block.shape[0] < chunk:
            block = jnp.concatenate([block, jnp.zeros((chunk-block.shape[0], n))])
        cols.append(np.asarray(jv(block))[:min(chunk, n-i)])
    return np.concatenate(cols).T   # J[:, i] = dG/dx_i


def evolve_mask(ns, m, channels):
    """Dofs / equations the solver evolves (vmec_params ``jmin2``/``jlam``, fixed boundary), as a flat bool mask.

    R, Z: shells 0..ns-2 for m = 0 and 1..ns-2 for m >= 1 (the axis row of m >= 1 is the extrapolated one, the edge
    row is the fixed boundary); lambda: shells 1..ns-1 (all m).  Dofs of a missing parity (e.g. Z m = 0 sine) have
    empty equations and are removed by the numerical emptiness test in ``solve_active``.
    """
    js = np.arange(ns)[:, None]
    rz = (js >= np.where(m == 0, 0, 1)[None, :]) & (js <= ns-2)
    lam = np.broadcast_to(js >= 1, rz.shape)
    return np.concatenate([(rz if c[0] in "RZ" else lam).ravel() for c in channels])


def solve_active(J, mask, rhs_list, rtol=1e-12):
    """Remove non-evolved and structurally empty dofs/equations, LU-solve the rest.  Returns solutions, info."""
    scale = np.abs(J).max()
    cols = (np.abs(J).max(0) > rtol*scale) & mask
    rows = (np.abs(J).max(1) > rtol*scale) & mask
    Ja = J[np.ix_(rows, cols)]
    info = {"n": int(J.shape[0]), "active_cols": int(cols.sum()), "active_rows": int(rows.sum())}
    if Ja.shape[0] != Ja.shape[1]:
        raise RuntimeError(f"active system not square: {Ja.shape}")
    s = np.linalg.svd(Ja, compute_uv=False)
    info.update(sigma_max=float(s[0]), sigma_min=float(s[-1]), cond=float(s[0]/s[-1]))
    sols = []
    for rhs in rhs_list:
        x = np.zeros(J.shape[0])
        x[cols] = np.linalg.solve(Ja, -rhs[rows])
        info.setdefault("residual", []).append(float(np.linalg.norm(Ja@x[cols]+rhs[rows])/max(np.linalg.norm(rhs[rows]), 1e-300)))
        sols.append(x)
    return sols, info, rows, cols


def attribution(case, ns, baseline="vmec", tcon0=1.0, axis_shells=3, channels=CHANNELS_SYM, score=True,
                out_parent=None):
    import vmex
    t0 = time.perf_counter()
    inp, rt, seed, info = M.exact_seed(case, ns, tcon0)
    rt = M.with_baseline(rt, seed, baseline)
    cache, res = M.fresh_cache(seed, rt)
    mn = int(np.asarray(seed.R_cos).shape[1])
    G = make_G(seed, rt, cache, channels)
    x0 = pack(seed, channels)
    g_full = np.asarray(G(x0))
    G_off = make_G(seed, rt, replace(cache, tcon=cache.tcon*0.0), channels)
    g_mhd = np.asarray(G_off(x0))
    g_con = g_full-g_mhd
    J = dense_jacobian(G, x0)
    t_jac = time.perf_counter()-t0
    # equation rows grouped by shell: index = channel*ns*mn + js*mn + k
    shell = np.tile(np.repeat(np.arange(ns), mn), len(channels))
    axis = shell < axis_shells
    sols, sinfo, rows, cols = solve_active(J, evolve_mask(ns, np.asarray(rt.modes.m), channels), [g_full, g_con, g_mhd, np.where(axis, g_mhd, 0.0), np.where(~axis, g_mhd, 0.0),
                                               np.where(axis, g_con, 0.0), np.where(~axis, g_con, 0.0)])
    names = ["total", "con", "mhd", "mhd_axis", "mhd_bulk", "con_axis", "con_bulk"]
    rec = {"case": case if isinstance(case, str) else case.name, "ns": ns, "baseline": baseline, "tcon0": tcon0,
           "axis_shells": axis_shells, "seconds_jacobian": t_jac, "solve": sinfo,
           "G_norms": {"total": float(np.linalg.norm(g_full)), "con": float(np.linalg.norm(g_con)), "mhd": float(np.linalg.norm(g_mhd)),
                       "mhd_axis": float(np.linalg.norm(g_mhd[axis])), "mhd_bulk": float(np.linalg.norm(g_mhd[~axis])),
                       "con_axis": float(np.linalg.norm(g_con[axis])), "con_bulk": float(np.linalg.norm(g_con[~axis]))}}
    rec["delta_norm"] = {n: float(np.linalg.norm(s)) for n, s in zip(names, sols)}
    # linearity check: total = con + mhd, mhd = axis + bulk
    rec["linearity"] = {"total_vs_con_plus_mhd": float(np.linalg.norm(sols[0]-sols[1]-sols[2])/np.linalg.norm(sols[0])),
                        "mhd_vs_axis_plus_bulk": float(np.linalg.norm(sols[2]-sols[3]-sols[4])/max(np.linalg.norm(sols[2]), 1e-300))}
    if score:
        rule = MS.rule_for(case if isinstance(case, str) else case.name, ns)
        rec["scores"], rec["profiles"] = {}, {}
        for name, d in zip(names, sols):
            st = unpack(x0+jnp.asarray(d), seed, channels)
            field = vmex.VmecInteriorField.from_state(inp, st)
            sc = MS.ss.score_state(field, {"m4": rule})
            rec["scores"][name] = {k: sc["scores"]["m4"]["all"][k] for k in ("B", "J", "force")}
            rec["scores"][name].update({f"{r}_{k}": sc["scores"]["m4"][r][k] for r in ("axis", "bulk") for k in ("B", "J")})
            rec["profiles"][name] = MS.cell_profiles(field, rule, ns)
    rec["seconds"] = time.perf_counter()-t0
    deltas = {n: s for n, s in zip(names, sols)}
    return rec, deltas, (inp, rt, seed, x0, channels)


ASYM_GROUPS = {   # name -> predicate(channel, m) on the equations (rows) of the full six-channel system
    "Z_cos_m0": lambda c, m: (c == "Z_cos") & (m == 0),
    "R_cos_m0": lambda c, m: (c == "R_cos") & (m == 0),
    "R_sin_m1": lambda c, m: (c == "R_sin") & (m == 1),
    "L_cos_m1": lambda c, m: (c == "L_cos") & (m == 1),
    "odd_all": lambda c, m: np.isin(c, ["R_sin", "Z_cos", "L_cos"]),
    "even_all": lambda c, m: np.isin(c, ["R_cos", "Z_sin", "L_sin"]),
}


def attribution_groups(case, ns, deck=None, baseline="vmec", tcon0=0.0, groups=ASYM_GROUPS, score=True, mpol=None):
    """Linear attribution of the solved-state error of an asymmetric case to blocks of the exact-state residual.

    Same linear algebra as :func:`attribution` with the six channels; the right-hand sides are the residual
    restricted to a (channel, m) block of ``groups``, plus the whole residual.  Returns the record and the deltas.
    """
    import vmex
    t0 = time.perf_counter()
    inp, rt, seed, info = M.exact_seed(case, ns, tcon0, deck=deck, mpol=mpol)
    rt = M.with_baseline(rt, seed, baseline)
    cache, _ = M.fresh_cache(seed, rt)
    channels = CHANNELS_ALL
    mn = int(np.asarray(seed.R_cos).shape[1])
    m = np.asarray(rt.modes.m)
    G = make_G(seed, rt, cache, channels)
    x0 = pack(seed, channels)
    g = np.asarray(G(x0))
    J = dense_jacobian(G, x0)
    chan = np.repeat(np.array(channels), ns*mn)
    mm = np.tile(m, ns*len(channels))
    rhs = {"total": g}
    for name, pred in groups.items():
        rhs[name] = np.where(pred(chan, mm), g, 0.0)
    names = list(rhs)
    sols, sinfo, rows, cols = solve_active(J, evolve_mask(ns, m, channels), [rhs[n] for n in names])
    rec = {"case": case if isinstance(case, str) else case.name, "ns": ns, "baseline": baseline, "tcon0": tcon0, "solve": sinfo,
           "G_norms": {n: float(np.linalg.norm(rhs[n])) for n in names},
           "delta_norm": {n: float(np.linalg.norm(s)) for n, s in zip(names, sols)}, "seconds_jacobian": time.perf_counter()-t0}
    if score:
        cname = case if isinstance(case, str) else case.name
        rule = MS.cp.volume_rule(MS.cases()[cname] if isinstance(case, str) else case, "m4", ns)
        rec["scores"] = {}
        for n, d in zip(names, sols):
            st = unpack(x0+jnp.asarray(d), seed, channels)
            sc = MS.ss.score_state(vmex.VmecInteriorField.from_state(inp, st), {"m4": rule})
            rec["scores"][n] = {k: sc["scores"]["m4"]["all"][k] for k in ("B", "J", "force")}
    rec["seconds"] = time.perf_counter()-t0
    return rec, dict(zip(names, sols)), (inp, rt, seed, x0, channels)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="solovev_symmetric")
    p.add_argument("--ns", type=int, nargs="+", required=True)
    p.add_argument("--baseline", nargs="+", default=["vmec"])
    p.add_argument("--tcon0", type=float, nargs="+", default=[1.0])
    p.add_argument("--axis-shells", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT/"results/mechanism/linear")
    p.add_argument("--asym", action="store_true", help="six-channel attribution by (channel, m) block (asymmetric cases)")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.asym:
        for ns in a.ns:
            for b in a.baseline:
                for t in a.tcon0:
                    rec, _, _ = attribution_groups(a.case, ns, None, b, t)
                    name = f"{rec['case']}-ns{ns}-tcon{t:g}-{b.replace(':', '')}-groups"
                    (a.out/f"{name}.json").write_text(json.dumps(rec, indent=1, allow_nan=False)+"\n")
                    print(name, "cond %.1e" % rec["solve"]["cond"], " ".join(f"{k} {v['B']:.2e}" for k, v in rec["scores"].items()), flush=True)
        return
    for ns in a.ns:
        for b in a.baseline:
            for t in a.tcon0:
                rec, _, _ = attribution(a.case, ns, b, t, a.axis_shells)
                name = f"{rec['case']}-ns{ns}-tcon{t:g}-{b.replace(':', '')}"
                (a.out/f"{name}.json").write_text(json.dumps(rec, indent=1, allow_nan=False)+"\n")
                sc = rec["scores"]
                print(name, "cond %.1e" % rec["solve"]["cond"], "B(total) %.2e con %.2e mhd %.2e axis %.2e bulk %.2e" % tuple(
                    sc[k]["B"] for k in ("total", "con", "mhd", "mhd_axis", "mhd_bulk")), flush=True)


if __name__ == "__main__":
    main()
