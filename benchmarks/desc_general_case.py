"""Solve one exact case in DESC and score it against the exact field everywhere.

Generalizes ``run_desc_coordinate.py`` (integer family only) to the sheared and
Solov'ev families, including the genuinely up-down-asymmetric Solov'ev case
(``sym=False``).  Boundary and initial interior geometry come from the exact
chart; lambda starts at zero and DESC solves for it.  Profiles: pressure and
prescribed iota from ``build_inputs.fit_profiles``, converted from powers of s
to even powers of rho.  Orientation follows the integer driver: the exact
counterclockwise R-Z angle is theta_ccw = -theta_DESC, so iota_DESC = -iota.

Scoring evaluates DESC at chosen (rho, theta, zeta) — rings from rho = 1e-4 to
0.99 and a tensor volume rule — maps to Cartesian points with DESC's R, Z and
compares B, J = curl B/mu0 and grad p with the exact field (JAX derivatives).
SI with B* = 1 T, L* = 1 m.  Runs in the DESC environment (Python 3.10).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import mu_0

import desc
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.profiles import PowerSeriesProfile

from analytic import cases, field, label_at_s, pressure, surface
from build_inputs import fit_profiles


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rho_series(poly_in_s):
    """Power series in s -> DESC PowerSeriesProfile in rho (even modes)."""
    coef = np.asarray(poly_in_s.coef, dtype=float)
    return coef, [2*k for k in range(len(coef))]


def exact_rpz(case, nodes):
    rho, theta, zeta = nodes.T
    labels = np.asarray([float(label_at_s(case, min(max(r*r, 0.0), 1.0))) for r in rho])
    xyz = np.asarray(surface(case, labels, -theta, zeta), dtype=float)
    return np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]), np.arctan2(xyz[:, 1], xyz[:, 0]), xyz[:, 2]))


def build(case, M, N, L, sym):
    bg = LinearGrid(rho=[1.0], M=2*M, N=2*N, NFP=case.nfp, sym=False)
    bnodes = np.asarray(bg.nodes)
    brpz = exact_rpz(case, bnodes)
    phase = np.max(np.abs(np.angle(np.exp(1j*(brpz[:, 1]-bnodes[:, 2])))))
    if phase > 1e-8:
        raise ValueError(f"exact boundary sampler does not return the requested toroidal angle ({phase:.2e})")
    boundary = FourierRZToroidalSurface.from_values(brpz, bnodes[:, 1], M=M, N=N, NFP=case.nfp, sym=sym)
    p_fit, iota_fit, _, fit_errors = fit_profiles(case)
    p_coef, p_modes = rho_series(p_fit)
    i_coef, i_modes = rho_series(iota_fit)
    from analytic import flux
    eq = Equilibrium(L=L, M=M, N=N, NFP=case.nfp, sym=sym, surface=boundary,
                     L_grid=2*L, M_grid=2*M, N_grid=2*N,
                     Psi=abs(float(flux(case, case.edge))),
                     pressure=PowerSeriesProfile(p_coef/mu_0, modes=p_modes),
                     iota=PowerSeriesProfile(-i_coef, modes=i_modes), ensure_nested=False)
    fg = LinearGrid(rho=np.linspace(0.025, 1, L+2), M=2*M, N=2*N, NFP=case.nfp, sym=False)
    fnodes = np.asarray(fg.nodes)
    frpz = exact_rpz(case, fnodes)
    eq.set_initial_guess(fnodes, frpz[:, 0], frpz[:, 2], np.zeros(len(fnodes)), ensure_nested=False)
    eq.axis = eq.get_axis()
    return eq, {k: float(v) for k, v in fit_errors.items()}


def exact_fields(case, xyz):
    f = lambda x: field(case, x)[0]  # noqa: E731
    p = lambda x: pressure(case, x)  # noqa: E731
    X = jnp.asarray(xyz)
    B = np.asarray(jax.vmap(f)(X))
    grad = np.asarray(jax.vmap(jax.jacfwd(f))(X))
    curl = np.stack([grad[:, 2, 1]-grad[:, 1, 2], grad[:, 0, 2]-grad[:, 2, 0], grad[:, 1, 0]-grad[:, 0, 1]], -1)
    gp = np.asarray(jax.vmap(jax.grad(p))(X))
    return B, curl/mu_0, gp/mu_0


def to_xyz(vector, phi):
    r, q, z = np.asarray(vector).T
    return np.column_stack((r*np.cos(phi)-q*np.sin(phi), r*np.sin(phi)+q*np.cos(phi), z))


def score(eq, case, nodes, weights=None):
    data = eq.compute(["R", "Z", "B", "J", "grad(p)", "sqrt(g)"], grid=Grid(nodes, sort=False), override_grid=True)
    phi = nodes[:, 2]
    R, Z = np.asarray(data["R"]), np.asarray(data["Z"])
    xyz = np.column_stack((R*np.cos(phi), R*np.sin(phi), Z))
    B, J, gp = (to_xyz(data[k], phi) for k in ("B", "J", "grad(p)"))
    Bx, Jx, gpx = exact_fields(case, xyz)
    jac = np.asarray(data["sqrt(g)"])
    w = np.ones(len(nodes)) if weights is None else weights*np.abs(jac)

    def rel(a, b):
        return float(np.sqrt(np.sum(w*np.sum((a-b)**2, 1))/np.sum(w*np.sum(b*b, 1))))

    force = np.cross(J, B)-gp
    return {"points": int(len(nodes)), "B_relative_l2": rel(B, Bx), "J_relative_l2": rel(J, Jx),
            "gradp_relative_l2": rel(gp, gpx),
            "force_over_gradp": float(np.sqrt(np.sum(w*np.sum(force**2, 1))/np.sum(w*np.sum(gpx**2, 1)))),
            "min_abs_sqrt_g": float(np.min(np.abs(jac)))}


def score_everywhere(eq, case, rings=(1e-4, 1e-3, 1e-2, 0.1, 0.3, 0.7, 0.99), n=16, volume=(16, 32, 32)):
    th = 2*np.pi*np.arange(n)/n
    ze = 2*np.pi*np.arange(n)/(n*case.nfp)
    T, Zt = np.meshgrid(th, ze, indexing="ij")
    out = {"rings": {}, "volume": {}}
    for rho in rings:
        out["rings"][f"{rho:g}"] = score(eq, case, np.column_stack((np.full(T.size, rho), T.ravel(), Zt.ravel())))
    nr, nt, nz = volume
    for rule in ("gauss", "midpoint"):
        if rule == "gauss":
            x, wr = leggauss(nr)
            r, wr = (x+1)/2, wr/2
        else:
            r, wr = (np.arange(nr)+0.5)/nr, np.full(nr, 1/nr)
        Rr, Tt, Zz = np.meshgrid(r, 2*np.pi*np.arange(nt)/nt, 2*np.pi*np.arange(nz)/(nz*case.nfp), indexing="ij")
        W = np.broadcast_to(wr[:, None, None], Rr.shape).ravel()
        out["volume"][rule] = score(eq, case, np.column_stack((Rr.ravel(), Tt.ravel(), Zz.ravel())), weights=W)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--M", type=int, default=8)
    parser.add_argument("--N", type=int, default=None, help="default M (0 for axisymmetric families)")
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--sym", choices=("auto", "true", "false"), default="auto")
    parser.add_argument("--maxiter", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    desc.set_device("cpu")
    case = cases()[args.case]
    axisymmetric = case.family == "solovev" or args.case.endswith("axisymmetric")
    N = args.N if args.N is not None else (0 if axisymmetric else args.M)
    L = args.L if args.L is not None else args.M+2
    sym = {"true": True, "false": False}.get(args.sym, not args.case.endswith("asymmetric"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    eq, fit_errors = build(case, args.M, N, L, sym)
    record = {"schema": 1, "evidence": "desc_exact_case_everywhere", "case": args.case,
              "resolution": {"L": L, "M": args.M, "N": N, "NFP": case.nfp, "sym": sym},
              "profile_fit_errors": fit_errors, "desc_version": desc.__version__,
              "jax": jax.__version__, "python": platform.python_version(),
              "script_sha256": sha256(__file__), "nested_initial_guess": bool(eq.is_nested())}
    record["projection"] = score_everywhere(eq, case)
    (args.output_dir/"record.partial.json").write_text(json.dumps(record, indent=2)+"\n")
    t1 = time.perf_counter()
    eq, info = eq.solve(ftol=1e-10, xtol=1e-10, gtol=1e-8, maxiter=args.maxiter, verbose=1)
    record["solver"] = {"success": bool(info.get("success", False)), "message": str(info.get("message", "")),
                        "iterations": int(info.get("nit", -1)), "solve_seconds": time.perf_counter()-t1}
    record["solved"] = score_everywhere(eq, case)
    h5 = args.output_dir/f"desc_{args.case}_M{args.M}.h5"
    eq.save(str(h5))
    record["equilibrium_artifact"] = {"path": h5.name, "sha256": sha256(h5)}
    record["elapsed_seconds"] = time.perf_counter()-t0
    (args.output_dir/"record.json").write_text(json.dumps(record, indent=2)+"\n")
    (args.output_dir/"record.partial.json").unlink()
    for stage in ("projection", "solved"):
        v = record[stage]["volume"]["gauss"]
        print(args.case, stage, "B %.2e J %.2e F %.2e" % (v["B_relative_l2"], v["J_relative_l2"], v["force_over_gradp"]),
              "axis(1e-4) B %.2e J %.2e" % (record[stage]["rings"]["0.0001"]["B_relative_l2"],
                                           record[stage]["rings"]["0.0001"]["J_relative_l2"]), flush=True)


if __name__ == "__main__":
    main()
