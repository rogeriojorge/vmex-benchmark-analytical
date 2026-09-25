"""C3: full-flux lambda versus the existing field-line (transport) lambda, sheared A.

Same exact surfaces, same odd grids, same zero-mean gauge.  The full-flux
potential (plan eqs. 7-8, handoff probe ``flux_potential.recover_lambda``, used
unmodified) uses both signed flux densities; the transport solve
(``project_sheared_straight_field._solve_lambda``) fixes only field-line
straightness.  Both lambdas are then used to rebuild the full Cartesian B from
the exact chart.  Reference only: no VMEX solve or projection is scored here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from analytic import ROOT, cases, field, flux, iota, label_at_s, surface
from evidence import capture_execution_source, reserve_run_directory, write_json

PROBES = ROOT/"docs/handoff/review-05e9473/probes"
sys.path.insert(0, str(PROBES))
from flux_potential import derivative, recover_lambda  # noqa: E402


def _chart(case, label, s, n, h):
    th = 2*np.pi*np.arange(n)[:, None]/n
    ph = 2*np.pi*np.arange(n)[None, :]/(n*case.nfp)
    xyz = np.asarray(surface(case, label, th, ph))
    R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
    shape = xyz.shape
    eR = np.broadcast_to(np.stack((np.cos(ph), np.sin(ph), np.zeros_like(ph)), -1), shape)
    ephi = np.broadcast_to(np.stack((-np.sin(ph), np.cos(ph), np.zeros_like(ph)), -1), shape)
    ez = np.zeros(shape)
    ez[..., 2] = 1
    xt = derivative(R, 0, 2*np.pi)[..., None]*eR + derivative(Z, 0, 2*np.pi)[..., None]*ez
    xp = (derivative(R, 1, 2*np.pi/case.nfp)[..., None]*eR + R[..., None]*ephi
          + derivative(Z, 1, 2*np.pi/case.nfp)[..., None]*ez)
    xs = (np.asarray(surface(case, float(label_at_s(case, s+h)), th, ph))
          - np.asarray(surface(case, float(label_at_s(case, s-h)), th, ph)))/(2*h)
    jac = np.sum(xs*np.cross(xt, xp), axis=-1)
    B = np.asarray(field(case, xyz)[0])
    return xt, xp, xs, jac, B


def _rebuild(lam, T, P, xt, xp, jac, nfp):
    dt = derivative(lam, 0, 2*np.pi)
    dp = derivative(lam, 1, 2*np.pi/nfp)
    return ((T-P*dp)[..., None]*xt + (P*(1+dt))[..., None]*xp)/jac[..., None]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surfaces", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument("--grids", type=int, nargs="+", default=(33, 65, 97))
    parser.add_argument("--radial-step", type=float, default=6.25e-6)
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/reference/full_flux_lambda")
    args = parser.parse_args(argv)
    import vmex
    from project_sheared_straight_field import _solve_lambda, _surface_rates
    run_id, out = reserve_run_directory(
        args.output_parent, f"compare-transport-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    here = Path(__file__).resolve().parent
    capture_execution_source(out, [Path(__file__), here/"analytic.py", here/"project_sheared_straight_field.py",
                                   PROBES/"flux_potential.py", here/"evidence.py"],
                             packages={"uwplasma/vmex": vmex.__file__})
    started = perf_counter()
    case = cases()["sheared_A"]
    if case.nfp != 2:
        raise SystemExit("transport solve hard-codes NFP=2")
    P = -float(flux(case, case.edge))/(2*np.pi)  # signed; chart Jacobian is negative
    rows = []
    for s in args.surfaces:
        label = float(label_at_s(case, s))
        transform = float(iota(case, label))
        T = P*transform
        for n in args.grids:
            xt, xp, xs, jac, B = _chart(case, label, s, n, args.radial_step)
            U = np.sum(B*np.cross(xp, xs), axis=-1)
            V = np.sum(B*np.cross(xs, xt), axis=-1)
            lam_flux, cert = recover_lambda(U, V, poloidal=T, toroidal=P, nfp=case.nfp,
                                            tolerance=5e-5)
            _, _, a, tangent = _surface_rates(case, label, n)
            lam_line, solve = _solve_lambda(a, transform)
            scale = np.linalg.norm(B)
            rows.append({
                "s": s, "grid": n, "iota": transform,
                "flux_certificate": cert, "transport_solve": solve,
                "tangent_relative_l2": tangent["tangent_relative_l2"],
                "lambda_rms": float(np.sqrt(np.mean(lam_flux**2))),
                "lambda_difference_max": float(np.max(np.abs(lam_flux-lam_line))),
                "lambda_difference_relative_l2": float(np.linalg.norm(lam_flux-lam_line)/np.linalg.norm(lam_flux)),
                "B_relative_l2_full_flux": float(np.linalg.norm(_rebuild(lam_flux, T, P, xt, xp, jac, case.nfp)-B)/scale),
                "B_relative_l2_transport": float(np.linalg.norm(_rebuild(lam_line, T, P, xt, xp, jac, case.nfp)-B)/scale),
                "B_relative_l2_lambda_zero": float(np.linalg.norm(_rebuild(0*lam_flux, T, P, xt, xp, jac, case.nfp)-B)/scale),
            })
            r = rows[-1]
            print(s, n, "%.2e %.2e %.2e %.2e" % (r["lambda_difference_relative_l2"], r["B_relative_l2_full_flux"],
                                                 r["B_relative_l2_transport"], r["B_relative_l2_lambda_zero"]))
    record = {"schema": 1, "run_id": run_id, "case": "sheared_A", "nfp": case.nfp,
              "evidence": "analytical_reference_only_no_vmex_solve",
              "toroidal_signed": P, "radial_difference_step": args.radial_step,
              "rows": rows, "elapsed_seconds": perf_counter()-started}
    write_json(out/"comparison.json", record, exclusive=True)
    print(run_id)


if __name__ == "__main__":
    main()
