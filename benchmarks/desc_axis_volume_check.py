"""DESC near-axis and volume scoring of a saved integer-family equilibrium.

The same test that exposed the VMEX axis defects: evaluate the native DESC
state at chosen (rho, theta, zeta), including rho -> 0, map to Cartesian
points through DESC's own R, Z, and compare B, J and grad p there with the
exact field.  No coordinate inversion is needed.  Volume norms use a tensor
Gauss rule in rho with DESC's |sqrt(g)|.  Runs in the DESC environment
(Python 3.10), so it does not import ``evidence.py``.

The saved DESC runs use SI with B* = 1 T and L* = 1 m: J = curl B / mu0 and
p = p_dimensionless / mu0 (see ``run_desc_coordinate.build``).
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
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.constants import mu_0

import desc
from desc.grid import Grid
from desc.io import load

from analytic import cases
from measurement import integer_exact_fields


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_xyz(vector, phi):
    r, p, z = np.asarray(vector).T
    return np.column_stack((r*np.cos(phi)-p*np.sin(phi), r*np.sin(phi)+p*np.cos(phi), z))


def evaluate(eq, nodes):
    data = eq.compute(["R", "Z", "B", "J", "grad(p)", "sqrt(g)"],
                      grid=Grid(nodes, sort=False), override_grid=True)
    phi = nodes[:, 2]
    R, Z = np.asarray(data["R"]), np.asarray(data["Z"])
    xyz = np.column_stack((R*np.cos(phi), R*np.sin(phi), Z))
    return (xyz, to_xyz(data["B"], phi), to_xyz(data["J"], phi),
            to_xyz(data["grad(p)"], phi), np.asarray(data["sqrt(g)"]))


def score(eq, case, nodes, weights=None):
    xyz, B, J, gp, jac = evaluate(eq, nodes)
    Bx, curl, gpx, _, label = integer_exact_fields(case, xyz)
    Jx, gpx = curl/mu_0, gpx/mu_0
    w = np.ones(len(nodes)) if weights is None else weights*np.abs(jac)

    def rel(a, b):
        return float(np.sqrt(np.sum(w*np.sum((a-b)**2, 1))/np.sum(w*np.sum(b*b, 1))))

    force = np.cross(J, B)-gp
    return {"points": int(len(nodes)), "B_relative_l2": rel(B, Bx), "J_relative_l2": rel(J, Jx),
            "gradp_relative_l2": rel(gp, gpx),
            "force_over_gradp": float(np.sqrt(np.sum(w*np.sum(force**2, 1))/np.sum(w*np.sum(gpx**2, 1)))),
            "flux_label_error_max": float(np.max(np.abs(label/case.edge-nodes[:, 0]**2))),
            "min_abs_sqrt_g": float(np.min(np.abs(jac)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equilibrium", type=Path, required=True)
    parser.add_argument("--case", default="integer_3d")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rings", type=float, nargs="+", default=(1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 0.7, 0.99))
    parser.add_argument("--angles", type=int, default=16)
    parser.add_argument("--volume", type=int, nargs=3, default=(16, 32, 32), metavar=("NR", "NT", "NZ"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output exists")
    desc.set_device("cpu")
    t0 = time.perf_counter()
    case = cases()[args.case]
    eq = load(str(args.equilibrium))
    n = args.angles
    th = 2*np.pi*np.arange(n)/n
    ze = 2*np.pi*np.arange(n)/(n*case.nfp)
    rings = {}
    for rho in args.rings:
        T, Zt = np.meshgrid(th, ze, indexing="ij")
        nodes = np.column_stack((np.full(T.size, rho), T.ravel(), Zt.ravel()))
        rings[f"{rho:g}"] = score(eq, case, nodes)
        print(rho, {k: "%.2e" % v for k, v in rings[f"{rho:g}"].items() if k != "points"}, flush=True)
    volume = {}
    nr, nt, nz = args.volume
    for rule in ("gauss", "midpoint"):
        if rule == "gauss":
            x, wr = leggauss(nr)
            r, wr = (x+1)/2, wr/2
        else:
            r, wr = (np.arange(nr)+0.5)/nr, np.full(nr, 1/nr)
        tt = 2*np.pi*np.arange(nt)/nt
        zz = 2*np.pi*np.arange(nz)/(nz*case.nfp)
        Rr, Tt, Zz = np.meshgrid(r, tt, zz, indexing="ij")
        W = np.broadcast_to(wr[:, None, None], Rr.shape).ravel()
        nodes = np.column_stack((Rr.ravel(), Tt.ravel(), Zz.ravel()))
        volume[rule] = score(eq, case, nodes, weights=W)
        print(rule, {k: "%.2e" % v for k, v in volume[rule].items() if k != "points"}, flush=True)
    record = {"schema": 1, "evidence": "desc_native_axis_and_volume_vs_exact",
              "case": args.case, "equilibrium": str(args.equilibrium),
              "equilibrium_sha256": sha256(args.equilibrium),
              "desc_version": desc.__version__, "jax": jax.__version__, "python": platform.python_version(),
              "script_sha256": sha256(__file__),
              "desc_resolution": {"L": int(eq.L), "M": int(eq.M), "N": int(eq.N), "NFP": int(eq.NFP)},
              "rings": rings, "volume": volume, "elapsed_seconds": time.perf_counter()-t0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
