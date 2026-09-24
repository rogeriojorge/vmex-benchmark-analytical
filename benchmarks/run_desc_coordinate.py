"""Run one matched analytical-coordinate case through native DESC.

This driver samples the exact integer-family boundary, optionally relabels its
interior poloidal coordinate with the compensating DESC lambda, solves with
the DESC force-balance objective, and scores at fixed Cartesian points shared
with the VMEX record. Projection and solved results are saved separately.
"""
import argparse
import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import scipy
from scipy.constants import mu_0

import desc
desc.set_device("gpu", gpuid=0)
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.profiles import PowerSeriesProfile

from analytic import cases, integer_field, surface
from score_samples import score


def cylindrical(xyz):
    return np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]),
                            np.arctan2(xyz[:, 1], xyz[:, 0]), xyz[:, 2]))


def vector_to_xyz(vector, phi):
    r, p, z = vector.T
    return np.column_stack((r*np.cos(phi)-p*np.sin(phi),
                            r*np.sin(phi)+p*np.cos(phi), z))


def desc_angle(case, xyz):
    """Recover the exact DESC poloidal label from the integer field-line labels."""
    a, b, c, _ = case.parameters
    B, psi = integer_field(xyz, a, b, c)
    x, y = xyz[:, 0]/a, xyz[:, 1]/b
    bx, by = np.asarray(B)[:, 0], np.asarray(B)[:, 1]
    u = (x*x+(bx/a)**2-1)/2
    v = (x*y+bx*by/(a*b))/2
    u0 = -(a*a-b*b)/(4*c*c)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    theta = 2*phi-np.arctan2(v, u-u0)
    rho = np.sqrt(np.maximum(np.asarray(psi), 0)/case.edge)
    return rho, theta, phi


def remap(theta_old, rho, phi, amplitude, m, n, nfp):
    """Solve theta_new=theta_old+u(theta_new), return new angle and lambda=-u."""
    s = rho*rho
    theta_new = np.asarray(theta_old).copy()
    for _ in range(12):
        u = amplitude*s*(1-s)*np.sin(m*theta_new-n*nfp*phi)
        theta_new = theta_old+u
    u = amplitude*s*(1-s)*np.sin(m*theta_new-n*nfp*phi)
    return theta_new, -u


def mapping(case, nodes, mode, amplitude, m, n):
    rho, theta, phi = nodes.T
    theta_geometry = theta.copy()
    if mode == "remapped":
        theta_new, lam = remap(theta, rho, phi, amplitude, m, n, case.nfp)
        theta, lam = theta_new, lam
    else:
        lam = np.zeros_like(rho)
    label = case.edge*rho*rho
    theta_ccw = -theta_geometry
    xyz = surface(case, label, theta_ccw, phi)
    rpz = cylindrical(xyz)
    return rpz, lam


def build(case, resolution, mode, amplitude, m_shift, n_shift):
    M = N = resolution
    L = resolution+2
    bg = LinearGrid(rho=[1.0], M=2*M, N=2*N, NFP=case.nfp,
                    sym=False)
    bnodes = np.asarray(bg.nodes)
    brpz, _ = mapping(case, bnodes, mode, amplitude, m_shift, n_shift)
    boundary = FourierRZToroidalSurface.from_values(
        brpz, bnodes[:, 1], M=M, N=N, NFP=case.nfp, sym=True)
    a, b, c, delta = case.parameters
    axis_pressure = -case.pressure_slope*delta
    eq = Equilibrium(
        L=L, M=M, N=N, NFP=case.nfp, sym=True, surface=boundary,
        L_grid=2*L, M_grid=2*M, N_grid=2*N,
        Psi=np.pi*a*b*c*delta,
        pressure=PowerSeriesProfile(
            [axis_pressure/mu_0, -axis_pressure/mu_0], modes=[0, 2]),
        iota=PowerSeriesProfile([2.0]), ensure_nested=False)
    fg = LinearGrid(rho=np.linspace(0.025, 1, L+2), M=2*M, N=2*N,
                    NFP=case.nfp, sym=False)
    fnodes = np.asarray(fg.nodes)
    frpz, flam = mapping(case, fnodes, mode, amplitude, m_shift, n_shift)
    desc_nodes = fnodes.copy()
    if mode == "remapped":
        desc_nodes[:, 1], flam = remap(fnodes[:, 1], fnodes[:, 0], fnodes[:, 2],
                                       amplitude, m_shift, n_shift, case.nfp)
    eq.set_initial_guess(desc_nodes, frpz[:, 0], frpz[:, 2], flam,
                         ensure_nested=False)
    eq.axis = eq.get_axis()
    return eq


def evaluate(eq, nodes, names):
    # Let DESC evaluate surface-integral dependencies on its proper quadrature
    # grids, then transfer them to these held-out points.
    data = eq.compute(names, grid=Grid(nodes, sort=False), override_grid=True)
    return {key: np.asarray(data[key]) for key in names}


def invert_points(eq, case, target_xyz, mode, amplitude, m_shift, n_shift):
    target = cylindrical(target_xyz)
    rho, theta, phi = desc_angle(case, target_xyz)
    if mode == "remapped":
        theta, _ = remap(theta, rho, phi, amplitude, m_shift, n_shift,
                         case.nfp)
    nodes = np.column_stack((rho, theta, phi))
    for _ in range(18):
        d = evaluate(eq, nodes, ["R", "Z", "R_r", "R_t", "Z_r", "Z_t"])
        dr, dz = target[:, 0]-d["R"], target[:, 2]-d["Z"]
        det = d["R_r"]*d["Z_t"]-d["R_t"]*d["Z_r"]
        if not np.all(np.isfinite(det)) or np.min(np.abs(det)) < 1e-12:
            raise RuntimeError("DESC coordinate inversion encountered a singular Jacobian")
        d_rho = (dr*d["Z_t"]-d["R_t"]*dz)/det
        d_theta = (d["R_r"]*dz-dr*d["Z_r"])/det
        nodes[:, 0] += d_rho
        nodes[:, 1] += d_theta
        if np.max(np.hypot(dr, dz)) < 2e-11:
            break
    final = evaluate(eq, nodes, ["R", "Z"])
    err = np.hypot(target[:, 0]-final["R"], target[:, 2]-final["Z"])
    if np.max(err) > 2e-8 or np.any(nodes[:, 0] <= 0) or np.any(nodes[:, 0] > 1+1e-8):
        raise RuntimeError(f"DESC held-out coordinate inversion failed: max={np.max(err):.3e}")
    return nodes, err


def spectral_width(eq, nfp):
    rho = np.linspace(0.15, 0.95, 7)
    theta = np.arange(64)*2*np.pi/64
    zeta = np.arange(32)*2*np.pi/(32*nfp)
    nodes = np.array(np.meshgrid(rho, theta, zeta, indexing="ij")).reshape(3, -1).T
    d = evaluate(eq, nodes, ["R", "Z"])
    nr, nt, nz = len(rho), len(theta), len(zeta)
    m = np.fft.fftfreq(nt, 1/nt)
    num = den = 0.0
    for key in ("R", "Z"):
        coeff = np.fft.fft2(d[key].reshape(nr, nt, nz), axes=(1, 2))/(nt*nz)
        mag = np.abs(coeff)**2
        num += float(np.sum(m[None, :, None]**2*mag))
        den += float(np.sum(mag))
    return float(np.sqrt(num/den))


def sample_score(eq, reference, args):
    nodes, coord_err = invert_points(
        eq, cases()[args.case], reference["xyz"], args.chart,
        args.amplitude, args.m, args.n)
    # Pointwise scoring uses an unsorted custom Cartesian-matched grid. DESC's
    # iota diagnostic is the prescribed profile in this setup, so it is not
    # reported as an independent transform measurement.
    d = evaluate(eq, nodes, ["B", "J", "grad(p)", "sqrt(g)"])
    phi = nodes[:, 2]
    B = vector_to_xyz(d["B"], phi)
    J = vector_to_xyz(d["J"], phi)
    gradp = vector_to_xyz(d["grad(p)"], phi)
    result_data = dict(reference)
    result_data["B"], result_data["J"], result_data["gradp"] = B, J, gradp
    result_data["xyz"] = reference["xyz"]
    result = score(result_data)
    result["max_coordinate_inversion_m"] = float(np.max(coord_err))
    result["min_sampled_jacobian"] = float(np.min(d["sqrt(g)"]))
    result["max_force_residual_Pa_per_m"] = float(np.max(
        np.linalg.norm(np.cross(J, B)-gradp, axis=1)))
    return result, result_data


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chart", choices=("base", "remapped"), required=True)
    parser.add_argument("--case", choices=("integer_axisymmetric", "integer_3d"),
                        default="integer_3d")
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--n", type=int, default=0)
    args = parser.parse_args()
    case = cases()[args.case]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.samples, allow_pickle=False) as data:
        reference = {key: data[key] for key in data.files}
    t0 = time.perf_counter()
    eq = build(case, args.resolution, args.chart, args.amplitude, args.m, args.n)
    if not eq.is_nested():
        raise RuntimeError("DESC projected initial guess is not nested")
    projected, projected_data = sample_score(eq, reference, args)
    result_path = args.output_dir/f"desc_integer_{args.chart}_L{args.resolution}.json"
    projected_path = args.output_dir/f"desc_integer_{args.chart}_L{args.resolution}_projection.npz"
    np.savez_compressed(projected_path, **projected_data)
    report = dict(
        schema=1, evidence="desc_coordinate_comparison", status="projected_only",
        chart=args.chart, parameters=dict(case=args.case, amplitude=args.amplitude,
                                           m=args.m, n=args.n, resolution=args.resolution,
                                           nfp=case.nfp, prescribed_iota=2.0,
                                           maxiter=args.maxiter),
        desc_version=desc.__version__, scipy_version=scipy.__version__,
        numpy_version=np.__version__, jax_version=jax.__version__,
        jax_devices=[str(d) for d in jax.devices()],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sample_sha256=sha256(args.samples), projected=projected,
        projected_artifact=str(projected_path.name),
        spectral_width_projected=spectral_width(eq, case.nfp),
        elapsed_projection_s=time.perf_counter()-t0)
    t1 = time.perf_counter()
    eq, info = eq.solve(ftol=1e-10, xtol=1e-10, gtol=1e-8,
                        maxiter=args.maxiter, verbose=1)
    solved, solved_data = sample_score(eq, reference, args)
    solved_path = args.output_dir/f"desc_integer_{args.chart}_L{args.resolution}_solved.npz"
    np.savez_compressed(solved_path, **solved_data)
    h5_path = args.output_dir/f"desc_integer_{args.chart}_L{args.resolution}.h5"
    eq.save(str(h5_path))
    report.update(
        status="solved" if info.get("success", False) else "solver_not_converged",
        solver={"success": bool(info.get("success", False)),
                "message": str(info.get("message", "")),
                "iterations": int(info.get("nit", -1)),
                "cost": finite_or_none(info.get("cost", np.nan)),
                "optimality": finite_or_none(info.get("optimality", np.nan))},
        solved=solved, solved_artifact=str(solved_path.name),
        equilibrium_artifact=str(h5_path.name),
        spectral_width_solved=spectral_width(eq, case.nfp),
        elapsed_solve_s=time.perf_counter()-t1,
        elapsed_total_s=time.perf_counter()-t0,
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if not info.get("success", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
