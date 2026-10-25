"""One DESC solve of an exact case with every protocol factor explicit, scored on the exact chart.

Protocol factors (each a flag, so the sheared-A iteration-cap question can be isolated one at a time):

* ``--seed``      ``zero`` (lambda = 0, the cold protocol of ``desc_general_case``) or ``exact``
                  (lambda from the exact field: B^zeta sqrt(g)/psi' - 1 = lambda_theta and
                  iota - B^theta sqrt(g)/psi' = lambda_zeta, integrated per shell in Fourier space and
                  fitted in the Zernike basis; the published sheared-DESC seed, generalised to any family);
* ``--closure``   ``iota`` (prescribed iota, ours) or ``current`` (prescribed enclosed toroidal current
                  from the independent Ampere integral, iota free; the published choice);
* ``--optimizer`` ``lsq-exact`` (DESC dense trust region) or ``matrix-free`` (SciPy TRF/LSMR with exact
                  JVP/VJP and the published algebraic fixed-boundary elimination; stellarator-symmetric only);
* ``--grid``      ``default`` (DESC's ConcentricGrid of L_grid, M_grid, N_grid) or ``quadrature``
                  (QuadratureGrid(L = 2L + 2) with full angular resolution on every shell);
* ``--N`` ``--L`` toroidal and radial resolution (``2M`` is accepted for N);
* ``--tol``       ``ours`` (ftol 1e-10, xtol 1e-10, gtol 1e-8) or ``published`` (ftol 1e-8, xtol 1e-10,
                  gtol 1e-5 on the eliminated scaled coefficients).

Everything else (exact boundary from the chart, profile fits, initial R/Z from the exact chart) is the
unchanged ``desc_general_case.build``.  The record keeps the per-iteration solver table with wall
timestamps, the native score of ``desc_general_case`` and the exact-chart score of ``common_norms``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

import desc
from desc.grid import ConcentricGrid, LinearGrid, QuadratureGrid
from desc.profiles import PowerSeriesProfile
from desc.transform import Transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analytic import cases, field, iota, label_at_s, surface  # noqa: E402
from build_inputs import fit_profiles  # noqa: E402
import common_norms as cn  # noqa: E402
from run_log import Tee, json_safe, parse_table, sha256  # noqa: E402
import desc_general_case as dgc  # noqa: E402

TOLERANCES = {"ours": {"ftol": 1e-10, "xtol": 1e-10, "gtol": 1e-8},
              "published": {"ftol": 1e-8, "xtol": 1e-10, "gtol": 1e-5}}


def current_profile(case, M):
    """Enclosed toroidal current (A) as a power series in rho^2 from the independent Ampere integral."""
    _, _, polI, errors = fit_profiles(case)
    coef = np.asarray(polI.coef, float)/mu_0
    return PowerSeriesProfile(coef, modes=[2*k for k in range(len(coef))]), float(errors["enclosed_current"])


def exact_lambda(eq, case, L, M, N):
    """Fit lambda from the exact B on the initial-guess geometry (see module docstring)."""
    nfp = case.nfp
    grid = LinearGrid(rho=np.linspace(0.04, 1, 2*L+2), M=2*M+3, N=2*N+4, NFP=nfp, sym=False)
    nodes = np.asarray(grid.nodes)
    data = eq.compute(["R", "Z", "sqrt(g)", "e^theta", "e^zeta"], grid=grid)
    rho, theta, phi = nodes.T
    R, Z = np.asarray(data["R"]), np.asarray(data["Z"])
    xyz = np.column_stack((R*np.cos(phi), R*np.sin(phi), Z))
    B = np.asarray(jax.vmap(lambda x: field(case, x)[0])(jnp.asarray(xyz)))
    Bc = np.column_stack((B[:, 0]*np.cos(phi)+B[:, 1]*np.sin(phi), -B[:, 0]*np.sin(phi)+B[:, 1]*np.cos(phi), B[:, 2]))
    jac = np.asarray(data["sqrt(g)"])
    b_theta = np.sum(Bc*np.asarray(data["e^theta"]), 1)
    b_zeta = np.sum(Bc*np.asarray(data["e^zeta"]), 1)
    psi_r = float(eq.Psi)*rho/np.pi
    ru = np.unique(rho)
    iota_d = -np.asarray([float(iota(case, label_at_s(case, min(r*r, 1.0)))) for r in ru])
    iota_n = iota_d[np.searchsorted(ru, rho)]
    target_theta = b_zeta*jac/psi_r-1
    target_zeta = iota_n-b_theta*jac/psi_r
    tu, zu = np.unique(theta), np.unique(phi)
    it, ir, iz = np.searchsorted(tu, theta), np.searchsorted(ru, rho), np.searchsorted(zu, phi)
    shape = (len(tu), len(ru), len(zu))

    def cube(v):
        out = np.zeros(shape)
        out[it, ir, iz] = v
        return out
    ft = np.fft.fftn(cube(target_theta), axes=(0, 2))
    fz = np.fft.fftn(cube(target_zeta), axes=(0, 2))
    mm = np.fft.fftfreq(len(tu), 1/len(tu))[:, None, None]
    nn = nfp*np.fft.fftfreq(len(zu), 1/len(zu))[None, None, :]
    den = mm**2+nn**2
    coef = (-1j*mm*ft-1j*nn*fz)/np.where(den, den, 1)
    values = np.fft.ifftn(coef, axes=(0, 2)).real[it, ir, iz]
    # consistency of the gradient data: the part of (target_theta, target_zeta) a scalar cannot represent
    lam_t = np.fft.ifftn(1j*mm*coef, axes=(0, 2)).real[it, ir, iz]
    lam_z = np.fft.ifftn(1j*nn*coef, axes=(0, 2)).real[it, ir, iz]
    inconsistency = float(np.sqrt(np.sum((lam_t-target_theta)**2+(lam_z-target_zeta)**2)
                                  / np.sum(target_theta**2+target_zeta**2)))
    eq.L_lmn = Transform(grid, eq.L_basis, build=False, build_pinv=True).fit(values)
    return {"grid_nodes": int(len(nodes)), "gradient_inconsistency_relative": inconsistency,
            "lambda_max_abs": float(np.max(np.abs(values)))}


def fast_exact_rpz(case, nodes):
    """``desc_general_case.exact_rpz`` with the flux-label root found once per distinct rho (same values)."""
    rho, theta, zeta = nodes.T
    uniq, inverse = np.unique(rho, return_inverse=True)
    labels = np.asarray([float(label_at_s(case, min(max(r*r, 0.0), 1.0))) for r in uniq])[inverse]
    xyz = np.asarray(surface(case, labels, -theta, zeta), dtype=float)
    return np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]), np.arctan2(xyz[:, 1], xyz[:, 0]), xyz[:, 2]))


def solve_grid_for(eq, kind, L):
    if kind == "quadrature":
        return QuadratureGrid(L=2*L+2, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP)
    return ConcentricGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=eq.sym, axis=False)


def matrix_free_solve(eq, grid, tol, maxiter, lsmr_maxiter=80, lsmr_tol=1e-5):
    """Published TRF/LSMR with exact JVP/VJP and algebraic fixed-boundary elimination.

    Identical to ``upstream/landreman_sheared_desc.solve_fixed_boundary_matrix_free`` except that the
    guard requiring a prescribed current is dropped, so the closure can be varied independently.
    """
    from scipy.optimize import least_squares
    from scipy.sparse.linalg import LinearOperator
    from desc.objectives import ForceBalance
    sys.path.insert(0, str(Path(__file__).resolve().parent/"upstream"))
    import landreman_sheared_desc as up
    coordinates = up.FixedBoundaryCoordinates(eq, precondition=True)
    objective = ForceBalance(eq, grid=grid)
    objective.build(verbose=1)
    residual = jax.jit(lambda y: objective.compute_scaled_error(coordinates.recover(y)))
    jvp = jax.jit(lambda y, d: jax.jvp(residual, (y,), (d,))[1])
    vjp = jax.jit(lambda y, c: jax.vjp(residual, y)[1](c)[0])
    counts = {"residual": 0, "jvp": 0, "vjp": 0, "jacobian": 0}

    def fun(y):
        counts["residual"] += 1
        return np.asarray(residual(jnp.asarray(y)), dtype=float)

    def jac(y):
        counts["jacobian"] += 1
        state = jnp.asarray(y)

        def matvec(v):
            counts["jvp"] += 1
            return np.asarray(jvp(state, jnp.asarray(np.asarray(v).reshape(-1))), dtype=float)

        def rmatvec(w):
            counts["vjp"] += 1
            return np.asarray(vjp(state, jnp.asarray(np.asarray(w).reshape(-1))), dtype=float)
        return LinearOperator((objective.dim_f, coordinates.dim), matvec=matvec, rmatvec=rmatvec, dtype=np.float64)

    result = least_squares(fun, coordinates.y0, jac=jac, method="trf", tr_solver="lsmr", x_scale=1.0,
                           ftol=tol["ftol"], xtol=tol["xtol"], gtol=tol["gtol"], max_nfev=maxiter, verbose=2,
                           tr_options={"maxiter": lsmr_maxiter, "atol": lsmr_tol, "btol": lsmr_tol, "regularize": True})
    coordinates.apply(eq, result.x)
    return eq, {"success": bool(result.success), "message": str(result.message),
                "nit": max(int(result.njev)-1, 0), "nfev": int(result.nfev), "njev": int(result.njev),
                "cost": float(result.cost), "optimality": float(result.optimality),
                "reduced_dimension": coordinates.dim, "residual_dimension": int(objective.dim_f),
                "operator_evaluations": counts, "lsmr_maxiter": lsmr_maxiter, "lsmr_tol": lsmr_tol}


def environment():
    return {"host": socket.gethostname(), "loadavg": list(os.getloadavg()), "cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "desc": desc.__version__, "jax": jax.__version__,
            "python": platform.python_version(), "avail_mem_gb": desc.config.get("avail_mem")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", required=True)
    p.add_argument("--M", type=int, required=True)
    p.add_argument("--N", default=None, help="integer, or 2M; default M (0 for axisymmetric families)")
    p.add_argument("--L", type=int, default=None, help="default M+2")
    p.add_argument("--sym", choices=("auto", "true", "false"), default="auto")
    p.add_argument("--seed", choices=("zero", "exact"), default="zero")
    p.add_argument("--closure", choices=("iota", "current"), default="iota")
    p.add_argument("--optimizer", choices=("lsq-exact", "matrix-free"), default="lsq-exact")
    p.add_argument("--grid", choices=("default", "quadrature"), default="default")
    p.add_argument("--tol", choices=tuple(TOLERANCES), default="ours")
    p.add_argument("--maxiter", type=int, default=150)
    p.add_argument("--levels", nargs="*", default=["q2"], choices=list(cn.LEVELS))
    p.add_argument("--avail-mem-gb", type=float, default=None, help="cap DESC's Jacobian chunk heuristic (math unchanged)")
    p.add_argument("--label", default="")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--score-only", type=Path, help="score a saved .h5 on the exact chart; no solve")
    a = p.parse_args(argv)
    desc.set_device("cpu")
    if a.avail_mem_gb:
        desc.config["avail_mem"] = a.avail_mem_gb
    case = cases()[a.case]
    if a.score_only:
        from desc.io import load
        eq = load(str(a.score_only))
        scores, times = cn.score_desc(eq, case, levels=a.levels)
        record = {"schema": 3, "evidence": "desc_exact_chart_rescore", "case": a.case, "artifact": {"path": str(a.score_only),
                  "sha256": sha256(a.score_only)}, "resolution": {"L": int(eq.L), "M": int(eq.M), "N": int(eq.N), "sym": bool(eq.sym)},
                  "chart": scores, "chart_seconds": times, "environment": environment()}
        a.output_dir.mkdir(parents=True, exist_ok=True)
        (a.output_dir/"record_chart.json").write_text(json.dumps(json_safe(record), indent=2, allow_nan=False)+"\n")
        return
    axisymmetric = case.family == "solovev" or a.case.endswith("axisymmetric")
    N = (2*a.M if a.N == "2M" else int(a.N)) if a.N is not None else (0 if axisymmetric else a.M)
    L = a.L if a.L is not None else a.M+2
    sym = {"true": True, "false": False}.get(a.sym, not case.lasym)
    if a.optimizer == "matrix-free" and not sym:
        raise SystemExit("the matrix-free elimination needs a stellarator-symmetric state")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    timings, t_all = {}, time.perf_counter()
    protocol = {"seed": a.seed, "closure": a.closure, "optimizer": a.optimizer, "grid": a.grid, "tol": a.tol,
                "tolerances": TOLERANCES[a.tol], "maxiter": a.maxiter, "maxiter_unit": "nfev" if a.optimizer == "matrix-free" else "iterations",
                "avail_mem_gb": a.avail_mem_gb}
    env0 = environment()
    t0 = time.perf_counter()
    dgc.exact_rpz = fast_exact_rpz  # identical values, one root find per distinct rho
    eq, fit_errors = dgc.build(case, a.M, N, L, sym)
    extra = {}
    if a.closure == "current":
        profile, err = current_profile(case, a.M)
        eq.current = profile
        eq.iota = None
        extra["current_profile_fit_error"] = err
    timings["build"] = time.perf_counter()-t0
    t0 = time.perf_counter()
    if a.seed == "exact":
        extra["exact_seed"] = exact_lambda(eq, case, L, a.M, N)
    timings["seed"] = time.perf_counter()-t0
    grid = solve_grid_for(eq, a.grid, L)
    record = {"schema": 3, "evidence": "desc_matched_run", "case": a.case, "label": a.label, "protocol": protocol,
              "resolution": {"L": L, "M": a.M, "N": N, "NFP": case.nfp, "sym": sym},
              "solve_grid": {"kind": a.grid, "num_nodes": int(grid.num_nodes), "num_rho": int(grid.num_rho)},
              "profile_fit_errors": fit_errors, **extra, "desc_version": desc.__version__, "jax": jax.__version__,
              "script_sha256": sha256(__file__), "environment_start": env0}
    tee = Tee(sys.stdout)
    t_solve, cpu_solve = time.perf_counter(), time.process_time()
    with contextlib.redirect_stdout(tee):
        if a.optimizer == "matrix-free":
            eq, info = matrix_free_solve(eq, grid, TOLERANCES[a.tol], a.maxiter)
        else:
            from desc.objectives import ForceBalance, ObjectiveFunction
            objective = None
            if a.grid != "default":
                objective = ObjectiveFunction(ForceBalance(eq, grid=grid), jac_chunk_size="auto")
            kw = {"objective": objective} if objective is not None else {}
            eq, info = eq.solve(maxiter=a.maxiter, optimizer="lsq-exact", verbose=2, **TOLERANCES[a.tol], **kw)
    timings["solve"] = time.perf_counter()-t_solve
    timings["solve_cpu"] = time.process_time()-cpu_solve
    h5 = a.output_dir/f"desc_{a.case}_M{a.M}.h5"
    eq.save(str(h5))
    table = parse_table(tee.lines)
    solver = {"success": bool(info.get("success", False)), "message": str(info.get("message", "")),
              "iterations": int(info.get("nit", len(table)-1 if table else -1)), "nfev": int(info.get("nfev", table[-1]["nfev"] if table else -1)),
              "solve_seconds": timings["solve"], "solve_cpu_seconds": timings["solve_cpu"], "initial_cost": table[0]["cost"] if table else None,
              "final_cost": info.get("cost", table[-1]["cost"] if table else None),
              "final_optimality": info.get("optimality", table[-1]["optimality"] if table else None)}
    for key in ("njev", "operator_evaluations", "reduced_dimension", "residual_dimension"):
        if key in info:
            solver[key] = info[key]
    record["solver"] = solver
    record["trajectory"] = table
    t0 = time.perf_counter()
    record["native"] = dgc.score_everywhere(eq, case)
    timings["score_native"] = time.perf_counter()-t0
    t0 = time.perf_counter()
    record["chart"], record["chart_seconds"] = cn.score_desc(eq, case, levels=a.levels)
    timings["score_chart"] = time.perf_counter()-t0
    record["nested_final"] = bool(eq.is_nested())
    record["equilibrium_artifact"] = {"path": str(h5.resolve()), "sha256": sha256(h5), "bytes": h5.stat().st_size}
    timings["total"] = time.perf_counter()-t_all
    record["timings"] = timings
    record["environment_end"] = environment()
    (a.output_dir/"record.json").write_text(json.dumps(json_safe(record), indent=2, allow_nan=False)+"\n")
    c = record["chart"][a.levels[0]]["all"]
    print(f"{a.case} M{a.M} {a.label}: B {c['B']:.2e} J {c['J']:.2e} F {c['force']:.2e} its {solver['iterations']} "
          f"success {solver['success']} solve {timings['solve']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
