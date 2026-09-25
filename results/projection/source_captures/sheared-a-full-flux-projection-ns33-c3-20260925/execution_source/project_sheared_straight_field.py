"""Derive and project a straight-field lambda map for the sheared-A family.

The exact surfaces use a physical cylindrical toroidal angle and a geometric
poloidal angle. This diagnostic solves the periodic field-line equation for
the coordinate correction, checks it on a finer grid, and projects the result
into the pinned VMEX state basis. It is a projection, not a recovered root.
"""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy.signal import resample
from scipy.sparse.linalg import LinearOperator, lsmr
import vmex
from vmex.core.fourier import mode_table, trig_tables
from vmex.core.residuals import m1_physical_to_constrained
from vmex.core.solver import SpectralState, evaluate_forces, prepare_runtime, resolution_from_input
from vmex.core.transforms import physical_to_internal_scale

from analytic import ROOT, cases, field, iota, label_at_s, surface
from evidence import reserve_run_directory, sha256_file, source_metadata, write_json
from native_samples import sample_native, split_samples
from score_samples import score

PIN = "b5f5267efc0795c4a49a224e321e9b370975c14c"
INPUT = ROOT / "inputs/input.sheared_A_current"
OUTPUT_PARENT = ROOT / "results/projection/sheared_A"
SOLVE_GRID = 48
CHECK_GRID = 64
MAXITER = 5000


def _derivative(values, axis, period):
    n = values.shape[axis]
    wave = 2 * np.pi * np.fft.fftfreq(n, d=period / n)
    shape = [1] * values.ndim
    shape[axis] = n
    return np.fft.ifft(
        np.fft.fft(values, axis=axis) * (1j * wave.reshape(shape)), axis=axis
    ).real


def _surface_rates(case, label, n):
    """Return a=dtheta/dphi and tangent-fit diagnostics on one exact surface."""
    theta = 2 * np.pi * np.arange(n) / n
    phi = (2 * np.pi / case.nfp) * np.arange(n) / n
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    xyz = np.asarray(surface(case, label, TH, PH), dtype=np.float64)
    major = np.hypot(xyz[..., 0], xyz[..., 1])
    height = xyz[..., 2]

    # R and Z are periodic over one field period. Cartesian x,y are not.
    Rtheta = _derivative(major, 0, 2 * np.pi)
    Ztheta = _derivative(height, 0, 2 * np.pi)
    Rphi = _derivative(major, 1, 2 * np.pi / case.nfp)
    Zphi = _derivative(height, 1, 2 * np.pi / case.nfp)
    eR = np.stack((np.cos(PH), np.sin(PH), np.zeros_like(PH)), axis=-1)
    ephi = np.stack((-np.sin(PH), np.cos(PH), np.zeros_like(PH)), axis=-1)
    ez = np.zeros_like(eR)
    ez[..., 2] = 1.0
    x_theta = Rtheta[..., None] * eR + Ztheta[..., None] * ez
    x_phi = Rphi[..., None] * eR + major[..., None] * ephi + Zphi[..., None] * ez
    B = np.asarray(field(case, xyz)[0], dtype=np.float64)

    g11 = np.sum(x_theta * x_theta, axis=-1)
    g12 = np.sum(x_theta * x_phi, axis=-1)
    g22 = np.sum(x_phi * x_phi, axis=-1)
    b1 = np.sum(x_theta * B, axis=-1)
    b2 = np.sum(x_phi * B, axis=-1)
    determinant = g11 * g22 - g12 * g12
    if np.min(determinant) <= 0 or not np.isfinite(determinant).all():
        raise FloatingPointError("exact surface has a nonpositive coordinate metric")
    Btheta = (b1 * g22 - b2 * g12) / determinant
    Bphi = (b2 * g11 - b1 * g12) / determinant
    if np.min(np.abs(Bphi)) < 1e-12:
        raise FloatingPointError("toroidal contravariant field crosses zero")
    a = Btheta / Bphi
    tangent = x_theta * Btheta[..., None] + x_phi * Bphi[..., None]
    tangent_error = np.linalg.norm(B - tangent) / np.linalg.norm(B)
    return TH, PH, a, {
        "tangent_relative_l2": float(tangent_error),
        "minimum_metric_determinant": float(np.min(determinant)),
        "minimum_abs_Bphi_coordinate": float(np.min(np.abs(Bphi))),
        "a_min": float(np.min(a)),
        "a_max": float(np.max(a)),
        "a_mean": float(np.mean(a)),
    }


def _solve_lambda(a, transform, *, atol=1e-12, maxiter=MAXITER):
    """Solve Dphi(lambda)+a*Dtheta(lambda)=iota-a with zero-mean gauge."""
    n = a.shape[0]
    if a.shape != (n, n):
        raise ValueError("field-line grid must be square")
    rhs = transform - a
    size = n * n

    def matvec(vector):
        lam = vector.reshape(n, n)
        value = _derivative(lam, 1, 2 * np.pi / 2) + a * _derivative(lam, 0, 2 * np.pi)
        return np.concatenate((value.ravel(), np.asarray([lam.mean()])))

    def rmatvec(vector):
        dual = vector[:-1].reshape(n, n)
        gauge = vector[-1]
        value = -_derivative(dual, 1, 2 * np.pi / 2)
        value -= _derivative(a * dual, 0, 2 * np.pi)
        value += gauge / size
        return value.ravel()

    operator = LinearOperator((size + 1, size), matvec=matvec, rmatvec=rmatvec,
                              dtype=np.float64)
    target = np.concatenate((rhs.ravel(), np.asarray([0.0])))
    result = lsmr(operator, target, atol=atol, btol=atol, conlim=1e12,
                  maxiter=maxiter)
    lam = result[0].reshape(n, n)
    defect = (_derivative(lam, 1, np.pi) + a * _derivative(lam, 0, 2 * np.pi)
              - rhs)
    return lam, {
        "istop": int(result[1]),
        "iterations": int(result[2]),
        "augmented_residual_norm": float(result[3]),
        "condition_estimate": float(result[6]),
        "mean_lambda": float(np.mean(lam)),
        "straightness_relative_l2": float(np.linalg.norm(defect) / np.linalg.norm(rhs)),
        "straightness_rms": float(np.linalg.norm(defect) / np.sqrt(size)),
        "straightness_max": float(np.max(np.abs(defect))),
    }


def _coefficients(values, modes):
    n = values.shape[0]
    spectrum = np.fft.fft2(values) / (n * n)
    cosine, sine = np.zeros(len(modes.m)), np.zeros(len(modes.m))
    for k, (m, toroidal_mode) in enumerate(zip(modes.m, modes.n)):
        m, toroidal_mode = int(m), int(toroidal_mode)
        factor = 1 if m == 0 and toroidal_mode == 0 else 2
        value = spectrum[m % n, (-toroidal_mode) % n]
        cosine[k] = factor * value.real
        sine[k] = -factor * value.imag
    return cosine, sine


def _fit_error(values, cosine, sine, modes, nfp):
    n = values.shape[0]
    theta = 2 * np.pi * np.arange(n) / n
    phi = (2 * np.pi / nfp) * np.arange(n) / n
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    fitted = np.zeros_like(values)
    for k, (m, toroidal_mode) in enumerate(zip(modes.m, modes.n)):
        angle = int(m) * TH - int(toroidal_mode) * nfp * PH
        fitted += cosine[k] * np.cos(angle) + sine[k] * np.sin(angle)
    difference = fitted - values
    return {
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(values)),
        "rms": float(np.linalg.norm(difference) / np.sqrt(values.size)),
        "max_abs": float(np.max(np.abs(difference))),
        "fitted_rms": float(np.linalg.norm(fitted) / np.sqrt(fitted.size)),
    }


def _rss_mib():
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-method", choices=("transport", "full_flux"), default="transport",
                        help="field-line LSMR (historical default) or the full-flux potential (plan eq. 8)")
    parser.add_argument("--flux-grid", type=int, default=49, help="odd grid for --lambda-method full_flux")
    parser.add_argument("--ns", type=int, default=17)
    args = parser.parse_args(argv)
    full_flux = args.lambda_method == "full_flux"
    if full_flux:
        from flux_lambda import exact_surface_lambda
    started = perf_counter()
    imported = source_metadata(vmex.__file__, "uwplasma/vmex",
                              importlib.metadata.version("vmex"))
    if imported.get("commit") != PIN:
        raise SystemExit("imported VMEX source differs from historical benchmark pin")
    inp = vmex.VmecInput.from_file(INPUT)
    case = cases()["sheared_A"]
    resolution = resolution_from_input(inp, ns=args.ns)
    runtime = prepare_runtime(inp, resolution)
    ns = int(resolution.ns)
    modes = mode_table(resolution.mpol, resolution.ntor)
    mode_scale = np.asarray(physical_to_internal_scale(modes, trig_tables(resolution)))
    s_full = np.asarray(runtime.setup.s_full, dtype=np.float64)
    phipf = np.asarray(runtime.setup.phipf, dtype=np.float64)
    lamscale = float(np.asarray(runtime.setup.lamscale))

    # Independent field-line solve convergence at a representative flux surface.
    convergence = []
    representative_s = 0.5
    representative_label = float(label_at_s(case, representative_s))
    representative_transform = float(iota(case, representative_label))
    for n in (24, 32, SOLVE_GRID):
        _, _, a, tangent = _surface_rates(case, representative_label, n)
        lam, solve = _solve_lambda(a, representative_transform)
        convergence.append({"grid": n, "tangent": tangent, "solve": solve,
                            "lambda_rms": float(np.linalg.norm(lam) / np.sqrt(lam.size))})

    grid = args.flux_grid if full_flux else SOLVE_GRID
    full_map = np.zeros((ns, grid, grid), dtype=np.float64)
    map_rows = []
    ext_cos = np.zeros((ns, len(modes.m)), dtype=np.float64)
    ext_sin = np.zeros_like(ext_cos)
    cosine_projection = np.zeros_like(ext_cos)
    for j, normalized_flux in enumerate(s_full):
        s_value = float(normalized_flux)
        if s_value <= 0:
            map_rows.append({"radial_index": j, "s": s_value,
                             "status": "axis_coefficient_closure"})
            continue
        label = float(label_at_s(case, s_value))
        transform = float(iota(case, label))
        TH, PH, a, tangent = _surface_rates(case, label, grid)
        if full_flux:
            lam, solve, _ = exact_surface_lambda(case, s_value, grid)
        else:
            lam, solve = _solve_lambda(a, transform)
        full_map[j] = lam
        cosine_projection[j], ext_sin[j] = _coefficients(lam, modes)
        # A stellarator-symmetric state has only sine lambda coefficients.
        ext_cos[j] = cosine_projection[j]
        fit = _fit_error(lam, ext_cos[j], ext_sin[j], modes, case.nfp)
        map_rows.append({"radial_index": j, "s": s_value,
                         "analytic_label": label, "iota": transform,
                         "tangent": tangent, "solve": solve,
                         "fourier_fit": fit})

    # VMEC's symmetric lambda closure copies m=0,n>0 to the axis row.
    m = np.asarray(modes.m, dtype=int)
    nmode = np.asarray(modes.n, dtype=int)
    axis_copy = (m == 0) & (nmode > 0)
    if ns > 1:
        ext_sin[0, axis_copy] = ext_sin[1, axis_copy]
        ext_cos[0, axis_copy] = ext_cos[1, axis_copy]

    # Physical WOUT lambda coefficients convert to state coefficients by
    # phipf/lamscale and VMEC's mode scaling (virtual_casing.py/wout.py).
    state_lambda_sin = (ext_sin * phipf[:, None] / lamscale
                        * mode_scale[None, :])
    state_lambda_cos = (ext_cos * phipf[:, None] / lamscale
                        * mode_scale[None, :])

    theta = 2 * np.pi * np.arange(CHECK_GRID) / CHECK_GRID
    phi = (2 * np.pi / case.nfp) * np.arange(CHECK_GRID) / CHECK_GRID
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    coefficients = np.zeros((4, ns, len(modes.m)), dtype=np.float64)
    native_fit_rows = []
    for j, normalized_flux in enumerate(s_full):
        label = float(label_at_s(case, float(normalized_flux)))
        xyz = np.asarray(surface(case, label, TH, PH), dtype=np.float64)
        major = np.hypot(xyz[..., 0], xyz[..., 1])
        spectra = [np.fft.fft2(values) / (CHECK_GRID**2)
                   for values in (major, xyz[..., 2])]
        for k, (mm, nn) in enumerate(zip(modes.m, modes.n)):
            mm, nn = int(mm), int(nn)
            factor = 1 if mm == 0 and nn == 0 else 2
            for offset, spectrum in ((0, spectra[0]), (2, spectra[1])):
                value = spectrum[mm, -nn]
                coefficients[offset, j, k] = factor * value.real
                coefficients[offset + 1, j, k] = -factor * value.imag
        native_fit_rows.append({"radial_index": j, "s": float(normalized_flux),
                                "geometry_grid": CHECK_GRID})

    R_cos, R_sin, Z_cos, Z_sin = (coefficients[q] * mode_scale[None, :]
                                   for q in range(4))
    R_cos, Z_sin, R_sin, Z_cos = m1_physical_to_constrained(
        R_cos, Z_sin, R_sin, Z_cos, modes=modes,
        lthreed=bool(resolution.lthreed), lasym=bool(resolution.lasym),
        lconm1=True)
    state = SpectralState(
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        L_cos=state_lambda_cos, L_sin=state_lambda_sin)
    gc, residuals, diagnostics = evaluate_forces(state, runtime)
    gc = jax.block_until_ready(gc)
    residual_norms = {name: float(np.linalg.norm(np.asarray(getattr(gc, name))))
                      for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")}
    residual_norms["all_blocks_l2"] = float(np.sqrt(sum(v * v for v in residual_norms.values())))
    invariants = {name: float(np.asarray(getattr(residuals, name)))
                  for name in ("fsqr", "fsqz", "fsql", "fedge", "gcr2", "gcz2", "gcl2")}

    samples = sample_native(inp, state, case, runtime=runtime)
    score_record = score(samples)
    reference, observations = split_samples(samples)
    prefix = "sheared-A-full-flux" if full_flux else "sheared-A-straight-field"
    run_id = datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%dT%H%M%S.%fZ")
    run_id, out = reserve_run_directory(OUTPUT_PARENT, run_id)
    seed_path = out / f"seed_ns{args.ns}.npz"
    np.savez_compressed(seed_path, **{
        name: np.asarray(getattr(state, name)) for name in
        ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    map_path = out / "straight_field_map.npz"
    np.savez_compressed(map_path, s_full=s_full, lambda_grid=full_map,
                        lambda_cos_coefficients=ext_cos,
                        lambda_sin_coefficients=ext_sin,
                        state_L_cos=state_lambda_cos,
                        state_L_sin=state_lambda_sin)
    reference_path, observation_path = out / "reference_samples.npz", out / "native_observations.npz"
    np.savez_compressed(reference_path, **reference)
    np.savez_compressed(observation_path, **observations)

    heldout = []
    for s_value in (0.25, 0.65, 0.94):
        label = float(label_at_s(case, s_value))
        transform = float(iota(case, label))
        _, _, a_hi, tangent_hi = _surface_rates(case, label, CHECK_GRID)
        if full_flux:
            lam_src, _, _ = exact_surface_lambda(case, s_value, grid)
        else:
            _, _, a_src, _ = _surface_rates(case, label, SOLVE_GRID)
            lam_src, _ = _solve_lambda(a_src, transform)
        lam_hi = resample(resample(lam_src, CHECK_GRID, axis=0), CHECK_GRID, axis=1)
        defect = (_derivative(lam_hi, 1, 2 * np.pi / case.nfp)
                  + a_hi * _derivative(lam_hi, 0, 2 * np.pi) - (transform - a_hi))
        cosine, sine = _coefficients(lam_src, modes)
        fit_error = _fit_error(lam_hi, cosine, sine, modes, case.nfp)
        heldout.append({"s": s_value, "tangent": tangent_hi,
                        "straightness_relative_l2": float(np.linalg.norm(defect)
                                                           / np.linalg.norm(transform-a_hi)),
                        "straightness_rms": float(np.linalg.norm(defect)
                                                   / np.sqrt(defect.size)),
                        "projected_fourier_fit": fit_error})

    report_path = out / "straight_field_projection.json"
    report = {
        "schema": 1,
        "evidence": ("exact_surface_full_flux_lambda_projection" if full_flux
                     else "exact_surface_fieldline_lambda_projection"),
        "lambda_method": args.lambda_method,
        "status": "projected_diagnostic",
        "accepted_recovery": False,
        "root_certified": False,
        "measurement_resolved": False,
        "source": source_metadata(vmex.__file__, "uwplasma/vmex",
                                  importlib.metadata.version("vmex")),
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/project_sheared_straight_field.py",
        "input": INPUT.relative_to(ROOT).as_posix(),
        "input_sha256": sha256_file(INPUT),
        "case": case.name,
        "vmec_resolution": {"ns": ns, "mpol": int(resolution.mpol),
                            "ntor": int(resolution.ntor), "nfp": int(resolution.nfp)},
        "closure": "current_prescribed",
        "toroidal_chart_period": 2 * np.pi / case.nfp,
        "lambda_gauge": "zero area mean on each exact surface",
        "lambda_map_method": "least-squares periodic solve of Dphi(lambda)+a*Dtheta(lambda)=iota-a",
        "lambda_internal_conversion": "physical_lambda * phipf / lamscale * physical_to_internal_scale",
        "solver_grid": grid,
        "heldout_grid": CHECK_GRID,
        "lsmr_maxiter": MAXITER,
        "fieldline_convergence": convergence,
        "radial_maps": map_rows,
        "heldout_validation": heldout,
        "native_fieldline_projection": native_fit_rows,
        "lambda_cosine_to_sine_coefficient_l2": float(
            np.linalg.norm(ext_cos) / max(np.linalg.norm(ext_sin), 1e-300)),
        "preconditioned_force_l2_by_block": residual_norms,
        "invariant_residuals": invariants,
        "native_score": score_record,
        "native_sample_count": int(samples["xyz"].shape[0]),
        "artifacts": {
            "seed": {"path": seed_path.name, "sha256": sha256_file(seed_path)},
            "lambda_map": {"path": map_path.name, "sha256": sha256_file(map_path)},
            "reference_samples": {"path": reference_path.name,
                                  "sha256": sha256_file(reference_path)},
            "native_observations": {"path": observation_path.name,
                                    "sha256": sha256_file(observation_path)},
        },
        "elapsed_seconds": perf_counter() - started,
        "host_peak_rss_mib": _rss_mib(),
        "environment": {"python": sys.version.split()[0], "jax": jax.__version__,
                        "numpy": np.__version__,
                        "devices": [str(device) for device in jax.devices()]},
    }
    write_json(report_path, report, exclusive=True)
    print(json.dumps({"run_id": run_id, "report": report_path.relative_to(ROOT).as_posix(),
                      "report_sha256": sha256_file(report_path),
                      "score": score_record, "residuals": invariants,
                      "lambda_symmetry_ratio": report["lambda_cosine_to_sine_coefficient_l2"],
                      "heldout": heldout, "elapsed_seconds": report["elapsed_seconds"],
                      "host_peak_rss_mib": report["host_peak_rss_mib"]}, indent=2))


if __name__ == "__main__":
    main()
