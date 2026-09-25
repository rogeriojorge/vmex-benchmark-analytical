"""Project the sheared-A exact surfaces onto the VMEX native state basis.

The projected state sets lambda to zero. It is a geometry projection only;
its magnetic-field and force residuals are measured and must not be treated as
an exact equilibrium seed or accepted recovery.
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
import vmex
from vmex.core.fourier import mode_table, trig_tables
from vmex.core.residuals import m1_physical_to_constrained
from vmex.core.solver import (
    SpectralState, evaluate_forces, prepare_runtime, resolution_from_input,
)
from vmex.core.transforms import physical_to_internal_scale

from analytic import ROOT, cases, label_at_s, surface
from build_inputs import FIELD_T, LENGTH_M, MU0
from evidence import reserve_run_directory, sha256_file, source_metadata, write_json
from native_samples import sample_native, split_samples
from score_samples import score

PIN = "b5f5267efc0795c4a49a224e321e9b370975c14c"
INPUT = ROOT/"inputs/input.sheared_A_current"
OUTPUT_PARENT = ROOT/"results/projection/sheared_A"
ANGULAR_GRID = 64


def _norm(state, name):
    return float(np.linalg.norm(np.asarray(getattr(state, name))))


def _array_sha(path):
    return sha256_file(path)


def main():
    started = perf_counter()
    source = source_metadata(vmex.__file__, "uwplasma/vmex",
                             importlib.metadata.version("vmex"))
    if source.get("commit") != PIN:
        raise SystemExit("imported VMEX source differs from historical benchmark pin")
    input_hash = sha256_file(INPUT)
    run_id = datetime.now(timezone.utc).strftime("sheared-A-projection-%Y%m%dT%H%M%S.%fZ")
    run_id, out = reserve_run_directory(OUTPUT_PARENT, run_id)
    inp = vmex.VmecInput.from_file(INPUT)
    case = cases()["sheared_A"]
    resolution = resolution_from_input(inp, ns=17)
    runtime = prepare_runtime(inp, resolution)
    ns = int(resolution.ns)
    modes = mode_table(resolution.mpol, resolution.ntor)
    scale = np.asarray(physical_to_internal_scale(modes, trig_tables(resolution)))
    s_full = np.asarray(runtime.setup.s_full, dtype=float)
    theta = 2*np.pi*np.arange(ANGULAR_GRID)/ANGULAR_GRID
    phi = 2*np.pi*np.arange(ANGULAR_GRID)/(ANGULAR_GRID*case.nfp)
    th_grid, ph_grid = np.meshgrid(theta, phi, indexing="ij")
    coefficients = np.zeros((4, ns, len(modes.m)), dtype=np.float64)
    max_surface_nonfinite = 0
    for j, normalized_flux in enumerate(s_full):
        label = float(label_at_s(case, normalized_flux))
        xyz = np.asarray(surface(case, label, th_grid, ph_grid), dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            max_surface_nonfinite += int(np.size(xyz)-np.count_nonzero(np.isfinite(xyz)))
            raise FloatingPointError(f"nonfinite exact surface at radial index {j}")
        major = np.hypot(xyz[..., 0], xyz[..., 1])
        spectra = [np.fft.fft2(values)/(ANGULAR_GRID**2)
                   for values in (major, xyz[..., 2])]
        for k, (m, n) in enumerate(zip(modes.m, modes.n)):
            m, n = int(m), int(n)
            factor = 1 if m == 0 and n == 0 else 2
            for offset, spectrum in ((0, spectra[0]), (2, spectra[1])):
                value = spectrum[m, -n]
                coefficients[offset, j, k] = factor*value.real
                coefficients[offset+1, j, k] = -factor*value.imag
    R_cos = coefficients[0]*scale[None, :]
    R_sin = coefficients[1]*scale[None, :]
    Z_cos = coefficients[2]*scale[None, :]
    Z_sin = coefficients[3]*scale[None, :]
    R_cos, Z_sin, R_sin, Z_cos = m1_physical_to_constrained(
        R_cos, Z_sin, R_sin, Z_cos, modes=modes,
        lthreed=bool(resolution.lthreed), lasym=bool(resolution.lasym),
        lconm1=True,
    )
    zeros = np.zeros_like(np.asarray(R_cos))
    state = SpectralState(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos,
                          Z_sin=Z_sin, L_cos=zeros, L_sin=zeros)
    state = jax.block_until_ready(state)

    edge_mismatch = {}
    for field_name, boundary_name in (
        ("R_cos", "boundary_R_cos"), ("R_sin", "boundary_R_sin"),
        ("Z_cos", "boundary_Z_cos"), ("Z_sin", "boundary_Z_sin"),
    ):
        difference = (np.asarray(getattr(state, field_name))[-1]
                      -np.asarray(getattr(runtime.setup, boundary_name)))
        physical = difference/scale
        edge_mismatch[field_name] = {
            "max_abs_coefficient_m": float(np.max(np.abs(physical), initial=0.0)),
            "l2_coefficient_m": float(np.linalg.norm(physical)),
        }

    gc, residuals, diagnostics = evaluate_forces(state, runtime)
    gc = jax.block_until_ready(gc)
    residual_record = {
        name: _norm(gc, name) for name in
        ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    }
    residual_record["all_blocks_l2"] = float(np.sqrt(sum(value**2 for key, value in
                                                           residual_record.items()
                                                           if key != "all_blocks_l2")))
    invariant = {name: float(np.asarray(getattr(residuals, name))) for name in
                 ("fsqr", "fsqz", "fsql", "fedge", "gcr2", "gcz2", "gcl2")}
    preconditioned = {name: float(np.asarray(getattr(diagnostics.preconditioned, name)))
                      for name in ("fsqr1", "fsqz1", "fsql1")}
    projection_path = out/"seed_ns17.npz"
    np.savez_compressed(projection_path, **{
        name: np.asarray(getattr(state, name)) for name in
        ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    })

    score_record = None
    sample_record = None
    score_error = None
    try:
        samples = sample_native(inp, state, case, runtime=runtime)
        score_record = score(samples)
        reference, observations = split_samples(samples)
        reference_path, observations_path = out/"reference_samples.npz", out/"native_observations.npz"
        np.savez_compressed(reference_path, **reference)
        np.savez_compressed(observations_path, **observations)
        sample_record = {
            "count": int(samples["xyz"].shape[0]),
            "positive_reference_volume_weights": bool(np.all(samples["weights"] > 0)),
            "native_flux_coordinates_finite": bool(np.all(np.isfinite(samples["s_native"]))),
            "reference_samples": {"path": reference_path.name,
                                  "sha256": _array_sha(reference_path)},
            "native_observations": {"path": observations_path.name,
                                    "sha256": _array_sha(observations_path)},
        }
    except Exception as error:
        score_error = {"type": type(error).__name__, "message": str(error)}

    report_path = out/"projection.json"
    report = {
        "schema": 1,
        "evidence": "lambda_zero_exact_geometry_projection",
        "status": ("projection_scored_diagnostic" if score_record is not None
                   else "projection_scoring_failed"),
        "accepted_recovery": False,
        "root_certified": False,
        "measurement_resolved": False,
        "source": source,
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/project_sheared_vmex.py",
        "input": INPUT.relative_to(ROOT).as_posix(),
        "input_sha256": input_hash,
        "case": case.name,
        "closure": "current_prescribed",
        "lambda_seed": "zero",
        "projection_description": "exact R/Z surfaces sampled on the VMEX full radial flux grid and Fourier projected; no lambda map was constructed",
        "ns": ns,
        "mpol": int(resolution.mpol),
        "ntor": int(resolution.ntor),
        "nfp": int(resolution.nfp),
        "angular_grid": ANGULAR_GRID,
        "mu0": MU0,
        "length_m": LENGTH_M,
        "field_t": FIELD_T,
        "exact_surface_nonfinite_count": max_surface_nonfinite,
        "input_edge_mismatch": edge_mismatch,
        "preconditioned_force_l2_by_block": residual_record,
        "invariant_residuals": invariant,
        "preconditioned_invariant_residuals": preconditioned,
        "jacobian_sign_changed": bool(np.asarray(diagnostics.jacobian_sign_changed)),
        "native_score": score_record,
        "sample_validation": sample_record,
        "score_error": score_error,
        "seed": {"path": projection_path.name, "sha256": _array_sha(projection_path)},
        "elapsed_seconds": perf_counter()-started,
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if platform.system() == "Darwin" else 1024)),
        "environment": {"python": sys.version.split()[0], "jax": jax.__version__,
                        "numpy": np.__version__,
                        "devices": [str(device) for device in jax.devices()]},
    }
    write_json(report_path, report, exclusive=True)
    print(json.dumps({"run_id": run_id, "report": report_path.relative_to(ROOT).as_posix(),
                      "report_sha256": sha256_file(report_path),
                      "status": report["status"], "score": score_record,
                      "score_error": score_error,
                      "residuals": invariant,
                      "edge_mismatch": edge_mismatch,
                      "elapsed_seconds": report["elapsed_seconds"],
                      "host_peak_rss_mib": report["host_peak_rss_mib"]}, indent=2))


if __name__ == "__main__":
    main()
