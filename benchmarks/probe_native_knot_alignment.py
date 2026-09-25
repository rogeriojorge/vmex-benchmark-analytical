"""Locate VMEX native-spline knots on an exact reference-coordinate ray."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import vmex
from vmex.core.extender import VmecInteriorField
from vmex.core.solver import SpectralState

from analytic import cases
from build_inputs import LENGTH_M
from evidence import sha256_file, source_metadata, write_json
from measurement import integer_surface_numpy, native_knot_crossings


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--state-npz", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True,
                        help="saved samples from the same state, used to bracket each knot")
    parser.add_argument("--measurement", type=Path, required=True,
                        help="measurement report that identifies the saved state and source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-vmex", required=True)
    parser.add_argument("--case", default="integer_3d")
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--phi", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=48)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.iterations < 24:
        raise SystemExit("use at least 24 vector bisection steps")
    inp = vmex.VmecInput.from_file(args.input)
    with np.load(args.state_npz, allow_pickle=False) as arrays:
        state = SpectralState(**{name: arrays[name] for name in
            ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    report = json.loads(args.measurement.read_text(encoding="utf-8"))
    state_hash = sha256_file(args.state_npz)
    saved_state_hash = report.get("artifacts", {}).get("spectral_state", {}).get("sha256")
    if saved_state_hash != state_hash:
        raise SystemExit("saved state hash does not match its measurement report")
    source = source_metadata(vmex.__file__, "uwplasma/vmex", importlib.metadata.version("vmex"))
    if source.get("commit") != args.expected_vmex:
        raise SystemExit("imported VMEX source does not match the requested source pin")
    case = cases()[args.case]
    ns = int(state.R_cos.shape[0])
    native_knots = np.arange(1, ns-1, dtype=float)/(ns-1)
    with np.load(args.samples, allow_pickle=False) as arrays:
        s_reference = np.asarray(arrays["s_reference"], dtype=float)
        s_native = np.asarray(arrays["s_native"], dtype=float)
    radial_count = len(np.unique(s_reference))
    if len(s_reference) % radial_count:
        raise SystemExit("saved sample grid does not have a complete radial ray")
    angular_count = len(s_reference)//radial_count
    sample_grid = next((row for row in report.get("rows", [])
                        if row.get("sample_count") == len(s_reference) and
                        row.get("nradial") == radial_count), {})
    ray_indices = np.arange(radial_count)*angular_count
    ray_reference = s_reference[ray_indices]
    ray_native = s_native[ray_indices]
    reference_linear = native_knot_crossings(ray_reference, ray_native, native_knots)
    native_order = np.argsort(ray_native, kind="stable")
    native_sorted = ray_native[native_order]
    reference_sorted = ray_reference[native_order]
    insertion = np.searchsorted(native_sorted, native_knots)
    if np.any(insertion == 0) or np.any(insertion == len(native_sorted)):
        raise SystemExit("saved radial ray does not bracket every interior native knot")
    lower = reference_sorted[insertion-1].copy()
    upper = reference_sorted[insertion].copy()
    field = VmecInteriorField.from_state(inp, state)
    start = time.perf_counter()
    for _ in range(args.iterations):
        midpoint = (lower+upper)/2
        q = np.column_stack((midpoint, np.full(ns-2, args.theta),
                             np.full(ns-2, args.phi)))
        xyz = integer_surface_numpy(case, q)*LENGTH_M
        measured_native = np.asarray(field.flux_coordinates(jnp.asarray(xyz)))[:, 0]
        below = measured_native < native_knots
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
    exact_crossings = (lower+upper)/2
    q = np.column_stack((exact_crossings, np.full(ns-2, args.theta),
                         np.full(ns-2, args.phi)))
    xyz = integer_surface_numpy(case, q)*LENGTH_M
    final_native = np.asarray(field.flux_coordinates(jnp.asarray(xyz)))[:, 0]
    elapsed = time.perf_counter()-start
    record = {
        "schema": 1,
        "status": "diagnostic_actual_native_knot_roots",
        "case": args.case,
        "state_source_sha256": report.get("state_source_sha256"),
        "spectral_state_sha256": state_hash,
        "input_sha256": sha256_file(args.input),
        "measurement_run_id": report.get("run_id"),
        "sample_grid_id": sample_grid.get("grid_id"),
        "sample_radial_rule": sample_grid.get("radial_rule"),
        "measurement_report_sha256": sha256_file(args.measurement),
        "sample_grid_sha256": sha256_file(args.samples),
        "vmex_source": source,
        "grid_radial_count": radial_count,
        "native_mesh_points": ns,
        "native_knots_uniform_in_s": True,
        "ray": {"theta": args.theta, "phi": args.phi},
        "root_method": "vector bisection of actual VMEX inverse flux map",
        "iterations_per_knot": args.iterations,
        "elapsed_seconds": elapsed,
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if platform.system() == "Darwin" else 1024)),
        "native_knots": native_knots.tolist(),
        "reference_s_at_native_knots": exact_crossings.tolist(),
        "native_residual_at_roots": (final_native-native_knots).tolist(),
        "max_abs_native_root_residual": float(np.max(np.abs(final_native-native_knots))),
        "linear_sampled_crossings_reference_s": reference_linear.tolist(),
        "linear_sampled_crossing_error": (reference_linear-exact_crossings).tolist(),
        "max_abs_linear_sampled_crossing_error": float(
            np.max(np.abs(reference_linear-exact_crossings))),
        "uniform_reference_knots": native_knots.tolist(),
        "max_native_crossing_shift_from_uniform_reference_knots": float(
            np.max(np.abs(exact_crossings-native_knots))),
        "reference_cell_partition_native_knot_aligned": False,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output, record, exclusive=True)
    print(json.dumps({
        "status": record["status"],
        "max_abs_native_root_residual": record["max_abs_native_root_residual"],
        "max_abs_linear_sampled_crossing_error": record[
            "max_abs_linear_sampled_crossing_error"],
        "max_native_crossing_shift_from_uniform_reference_knots": record[
            "max_native_crossing_shift_from_uniform_reference_knots"],
        "elapsed_seconds": elapsed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
