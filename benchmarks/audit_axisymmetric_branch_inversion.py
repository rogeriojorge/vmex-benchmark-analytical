"""Audit physical-field coordinate inversion on saved NS65 branch states."""
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import vmex
from vmex.core import implicit
from vmex.core.extender import _cartesian_derivative, _invert_coordinates
from vmex.core.virtual_casing import _state_field_spectra

from analytic import ROOT, cases
from axisymmetric_root_response import PARAMETER_INDEX, _input_for_parameters
from evidence import reserve_run_directory, sha256_file, source_metadata, write_json

PIN = "b5f5267efc0795c4a49a224e321e9b370975c14c"
RUN = ROOT/"results/vmex/response_runs/axisym-root-response-20260924T232219.753772Z"
REPORT = RUN/"response.json"
ARRAYS = RUN/"response_ns65.npz"
INPUT = ROOT/"inputs/input.integer_axisymmetric_current"
OUTPUT_PARENT = ROOT/"results/audit/axisym_branch_inversion"
STEPS = (1e-3, 3e-4, 1e-4)
ITERATIONS = (8, 12, 24)


def _state(arrays, prefix):
    return implicit.SpectralState(**{
        name: jnp.asarray(arrays[f"{prefix}_{name}"])
        for name in implicit._STATE_FIELDS
    })


def main():
    started = perf_counter()
    source = source_metadata(vmex.__file__, "uwplasma/vmex",
                             importlib.metadata.version("vmex"))
    if source.get("commit") != PIN:
        raise SystemExit("imported VMEX source differs from historical benchmark pin")
    run_id = datetime.now(timezone.utc).strftime("axisym-branch-inversion-%Y%m%dT%H%M%S.%fZ")
    run_id, out = reserve_run_directory(OUTPUT_PARENT, run_id)
    report = json.loads(REPORT.read_text())
    arrays = np.load(ARRAYS, allow_pickle=False)
    ns = 65
    case = cases()["integer_axisymmetric"]
    parameters = np.asarray(case.parameters, dtype=float)
    parsed = vmex.VmecInput.from_file(INPUT)
    base_input, degrees, _ = _input_for_parameters(parsed, parameters)
    base_input = replace(base_input, ns_array=np.asarray([ns]),
                         ftol_array=np.asarray([1e-12]),
                         niter_array=np.asarray([10000]))
    cfg = implicit.make_config(base_input, ns=ns, ftol=1e-12,
                               max_iterations=10000, refine_tol=1e-11,
                               adjoint_tol=1e-10)
    points = jnp.asarray(arrays["points_xyz_m"])
    initial_flux = jnp.asarray(arrays["initial_vmex_flux_coordinates"])
    tangent = jnp.asarray(arrays["dB_delta"])
    base_state = _state(arrays, "state_base")

    def field_evaluator(iterations):
        @jax.jit
        def evaluate(state, runtime):
            spectra = _state_field_spectra(base_input, state, runtime)
            coordinates, valid = _invert_coordinates(
                spectra, points, newton_iterations=iterations,
                initial_flux=initial_flux,
            )
            field = _cartesian_derivative(spectra, 0, coordinates, valid)
            return coordinates, valid, field
        return evaluate

    evaluators = {iterations: field_evaluator(iterations) for iterations in ITERATIONS}
    base_params = implicit.params_from_input(base_input)
    base_fields = {}
    base_valid = {}
    base_runtime = implicit.runtime_from_params(base_params, cfg)
    for iterations, evaluator in evaluators.items():
        coordinates, valid, field = evaluator(base_state, base_runtime)
        base_fields[iterations] = np.asarray(jax.block_until_ready(field))
        base_valid[iterations] = int(np.count_nonzero(np.asarray(valid)))

    branch_rows = []
    coordinate_array = {}
    fd_array = {}
    scale = 1.0  # FIELD_T/LENGTH_M from the benchmark convention
    for step in STEPS:
        plus_parameters = parameters.copy()
        minus_parameters = parameters.copy()
        plus_parameters[PARAMETER_INDEX["delta"]] += step
        minus_parameters[PARAMETER_INDEX["delta"]] -= step
        plus_input, plus_degrees, _ = _input_for_parameters(parsed, plus_parameters)
        minus_input, minus_degrees, _ = _input_for_parameters(parsed, minus_parameters)
        if plus_degrees != degrees or minus_degrees != degrees:
            raise ValueError("profile/boundary basis changed inside audit ladder")
        plus_input = replace(plus_input, ns_array=base_input.ns_array,
                             ftol_array=base_input.ftol_array,
                             niter_array=base_input.niter_array)
        minus_input = replace(minus_input, ns_array=base_input.ns_array,
                              ftol_array=base_input.ftol_array,
                              niter_array=base_input.niter_array)
        plus_params = implicit.params_from_input(plus_input)
        minus_params = implicit.params_from_input(minus_input)
        plus_runtime = implicit.runtime_from_params(plus_params, cfg)
        minus_runtime = implicit.runtime_from_params(minus_params, cfg)
        plus_state = _state(arrays, f"state_delta_h{step:.0e}_plus")
        minus_state = _state(arrays, f"state_delta_h{step:.0e}_minus")
        for iterations, evaluator in evaluators.items():
            plus_coords, plus_valid, plus_B = evaluator(plus_state, plus_runtime)
            minus_coords, minus_valid, minus_B = evaluator(minus_state, minus_runtime)
            plus_coords = np.asarray(jax.block_until_ready(plus_coords))
            minus_coords = np.asarray(jax.block_until_ready(minus_coords))
            plus_valid = np.asarray(plus_valid)
            minus_valid = np.asarray(minus_valid)
            plus_B = np.asarray(jax.block_until_ready(plus_B))
            minus_B = np.asarray(jax.block_until_ready(minus_B))
            fd = (plus_B-minus_B)/(2*step)
            fd_array[f"fd_h{step:.0e}_iter{iterations}"] = fd
            coordinate_array[f"plus_h{step:.0e}_iter{iterations}"] = plus_coords
            coordinate_array[f"minus_h{step:.0e}_iter{iterations}"] = minus_coords
            branch_rows.append({
                "step": step,
                "newton_iterations": iterations,
                "valid_plus": int(np.count_nonzero(plus_valid)),
                "valid_minus": int(np.count_nonzero(minus_valid)),
                "point_count": int(points.shape[0]),
                "B_branch_fd_l2_over_fixed_scale": float(np.linalg.norm(fd)/scale),
                "B_branch_fd_minus_jvp_over_fixed_scale": float(np.linalg.norm(fd-tangent)/scale),
            })
    by_step = {}
    for step in STEPS:
        base = fd_array[f"fd_h{step:.0e}_iter12"]
        by_step[f"{step:g}"] = {
            "fd_iter8_vs_iter12_over_fixed_scale": float(np.linalg.norm(
                fd_array[f"fd_h{step:.0e}_iter8"]-base)/scale),
            "fd_iter24_vs_iter12_over_fixed_scale": float(np.linalg.norm(
                fd_array[f"fd_h{step:.0e}_iter24"]-base)/scale),
        }
    arrays_path = out/"inversion_audit_ns65.npz"
    np.savez_compressed(arrays_path, **fd_array, **coordinate_array,
                        **{f"base_B_iter{k}": v for k,v in base_fields.items()})
    record = {
        "schema": 1,
        "evidence": "saved_branch_state_coordinate_inversion_audit",
        "status": "diagnostic_field_reconstruction_check",
        "accepted_derivative": False,
        "source": source,
        "source_response_report": REPORT.relative_to(ROOT).as_posix(),
        "source_response_report_sha256": sha256_file(REPORT),
        "source_response_arrays_sha256": sha256_file(ARRAYS),
        "ns": ns,
        "centered_steps": STEPS,
        "newton_iteration_counts": ITERATIONS,
        "point_count": int(points.shape[0]),
        "base_valid_points": base_valid,
        "branch_rows": branch_rows,
        "stepwise_fd_reconstruction_changes": by_step,
        "fixed_scale": "FIELD_T/LENGTH_M = 1 T/m per parameter unit",
        "saved_arrays": {"path": arrays_path.name,
                         "sha256": sha256_file(arrays_path)},
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/audit_axisymmetric_branch_inversion.py",
        "elapsed_seconds": perf_counter()-started,
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if sys.platform == "darwin" else 1024)),
    }
    report_path = out/"inversion_audit.json"
    write_json(report_path, record, exclusive=True)
    print(json.dumps({"run_id": run_id,
                      "report": report_path.relative_to(ROOT).as_posix(),
                      "report_sha256": sha256_file(report_path),
                      "base_valid_points": base_valid,
                      "branches": branch_rows,
                      "stepwise_fd_reconstruction_changes": by_step,
                      "elapsed_seconds": record["elapsed_seconds"],
                      "host_peak_rss_mib": record["host_peak_rss_mib"]}, indent=2))


if __name__ == "__main__":
    main()
