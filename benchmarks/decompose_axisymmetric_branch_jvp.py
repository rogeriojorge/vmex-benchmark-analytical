"""Compare field reconstruction JVPs along the measured NS65 branch state FD."""
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

from analytic import ROOT, cases
from axisymmetric_root_response import _input_for_parameters, _physical_field_function
from evidence import reserve_run_directory, sha256_file, source_metadata, write_json

PIN = "b5f5267efc0795c4a49a224e321e9b370975c14c"
RESPONSE_DIR = ROOT/"results/vmex/response_runs/axisym-root-response-20260924T232219.753772Z"
RESPONSE_REPORT = RESPONSE_DIR/"response.json"
RESPONSE_ARRAYS = RESPONSE_DIR/"response_ns65.npz"
FINE_DIR = ROOT/"results/audit/axisym_branch_fine_steps/axisym-branch-fine-20260925T000429.848273Z"
FINE_REPORT = FINE_DIR/"fine_branch_probe.json"
FINE_ARRAYS = FINE_DIR/"fine_branch_states_ns65.npz"
INPUT = ROOT/"inputs/input.integer_axisymmetric_current"
OUT_PARENT = ROOT/"results/audit/axisym_branch_jvp_decomposition"
STEP = 1e-5
FIELDS = implicit._STATE_FIELDS


def _norm(value):
    return float(np.linalg.norm(np.asarray(value)))


def _score(value):
    return {"signed": np.asarray(value).tolist(), "l2_over_fixed_scale": _norm(value)}


def main():
    start = perf_counter()
    source = source_metadata(vmex.__file__, "uwplasma/vmex",
                             importlib.metadata.version("vmex"))
    if source.get("commit") != PIN:
        raise SystemExit("imported VMEX source differs from historical benchmark pin")
    run_id = datetime.now(timezone.utc).strftime("axisym-branch-jvp-%Y%m%dT%H%M%S.%fZ")
    run_id, out = reserve_run_directory(OUT_PARENT, run_id)
    parsed = vmex.VmecInput.from_file(INPUT)
    case = cases()["integer_axisymmetric"]
    base_parameters = np.asarray(case.parameters, dtype=float)
    base_input, _, _ = _input_for_parameters(parsed, base_parameters)
    base_input = replace(base_input, ns_array=np.asarray([65]),
                         ftol_array=np.asarray([1e-12]),
                         niter_array=np.asarray([10000]))
    cfg = implicit.make_config(base_input, ns=65, ftol=1e-12,
                               max_iterations=10000, refine_tol=1e-11,
                               adjoint_tol=1e-10)
    base_params = implicit.params_from_input(base_input)
    # Materialize the runtime template before tracing the field JVP.
    implicit.runtime_from_params(base_params, cfg)
    original = np.load(RESPONSE_ARRAYS, allow_pickle=False)
    fine = np.load(FINE_ARRAYS, allow_pickle=False)
    base_state = implicit.SpectralState(**{
        name: jnp.asarray(original[f"state_base_{name}"]) for name in FIELDS})
    plus_state = implicit.SpectralState(**{
        name: jnp.asarray(fine[f"state_{STEP:.0e}_plus_{name}"]) for name in FIELDS})
    minus_state = implicit.SpectralState(**{
        name: jnp.asarray(fine[f"state_{STEP:.0e}_minus_{name}"]) for name in FIELDS})
    state_dot = jax.tree.map(lambda plus, minus: (plus-minus)/(2*STEP),
                             plus_state, minus_state)
    plus_input = vmex.VmecInput.from_file(FINE_DIR/f"input_delta_ns65_h{STEP:.0e}_plus.indata")
    minus_input = vmex.VmecInput.from_file(FINE_DIR/f"input_delta_ns65_h{STEP:.0e}_minus.indata")
    plus_params = implicit.params_from_input(plus_input)
    minus_params = implicit.params_from_input(minus_input)
    params_dot = jax.tree.map(lambda plus, minus: (plus-minus)/(2*STEP),
                              plus_params, minus_params)
    points = jnp.asarray(original["points_xyz_m"])
    initial_flux = jnp.asarray(original["initial_vmex_flux_coordinates"])
    field_fn = _physical_field_function(base_input, cfg, points, initial_flux)
    fd = jnp.asarray(fine[f"B_fd_h{STEP:.0e}"])
    zeros_state = jax.tree.map(jnp.zeros_like, base_state)

    total_dot = jax.jvp(field_fn, (base_state, base_params),
                        (state_dot, params_dot))[1]
    param_dot = jax.jvp(field_fn, (base_state, base_params),
                        (zeros_state, params_dot))[1]
    state_contributions = {}
    for active_name in ("R_cos", "Z_sin", "L_sin"):
        component = dict((name, jnp.zeros_like(getattr(base_state, name))) for name in FIELDS)
        component[active_name] = getattr(state_dot, active_name)
        component = implicit.SpectralState(**component)
        state_contributions[active_name] = jax.jvp(
            field_fn, (base_state, base_params),
            (component, jax.tree.map(jnp.zeros_like, base_params)))[1]
    component_sum = param_dot
    for value in state_contributions.values():
        component_sum = component_sum+value
    values = jax.block_until_ready((fd, total_dot, param_dot, component_sum,
                                    *state_contributions.values()))
    fd, total_dot, param_dot, component_sum, *component_values = values
    total_dot, param_dot, component_sum = map(np.asarray,
                                              (total_dot, param_dot, component_sum))
    fd = np.asarray(fd)
    component_records = {"direct_input_and_profiles": _score(param_dot)}
    arrays = {"branch_B_fd": fd, "branch_state_and_input_jvp": total_dot,
              "direct_input_and_profiles_jvp": param_dot}
    for name, value in zip(state_contributions, component_values):
        val = np.asarray(value)
        component_records[name] = _score(val)
        arrays[f"{name}_jvp"] = val
    arrays["component_sum"] = component_sum
    arrays_path = out/"branch_jvp_components_ns65.npz"
    np.savez_compressed(arrays_path, **arrays)
    matching_rows = [row for row in json.loads(FINE_REPORT.read_text())["branch_rows"]
                     if row["step"] == STEP]
    root_residuals = [float(row[side]["residual_after_anchor"])
                      for row in matching_rows for side in ("plus_root", "minus_root")]
    if len(root_residuals) != 2:
        raise RuntimeError(f"expected both saved branch roots at h={STEP:g}")
    record = {
        "schema": 1,
        "evidence": "field_reconstruction_jvp_along_measured_branch_state_fd",
        "status": "diagnostic_branch_response_decomposition",
        "accepted_derivative": False,
        "source": source,
        "response_report": RESPONSE_REPORT.relative_to(ROOT).as_posix(),
        "response_report_sha256": sha256_file(RESPONSE_REPORT),
        "response_arrays_sha256": sha256_file(RESPONSE_ARRAYS),
        "fine_branch_report": FINE_REPORT.relative_to(ROOT).as_posix(),
        "fine_branch_report_sha256": sha256_file(FINE_REPORT),
        "fine_branch_arrays_sha256": sha256_file(FINE_ARRAYS),
        "input_sha256": sha256_file(INPUT),
        "ns": 65, "direction": "delta", "step": STEP,
        "valid_saved_branch_root_residual_max": max(root_residuals),
        "branch_B_fd": _score(fd),
        "branch_state_and_input_jvp": _score(total_dot),
        "jvp_minus_fd_l2_over_fixed_scale": _norm(total_dot-fd),
        "residual_level_jvp": _score(np.asarray(original["dB_delta"])),
        "jvp_using_measured_branch_state": component_records,
        "component_sum_closure_l2": _norm(component_sum-total_dot),
        "arrays": {"path": arrays_path.name, "sha256": sha256_file(arrays_path)},
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/decompose_axisymmetric_branch_jvp.py",
        "elapsed_seconds": perf_counter()-start,
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if sys.platform == "darwin" else 1024)),
    }
    report_path = out/"branch_jvp_decomposition.json"
    write_json(report_path, record, exclusive=True)
    print(json.dumps({"run_id": run_id,
                      "report": report_path.relative_to(ROOT).as_posix(),
                      "report_sha256": sha256_file(report_path),
                      "branch_B_fd_l2_over_fixed_scale": _norm(fd),
                      "branch_state_and_input_jvp_l2_over_fixed_scale": _norm(total_dot),
                      "jvp_minus_fd_l2_over_fixed_scale": _norm(total_dot-fd),
                      "residual_level_jvp_l2_over_fixed_scale": _norm(np.asarray(original["dB_delta"])),
                      "components": component_records,
                      "elapsed_seconds": record["elapsed_seconds"],
                      "host_peak_rss_mib": record["host_peak_rss_mib"]}, indent=2))


if __name__ == "__main__":
    main()
