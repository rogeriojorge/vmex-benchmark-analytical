"""Probe finer independent cold-root differences for the NS65 null direction."""
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
from vmex.core.fourier import mode_table
from vmex.core.virtual_casing import _state_field_spectra

from analytic import ROOT, cases
from axisymmetric_root_response import PARAMETER_INDEX, _input_for_parameters, _root_and_anchor
from evidence import reserve_run_directory, sha256_file, source_metadata, write_json

PIN = "b5f5267efc0795c4a49a224e321e9b370975c14c"
RUN = ROOT/"results/vmex/response_runs/axisym-root-response-20260924T232219.753772Z"
REPORT = RUN/"response.json"
ARRAYS = RUN/"response_ns65.npz"
INPUT = ROOT/"inputs/input.integer_axisymmetric_current"
OUTPUT_PARENT = ROOT/"results/audit/axisym_branch_fine_steps"
STEPS = (3e-5, 1e-5)


def _state_from_npz(arrays, prefix):
    return implicit.SpectralState(**{
        name: jnp.asarray(arrays[f"{prefix}_{name}"])
        for name in implicit._STATE_FIELDS
    })


def main():
    start = perf_counter()
    source = source_metadata(vmex.__file__, "uwplasma/vmex",
                             importlib.metadata.version("vmex"))
    if source.get("commit") != PIN:
        raise SystemExit("imported VMEX source differs from historical benchmark pin")
    run_id = datetime.now(timezone.utc).strftime("axisym-branch-fine-%Y%m%dT%H%M%S.%fZ")
    run_id, out = reserve_run_directory(OUTPUT_PARENT, run_id)
    parsed = vmex.VmecInput.from_file(INPUT)
    case = cases()["integer_axisymmetric"]
    parameters = np.asarray(case.parameters, dtype=float)
    base_input, base_degrees, _ = _input_for_parameters(parsed, parameters)
    base_input = replace(base_input, ns_array=np.asarray([65]),
                         ftol_array=np.asarray([1e-12]),
                         niter_array=np.asarray([10000]))
    cfg = implicit.make_config(base_input, ns=65, ftol=1e-12,
                               max_iterations=10000, refine_tol=1e-11,
                               adjoint_tol=1e-10)
    saved = np.load(ARRAYS, allow_pickle=False)
    points = jnp.asarray(saved["points_xyz_m"])
    initial_flux = jnp.asarray(saved["initial_vmex_flux_coordinates"])
    base_state = _state_from_npz(saved, "state_base")
    state_tangent = {name: np.asarray(saved[f"state_tangent_delta_{name}"])
                     for name in implicit._STATE_FIELDS}
    dB_jvp = np.asarray(saved["dB_delta"])
    previous_fd = np.asarray(saved["dB_fd_delta_h1e-04"])
    modes = mode_table(13, 0)
    base_params = implicit.params_from_input(base_input)
    base_runtime = implicit.runtime_from_params(base_params, cfg)

    @jax.jit
    def sample_field(state, runtime):
        spectra = _state_field_spectra(base_input, state, runtime)
        coordinates, valid = _invert_coordinates(
            spectra, points, newton_iterations=12, initial_flux=initial_flux)
        return valid, _cartesian_derivative(spectra, 0, coordinates, valid)

    base_valid, base_B = sample_field(base_state, base_runtime)
    base_valid = np.asarray(jax.block_until_ready(base_valid))
    base_B = np.asarray(jax.block_until_ready(base_B))
    rows = []
    arrays_out = {"base_B": base_B}
    for step in STEPS:
        pair_states = []
        pair_roots = []
        pair_fields = []
        pair_valid = []
        variant_hashes = {}
        for sign in (1, -1):
            trial = parameters.copy()
            trial[PARAMETER_INDEX["delta"]] += sign*step
            trial_input, degrees, _ = _input_for_parameters(parsed, trial)
            if degrees != base_degrees:
                raise ValueError("profile/boundary basis changed inside fine-step probe")
            trial_input = replace(trial_input, ns_array=base_input.ns_array,
                                  ftol_array=base_input.ftol_array,
                                  niter_array=base_input.niter_array)
            deck_path = out/f"input_delta_ns65_h{step:.0e}_{'plus' if sign > 0 else 'minus'}.indata"
            trial_input.to_indata(deck_path)
            variant_hashes[f"{'plus' if sign > 0 else 'minus'}_input"] = {
                "path": deck_path.name,
                "sha256": sha256_file(deck_path),
            }
            trial_params = implicit.params_from_input(trial_input)
            runtime = implicit.runtime_from_params(trial_params, cfg)
            state, mask, root = _root_and_anchor(cfg, trial_params)
            valid, field = sample_field(state, runtime)
            valid, field = jax.block_until_ready((valid, field))
            state_np = {name: np.asarray(getattr(state, name))
                        for name in implicit._STATE_FIELDS}
            for name, values in state_np.items():
                arrays_out[f"state_{step:.0e}_{'plus' if sign>0 else 'minus'}_{name}"] = values
            pair_states.append(state_np)
            pair_roots.append(root)
            pair_fields.append(np.asarray(field))
            pair_valid.append(np.asarray(valid))
            variant_hashes[f"{'plus' if sign > 0 else 'minus'}_settings"] = {
                "input_parameter_vector": trial.tolist(),
                "input_am": np.asarray(trial_input.am).tolist(),
                "input_ac": np.asarray(trial_input.ac).tolist(),
                "curtor": float(trial_input.curtor),
            }
        fd = (pair_fields[0]-pair_fields[1])/(2*step)
        arrays_out[f"B_fd_h{step:.0e}"] = fd
        state_rows = {}
        for name in implicit._STATE_FIELDS:
            state_fd = (pair_states[0][name]-pair_states[1][name])/(2*step)
            difference = state_fd-state_tangent[name]
            per_mode = []
            for mode_index, mode in enumerate(modes.m):
                per_mode.append({"m": int(mode),
                                 "difference_l2": float(np.linalg.norm(difference[:, mode_index])),
                                 "branch_fd_l2": float(np.linalg.norm(state_fd[:, mode_index])),
                                 "tangent_l2": float(np.linalg.norm(state_tangent[name][:, mode_index]))})
            state_rows[name] = {
                "branch_fd_l2": float(np.linalg.norm(state_fd)),
                "tangent_l2": float(np.linalg.norm(state_tangent[name])),
                "difference_l2": float(np.linalg.norm(difference)),
                "difference_l2_over_state_scale": float(np.linalg.norm(difference)/
                                                          max(np.linalg.norm(state_tangent[name]),1e-30)),
                "modes": per_mode,
            }
        rows.append({
            "step": step,
            "valid_plus": int(np.count_nonzero(pair_valid[0])),
            "valid_minus": int(np.count_nonzero(pair_valid[1])),
            "point_count": int(points.shape[0]),
            "B_branch_fd_l2_over_fixed_scale": float(np.linalg.norm(fd)),
            "B_branch_fd_minus_jvp_over_fixed_scale": float(np.linalg.norm(fd-dB_jvp)),
            "B_branch_fd_change_from_h1e-04_over_fixed_scale": float(np.linalg.norm(fd-previous_fd)),
            "plus_root": pair_roots[0], "minus_root": pair_roots[1],
            "variants": variant_hashes,
            "state_fd_by_field": state_rows,
        })
    arrays_path = out/"fine_branch_states_ns65.npz"
    np.savez_compressed(arrays_path, **arrays_out)
    report_path = out/"fine_branch_probe.json"
    record = {
        "schema": 1,
        "evidence": "independent_cold_branch_fd_fine_steps",
        "status": "diagnostic_unresolved_branch_response",
        "accepted_derivative": False,
        "source": source,
        "source_response_report": REPORT.relative_to(ROOT).as_posix(),
        "source_response_report_sha256": sha256_file(REPORT),
        "source_response_arrays_sha256": sha256_file(ARRAYS),
        "input": INPUT.relative_to(ROOT).as_posix(),
        "input_sha256": sha256_file(INPUT),
        "ns": 65, "ftol": 1e-12,
        "base_valid_point_count": int(np.count_nonzero(base_valid)),
        "steps": STEPS,
        "fixed_field_scale": "FIELD_T/LENGTH_M = 1 T/m per parameter unit",
        "branch_rows": rows,
        "arrays": {"path": arrays_path.name, "sha256": sha256_file(arrays_path)},
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/probe_axisymmetric_branch_fine_steps.py",
        "elapsed_seconds": perf_counter()-start,
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if sys.platform == "darwin" else 1024)),
    }
    write_json(report_path, record, exclusive=True)
    print(json.dumps({"run_id": run_id,
                      "report": report_path.relative_to(ROOT).as_posix(),
                      "report_sha256": sha256_file(report_path),
                      "branches": [{k: v for k,v in row.items() if k in (
                          "step", "valid_plus", "valid_minus", "B_branch_fd_l2_over_fixed_scale",
                          "B_branch_fd_minus_jvp_over_fixed_scale",
                          "B_branch_fd_change_from_h1e-04_over_fixed_scale",
                          "plus_root", "minus_root")}
                          for row in rows],
                      "elapsed_seconds": record["elapsed_seconds"],
                      "host_peak_rss_mib": record["host_peak_rss_mib"]}, indent=2))


if __name__ == "__main__":
    main()
