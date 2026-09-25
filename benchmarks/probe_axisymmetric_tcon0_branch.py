"""Compare NS65 integer-family delta branches at TCON0=0 and the default."""
from dataclasses import replace
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
import jax.numpy as jnp
import numpy as np
import vmex
from vmex.core import implicit
from vmex.core.extender import _cartesian_derivative, _invert_coordinates
from vmex.core.virtual_casing import _state_field_spectra

from analytic import ROOT, cases
from axisymmetric_root_response import (
    PARAMETER_INDEX, _input_for_parameters, _root_and_anchor,
)
from evidence import reserve_run_directory, sha256_file, source_metadata, write_json

PIN = "b5f5267efc0795c4a49a224e321e9b370975c14c"
NS = 65
STEP = 1e-5
TCON0 = 0.0
RESPONSE_DIR = ROOT / "results/vmex/response_runs/axisym-root-response-20260924T232219.753772Z"
FINE_DIR = ROOT / "results/audit/axisym_branch_fine_steps/axisym-branch-fine-20260925T000429.848273Z"
RELABEL_DIR = ROOT / "results/audit/axisym_radial_relabel/axisym-relabel-field-20260925T010115.952375Z"
INPUT = ROOT / "inputs/input.integer_axisymmetric_current"
OUTPUT_PARENT = ROOT / "results/audit/axisym_branch_tcon0"
STATE_FIELDS = implicit._STATE_FIELDS


def _rss_mib():
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor)


def _state_arrays(state):
    return {name: np.asarray(getattr(state, name)) for name in STATE_FIELDS}


def main():
    started = perf_counter()
    source = source_metadata(vmex.__file__, "uwplasma/vmex",
                             importlib.metadata.version("vmex"))
    if source.get("commit") != PIN:
        raise SystemExit("imported VMEX source differs from the historical pin")
    run_id = datetime.now(timezone.utc).strftime("axisym-tcon0-branch-%Y%m%dT%H%M%S.%fZ")
    run_id, out = reserve_run_directory(OUTPUT_PARENT, run_id)

    parsed = vmex.VmecInput.from_file(INPUT)
    case = cases()["integer_axisymmetric"]
    parameters = np.asarray(case.parameters, dtype=np.float64)
    template, base_degrees, _ = _input_for_parameters(parsed, parameters)
    controls = {"ns_array": np.asarray([NS]), "ftol_array": np.asarray([1e-12]),
                "niter_array": np.asarray([10000]), "tcon0": TCON0}
    base_input = replace(template, **controls)
    cfg = implicit.make_config(base_input, ns=NS, ftol=1e-12,
                               max_iterations=10000, refine_tol=1e-11,
                               adjoint_tol=1e-10)
    base_params = implicit.params_from_input(base_input)

    point_archive = np.load(RELABEL_DIR / "radial_relabel_field_ns65.npz",
                            allow_pickle=False)
    points = jnp.asarray(point_archive["points_xyz_m"])
    reference_s = np.asarray(point_archive["reference_s"])
    fine = np.load(FINE_DIR / "fine_branch_states_ns65.npz", allow_pickle=False)
    default_branch_fd = np.asarray(point_archive["branch_B_fd"])
    decks = {}
    root_rows = []
    state_records = {}
    parameter_records = {}

    cases_to_solve = [("base", parameters.copy(), base_input, base_params)]
    for sign, label in ((1, "plus"), (-1, "minus")):
        trial = parameters.copy()
        trial[PARAMETER_INDEX["delta"]] += sign * STEP
        trial_input, degrees, gate = _input_for_parameters(parsed, trial)
        if degrees != base_degrees:
            raise ValueError("profile or boundary basis changed under delta")
        trial_input = replace(trial_input, **controls)
        params = implicit.params_from_input(trial_input)
        cases_to_solve.append((label, trial, trial_input, params))
        parameter_records[label] = {"parameters": trial.tolist(), "fit_gate": gate}

    for label, trial, trial_input, params in cases_to_solve:
        deck_path = out / f"input_delta_ns{NS}_{label}.indata"
        trial_input.to_indata(deck_path)
        decks[label] = {"path": deck_path.name, "sha256": sha256_file(deck_path)}
        try:
            state, mask, root = _root_and_anchor(cfg, params)
            state = jax.block_until_ready(state)
            state_records[label] = _state_arrays(state)
            root_status = ("root_certified"
                           if root["residual_after_anchor"] <= cfg.refine_tol
                           else "residual_unresolved")
            receipt = {
                "schema": 1, "case": label, "status": root_status,
                "tcon0": TCON0, "root": root,
                "input": decks[label],
                "state": {"path": f"state_{label}.npz"},
            }
            state_path = out / f"state_{label}.npz"
            np.savez_compressed(state_path, **state_records[label])
            receipt["state"]["sha256"] = sha256_file(state_path)
            write_json(out / f"root_{label}.json", receipt, exclusive=True)
            root_rows.append(receipt)
        except Exception as error:
            root_rows.append({"schema": 1, "case": label,
                              "status": "solver_failed",
                              "input": decks[label],
                              "error_type": type(error).__name__,
                              "error": str(error)})

    record = {
        "schema": 1,
        "evidence": "matched_axisymmetric_delta_branch_tcon0_control",
        "status": "diagnostic",
        "accepted_derivative": False,
        "accepted_coordinate_relabel": False,
        "source": source,
        "input_sha256": sha256_file(INPUT),
        "base_input_fit_gate": base_degrees,
        "tcon0": TCON0,
        "ns": NS,
        "ftol": 1e-12,
        "max_iterations": 10000,
        "delta_centered_step": STEP,
        "reference_arrays_sha256": sha256_file(RELABEL_DIR / "radial_relabel_field_ns65.npz"),
        "reference_point_count": int(points.shape[0]),
        "reference_s": reference_s.tolist(),
        "inputs": decks,
        "perturbed_parameter_vectors": parameter_records,
        "roots": root_rows,
        "root_residual_certificate_tolerance": float(cfg.refine_tol),
        "all_roots_returned": len(root_rows) == 3 and all(
            row["status"] in ("root_certified", "residual_unresolved")
            for row in root_rows),
        "all_roots_certified": len(root_rows) == 3 and all(
            row["status"] == "root_certified" for row in root_rows),
    }
    if record["all_roots_returned"]:
        runtimes = {}
        for label, _, _, params in cases_to_solve:
            runtimes[label] = implicit.runtime_from_params(params, cfg)
        base_runtime = runtimes["base"]
        base_state = implicit.SpectralState(**{
            name: jnp.asarray(state_records["base"][name]) for name in STATE_FIELDS
        })
        base_field = vmex.VmecInteriorField.from_state(
            base_input, base_state, runtime=base_runtime)
        initial_flux = base_field.flux_coordinates(points)

        @jax.jit
        def sample(state, runtime):
            spectra = _state_field_spectra(base_input, state, runtime)
            coordinates, valid = _invert_coordinates(
                spectra, points, newton_iterations=12, initial_flux=initial_flux)
            return valid, _cartesian_derivative(spectra, 0, coordinates, valid)

        fields, validity = {}, {}
        for label, _, _, params in cases_to_solve:
            state = implicit.SpectralState(**{
                name: jnp.asarray(state_records[label][name]) for name in STATE_FIELDS
            })
            valid, value = sample(state, runtimes[label])
            valid, value = jax.block_until_ready((valid, value))
            fields[label] = np.asarray(value)
            validity[label] = np.asarray(valid)
        branch_fd = (fields["plus"] - fields["minus"]) / (2 * STEP)
        record.update({
            "all_field_inversions_valid": bool(all(v.all() for v in validity.values())),
            "valid_counts": {key: int(np.count_nonzero(value))
                             for key, value in validity.items()},
            "branch_B_fd_l2_fixed_scale": float(np.linalg.norm(branch_fd)),
            "branch_B_fd_minus_exact_null_l2": float(np.linalg.norm(branch_fd)),
            "branch_B_fd_minus_default_tcon0_l2": float(
                np.linalg.norm(branch_fd - default_branch_fd)),
            "default_tcon0_branch_B_fd_l2": float(np.linalg.norm(default_branch_fd)),
        })
        arrays_path = out / "branch_tcon0_ns65.npz"
        np.savez_compressed(arrays_path, points_xyz_m=np.asarray(points),
                            reference_s=reference_s,
                            initial_flux=np.asarray(initial_flux),
                            B_base=fields["base"], B_plus=fields["plus"],
                            B_minus=fields["minus"], B_fd=branch_fd,
                            B_fd_default_tcon0=default_branch_fd,
                            valid_base=validity["base"], valid_plus=validity["plus"],
                            valid_minus=validity["minus"])
        record["arrays"] = {"path": arrays_path.name,
                            "sha256": sha256_file(arrays_path)}
    else:
        record["all_field_inversions_valid"] = False

    record.update({
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/probe_axisymmetric_tcon0_branch.py",
        "elapsed_seconds": perf_counter() - started,
        "host_peak_rss_mib": _rss_mib(),
    })
    report_path = out / "tcon0_branch_probe.json"
    write_json(report_path, record, exclusive=True)
    print(json.dumps({key: record.get(key) for key in (
        "status", "all_roots_returned", "all_roots_certified",
        "all_field_inversions_valid",
        "branch_B_fd_l2_fixed_scale", "branch_B_fd_minus_exact_null_l2",
        "branch_B_fd_minus_default_tcon0_l2", "elapsed_seconds",
        "host_peak_rss_mib")}, indent=2))


if __name__ == "__main__":
    main()
