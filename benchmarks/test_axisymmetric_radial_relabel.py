"""Test whether the NS65 null-direction gap is a simple radial relabeling."""
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
NS = 65
STEP = 1e-5
RESPONSE_DIR = ROOT/"results/vmex/response_runs/axisym-root-response-20260924T232219.753772Z"
RESPONSE_REPORT = RESPONSE_DIR/"response.json"
RESPONSE_ARRAYS = RESPONSE_DIR/"response_ns65.npz"
FINE_DIR = ROOT/"results/audit/axisym_branch_fine_steps/axisym-branch-fine-20260925T000429.848273Z"
FINE_REPORT = FINE_DIR/"fine_branch_probe.json"
FINE_ARRAYS = FINE_DIR/"fine_branch_states_ns65.npz"
INPUT = ROOT/"inputs/input.integer_axisymmetric_current"
OUT_PARENT = ROOT/"results/audit/axisym_radial_relabel"
FIELDS = implicit._STATE_FIELDS


def _norm(value):
    return float(np.linalg.norm(np.asarray(value)))


def _state_from_archive(archive, prefix):
    return implicit.SpectralState(**{
        name: jnp.asarray(archive[f"{prefix}{name}"]) for name in FIELDS
    })


def _add(left, right):
    return jax.tree.map(lambda a, b: a+b, left, right)


def _subtract(left, right):
    return jax.tree.map(lambda a, b: a-b, left, right)


def _score(value):
    return {"l2_over_fixed_scale": _norm(value), "signed": np.asarray(value).tolist()}


def main():
    start = perf_counter()
    source = source_metadata(vmex.__file__, "uwplasma/vmex",
                             importlib.metadata.version("vmex"))
    if source.get("commit") != PIN:
        raise SystemExit("imported VMEX source differs from the historical benchmark pin")
    run_id = datetime.now(timezone.utc).strftime("axisym-radial-relabel-%Y%m%dT%H%M%S.%fZ")
    run_id, out = reserve_run_directory(OUT_PARENT, run_id)
    case = cases()["integer_axisymmetric"]
    parsed = vmex.VmecInput.from_file(INPUT)
    base_input, _, _ = _input_for_parameters(parsed, np.asarray(case.parameters))
    base_input = replace(base_input, ns_array=np.asarray([NS]),
                         ftol_array=np.asarray([1e-12]),
                         niter_array=np.asarray([10000]))
    cfg = implicit.make_config(base_input, ns=NS, ftol=1e-12,
                               max_iterations=10000, refine_tol=1e-11,
                               adjoint_tol=1e-10)
    base_params = implicit.params_from_input(base_input)
    implicit.runtime_from_params(base_params, cfg)

    response = np.load(RESPONSE_ARRAYS, allow_pickle=False)
    fine = np.load(FINE_ARRAYS, allow_pickle=False)
    base_state = _state_from_archive(response, "state_base_")
    residual_tangent = _state_from_archive(response, "state_tangent_delta_")
    plus_state = _state_from_archive(fine, f"state_{STEP:.0e}_plus_")
    minus_state = _state_from_archive(fine, f"state_{STEP:.0e}_minus_")
    branch_state_tangent = jax.tree.map(
        lambda plus, minus: (plus-minus)/(2*STEP), plus_state, minus_state)
    state_gap = _subtract(branch_state_tangent, residual_tangent)

    # VMEX's implicit full-mesh radial coordinate is uniform normalized flux s.
    s = np.linspace(0.0, 1.0, NS)
    base_R1 = np.asarray(base_state.R_cos)[:, 1]
    base_Z1 = np.asarray(base_state.Z_sin)[:, 1]
    dR1_ds = np.gradient(base_R1, s, edge_order=2)
    dZ1_ds = np.gradient(base_Z1, s, edge_order=2)
    gap_R1 = np.asarray(state_gap.R_cos)[:, 1]
    gap_Z1 = np.asarray(state_gap.Z_sin)[:, 1]
    denominator = dR1_ds*dR1_ds+dZ1_ds*dZ1_ds
    eta = np.divide(gap_R1*dR1_ds+gap_Z1*dZ1_ds, denominator,
                    out=np.zeros_like(denominator), where=denominator > 1e-24)
    eta = np.where(np.isfinite(eta), eta, 0.0)

    radial_derivative = jax.tree.map(
        lambda value: jnp.asarray(np.gradient(np.asarray(value), s, axis=0,
                                               edge_order=2)), base_state)
    radial_relabel_tangent = jax.tree.map(
        lambda derivative: jnp.asarray(eta)[:, None]*derivative, radial_derivative)
    candidate_tangent = _add(residual_tangent, radial_relabel_tangent)
    predicted_gap = radial_relabel_tangent
    unrepresented_gap = _subtract(state_gap, predicted_gap)

    plus_input = vmex.VmecInput.from_file(FINE_DIR/f"input_delta_ns{NS}_h{STEP:.0e}_plus.indata")
    minus_input = vmex.VmecInput.from_file(FINE_DIR/f"input_delta_ns{NS}_h{STEP:.0e}_minus.indata")
    params_dot = jax.tree.map(
        lambda plus, minus: (plus-minus)/(2*STEP),
        implicit.params_from_input(plus_input), implicit.params_from_input(minus_input))
    points = jnp.asarray(response["points_xyz_m"])
    initial_flux = jnp.asarray(response["initial_vmex_flux_coordinates"])
    field_fn = _physical_field_function(base_input, cfg, points, initial_flux)
    branch_B_fd = jnp.asarray(fine[f"B_fd_h{STEP:.0e}"])
    field_tangents = jax.block_until_ready((
        branch_B_fd,
        jax.jvp(field_fn, (base_state, base_params),
                (residual_tangent, params_dot))[1],
        jax.jvp(field_fn, (base_state, base_params),
                (candidate_tangent, params_dot))[1],
        jax.jvp(field_fn, (base_state, base_params),
                (branch_state_tangent, params_dot))[1],
    ))
    branch_B_fd, residual_B_jvp, relabel_B_jvp, branch_state_B_jvp = map(
        np.asarray, field_tangents)

    m1_active = np.zeros((NS, 13), dtype=bool)
    m1_active[:, 1] = True
    m1_gap_norm = np.sqrt(sum(
        np.linalg.norm(np.asarray(getattr(state_gap, name))[m1_active])**2
        for name in ("R_cos", "Z_sin")))
    m1_fit_gap = np.sqrt(np.linalg.norm(gap_R1-eta*dR1_ds)**2+
                         np.linalg.norm(gap_Z1-eta*dZ1_ds)**2)
    m1_all_gap = np.sqrt(np.linalg.norm(gap_R1)**2+np.linalg.norm(gap_Z1)**2)
    matching_rows = [row for row in json.loads(FINE_REPORT.read_text())["branch_rows"]
                     if row["step"] == STEP]
    root_residuals = [float(row[side]["residual_after_anchor"])
                      for row in matching_rows for side in ("plus_root", "minus_root")]
    if len(root_residuals) != 2:
        raise RuntimeError("expected both saved centered-branch roots")

    arrays = {
        "radial_coordinate_s": s,
        "fitted_radial_shift_eta": eta,
        "m1_geometry_gap_R_cos": gap_R1,
        "m1_geometry_gap_Z_sin": gap_Z1,
        "m1_geometry_radial_fit_R_cos": eta*dR1_ds,
        "m1_geometry_radial_fit_Z_sin": eta*dZ1_ds,
        "branch_B_fd": branch_B_fd,
        "residual_tangent_B_jvp": residual_B_jvp,
        "radial_relabel_candidate_B_jvp": relabel_B_jvp,
        "measured_branch_state_B_jvp": branch_state_B_jvp,
    }
    arrays_path = out/"radial_relabel_test_ns65.npz"
    np.savez_compressed(arrays_path, **arrays)
    record = {
        "schema": 1,
        "evidence": "axisymmetric_m1_radial_coordinate_relabel_candidate",
        "status": "diagnostic_candidate_not_gauge_certified",
        "accepted_derivative": False,
        "accepted_coordinate_relabel": False,
        "source": source,
        "response_report": RESPONSE_REPORT.relative_to(ROOT).as_posix(),
        "response_report_sha256": sha256_file(RESPONSE_REPORT),
        "response_arrays_sha256": sha256_file(RESPONSE_ARRAYS),
        "fine_branch_report": FINE_REPORT.relative_to(ROOT).as_posix(),
        "fine_branch_report_sha256": sha256_file(FINE_REPORT),
        "fine_branch_arrays_sha256": sha256_file(FINE_ARRAYS),
        "input_sha256": sha256_file(INPUT),
        "ns": NS, "direction": "delta", "step": STEP,
        "radial_coordinate": "uniform normalized toroidal flux s",
        "radial_shift_fit": "least squares to m=1 Rcos and Zsin state gap versus d(base geometry)/ds",
        "m1_geometry_fit_relative_residual": float(m1_fit_gap/m1_all_gap),
        "m1_geometry_gap_l2": float(m1_all_gap),
        "m1_geometry_unfit_gap_l2": float(m1_fit_gap),
        "full_state_gap_l2": float(np.sqrt(sum(
            np.linalg.norm(np.asarray(getattr(state_gap, name)))**2 for name in FIELDS))),
        "full_state_gap_after_radial_fit_l2": float(np.sqrt(sum(
            np.linalg.norm(np.asarray(getattr(unrepresented_gap, name)))**2
            for name in FIELDS))),
        "lambda_sin_gap_l2": _norm(state_gap.L_sin),
        "lambda_sin_gap_after_radial_fit_l2": _norm(unrepresented_gap.L_sin),
        "branch_root_residual_max": max(root_residuals),
        "branch_B_fd": _score(branch_B_fd),
        "residual_tangent_B_jvp": _score(residual_B_jvp),
        "radial_relabel_candidate_B_jvp": _score(relabel_B_jvp),
        "measured_branch_state_B_jvp": _score(branch_state_B_jvp),
        "residual_tangent_B_error": _norm(residual_B_jvp-branch_B_fd),
        "radial_relabel_candidate_B_error": _norm(relabel_B_jvp-branch_B_fd),
        "measured_branch_state_B_error": _norm(branch_state_B_jvp-branch_B_fd),
        "arrays": {"path": arrays_path.name, "sha256": sha256_file(arrays_path)},
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/test_axisymmetric_radial_relabel.py",
        "elapsed_seconds": perf_counter()-start,
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if sys.platform == "darwin" else 1024)),
    }
    report_path = out/"radial_relabel_test.json"
    write_json(report_path, record, exclusive=True)
    print(json.dumps({key: record[key] for key in (
        "status", "m1_geometry_fit_relative_residual", "full_state_gap_l2",
        "full_state_gap_after_radial_fit_l2", "lambda_sin_gap_l2",
        "lambda_sin_gap_after_radial_fit_l2", "residual_tangent_B_error",
        "radial_relabel_candidate_B_error", "measured_branch_state_B_error",
        "elapsed_seconds", "host_peak_rss_mib")}, indent=2))


if __name__ == "__main__":
    main()
