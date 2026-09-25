"""Extend the radial-relabel candidate test to 96 exact fixed Cartesian points."""
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

from analytic import ROOT, cases, label_at_s, surface
from axisymmetric_root_response import _input_for_parameters
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


def _state(archive, prefix):
    return implicit.SpectralState(**{
        name: jnp.asarray(archive[f"{prefix}{name}"]) for name in FIELDS
    })


def _norm(value):
    return float(np.linalg.norm(np.asarray(value)))


def _score(value):
    return {"l2_over_fixed_scale": _norm(value), "signed": np.asarray(value).tolist()}


def _sample_points(case):
    radial_s = np.asarray((0.10, 0.25, 0.45, 0.65, 0.82, 0.94))
    theta = 2*np.pi*np.arange(16)/16
    s, t = np.meshgrid(radial_s, theta, indexing="ij")
    label = label_at_s(case, s)
    xyz = surface(case, label, t, np.full_like(t, 0.37))
    return np.asarray(xyz, dtype=float).reshape((-1, 3)), s.reshape(-1)


def main():
    start = perf_counter()
    source = source_metadata(vmex.__file__, "uwplasma/vmex",
                             importlib.metadata.version("vmex"))
    if source.get("commit") != PIN:
        raise SystemExit("imported VMEX source differs from the historical benchmark pin")
    run_id = datetime.now(timezone.utc).strftime("axisym-relabel-field-%Y%m%dT%H%M%S.%fZ")
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
    runtime = implicit.runtime_from_params(base_params, cfg)

    response = np.load(RESPONSE_ARRAYS, allow_pickle=False)
    fine = np.load(FINE_ARRAYS, allow_pickle=False)
    base_state = _state(response, "state_base_")
    tangent = _state(response, "state_tangent_delta_")
    plus_state = _state(fine, f"state_{STEP:.0e}_plus_")
    minus_state = _state(fine, f"state_{STEP:.0e}_minus_")
    branch_state_tangent = jax.tree.map(
        lambda plus, minus: (plus-minus)/(2*STEP), plus_state, minus_state)
    state_gap = jax.tree.map(lambda a, b: a-b, branch_state_tangent, tangent)

    s_grid = np.linspace(0.0, 1.0, NS)
    dR_ds = np.gradient(np.asarray(base_state.R_cos)[:, 1], s_grid, edge_order=2)
    dZ_ds = np.gradient(np.asarray(base_state.Z_sin)[:, 1], s_grid, edge_order=2)
    gap_R = np.asarray(state_gap.R_cos)[:, 1]
    gap_Z = np.asarray(state_gap.Z_sin)[:, 1]
    denom = dR_ds*dR_ds+dZ_ds*dZ_ds
    eta = np.divide(gap_R*dR_ds+gap_Z*dZ_ds, denom,
                    out=np.zeros_like(denom), where=denom > 1e-24)
    eta = np.where(np.isfinite(eta), eta, 0.0)
    radial_base = jax.tree.map(
        lambda value: jnp.asarray(np.gradient(np.asarray(value), s_grid, axis=0,
                                               edge_order=2)), base_state)
    radial_correction = jax.tree.map(
        lambda value: jnp.asarray(eta)[:, None]*value, radial_base)
    relabel_candidate = jax.tree.map(
        lambda a, b: a+b, tangent, radial_correction)

    plus_input = vmex.VmecInput.from_file(FINE_DIR/f"input_delta_ns{NS}_h{STEP:.0e}_plus.indata")
    minus_input = vmex.VmecInput.from_file(FINE_DIR/f"input_delta_ns{NS}_h{STEP:.0e}_minus.indata")
    plus_params = implicit.params_from_input(plus_input)
    minus_params = implicit.params_from_input(minus_input)
    params_dot = jax.tree.map(lambda plus, minus: (plus-minus)/(2*STEP),
                              plus_params, minus_params)

    points_np, reference_s = _sample_points(case)
    points = jnp.asarray(points_np)
    base_field = vmex.VmecInteriorField.from_state(base_input, base_state, runtime=runtime)
    initial_flux = base_field.flux_coordinates(points)
    jax.block_until_ready(initial_flux)

    def _coordinates(state, params):
        local_runtime = implicit.runtime_from_params(params, cfg)
        spectra = _state_field_spectra(base_input, state, local_runtime)
        return _invert_coordinates(spectra, points, newton_iterations=12,
                                   initial_flux=initial_flux)

    def _field(state, params):
        local_runtime = implicit.runtime_from_params(params, cfg)
        spectra = _state_field_spectra(base_input, state, local_runtime)
        coordinates, valid = _invert_coordinates(
            spectra, points, newton_iterations=12, initial_flux=initial_flux)
        return _cartesian_derivative(spectra, 0, coordinates, valid)

    coordinates_fn = jax.jit(_coordinates)
    field_fn = jax.jit(_field)
    plus_B = field_fn(plus_state, plus_params)
    minus_B = field_fn(minus_state, minus_params)
    plus_coordinates, plus_valid = coordinates_fn(plus_state, plus_params)
    minus_coordinates, minus_valid = coordinates_fn(minus_state, minus_params)
    base_coordinates, base_valid = coordinates_fn(base_state, base_params)
    branch_B_fd = (plus_B-minus_B)/(2*STEP)
    residual_B_jvp = jax.jvp(_field, (base_state, base_params), (tangent, params_dot))[1]
    relabel_B_jvp = jax.jvp(_field, (base_state, base_params),
                            (relabel_candidate, params_dot))[1]
    measured_state_B_jvp = jax.jvp(
        _field, (base_state, base_params), (branch_state_tangent, params_dot))[1]
    candidate_plus_state = jax.tree.map(
        lambda value, direction: value + STEP*direction, base_state, relabel_candidate)
    candidate_minus_state = jax.tree.map(
        lambda value, direction: value - STEP*direction, base_state, relabel_candidate)
    candidate_plus_B = field_fn(candidate_plus_state, plus_params)
    candidate_minus_B = field_fn(candidate_minus_state, minus_params)
    candidate_plus_coordinates, candidate_plus_valid = coordinates_fn(
        candidate_plus_state, plus_params)
    candidate_minus_coordinates, candidate_minus_valid = coordinates_fn(
        candidate_minus_state, minus_params)
    candidate_B_fd = (candidate_plus_B-candidate_minus_B)/(2*STEP)
    values = jax.block_until_ready((branch_B_fd, residual_B_jvp, relabel_B_jvp,
                                   measured_state_B_jvp, base_coordinates,
                                   plus_coordinates, minus_coordinates,
                                   base_valid, plus_valid, minus_valid,
                                   candidate_B_fd, candidate_plus_coordinates,
                                   candidate_minus_coordinates,
                                   candidate_plus_valid, candidate_minus_valid))
    (branch_B_fd, residual_B_jvp, relabel_B_jvp, measured_state_B_jvp,
     base_coordinates, plus_coordinates, minus_coordinates,
     base_valid, plus_valid, minus_valid, candidate_B_fd,
     candidate_plus_coordinates, candidate_minus_coordinates,
     candidate_plus_valid, candidate_minus_valid) = values
    valid = np.asarray(base_valid) & np.asarray(plus_valid) & np.asarray(minus_valid)
    native_rho = np.asarray(base_coordinates)[:, 0]
    native_s = native_rho**2
    label_error = native_s-reference_s
    matching_rows = [row for row in json.loads(FINE_REPORT.read_text())["branch_rows"]
                     if row["step"] == STEP]
    root_residuals = [float(row[side]["residual_after_anchor"])
                      for row in matching_rows for side in ("plus_root", "minus_root")]
    if len(root_residuals) != 2:
        raise RuntimeError("expected both saved centered-branch roots")

    arrays = {
        "points_xyz_m": points_np,
        "reference_s": reference_s,
        "base_native_rho": native_rho,
        "base_native_s": native_s,
        "base_native_coordinates": np.asarray(base_coordinates),
        "plus_native_coordinates": np.asarray(plus_coordinates),
        "minus_native_coordinates": np.asarray(minus_coordinates),
        "base_valid": np.asarray(base_valid),
        "plus_valid": np.asarray(plus_valid),
        "minus_valid": np.asarray(minus_valid),
        "fitted_radial_shift_eta": eta,
        "branch_B_fd": np.asarray(branch_B_fd),
        "residual_tangent_B_jvp": np.asarray(residual_B_jvp),
        "radial_relabel_candidate_B_jvp": np.asarray(relabel_B_jvp),
        "measured_branch_state_B_jvp": np.asarray(measured_state_B_jvp),
        "radial_relabel_candidate_centered_B_fd": np.asarray(candidate_B_fd),
        "candidate_plus_coordinates": np.asarray(candidate_plus_coordinates),
        "candidate_minus_coordinates": np.asarray(candidate_minus_coordinates),
        "candidate_plus_valid": np.asarray(candidate_plus_valid),
        "candidate_minus_valid": np.asarray(candidate_minus_valid),
    }
    arrays_path = out/"radial_relabel_field_ns65.npz"
    np.savez_compressed(arrays_path, **arrays)
    record = {
        "schema": 1,
        "evidence": "axisymmetric_radial_relabel_physical_field_test",
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
        "sample_count": int(points_np.shape[0]),
        "sample_grid": "six normalized-flux surfaces times sixteen analytic poloidal angles at fixed toroidal angle",
        "all_base_plus_minus_inversions_valid": bool(valid.all()),
        "candidate_plus_minus_inversions_valid": bool(
            np.asarray(candidate_plus_valid).all() and np.asarray(candidate_minus_valid).all()),
        "base_native_flux_label_error_definition": "rho from native inversion squared to compare normalized toroidal flux s",
        "base_native_flux_label_error_max": float(np.max(np.abs(label_error))),
        "base_native_flux_label_error_rms": float(np.sqrt(np.mean(label_error**2))),
        "branch_root_residual_max": max(root_residuals),
        "branch_B_fd": _norm(branch_B_fd),
        "residual_tangent_B_jvp": _norm(residual_B_jvp),
        "radial_relabel_candidate_B_jvp": _norm(relabel_B_jvp),
        "radial_relabel_candidate_centered_B_fd": _norm(candidate_B_fd),
        "radial_relabel_candidate_centered_fd_vs_jvp_error": _norm(candidate_B_fd-relabel_B_jvp),
        "radial_relabel_candidate_physical_null_change": _norm(candidate_B_fd),
        "analytic_delta_field_derivative": 0.0,
        "measured_branch_state_B_jvp": _norm(measured_state_B_jvp),
        "residual_tangent_B_error": _norm(residual_B_jvp-branch_B_fd),
        "radial_relabel_candidate_B_error": _norm(relabel_B_jvp-branch_B_fd),
        "measured_branch_state_B_error": _norm(measured_state_B_jvp-branch_B_fd),
        "arrays": {"path": arrays_path.name, "sha256": sha256_file(arrays_path)},
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/validate_axisymmetric_relabel_field.py",
        "elapsed_seconds": perf_counter()-start,
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if sys.platform == "darwin" else 1024)),
    }
    report_path = out/"radial_relabel_field_test.json"
    write_json(report_path, record, exclusive=True)
    print(json.dumps({key: record[key] for key in (
        "status", "sample_count", "all_base_plus_minus_inversions_valid",
        "base_native_flux_label_error_max", "branch_root_residual_max",
        "branch_B_fd", "residual_tangent_B_error",
        "radial_relabel_candidate_B_error", "radial_relabel_candidate_centered_B_fd",
        "radial_relabel_candidate_centered_fd_vs_jvp_error",
        "radial_relabel_candidate_physical_null_change", "measured_branch_state_B_error",
        "elapsed_seconds", "host_peak_rss_mib")}, indent=2))


if __name__ == "__main__":
    main()
