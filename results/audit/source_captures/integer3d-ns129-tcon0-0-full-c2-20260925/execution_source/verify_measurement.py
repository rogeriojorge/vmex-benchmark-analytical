"""Independently certify physical scoring on one saved symmetric VMEX state.

The exact B/J/grad-p oracle uses the supplement's NumPy Cartesian equations
and complex-step derivatives. Route A uses VMEX's Cartesian field API. Route B
evaluates the same VMEX spectral representation through its forward flux chart.
No solve is performed and source states are never rewritten.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import platform

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import vmex
from vmex.core import profiles
from vmex.core.extender import (
    VmecInteriorField,
    _cartesian_derivative,
    _invert_coordinates,
    _position_and_field,
    _prepared,
)
from vmex.core.restart import state_from_wout
from vmex.core.solver import SpectralState
from vmex.core.wout import read_wout

from analytic import ROOT, cases
from build_inputs import FIELD_T, LENGTH_M, MU0
from evidence import (
    acceptance_state, checkpoint_grid, interruption_receipt, normalize_solver_report,
    parent_comparison_rows as _parent_comparison_rows,
    reserve_run_directory, sha256_file, source_metadata, write_json,
)
from measurement import (
    TARGETS, composite_spread as _composite_spread,
    DEFAULT_GRID_SHIFT, deterministic_sample_indices,
    exact_volume_m3,
    integer_exact_fields,
    native_radial_probes,
    reference_grid,
    weighted_square_sum,
)


PINNED_VMEX = "b5f5267efc0795c4a49a224e321e9b370975c14c"
SHIFTED = DEFAULT_GRID_SHIFT
FULL_GRIDS = [
    ("legacy96", 3, 8, 4, 0.0, 0.0, "gauss"),
    ("base_8x32x32", 8, 32, 32, 0.0, 0.0, "gauss"),
    ("base_shifted", 8, 32, 32, *SHIFTED, "gauss"),
    ("radial_16x32x32", 16, 32, 32, 0.0, 0.0, "gauss"),
    ("theta_64", 8, 64, 32, 0.0, 0.0, "gauss"),
    ("phi_64", 8, 32, 64, 0.0, 0.0, "gauss"),
    ("full_16x64x64", 16, 64, 64, 0.0, 0.0, "gauss"),
    ("full_shifted", 16, 64, 64, *SHIFTED, "gauss"),
    ("full_midpoint", 16, 64, 64, 0.0, 0.0, "midpoint"),
]
SMOKE_GRIDS = [
    ("legacy96", 3, 8, 4, 0.0, 0.0, "gauss"),
    ("smoke_4x8x8", 4, 8, 8, 0.0, 0.0, "gauss"),
    ("smoke_shifted", 4, 8, 8, *SHIFTED, "gauss"),
]
RADIAL_EXTENSION_GRIDS = [
    ("full_16x64x64", 16, 64, 64, 0.0, 0.0, "gauss"),
    ("full_shifted", 16, 64, 64, *SHIFTED, "gauss"),
    ("full_midpoint", 16, 64, 64, 0.0, 0.0, "midpoint"),
    ("radial_32x64x64", 32, 64, 64, 0.0, 0.0, "gauss"),
    ("radial_32_shifted", 32, 64, 64, *SHIFTED, "gauss"),
    ("radial_32_midpoint", 32, 64, 64, 0.0, 0.0, "midpoint"),
]
RADIAL64_EXTENSION_GRIDS = [
    ("radial_64x64x64", 64, 64, 64, 0.0, 0.0, "gauss"),
    ("radial_64_shifted", 64, 64, 64, *SHIFTED, "gauss"),
    ("radial_64_midpoint", 64, 64, 64, 0.0, 0.0, "midpoint"),
]
RADIAL128_EXTENSION_GRIDS = [
    ("radial_128x64x64", 128, 64, 64, 0.0, 0.0, "gauss"),
    ("radial_128_shifted", 128, 64, 64, *SHIFTED, "gauss"),
    ("radial_128_midpoint", 128, 64, 64, 0.0, 0.0, "midpoint"),
]
COMPOSITE_GRIDS = [
    ("cell2_gauss", 64, 64, 64, 0.0, 0.0, "cell_gauss", 32, 2),
    ("cell2_shifted", 64, 64, 64, *SHIFTED, "cell_gauss", 32, 2),
    ("cell2_midpoint", 64, 64, 64, 0.0, 0.0, "cell_midpoint", 32, 2),
    ("cell4_gauss", 128, 64, 64, 0.0, 0.0, "cell_gauss", 32, 4),
    ("cell4_shifted", 128, 64, 64, *SHIFTED, "cell_gauss", 32, 4),
    ("cell4_midpoint", 128, 64, 64, 0.0, 0.0, "cell_midpoint", 32, 4),
]


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wout", type=Path)
    source.add_argument("--state-npz", type=Path,
                        help="six-array SpectralState, such as an analytical projection seed")
    parser.add_argument("--legacy-samples", type=Path)
    parser.add_argument("--state-id", required=True)
    parser.add_argument("--output-parent", type=Path,
                        default=ROOT / "results/audit/measurement")
    parser.add_argument("--profile", choices=("smoke", "full", "radial-extension",
                                               "radial64", "radial128", "composite"),
                        default="full")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--expected-vmex", default=PINNED_VMEX)
    parser.add_argument("--parent-run-id")
    return parser


def _rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024**2 if sys.platform == "darwin" else 1024))


def _load_state(args, inp):
    if args.wout is not None:
        wout = read_wout(args.wout)
        state = state_from_wout(wout, inp=inp)
        return state, args.wout.resolve(), sha256_file(args.wout)
    with np.load(args.state_npz, allow_pickle=False) as arrays:
        state = SpectralState(**{name: arrays[name] for name in
            ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    path = args.state_npz.resolve()
    return state, path, sha256_file(path)


def _curl(gradB: np.ndarray) -> np.ndarray:
    return np.stack((gradB[:, 2, 1]-gradB[:, 1, 2],
                     gradB[:, 0, 2]-gradB[:, 2, 0],
                     gradB[:, 1, 0]-gradB[:, 0, 1]), axis=-1)


def _profile_pressure(inp, s):
    return profiles.pressure(
        inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, s,
        pres_scale=inp.pres_scale, bloat=inp.bloat,
        spres_ped=inp.spres_ped,
    )


def _native_target_grid(field, case, native_s, position_fn, *, ntheta=8, nphi=8):
    """Make equal-weight diagnostics at native radial cells and their knots."""
    native_s = np.asarray(native_s, dtype=float)
    if np.any(native_s <= 0) or np.any(native_s >= 1):
        raise ValueError("targeted native flux labels must lie strictly inside (0, 1)")
    theta = 2*np.pi*np.arange(ntheta)/ntheta
    phi = (2*np.pi/case.nfp)*np.arange(nphi)/nphi
    s, th, ph = np.meshgrid(native_s, theta, phi, indexing="ij")
    q = np.column_stack((np.sqrt(s.ravel()), th.ravel(), ph.ravel()))
    xyz = np.asarray(position_fn(jnp.asarray(q)))
    reference_values = integer_exact_fields(case, xyz/LENGTH_M)
    s_reference = reference_values[-1]/case.edge
    return {
        "xyz": xyz,
        "weights": np.full(len(xyz), 1/len(xyz)),
        "s_reference": s_reference,
        "coordinates": q,
        "s_native_target": s.ravel(),
        "volume_quadrature": False,
    }


def _make_evaluators(field, inp):
    """Build each JAX transformation once per state and reuse its shape cache."""
    spectra = _prepared(field.spectra)

    def inverse_coordinate(point, q):
        roots, _ = _invert_coordinates(
            spectra, point[None, :], newton_iterations=field.newton_iterations,
            initial_flux=q[None, :],
        )
        rho, theta, phi = roots[0]
        return jnp.stack((rho*rho, theta, phi))

    return {
        "coordinate_jacobian": jax.jit(jax.vmap(
            jax.jacfwd(inverse_coordinate, argnums=0), in_axes=(0, 0))),
        "dpds": jax.jit(jax.vmap(jax.grad(lambda s: _profile_pressure(inp, s)))),
        "position": jax.jit(jax.vmap(lambda q: _position_and_field(spectra, q)[0])),
        "field": jax.jit(lambda q: _cartesian_derivative(
            spectra, 0, q, jnp.ones((q.shape[0],), dtype=bool))),
        "gradB": jax.jit(lambda q: _cartesian_derivative(
            spectra, 1, q, jnp.ones((q.shape[0],), dtype=bool))),
        "chart_jacobian": jax.jit(jax.vmap(jax.jacfwd(
            lambda q: _position_and_field(spectra, q)[0]))),
    }


def _evaluate_grid(field, inp, case, grid, evaluators, chunk_size, keep_arrays=False):
    """Evaluate in bounded chunks and retain only sufficient statistics by default."""
    xyz = np.asarray(grid["xyz"], dtype=float)
    weights = np.asarray(grid["weights"], dtype=float)
    s_reference = np.asarray(grid["s_reference"], dtype=float)
    count = len(xyz)
    sums = {}
    maxima = {}
    weight_sum = 0.0
    s_label_square_sum = 0.0
    position_error_square_sum = 0.0
    inverse_identity_max = 0.0
    jacobian_abs_min = np.inf
    jacobian_abs_max = 0.0
    jacobian_condition_max = 0.0
    jacobian_sign = None
    position_backward_max = 0.0
    label_difference_max = 0.0
    jacobian_orientation_consistent = True
    samples = {} if keep_arrays else None
    sample_indices = deterministic_sample_indices(count) if keep_arrays else None

    def add_norm(name, values, sample_weights):
        sums[name] = sums.get(name, 0.0) + weighted_square_sum(values, sample_weights)

    def add_max(name, values):
        maximum = float(np.max(values))
        maxima[name] = max(maxima.get(name, -np.inf), maximum)

    def score_summary(prefix, force_key, grad_key):
        fixed_scale = FIELD_T**2/(MU0*LENGTH_M)
        return {
            "field_relative_l2": float(np.sqrt(sums[f"{prefix}_field_error"] /
                                                sums[f"{prefix}_field_reference"])),
            "current_relative_l2": float(np.sqrt(sums[f"{prefix}_current_error"] /
                                                  sums[f"{prefix}_current_reference"])),
            "gradp_relative_l2": float(np.sqrt(sums[f"{prefix}_gradp_error"] /
                                                sums[f"{prefix}_gradp_reference"])),
            "force_pressure_scale": float(np.sqrt(sums[force_key]/sums[grad_key])),
            "force_fixed_magnetic_scale": float(np.sqrt(sums[force_key]/weight_sum)/fixed_scale),
            "field_max_abs_T": maxima[f"{prefix}_field_max"],
            "current_max_abs_A_per_m2": maxima[f"{prefix}_current_max"],
            "force_max_abs_Pa_per_m": maxima[f"{prefix}_force_max"],
            "divergence_max_abs_per_m": maxima[f"{prefix}_divergence_max"],
        }

    for first in range(0, count, chunk_size):
        last = min(first+chunk_size, count)
        chunk_weights = weights[first:last]
        points = jnp.asarray(xyz[first:last])
        s_ref = s_reference[first:last]
        coords = np.asarray(field.flux_coordinates(points))
        rho = np.sqrt(coords[:, 0])
        q_native = jnp.asarray(np.column_stack((rho, coords[:, 1], coords[:, 2])))
        position = np.asarray(evaluators["position"](q_native))
        field.set_points_flux(jnp.asarray(coords))
        A_B = np.asarray(field.B())
        A_gradB = np.asarray(field.gradB())
        A_coord_jac = np.asarray(evaluators["coordinate_jacobian"](position, q_native))
        A_dpds = np.asarray(evaluators["dpds"](jnp.asarray(coords[:, 0])))
        B_field = np.asarray(evaluators["field"](q_native))
        B_gradB = np.asarray(evaluators["gradB"](q_native))
        B_chart_jac = np.asarray(evaluators["chart_jacobian"](q_native))
        B_dpds = np.asarray(evaluators["dpds"](jnp.asarray(coords[:, 0])))

        A_J = _curl(A_gradB)/MU0
        A_gradp = A_dpds[:, None]*A_coord_jac[:, 0, :]
        q_gradient = np.zeros((len(coords), 3), dtype=float)
        q_gradient[:, 0] = 2*rho*B_dpds
        B_gradp = np.linalg.solve(
            B_chart_jac.swapaxes(1, 2), q_gradient[..., None])[..., 0]
        B_J = _curl(B_gradB)/MU0

        D_x_rho = A_coord_jac.copy()
        D_x_rho[:, 0, :] /= 2*rho[:, None]
        identity_error = np.einsum("nij,njk->nik", B_chart_jac, D_x_rho)-np.eye(3)[None, :, :]
        forward_error = position-xyz[first:last]
        position_backward_max = max(position_backward_max, float(np.max(np.linalg.norm(forward_error, axis=-1))))
        label_difference_max = max(label_difference_max, float(np.max(np.abs(coords[:, 0]-s_ref))))
        exact_A_values = integer_exact_fields(case, xyz[first:last]/LENGTH_M)
        exact_A = {
            "B": FIELD_T*exact_A_values[0],
            "J": FIELD_T/(MU0*LENGTH_M)*exact_A_values[1],
            "gradp": FIELD_T**2/(MU0*LENGTH_M)*exact_A_values[2],
        }
        exact_B_values = integer_exact_fields(case, position/LENGTH_M)
        exact_B = {
            "B": FIELD_T*exact_B_values[0],
            "J": FIELD_T/(MU0*LENGTH_M)*exact_B_values[1],
            "gradp": FIELD_T**2/(MU0*LENGTH_M)*exact_B_values[2],
        }
        pressure_scale = exact_A["gradp"]
        route_values = {
            "A": (A_B, A_J, A_gradp, A_gradB, exact_A),
            "B": (B_field, B_J, B_gradp, B_gradB, exact_B),
        }
        for label, (B, J, gradp, gradB, exact) in route_values.items():
            prefix = f"route_{label}"
            force = np.cross(J, B)-gradp
            for metric, actual, expected in (
                ("field", B, exact["B"]),
                ("current", J, exact["J"]),
                ("gradp", gradp, exact["gradp"]),
            ):
                add_norm(f"{prefix}_{metric}_error", actual-expected, chunk_weights)
                add_norm(f"{prefix}_{metric}_reference", expected, chunk_weights)
                add_max(f"{prefix}_{metric}_max", np.linalg.norm(actual-expected, axis=-1))
            add_norm(f"{prefix}_force_error", force, chunk_weights)
            add_max(f"{prefix}_force_max", np.linalg.norm(force, axis=-1))
            add_max(f"{prefix}_divergence_max", np.abs(np.trace(gradB, axis1=1, axis2=2)))

        for name, difference, actual in (
            ("field", A_B-B_field, A_B),
            ("current", A_J-B_J, A_J),
            ("gradp", A_gradp-B_gradp, A_gradp),
        ):
            add_norm(f"route_difference_{name}_error", difference, chunk_weights)
            add_norm(f"route_difference_{name}_reference", actual, chunk_weights)
        force_A = np.cross(A_J, A_B)-A_gradp
        force_B = np.cross(B_J, B_field)-B_gradp
        add_norm("route_difference_force_error", force_A-force_B, chunk_weights)
        add_norm("route_difference_force_reference", pressure_scale, chunk_weights)

        weight_sum += float(np.sum(chunk_weights))
        position_error_square_sum += float(np.sum(
            chunk_weights*np.sum(forward_error*forward_error, axis=-1)))
        s_delta = coords[:, 0]-s_ref
        s_label_square_sum += float(np.sum(s_delta*s_delta))
        inverse_identity_max = max(inverse_identity_max, float(np.max(np.abs(identity_error))))
        det = np.linalg.det(B_chart_jac)
        det_signs = np.sign(det)
        if jacobian_sign is None:
            jacobian_sign = int(det_signs[0])
        jacobian_orientation_consistent &= bool(np.all(det_signs == jacobian_sign))
        jacobian_abs_min = min(jacobian_abs_min, float(np.min(np.abs(det))))
        jacobian_abs_max = max(jacobian_abs_max, float(np.max(np.abs(det))))
        jacobian_condition_max = max(jacobian_condition_max,
                                     float(np.max(np.linalg.cond(B_chart_jac))))

        if keep_arrays:
            retained = sample_indices[(sample_indices >= first) & (sample_indices < last)]-first
            chunk_samples = {
                "xyz": xyz[first:last], "weights": chunk_weights,
                "s_reference": s_ref, "B_route_A": A_B, "J_route_A": A_J,
                "gradp_route_A": A_gradp, "s_native": coords[:, 0],
                "B_route_B": B_field, "J_route_B": B_J,
                "gradp_route_B": B_gradp,
                "B_exact_route_A": exact_A["B"], "J_exact_route_A": exact_A["J"],
                "gradp_exact_route_A": exact_A["gradp"],
            }
            for name, values in chunk_samples.items():
                samples.setdefault(name, []).append(values[retained])

    if weight_sum <= 0:
        raise ValueError("grid must have positive total weight")
    scores_A = score_summary("route_A", "route_A_force_error", "route_A_gradp_reference")
    scores_B = score_summary("route_B", "route_B_force_error", "route_B_gradp_reference")
    route_difference = {
        metric: float(np.sqrt(sums[f"route_difference_{metric}_error"] /
                              sums[f"route_difference_{metric}_reference"]))
        for metric in ("field", "current", "gradp", "force")
    }
    volume_quadrature = bool(grid.get("volume_quadrature", True))
    expected_volume = exact_volume_m3(case, LENGTH_M) if volume_quadrature else None
    result = {
        "sample_count": count,
        "sample_measure": ("full_torus_physical_volume" if volume_quadrature
                           else "targeted_equal_weight_point_diagnostic"),
        "physical_volume_m3": weight_sum if volume_quadrature else None,
        "exact_volume_m3": expected_volume,
        "volume_relative_error": (
            abs(weight_sum-expected_volume)/expected_volume if volume_quadrature else None
        ),
        "angular_measure": "one field period with exact NFP replication to full torus",
        "reference_jacobian_orientation_consistent": jacobian_orientation_consistent,
        "reference_jacobian_sign": jacobian_sign,
        "native_jacobian_min_abs_m3": jacobian_abs_min,
        "native_jacobian_max_abs_m3": jacobian_abs_max,
        "native_jacobian_condition_max": jacobian_condition_max,
        "route_A": scores_A,
        "route_B": scores_B,
        "route_A_B_difference": route_difference,
        "coordinate_inversion": {
            "position_backward_error_max_m": position_backward_max,
            "position_backward_error_rms_m": float(np.sqrt(position_error_square_sum/weight_sum)),
            "inverse_jacobian_identity_max_abs": inverse_identity_max,
            "s_native_minus_reference_max_abs": label_difference_max,
            "s_native_minus_reference_rms": float(np.sqrt(s_label_square_sum/count)),
        },
    }
    if keep_arrays:
        result["samples"] = {name: np.concatenate(chunks, axis=0)
                             for name, chunks in samples.items()}
        if "s_native_target" in grid:
            result["samples"]["s_native_target"] = np.asarray(grid["s_native_target"])[sample_indices]
        result["samples_metadata"] = {
            "full_sample_count": count,
            "retained_sample_count": int(len(sample_indices)),
            "selection": "uniform_index_spacing_with_endpoints",
        }
    return result


def _legacy_comparison(grid, result, legacy_path):
    if legacy_path is None or result.get("samples") is None:
        return None
    with np.load(legacy_path, allow_pickle=False) as old:
        old_xyz = np.asarray(old["xyz"])
        old_B = np.asarray(old["B"])
        old_J = np.asarray(old["J"])
        old_gp = np.asarray(old["gradp"])
        old_weights = np.asarray(old["weights"])
    arrays = result["samples"]
    if old_xyz.shape != arrays["xyz"].shape:
        return {"status": "point_count_mismatch", "legacy_sample_count": len(old_xyz)}
    return {
        "status": "compared",
        "max_position_difference_m": float(np.max(np.linalg.norm(old_xyz-arrays["xyz"], axis=-1))),
        "B_relative_rms_difference": float(np.linalg.norm(old_B-arrays["B_route_A"])
                                              / max(np.linalg.norm(old_B), np.finfo(float).tiny)),
        "J_relative_rms_difference": float(np.linalg.norm(old_J-arrays["J_route_A"])
                                              / max(np.linalg.norm(old_J), np.finfo(float).tiny)),
        "gradp_relative_rms_difference": float(np.linalg.norm(old_gp-arrays["gradp_route_A"])
                                                  / max(np.linalg.norm(old_gp), np.finfo(float).tiny)),
        "legacy_volume_m3": float(np.sum(old_weights)),
        "new_full_torus_volume_m3": float(np.sum(arrays["weights"])),
        "legacy_sample_sha256": sha256_file(legacy_path),
    }


def _write_npz_checkpoint(path: Path, arrays: dict[str, np.ndarray]) -> dict:
    """Write a sample artifact once, returning a path/hash receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        np.savez_compressed(output, **arrays)
    return {"path": path.relative_to(path.parents[1]).as_posix(),
            "sha256": sha256_file(path)}


def _fine_spread(rows):
    groups = {}
    for row in rows:
        if (row.get("sample_measure") == "full_torus_physical_volume" and
                row.get("ntheta") == 64 and row.get("nphi_one_period") == 64):
            groups.setdefault(row["nradial"], []).append(row)
    fine = None
    required = {
        ("gauss", 0.0, 0.0),
        ("gauss", SHIFTED[0], SHIFTED[1]),
        ("midpoint", 0.0, 0.0),
    }
    for radial_count in sorted(groups, reverse=True):
        candidates = groups[radial_count]
        signatures = {(row["radial_rule"], row["theta_shift_fraction"],
                       row["phi_shift_fraction"]) for row in candidates}
        if required <= signatures:
            fine = [next(row for row in candidates if
                         (row["radial_rule"], row["theta_shift_fraction"],
                          row["phi_shift_fraction"]) == signature)
                    for signature in required]
            break
    if fine is None:
        return None, False
    spread = {}
    resolved = True
    for metric, limit in TARGETS.items():
        values = [row["route_A"][metric] for row in fine]
        amount = max(values)-min(values)
        spread[metric] = amount
        if amount > 0.1*limit:
            resolved = False
    fine_radial = max(row["nradial"] for row in fine)
    lower_radials = [radial_count for radial_count in groups if radial_count < fine_radial]
    if lower_radials:
        lower_radial = max(lower_radials)
        fine_zero = next(row for row in fine if
                         row["radial_rule"] == "gauss" and
                         row["theta_shift_fraction"] == 0.0 and
                         row["phi_shift_fraction"] == 0.0)
        lower_zero = next((row for row in groups[lower_radial] if
                           row["radial_rule"] == "gauss" and
                           row["theta_shift_fraction"] == 0.0 and
                           row["phi_shift_fraction"] == 0.0), None)
        level_change = {}
        if lower_zero is not None:
            for metric, limit in TARGETS.items():
                amount = abs(fine_zero["route_A"][metric]-lower_zero["route_A"][metric])
                level_change[metric] = amount
                if amount > 0.1*limit:
                    resolved = False
        else:
            level_change = None
            resolved = False
    else:
        level_change = None
        resolved = False
    spread["previous_radial_level_change"] = level_change
    if max(row["volume_relative_error"] for row in fine) > 1e-10:
        resolved = False
    for row in fine:
        for metric, limit in TARGETS.items():
            if row["route_A_B_difference"][metric] > 0.1*limit:
                resolved = False
    return spread, resolved






def main(argv=None):
    args = _parser().parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit("input deck does not exist")
    if args.chunk_size < 1:
        raise SystemExit("chunk size must be positive")
    case_row = json.loads((input_path.parent/"manifest.json").read_text())["records"]
    case_row = next((row for row in case_row if row["file"] == input_path.name), None)
    if case_row is None:
        raise SystemExit("input deck has no generated manifest row")
    case = cases()[case_row["case"]]
    if case.family != "integer":
        raise SystemExit("R1 currently supports the integer family; sheared sampling follows after chart review")
    inp = vmex.VmecInput.from_file(input_path)
    if inp.lfreeb:
        raise SystemExit("R1 fixed-state measurement requires a fixed-boundary input")
    state, _state_source, state_source_sha = _load_state(args, inp)
    imported_source = source_metadata(
        vmex.__file__, "uwplasma/vmex", importlib.metadata.version("vmex"))
    if imported_source.get("commit") != args.expected_vmex:
        raise SystemExit(
            f"wrong VMEX source pin: expected {args.expected_vmex}, "
            f"loaded {imported_source.get('commit')}"
        )
    field = VmecInteriorField.from_state(inp, state)
    evaluators = _make_evaluators(field, inp)
    source_report = (args.wout.parent/"forward.json") if args.wout is not None else None
    solver_report = None
    if source_report is not None and source_report.is_file():
        solver_report = json.loads(source_report.read_text(encoding="utf-8"))
    normalized_solver_report = (None if solver_report is None else
                                normalize_solver_report(solver_report))
    solver_provenance = normalized_solver_report
    solver_source_matches = (
        normalized_solver_report is not None and
        normalized_solver_report.get("source_commit") == imported_source.get("commit")
    )
    solver_converged = (
        normalized_solver_report.get("solver_converged") if solver_source_matches else None
    )
    runtime = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "numpy_version": np.__version__,
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_backend": jax.default_backend(),
        "devices": sorted({str(device.device_kind) for device in jax.devices()}),
    }
    benchmark_files = (
        Path(__file__),
        Path(__file__).with_name("measurement.py"),
        Path(__file__).with_name("evidence.py"),
        Path(__file__).with_name("analytic.py"),
        Path(__file__).with_name("build_inputs.py"),
        input_path.parent/"manifest.json",
    )
    benchmark_code_sha256 = {
        path.resolve().relative_to(ROOT.resolve()).as_posix(): sha256_file(path)
        for path in benchmark_files
    }

    parent_report = None
    comparison_parent = None
    parent_comparison_rows = None
    if args.profile in {"radial64", "radial128", "composite"}:
        if not args.parent_run_id:
            raise SystemExit(f"{args.profile} requires a complete parent measurement run")
        parent_path = args.output_parent/args.parent_run_id/"measurement.json"
        if not parent_path.is_file():
            raise SystemExit("parent measurement report is missing")
        parent_report = json.loads(parent_path.read_text(encoding="utf-8"))
        expected_parent_identity = (
            imported_source.get("commit"), case.name, state_source_sha,
            sha256_file(input_path),
        )
        if parent_report.get("run_id") != args.parent_run_id:
            raise SystemExit("parent report does not match the source, input and saved state")
        parent_comparison_rows = _parent_comparison_rows(
            args.output_parent, parent_report, expected_parent_identity)
        parent_spread, _ = _fine_spread(parent_comparison_rows)
        if (parent_report.get("status") not in {"measurement_diagnostic", "measurement_resolved"} or
                parent_spread is None or parent_spread.get("previous_radial_level_change") is None):
            raise SystemExit("parent ancestry lacks a complete prior-radial comparison")
        comparison_parent = {
            "run_id": parent_report["run_id"],
            "report_sha256": sha256_file(parent_path),
            "preceding_radial_level_spread": parent_spread,
        }

    run_id, out = reserve_run_directory(args.output_parent, args.state_id)
    (out/"input.indata").write_bytes(input_path.read_bytes())
    state_path = out/"spectral_state.npz"
    np.savez_compressed(state_path, **{
        name: np.asarray(getattr(state, name))
        for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
    })
    if args.profile == "full":
        grids = FULL_GRIDS
    elif args.profile == "radial-extension":
        grids = RADIAL_EXTENSION_GRIDS
    elif args.profile == "radial64":
        grids = RADIAL64_EXTENSION_GRIDS
    elif args.profile == "radial128":
        grids = RADIAL128_EXTENSION_GRIDS
    elif args.profile == "composite":
        grids = [(name, nr, nt, nphi, tshift, pshift, rule, cells, order)
                 for (name, nr, nt, nphi, tshift, pshift, rule, cells, order)
                 in COMPOSITE_GRIDS]
        grids = [(name, nr, nt, nphi, tshift, pshift, rule, state.R_cos.shape[0]-1, order)
                 for (name, nr, nt, nphi, tshift, pshift, rule, _cells, order) in grids]
    else:
        grids = SMOKE_GRIDS
    grid_ids = [row[0] for row in grids]
    if args.profile != "smoke":
        grid_ids.extend(
            f"target_{group_id}"
            for group_id in native_radial_probes(state.R_cos.shape[0])
        )
    running = {
        "schema": 1,
        "run_id": run_id,
        "parent_run_id": args.parent_run_id,
        "comparison_parent": comparison_parent,
        "status": "running",
        "profile": args.profile,
        "case": case.name,
        "state_id": args.state_id,
        "vmex_source": imported_source,
        "benchmark_code_sha256": benchmark_code_sha256,
        "runtime": runtime,
        "state_source_sha256": state_source_sha,
        "state_source_kind": "saved_wout" if args.wout is not None else "saved_spectral_state",
        "input_sha256": sha256_file(input_path),
        "solver_report_sha256": (
            sha256_file(source_report) if source_report is not None and source_report.is_file() else None
        ),
        "solver_provenance": solver_provenance,
        "solver_source_matches_imported": solver_source_matches,
        "solver_converged": solver_converged,
        "input_boundary_fit_m": case_row.get("boundary_max_error_m"),
        "input_omitted_basis_max_m": case_row.get("symmetric_omitted_coeff_m"),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "grid_ids": grid_ids,
        "host_peak_rss_mib": None,
        "device_memory_peak_mib": None,
    }
    write_json(out/"measurement.json", running, exclusive=True)
    (out/"input.indata").chmod(0o444)
    (state_path).chmod(0o444)
    measurement_start = perf_counter()
    rows = []
    completed_grid_ids = []
    finest_sample_artifact = None
    targeted_sample_artifacts = {}

    def persist_completed_grid(row):
        receipt = checkpoint_grid(out, row["grid_id"], row)
        completed_grid_ids.append(row["grid_id"])
        running.update(
            status="running",
            completed_grids=list(completed_grid_ids),
            latest_grid_checkpoint=receipt,
            checkpoint_manifest_sha256=sha256_file(out/"grid_checkpoints"/"checkpoints.json"),
        )
        write_json(out/"measurement.json", running)

    try:
        for grid_spec in grids:
            grid_id, nr, nt, np_, theta_shift, phi_shift, radial_rule = grid_spec[:7]
            radial_cells, radial_order = grid_spec[7:] if len(grid_spec) == 9 else (None, 1)
            sample_start = perf_counter()
            grid = reference_grid(
                case, nradial=nr, ntheta=nt, nphi=np_,
                theta_shift=theta_shift, phi_shift=phi_shift,
                radial_rule=radial_rule, radial_cells=radial_cells,
                radial_order=radial_order, length_m=LENGTH_M,
            )
            keep = grid_id in {"legacy96", "full_16x64x64", "radial_32x64x64",
                               "radial_64x64x64", "radial_128x64x64", "cell4_gauss"}
            result = _evaluate_grid(field, inp, case, grid, evaluators, args.chunk_size,
                                    keep_arrays=keep)
            sample_artifact = None
            if keep:
                arrays = result.pop("samples")
                sample_artifact = _write_npz_checkpoint(
                    out/"grid_checkpoints"/f"{grid_id}_samples.npz", arrays)
                if grid_id == "legacy96":
                    result["legacy_sample_reproduction"] = _legacy_comparison(
                        grid, {**result, "samples": arrays}, args.legacy_samples)
                if grid_id in {"full_16x64x64", "radial_32x64x64", "radial_64x64x64",
                               "radial_128x64x64", "cell4_gauss"}:
                    finest_sample_artifact = sample_artifact
                arrays = None
            row = {
                "grid_id": grid_id,
                "nradial": nr,
                "ntheta": nt,
                "nphi_one_period": np_,
                "theta_shift_fraction": theta_shift,
                "phi_shift_fraction": phi_shift,
                "radial_rule": radial_rule,
                "radial_cells": radial_cells,
                "radial_order": radial_order if radial_rule.startswith("cell_") else None,
                "sampling_score_seconds": perf_counter()-sample_start,
                "samples_artifact": sample_artifact,
                **result,
            }
            rows.append(row)
            persist_completed_grid(row)
            print(json.dumps({"grid_id": grid_id,
                              "route_A": result["route_A"],
                              "volume_relative_error": result["volume_relative_error"]}),
                  flush=True)
        if args.profile != "smoke":
            radial_probes = native_radial_probes(state.R_cos.shape[0])
            for group_id, native_s in radial_probes.items():
                grid_id = f"target_{group_id}"
                sample_start = perf_counter()
                try:
                    target_grid = _native_target_grid(
                        field, case, native_s, evaluators["position"])
                    result = _evaluate_grid(field, inp, case, target_grid, evaluators,
                                            args.chunk_size, keep_arrays=True)
                    arrays = result.pop("samples")
                    sample_artifact = _write_npz_checkpoint(
                        out/"grid_checkpoints"/f"{grid_id}_samples.npz", arrays)
                    targeted_sample_artifacts[group_id] = sample_artifact
                    arrays = None
                    row = {
                        "grid_id": grid_id,
                        "status": "measured",
                        "native_s_targets": native_s,
                        "native_radial_mesh_points": int(state.R_cos.shape[0]),
                        "n_theta": 8,
                        "n_phi_one_period": 8,
                        "sampling_score_seconds": perf_counter()-sample_start,
                        "samples_artifact": sample_artifact,
                        **result,
                    }
                    rows.append(row)
                    persist_completed_grid(row)
                    print(json.dumps({"grid_id": grid_id,
                                      "route_A": result["route_A"],
                                      "route_A_B_difference": result["route_A_B_difference"]}),
                          flush=True)
                except Exception as exc:
                    rows.append({
                        "grid_id": grid_id,
                        "status": "targeted_sampling_failed",
                        "native_s_targets": native_s,
                        "native_radial_mesh_points": int(state.R_cos.shape[0]),
                        "failure": {"type": type(exc).__name__, "message": str(exc)},
                        "sampling_score_seconds": perf_counter()-sample_start,
                    })
                    persist_completed_grid(rows[-1])
    except KeyboardInterrupt as exc:
        receipt = interruption_receipt(
            out, run_id=run_id, completed_grids=completed_grid_ids, error=exc)
        running.update(
            status="measurement_interrupted",
            interruption=receipt,
            completed_grids=list(completed_grid_ids),
            rows=rows,
            host_peak_rss_mib=_rss_mib(),
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(out/"measurement.json", running)
        raise
    except Exception as exc:
        running.update(
            status="measurement_failed",
            failure={"type": type(exc).__name__, "message": str(exc)},
            rows=rows,
            host_peak_rss_mib=_rss_mib(),
            completed_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(out/"measurement.json", running)
        raise

    comparison_rows = (parent_comparison_rows + rows
                       if parent_comparison_rows is not None else rows)
    if args.profile == "composite":
        spread, grid_measurement_resolved = _composite_spread(rows)
    else:
        spread, grid_measurement_resolved = _fine_spread(comparison_rows)
    targeted_rows = [row for row in rows if row["grid_id"].startswith("target_")]
    targeted_sampling_complete = (
        args.profile == "smoke" or
        len(targeted_rows) == len(native_radial_probes(state.R_cos.shape[0])) and
        all(row.get("status") == "measured" for row in targeted_rows)
    )
    targeted_thresholds_met = (
        args.profile == "smoke" or
        targeted_sampling_complete and all(
            all(row["route_A"][metric] <= target for metric, target in TARGETS.items())
            for row in targeted_rows
        )
    )
    targeted_route_agreement = (
        args.profile == "smoke" or
        targeted_sampling_complete and all(
            all(row["route_A_B_difference"][metric] <= 0.1*target
                for metric, target in TARGETS.items())
            for row in targeted_rows
        )
    )
    measurement_resolved = (grid_measurement_resolved and targeted_sampling_complete
                            and targeted_route_agreement)
    sample_artifact = finest_sample_artifact
    target_artifact = targeted_sample_artifacts or None
    volume_rows = [row for row in rows if row.get("sample_measure") == "full_torus_physical_volume"]
    best = (next(row for row in volume_rows if row["grid_id"] == "cell4_gauss")
            if args.profile == "composite" else max(volume_rows, key=lambda row: (row["nradial"], row["ntheta"],
                                             row["nphi_one_period"],
                                             row["radial_rule"] == "gauss",
                                             row["theta_shift_fraction"] == 0.0 and
                                             row["phi_shift_fraction"] == 0.0)))
    pointwise = (all(best["route_A"][metric] <= target for metric, target in TARGETS.items())
                 and targeted_thresholds_met)
    root_certified = False
    representation_resolved = (
        case_row.get("boundary_max_error_m", float("inf")) <= 1e-6 and
        case_row.get("symmetric_omitted_coeff_m", float("inf")) <= 1e-10
    )
    gates = acceptance_state(
        solver_converged=bool(solver_converged),
        pointwise_thresholds_met=pointwise,
        measurement_resolved=measurement_resolved,
        representation_resolved=representation_resolved,
        root_certified=root_certified,
    )
    summary = {
        **running,
        "status": "measurement_resolved" if measurement_resolved else "measurement_diagnostic",
        "rows": rows,
        "targets": TARGETS,
        "fine_grid_score_spread": spread,
        "grid_measurement_resolved": grid_measurement_resolved,
        "targeted_sampling_complete": targeted_sampling_complete,
        "targeted_route_agreement": targeted_route_agreement,
        "targeted_pointwise_thresholds_met": targeted_thresholds_met,
        "measurement_resolved": measurement_resolved,
        "solver_converged": solver_converged,
        "solver_source_matches_imported": solver_source_matches,
        "solver_provenance": solver_provenance,
        "solver_report_sha256": running["solver_report_sha256"],
        "pointwise_thresholds_met": pointwise,
        "representation_resolved": representation_resolved,
        "root_certified": root_certified,
        "accepted": gates["accepted"],
        "angular_measure": "one physical field period, replicated by NFP to the full torus",
        "reference_oracle": "Landreman supplement integer Cartesian equations; NumPy complex-step derivatives",
        "route_A": "VMEX VmecInteriorField Cartesian B, gradB, and flux-coordinate API",
        "route_B": "same VMEX native spectral representation through its forward flux chart and chain rule",
        "uncertainty_note": "Grid/shift spread is an empirical measurement estimate, not a rigorous bound.",
        "elapsed_seconds": perf_counter()-measurement_start,
        "host_peak_rss_mib": _rss_mib(),
        "device_memory_peak_mib": None,
        "artifacts": {
            "input_copy": {"path": "input.indata", "sha256": sha256_file(out/"input.indata")},
            "spectral_state": {"path": state_path.name, "sha256": sha256_file(state_path)},
            "finest_samples": sample_artifact,
            "targeted_samples": target_artifact,
        },
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out/"measurement.json", summary)
    print(json.dumps({"run_id": run_id, "status": summary["status"],
                      "measurement_resolved": measurement_resolved,
                      "accepted": summary["accepted"],
                      "output_run_id": run_id}, indent=2), flush=True)


if __name__ == "__main__":
    main()
