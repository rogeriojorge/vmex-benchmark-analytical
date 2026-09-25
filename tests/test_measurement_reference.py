import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from analytic import cases, differential_fields, surface, volume
from evidence import parent_comparison_rows
from measurement import (TARGETS, composite_spread, integer_exact_fields,
                         decompose_physical_force_error,
                         integer_surface_jax, integer_surface_numpy,
                         compare_pressure_profiles, deterministic_sample_indices,
                         native_knot_crossings,
                         native_radial_probes, reference_grid, weighted_square_sum)


def test_integer_chart_matches_analytical_surface_and_full_torus_volume():
    case = cases()["integer_3d"]
    grid = reference_grid(case, nradial=8, ntheta=32, nphi=32,
                          theta_shift=0.173, phi_shift=0.371)
    expected_volume = volume(case)
    assert grid["nfp_replication"] == case.nfp == 2
    np.testing.assert_allclose(np.sum(grid["weights"]), expected_volume,
                               rtol=2e-12, atol=0)

    coordinates = grid["coordinates"][[0, 97, 1234, -1]]
    expected_xyz = surface(case, case.edge*coordinates[:, 0],
                           coordinates[:, 1], coordinates[:, 2])
    np.testing.assert_allclose(grid["xyz"][[0, 97, 1234, -1]], expected_xyz,
                               rtol=2e-14, atol=2e-14)
    chart_coordinates = grid["coordinates"][[0, 97, 1234, -1]]
    np.testing.assert_allclose(
        integer_surface_numpy(case, chart_coordinates),
        jax.vmap(lambda q: integer_surface_jax(case, q))(jnp.asarray(chart_coordinates)),
        rtol=2e-14, atol=2e-14,
    )


def test_numpy_complex_step_fields_match_jax_derivative_reference():
    case = cases()["integer_3d"]
    coordinates = np.array([[0.24, 0.37, 0.61],
                            [0.63, 1.20, 2.30],
                            [0.86, 4.80, 5.90]])
    xyz = surface(case, case.edge*coordinates[:, 0],
                  coordinates[:, 1], coordinates[:, 2])
    independent = integer_exact_fields(case, xyz)
    automatic = differential_fields(case, xyz)
    for actual, expected in zip(independent, automatic):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=3e-14)


def test_reference_volume_matches_one_period_and_full_torus_formulations():
    case = cases()["integer_3d"]
    one_period = reference_grid(case, nradial=8, ntheta=32, nphi=32)
    full_torus = reference_grid(replace(case, nfp=1), nradial=8, ntheta=32, nphi=64)
    assert np.isclose(one_period["weights"].sum(), volume(case), rtol=1e-11)
    assert np.isclose(full_torus["weights"].sum(), volume(case), rtol=1e-11)
    np.testing.assert_allclose(one_period["weights"].sum(),
                               full_torus["weights"].sum(), rtol=1e-12, atol=0)


def test_knot_aligned_composite_grid_covers_each_cell_and_full_torus():
    case = cases()["integer_3d"]
    cells, order = 32, 4
    grid = reference_grid(case, nradial=cells*order, ntheta=64, nphi=64,
                          radial_rule="cell_gauss", radial_cells=cells,
                          radial_order=order)
    radial = np.unique(grid["coordinates"][:, 0]).reshape(cells, order)
    local = radial*cells-np.arange(cells)[:, None]
    assert np.all((local > 0) & (local < 1))
    assert grid["radial_cells"] == cells
    assert grid["radial_order"] == order
    np.testing.assert_allclose(grid["weights"].sum(), volume(case), rtol=1e-10, atol=0)


def test_composite_resolution_requires_order_spread_level_change_and_route_agreement():
    signatures = (("cell_gauss", 0.0, 0.0),
                  ("cell_gauss", 0.173, 0.371),
                  ("cell_midpoint", 0.0, 0.0))
    rows = []
    for order in (2, 4):
        for index, (rule, theta_shift, phi_shift) in enumerate(signatures):
            route_a = {metric: limit*(0.2+0.02*(order == 4)+0.02*index)
                       for metric, limit in TARGETS.items()}
            rows.append({
                "sample_measure": "full_torus_physical_volume",
                "radial_rule": rule,
                "radial_order": order,
                "radial_cells": 32,
                "theta_shift_fraction": theta_shift,
                "phi_shift_fraction": phi_shift,
                "route_A": route_a,
                "route_A_B_difference": {metric: limit*0.01
                                         for metric, limit in TARGETS.items()},
                "volume_relative_error": 1e-13,
            })

    spread, resolved = composite_spread(rows)
    assert resolved
    assert spread["native_cell_count"] == 32
    assert set(spread["by_order"]) == {"2", "4"}

    rows[-1]["route_A"]["field_relative_l2"] += 0.2*TARGETS["field_relative_l2"]
    _, resolved = composite_spread(rows)
    assert not resolved


def test_native_radial_probes_are_one_sided_and_inside_domain():
    probes = native_radial_probes(129)
    axis = np.asarray(probes["near_axis"])
    edge = np.asarray(probes["near_edge"])
    knots = np.asarray(probes["native_knots_and_cell_interiors"])
    assert np.all((axis > 0) & (axis < 0.02))
    assert np.all((edge > 0.9) & (edge < 1))
    assert np.all((knots > 0) & (knots < 1))
    assert len(np.unique(knots)) == len(knots)
    np.testing.assert_allclose(knots[4:6],
                               [0.5-1e-5/128, 0.5+1e-5/128], rtol=0, atol=1e-15)


def test_pressure_audit_separates_profile_fit_from_coordinate_label_shift():
    exact = np.array([1.0, 2.0])
    fitted = np.array([1.0, 2.1])
    native = np.array([1.2, 2.1])
    result = compare_pressure_profiles(native, fitted, exact, np.ones(2), 1.0)
    np.testing.assert_allclose(result["input_profile_fit_rms_fixed_scale"], np.sqrt(0.005))
    np.testing.assert_allclose(result["native_label_shift_rms_fixed_scale"], np.sqrt(0.02))
    np.testing.assert_allclose(result["combined_at_physical_points_rms_fixed_scale"],
                               np.sqrt(0.025))


def test_parent_comparison_rows_checks_hashes_and_preserves_ancestry(tmp_path):
    import hashlib
    import json

    identity = {
        "vmex_source": {"commit": "source-pin"},
        "case": "integer_3d",
        "state_source_sha256": "a"*64,
        "input_sha256": "b"*64,
        "status": "measurement_diagnostic",
    }
    parent_dir = tmp_path/"radial32"
    parent_dir.mkdir()
    parent = {**identity, "run_id": "radial32", "comparison_parent": None,
              "rows": [{"grid_id": "radial16"}, {"grid_id": "radial32"}]}
    parent_path = parent_dir/"measurement.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    child = {**identity, "run_id": "radial64",
             "comparison_parent": {"run_id": "radial32", "report_sha256": parent_sha},
             "rows": [{"grid_id": "radial64"}]}

    rows = parent_comparison_rows(
        tmp_path, child, ("source-pin", "integer_3d", "a"*64, "b"*64))
    assert [row["grid_id"] for row in rows] == ["radial16", "radial32", "radial64"]

    parent_path.write_text(parent_path.read_text()+" ", encoding="utf-8")
    with pytest.raises(SystemExit, match="report hash"):
        parent_comparison_rows(
            tmp_path, child, ("source-pin", "integer_3d", "a"*64, "b"*64))


def test_actual_saved_vmex_ray_crosses_native_knots_off_reference_knots():
    import hashlib
    import json

    record_path = ROOT/"results/audit/native_knot_alignment_ns33_radial32.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    source_samples = ROOT/"results/audit/measurement_gpu/ns33-projected-default-gpu-radial32-v1/finest_samples.npz"
    assert hashlib.sha256(source_samples.read_bytes()).hexdigest() == record["source_samples_sha256"]
    crossings = native_knot_crossings(
        record["reference_s_samples"], record["native_s_samples"], record["native_knots"])
    np.testing.assert_allclose(crossings, record["reference_s_at_native_knots"], rtol=0, atol=2e-15)
    assert record["vmex_source"]["commit"] == "b5f5267efc0795c4a49a224e321e9b370975c14c"
    assert record["reference_cell_partition_native_knot_aligned"] is False
    assert record["maximum_reference_crossing_shift"] > 1e-3

    exact_path = ROOT/"results/audit/native_knot_alignment_ns33_exact_roots.json"
    exact = json.loads(exact_path.read_text(encoding="utf-8"))
    radial64_samples = ROOT/"results/audit/measurement_gpu/ns33-projected-default-gpu-radial64-v1/finest_samples.npz"
    assert hashlib.sha256(radial64_samples.read_bytes()).hexdigest() == exact["sample_grid_sha256"]
    assert exact["vmex_source"]["commit"] == record["vmex_source"]["commit"]
    assert exact["sample_radial_rule"] == "gauss"
    assert exact["native_mesh_points"] == 33
    assert exact["max_abs_native_root_residual"] < 1e-13
    assert exact["max_abs_linear_sampled_crossing_error"] < 3e-6
    assert exact["max_native_crossing_shift_from_uniform_reference_knots"] > 3e-3
    coarse_error = np.max(np.abs(
        np.asarray(record["reference_s_at_native_knots"])-
        np.asarray(exact["reference_s_at_native_knots"])))
    assert coarse_error < 1e-5


def test_native_knot_crossings_reject_nonmonotone_or_unbracketed_rays():
    with pytest.raises(ValueError, match="strictly increasing"):
        native_knot_crossings([0.1, 0.2, 0.3], [0.1, 0.3, 0.2], [0.15])
    with pytest.raises(ValueError, match="bracketed"):
        native_knot_crossings([0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.4])


def test_streamed_weighted_square_sufficient_statistics_match_batch_reduction():
    rng = np.random.default_rng(14)
    values = rng.normal(size=(101, 3))
    weights = rng.uniform(0.1, 2.0, size=101)
    batch = weighted_square_sum(values, weights)
    streamed = sum(weighted_square_sum(values[first:first+13], weights[first:first+13])
                   for first in range(0, len(values), 13))
    np.testing.assert_allclose(streamed, batch, rtol=3e-16, atol=0)
    with pytest.raises(ValueError, match="incompatible shapes"):
        weighted_square_sum(values, weights[:-1])


def test_sample_retention_is_deterministic_bounded_and_endpoint_inclusive():
    indices = deterministic_sample_indices(100_000, maximum=257)
    assert len(indices) == 257
    assert indices[0] == 0 and indices[-1] == 99_999
    np.testing.assert_array_equal(indices, deterministic_sample_indices(100_000, maximum=257))
    np.testing.assert_array_equal(deterministic_sample_indices(5, maximum=257), np.arange(5))


def test_physical_force_error_decomposition_reconstructs_vector_identity():
    rng = np.random.default_rng(81)
    fields = [rng.normal(size=(11, 3)) for _ in range(6)]
    result = decompose_physical_force_error(*fields)
    pieces = sum(result[name] for name in (
        "dJ_cross_B_reference", "J_reference_cross_dB", "dJ_cross_dB", "minus_dgradp"))
    np.testing.assert_allclose(pieces, result["total_force_error"], rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(result["identity_defect"], 0.0, atol=2e-15)
    with pytest.raises(ValueError, match="shape"):
        decompose_physical_force_error(*fields[:-1], fields[-1][:, :2])
