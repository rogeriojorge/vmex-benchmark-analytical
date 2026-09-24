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
from measurement import (integer_exact_fields, integer_surface_jax, integer_surface_numpy,
                         compare_pressure_profiles, native_radial_probes, reference_grid)
from verify_measurement import _composite_spread, _parent_comparison_rows


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
    from verify_measurement import TARGETS

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

    spread, resolved = _composite_spread(rows)
    assert resolved
    assert spread["native_cell_count"] == 32
    assert set(spread["by_order"]) == {"2", "4"}

    rows[-1]["route_A"]["field_relative_l2"] += 0.2*TARGETS["field_relative_l2"]
    _, resolved = _composite_spread(rows)
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

    rows = _parent_comparison_rows(
        tmp_path, child, ("source-pin", "integer_3d", "a"*64, "b"*64))
    assert [row["grid_id"] for row in rows] == ["radial16", "radial32", "radial64"]

    parent_path.write_text(parent_path.read_text()+" ", encoding="utf-8")
    with pytest.raises(SystemExit, match="report hash"):
        _parent_comparison_rows(
            tmp_path, child, ("source-pin", "integer_3d", "a"*64, "b"*64))
