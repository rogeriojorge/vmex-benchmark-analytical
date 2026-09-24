import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from evidence import (  # noqa: E402
    acceptance_state,
    desc_native_flux_label,
    failure_record,
    reserve_run_directory,
    single_source,
    validate_point_cloud,
    write_json,
)
from plot_tcon_ladder import generate  # noqa: E402
from score_samples import point_cloud, score  # noqa: E402


def test_run_directory_is_never_reused(tmp_path):
    run_id, path = reserve_run_directory(tmp_path, "case-001")
    assert run_id == "case-001"
    assert path.is_dir()
    with pytest.raises(FileExistsError):
        reserve_run_directory(tmp_path, "case-001")


def test_failed_record_keeps_missing_memory_unknown_and_valid_json(tmp_path):
    record = failure_record(
        run_id="failed-001",
        error=RuntimeError("bounded test failure"),
        peak_rss_mib=None,
        requested={"ftol": 1e-10},
        effective={"ftol_array": [1e-10]},
        source={"commit": "abc123"},
    )
    path = tmp_path / "failed.json"
    write_json(path, record, exclusive=True)
    saved = json.loads(path.read_text())
    assert saved["host_peak_rss_mib"] is None
    assert saved["device_memory_peak_mib"] is None
    assert saved["accepted"] is False


def test_capped_or_unresolved_state_cannot_be_accepted():
    capped = acceptance_state(
        solver_converged=False,
        pointwise_thresholds_met=True,
        measurement_resolved=True,
        representation_resolved=True,
        root_certified=True,
    )
    unresolved = acceptance_state(
        solver_converged=True,
        pointwise_thresholds_met=True,
        measurement_resolved=False,
        representation_resolved=True,
        root_certified=True,
    )
    assert not capped["accepted"]
    assert not unresolved["accepted"]


def test_requested_and_effective_controls_are_both_stored():
    from evidence import controls_record

    controls = controls_record(
        {"ftol": 1e-10, "ns": 129},
        {"ftol_array": [1e-10], "ns_array": [129]},
    )
    assert controls["requested"]["ftol"] == 1e-10
    assert controls["effective"]["ns_array"] == [129]


def test_mixed_or_missing_source_ids_are_rejected():
    with pytest.raises(ValueError):
        single_source([{"source_commit": "one"}, {"source_commit": "two"}])
    with pytest.raises(ValueError):
        single_source([{"source_commit": "one"}, {"source_commit": None}])


def test_reference_cloud_is_neutral_and_desc_label_is_its_own():
    cloud = {
        "case_name": "integer_3d",
        "xyz": np.zeros((3, 3)),
        "weights": np.ones(3),
        "s_reference": np.array([0.1, 0.3, 0.8]),
        "length_m": 1.0,
        "field_t": 1.0,
        "mu0": 1.0,
    }
    validate_point_cloud(cloud)
    with pytest.raises(ValueError, match="solver fields"):
        validate_point_cloud({**cloud, "vmex_s": np.ones(3)})
    np.testing.assert_allclose(desc_native_flux_label(np.array([0.2, 0.5, 1.0])),
                               [0.04, 0.25, 1.0], rtol=1e-15, atol=0)
    historical = point_cloud({**cloud, "s": cloud["s_reference"], "vmex_s": np.ones(3)})
    assert "vmex_s" not in historical
    assert "s_native" not in historical


def test_new_scorer_reports_metrics_without_acceptance_shortcut():
    from analytic import cases, differential_fields, surface

    case = cases()["integer_3d"]
    labels = np.array([0.15, 0.4, 0.7, 0.85]) * case.edge
    xyz = surface(case, labels, np.linspace(0.2, 5.0, 4), np.linspace(0.1, 1.2, 4))
    B, J, gradp, _, _ = differential_fields(case, xyz)
    result = score({
        "case_name": case.name,
        "xyz": xyz,
        "B": B,
        "J": J,
        "gradp": gradp,
        "weights": np.ones(len(xyz)),
        "s_reference": labels / case.edge,
        "s_native": labels / case.edge,
        "length_m": 1.0,
        "field_t": 1.0,
        "mu0": 1.0,
    })
    assert result["field_relative_l2"] < 1e-14
    assert result["native_flux_label_minus_reference_max_abs"] == 0.0
    assert "accepted" not in result


def _write_tcon_record(root, dirname, tcon0, initialization):
    run_dir = root / "results/vmex" / dirname
    run_dir.mkdir(parents=True)
    report = {
        "input": "inputs/input.integer_3d_iota",
        "ns_override": 33,
        "tcon0_effective": tcon0,
        "initialization": initialization,
        "vmex_commit": "b5f5267efc0795c4a49a224e321e9b370975c14c",
        "native_score": {
            "field_relative_l2": 3e-5,
            "current_relative_l2": 4e-3,
            "force_pressure_scale": 2e-2,
            "accepted": False,
        },
        "iterations": 77,
        "jax_version": "historical-injected",
        "command": "historical-injected",
    }
    raw = json.dumps(report, indent=2).encode()
    path = run_dir / "forward.json"
    path.write_bytes(raw)
    (run_dir / "native_scores.json").write_text(json.dumps({"accepted": False}))
    return path, raw


def test_tcon_rendering_does_not_mutate_raw_reports(tmp_path):
    first, first_bytes = _write_tcon_record(
        tmp_path, "projected_default", 1.0, "projected")
    second, second_bytes = _write_tcon_record(
        tmp_path, "cold_zero", 0.0, "cold")
    generate(tmp_path)
    assert first.read_bytes() == first_bytes
    assert second.read_bytes() == second_bytes
    summary = json.loads((tmp_path / "results/audit/tcon_reinterpretation/ns33_summary.json").read_text())
    assert all("iterations" not in row for row in summary["records"])
    assert summary["excluded_fields"]
    assert (tmp_path / "figures/vmex_tcon_ladder_ns33_reinterpreted.png").is_file()
