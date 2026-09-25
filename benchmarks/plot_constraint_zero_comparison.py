"""Plot the matched NS129 TCON0 diagnostic from immutable run reports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT
from evidence import sha256_file, write_json


DEFAULT = ROOT / "results/vmex/integer_3d_iota_ns129_niter3000_projected_ftol1e-10_tcon0default"
ZERO = ROOT / "results/vmex/runs/integer3d-ns129-zero-t0t1-20260924"
INPUT = ROOT / "inputs/input.integer_3d_iota"
OUTPUT = ROOT / "figures/vmex_ns129_tcon0_zero_diagnostic.png"
MANIFEST = ROOT / "results/vmex/ns129_tcon0_zero_figure_manifest.json"
METRICS = (
    ("field_relative_l2", r"$E_B$", 1e-5),
    ("current_relative_l2", r"$E_J$", 1e-3),
    ("gradp_relative_l2", r"$E_{\nabla p}$", 1e-3),
    ("force_pressure_scale", r"$E_{F,p}$", 1e-3),
)


def main() -> None:
    default_report_path = DEFAULT / "forward.json"
    zero_report_path = ZERO / "forward.json"
    default_report = json.loads(default_report_path.read_text())
    zero_report = json.loads(zero_report_path.read_text())
    if default_report.get("vmex_commit") != zero_report.get("source", {}).get("commit"):
        raise ValueError("comparison must use the same VMEX source pin")
    if default_report.get("seed_sha256") != zero_report.get("seed_sha256"):
        raise ValueError("comparison must use the same projected seed")
    if sha256_file(INPUT) != zero_report.get("input_sha256"):
        raise ValueError("the current deck no longer matches the zero-TCON run")
    if default_report.get("ns_override") != zero_report.get("controls", {}).get(
        "effective", {}).get("ns_array", [None])[0]:
        raise ValueError("comparison must use the same radial resolution")

    default_samples_path = DEFAULT / "native_samples.npz"
    zero_points_path = ZERO / "point_cloud.npz"
    with np.load(default_samples_path, allow_pickle=False) as default_samples, \
            np.load(zero_points_path, allow_pickle=False) as zero_points:
        if default_samples["xyz"].shape != zero_points["xyz"].shape:
            raise ValueError("comparison grids have different point counts")
        sample_count = len(default_samples["xyz"])
        position_difference = float(np.max(np.abs(default_samples["xyz"]-zero_points["xyz"])))
        weight_scale = np.maximum(np.abs(default_samples["weights"]), np.finfo(float).tiny)
        weight_relative_difference = float(np.max(
            np.abs(default_samples["weights"]-zero_points["weights"])/weight_scale))
        grid_hash = hashlib.sha256(
            np.ascontiguousarray(zero_points["xyz"]).tobytes()
            + np.ascontiguousarray(zero_points["weights"]).tobytes()
        ).hexdigest()
    if position_difference > 1e-12 or weight_relative_difference > 1e-9:
        raise ValueError("the two saved grids are not equivalent to roundoff")

    default_scores = default_report["native_score"]
    zero_scores = zero_report["native_score"]
    colors = ("#667085", "#007C91")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), constrained_layout=True)
    for axis, (key, title, target) in zip(axes.flat, METRICS):
        values = (default_scores[key], zero_scores[key])
        bars = axis.bar((0, 1), values, width=0.62, color=colors, zorder=3)
        axis.axhline(target, color="#B54708", linestyle="--", linewidth=1.2,
                     label=f"initial target {target:.0e}", zorder=2)
        axis.set_yscale("log")
        axis.set_title(title, fontsize=14, pad=8)
        axis.set_xticks((0, 1), ("TCON0 = 1", "TCON0 = 0"))
        axis.set_ylabel("relative L2" if key != "force_pressure_scale" else
                        "RMS force / exact grad-p RMS")
        axis.grid(axis="y", which="both", color="#E4E7EC", linewidth=0.7, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc="best")
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.2e}", (bar.get_x()+bar.get_width()/2, value),
                          xytext=(0, 5), textcoords="offset points", ha="center",
                          va="bottom", fontsize=9, color="#344054")
    fig.suptitle("Projected NS129 root: constraint-strength diagnostic", fontsize=16,
                 fontweight="semibold")
    fig.text(0.5, -0.015,
             "Historical VMEX pin · FTOL 1×10⁻¹⁰ · same input and seed · 96-point score; "
             "measurement unresolved, neither result accepted",
             ha="center", va="top", fontsize=9, color="#475467")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "schema": 1,
        "evidence": "matched_constraint_strength_diagnostic_figure",
        "status": "diagnostic_measurement_unresolved",
        "comparison": {
            "vmex_commit": default_report["vmex_commit"],
            "input_sha256": zero_report["input_sha256"],
            "seed_sha256": zero_report["seed_sha256"],
            "ns": zero_report["controls"]["effective"]["ns_array"][0],
            "ftol": zero_report["controls"]["effective"]["ftol_array"][0],
            "iterations": {
                "tcon0_1": default_report["iterations"],
                "tcon0_0": zero_report["iterations"],
            },
            "solver_converged": {
                "tcon0_1": default_report["converged"],
                "tcon0_0": zero_report["solver_converged"],
            },
            "sample_count": sample_count,
            "max_abs_position_difference_m": position_difference,
            "max_relative_weight_difference": weight_relative_difference,
            "common_grid_content_sha256": grid_hash,
            "scores": {
                "tcon0_1": {key: default_scores[key] for key, _, _ in METRICS},
                "tcon0_0": {key: zero_scores[key] for key, _, _ in METRICS},
            },
        },
        "source_artifacts": {
            "default_report": {"path": str(default_report_path.relative_to(ROOT)),
                               "sha256": sha256_file(default_report_path)},
            "zero_report": {"path": str(zero_report_path.relative_to(ROOT)),
                            "sha256": sha256_file(zero_report_path)},
            "default_samples": {"path": str(default_samples_path.relative_to(ROOT)),
                                "sha256": sha256_file(default_samples_path)},
            "zero_point_cloud": {"path": str(zero_points_path.relative_to(ROOT)),
                                 "sha256": sha256_file(zero_points_path)},
            "figure": {"path": str(OUTPUT.relative_to(ROOT)),
                       "sha256": sha256_file(OUTPUT)},
        },
        "plot_script_sha256": sha256_file(Path(__file__)),
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({"figure": str(OUTPUT.relative_to(ROOT)),
                      "figure_sha256": sha256_file(OUTPUT),
                      "manifest": str(MANIFEST.relative_to(ROOT)),
                      "grid_position_difference_m": position_difference,
                      "grid_weight_relative_difference": weight_relative_difference},
                     indent=2))


if __name__ == "__main__":
    main()
