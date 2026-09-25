"""Plot the saved, matched-point NS17 sheared-A projection and capped states."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evidence import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
PROJECTION_DIR = ROOT/"results/projection/sheared_A/sheared-A-projection-20260924T235541.410475Z"
COLD_DIR = ROOT/"results/vmex/runs/sheared-a-ns17-historical-current-terminal-20260924"
PROJECTED_DIR = ROOT/"results/vmex/runs/sheared-a-ns17-projected-terminal-20260924"
OUTPUT = ROOT/"figures/vmex_sheared_a_ns17_diagnostic.png"
MANIFEST = ROOT/"results/vmex/sheared_a_ns17_figure_manifest.json"
PIN = "b5f5267efc0795c4a49a224e321e9b370975c14c"
SERIES = (
    ("lambda_zero_projection", "Exact R/Z projection\n$\\lambda=0$", "#98A2B3"),
    ("cold_terminal", "Cold start\n(capped terminal)", "#0072B2"),
    ("projected_terminal", "Projected start\n(capped terminal)", "#D55E00"),
)
METRICS = (
    ("field_relative_l2", r"$E_B$", 1e-5),
    ("current_relative_l2", r"$E_J$", 1e-3),
    ("gradp_relative_l2", r"$E_{\nabla p}$", 1e-3),
    ("force_pressure_scale", r"$E_{F,p}$", 1e-3),
    ("native_flux_label_minus_reference_max_abs", "max |sVMEX − sref|", None),
)


def _load():
    projection_path = PROJECTION_DIR/"projection.json"
    cold_path = COLD_DIR/"forward.json"
    projected_path = PROJECTED_DIR/"forward.json"
    projection = json.loads(projection_path.read_text())
    cold = json.loads(cold_path.read_text())
    projected = json.loads(projected_path.read_text())
    reports = (projection, cold, projected)
    if any(report["source"]["commit"] != PIN for report in reports):
        raise ValueError("all states must use the pinned historical VMEX source")
    if len({report["input_sha256"] for report in reports}) != 1:
        raise ValueError("all states must use the same processed input")
    if any(report.get("ns") != 17 for report in (projection,)):
        raise ValueError("projection must be the NS17 record")
    for report in (cold, projected):
        controls = report["controls"]["effective"]
        if controls["ns_array"][0] != 17 or controls["niter_array"][0] != 10000:
            raise ValueError("terminal solve is not the matched NS17 bounded experiment")
        if report["solver_converged"] or report["accepted"]:
            raise ValueError("this diagnostic figure expects capped, unaccepted solver states")
    if projection["accepted_recovery"] or projection["root_certified"]:
        raise ValueError("projection must remain explicitly unaccepted")

    point_paths = (PROJECTION_DIR/"reference_samples.npz",
                   COLD_DIR/"point_cloud.npz", PROJECTED_DIR/"point_cloud.npz")
    point_hashes = [sha256_file(path) for path in point_paths]
    if len(set(point_hashes)) != 1:
        raise ValueError("the three states do not share one byte-identical sample grid")
    scores = (projection["native_score"], cold["native_score"],
              projected["native_score"])
    if any(score["sample_count"] != 96 for score in scores):
        raise ValueError("expected 96 matched physical samples")
    for score in scores:
        if score.get("evidence") != "physical_samples_vs_analytic":
            raise ValueError("unexpected score evidence type")
    return (reports, (projection_path, cold_path, projected_path), point_paths,
            point_hashes, scores)


def main():
    reports, report_paths, point_paths, point_hashes, scores = _load()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.3), constrained_layout=True)
    for axis, (key, title, target) in zip(axes.flat[:5], METRICS):
        values = [score[key] for score in scores]
        bars = axis.bar(np.arange(3), values, width=0.66,
                        color=[color for _, _, color in SERIES], zorder=3)
        if target is not None:
            axis.axhline(target, color="#B54708", linestyle="--", linewidth=1.15,
                         label=f"initial target {target:.0e}", zorder=2)
        axis.set_yscale("log")
        axis.set_title(title, fontsize=13, pad=8)
        axis.set_xticks(np.arange(3), [label for _, label, _ in SERIES], fontsize=8)
        axis.grid(axis="y", which="both", color="#E4E7EC", linewidth=0.7, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        if target is not None:
            axis.legend(frameon=False, fontsize=7, loc="best")
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.2e}", (bar.get_x()+bar.get_width()/2, value),
                          xytext=(0, 4), textcoords="offset points", ha="center",
                          va="bottom", fontsize=8, color="#344054")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02, 0.96,
        "Interpretation\n\n"
        "Both solver runs reached the 10,000-iteration cap. Neither root is certified.\n\n"
        "The exact-surface projection matches R/Z geometry but sets lambda = 0; it is not a straight-field map.\n\n"
        "Starting from that projection worsens B, J, pressure-gradient, and flux-label scores. The similar force ratio does not make it a recovered equilibrium.",
        va="top", ha="left", fontsize=10.2, linespacing=1.35,
        color="#344054", wrap=True,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F8FAFC",
              "edgecolor": "#D0D5DD"})
    fig.suptitle("Sheared-A current-closure test at NS=17", fontsize=16,
                 fontweight="semibold")
    fig.text(0.5, -0.015,
             "Historical VMEX pin · same input and 96 physical points · capped states shown as diagnostics",
             ha="center", va="top", fontsize=9, color="#475467")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_json(MANIFEST, {
        "schema": 1,
        "evidence": "matched_saved_sheared_a_projection_and_terminal_scores",
        "status": "diagnostic_no_recovery",
        "source_reports": [path.relative_to(ROOT).as_posix() for path in report_paths],
        "source_report_sha256": [sha256_file(path) for path in report_paths],
        "sample_grids": [path.relative_to(ROOT).as_posix() for path in point_paths],
        "sample_grid_sha256": point_hashes,
        "plot_script": Path(__file__).relative_to(ROOT).as_posix(),
        "plot_script_sha256": sha256_file(Path(__file__)),
        "figure": OUTPUT.relative_to(ROOT).as_posix(),
        "figure_sha256": sha256_file(OUTPUT),
        "vmex_commit": PIN,
        "ns": 17,
        "sample_count": 96,
        "series": [key for key, _, _ in SERIES],
        "metrics": [key for key, _, _ in METRICS],
        "all_solver_roots_certified": False,
    })
    print(json.dumps({"figure": OUTPUT.relative_to(ROOT).as_posix(),
                      "figure_sha256": sha256_file(OUTPUT),
                      "manifest": MANIFEST.relative_to(ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()
