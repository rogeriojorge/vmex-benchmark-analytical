"""Render an immutable fixed-state R1 measurement report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from analytic import ROOT
from evidence import sha256_file, write_json


METRICS = (
    ("field_relative_l2", r"$E_B$", 1e-5),
    ("current_relative_l2", r"$E_J$", 1e-3),
    ("gradp_relative_l2", r"$E_{\nabla p}$", 1e-3),
    ("force_pressure_scale", r"$E_{F,p}$", 1e-3),
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"figures")
    parser.add_argument("--manifest-dir", type=Path,
                        default=ROOT/"results/audit/measurement_figures")
    return parser


def generate(measurement_dir: Path, output_dir: Path, manifest_dir: Path) -> tuple[Path, Path]:
    run_dir = measurement_dir.resolve()
    report_path = run_dir/"measurement.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    volume_rows = [row for row in report["rows"]
                   if row.get("sample_measure") == "full_torus_physical_volume"]
    targeted_rows = [row for row in report["rows"]
                     if row.get("grid_id", "").startswith("target_") and
                     row.get("status") == "measured"]
    labels = []
    for row in volume_rows:
        if row.get("sample_measure") != "full_torus_physical_volume":
            labels.append("legacy 96")
            continue
        label = (f"{row['nradial']}×{row['ntheta']}×"
                 f"{row['nphi_one_period']}")
        if row["radial_rule"].startswith("cell_"):
            label = f"{row['radial_order']} / cell\n{label}"
        if row["theta_shift_fraction"] != 0 or row["phi_shift_fraction"] != 0:
            label += " shift"
        if row["radial_rule"] in {"midpoint", "cell_midpoint"}:
            label += " midpoint"
        labels.append(label)
    x = np.arange(len(volume_rows))
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "-", report["run_id"]).strip("-")
    output_dir, manifest_dir = Path(output_dir), Path(manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir/f"vmex_r1_{safe_id}.png"
    manifest_path = manifest_dir/f"{safe_id}_figure_manifest.json"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 9.2))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.23, top=0.81,
                        wspace=0.24, hspace=0.48)
    axes = axes.ravel()
    colors = {"gauss": "#2463A6", "midpoint": "#C56A19",
              "cell_gauss": "#2463A6", "cell_midpoint": "#C56A19"}
    for axis, (metric, title, target) in zip(axes[:4], METRICS):
        for index, row in enumerate(volume_rows):
            rule = row["radial_rule"]
            shifted = (row["theta_shift_fraction"] != 0 or row["phi_shift_fraction"] != 0)
            marker = "^" if shifted else (
                "D" if rule in {"midpoint", "cell_midpoint"} else "o")
            axis.scatter(index, row["route_A"][metric], marker=marker,
                         s=58, color=colors[rule], edgecolor="white", linewidth=0.6,
                         zorder=3)
        axis.axhline(target, color="#9C3434", linestyle="--", linewidth=1.1,
                     label="initial target")
        axis.set_yscale("log")
        axis.set_ylabel("relative L2" if metric != "force_pressure_scale" else
                        "RMS force / exact pressure-gradient RMS")
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=38, ha="right")
        axis.tick_params(axis="x", labelsize=7.2, pad=3)
        axis.grid(axis="y", which="both", color="#D9DEE4", linewidth=0.7)

    target_axis = axes[4]
    target_label_map = {
        "target_near_axis": "near axis",
        "target_near_edge": "near edge",
        "target_native_knots_and_cell_interiors": "knots / cell interiors",
    }
    target_labels = [target_label_map.get(
        row["grid_id"], row["grid_id"].removeprefix("target_").replace("_", " "))
        for row in targeted_rows]
    target_x = np.arange(len(targeted_rows))
    for metric, title, target in (METRICS[0], METRICS[1], METRICS[3]):
        target_axis.plot(target_x, [row["route_A"][metric] for row in targeted_rows],
                         marker={"field_relative_l2": "o", "current_relative_l2": "s",
                                 "force_pressure_scale": "^"}[metric],
                         linewidth=1.1, markersize=5, label=title)
    target_axis.set_yscale("log")
    target_axis.set_xticks(target_x, target_labels, rotation=20, ha="right")
    target_axis.tick_params(axis="x", labelsize=8, pad=4)
    target_axis.set_title("Targeted native-s diagnostics (equal point weights)")
    target_axis.set_ylabel("diagnostic score; not a volume norm")
    target_axis.grid(axis="y", which="both", color="#D9DEE4", linewidth=0.7)
    target_axis.legend(frameon=False, fontsize=8)

    route_axis = axes[5]
    route_metrics = ("field_relative_l2", "current_relative_l2",
                     "gradp_relative_l2", "force_pressure_scale")
    route_names = (r"$B$", r"$J$", r"$\nabla p$", r"$F$",)
    route_max = [max(row["route_A_B_difference"][metric] for row in report["rows"]
                     if "route_A_B_difference" in row) for metric in route_metrics]
    route_axis.bar(np.arange(4), route_max, color="#4C836B", width=0.65)
    route_axis.set_yscale("log")
    route_axis.set_xticks(np.arange(4), route_names)
    route_axis.set_title("Largest Route A / Route B disagreement")
    route_axis.set_ylabel("relative difference; force on pressure scale")
    route_axis.grid(axis="y", which="both", color="#D9DEE4", linewidth=0.7)
    route_axis.set_axisbelow(True)

    radial_spread = report.get("fine_grid_score_spread") or {}
    solver = report.get("solver_provenance") or {}
    spread_values = (
        ("B", radial_spread.get("field_relative_l2")),
        ("J", radial_spread.get("current_relative_l2")),
        ("force", radial_spread.get("force_pressure_scale")),
    )
    spread_text = " · ".join(
        f"{name} {value:.2g}" for name, value in spread_values if value is not None
    ) or "not yet resolved"
    is_composite = report.get("profile") == "composite"
    title = ("VMEX knot-aligned fixed-state physical-field scoring" if is_composite else
             "VMEX integer-3D fixed-state physical-field scoring")
    fig.suptitle(title,
                 y=0.975, fontsize=15, fontweight="semibold")
    fig.text(
        0.5, 0.925,
        f"NS{solver.get('ns_override', '—')} projected state · "
        f"source {report['vmex_source']['commit'][:7]} · "
        f"FTOL={solver.get('ftol')} · TCON0={solver.get('tcon0_effective')} · "
        f"{report['status'].replace('_', ' ')} · accepted={report['accepted']}",
        ha="center", va="center", fontsize=10, color="#343A40",
    )
    legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#2463A6",
                   markeredgecolor="white", markersize=7, label="Gauss"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="#2463A6",
                   markeredgecolor="white", markersize=7, label="shifted Gauss"),
            Line2D([0], [0], marker="D", color="none", markerfacecolor="#C56A19",
                   markeredgecolor="white", markersize=6.5, label="radial midpoint"),
    ]
    if is_composite:
        legend_handles[0] = Line2D([0], [0], marker="o", color="none",
                                   markerfacecolor="#2463A6", markeredgecolor="white",
                                   markersize=7, label="cell Gauss")
        legend_handles[2] = Line2D([0], [0], marker="D", color="none",
                                   markerfacecolor="#C56A19", markeredgecolor="white",
                                   markersize=6.5, label="cell midpoint")
    fig.legend(
        handles=legend_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=3, frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5, 0.105,
        ("Labels give radial points per native uniform-s cell and (Nr, Nθ, Nφ per field period); "
         "shift marks offset angular nodes; midpoint marks cell midpoints." if is_composite else
         "Labels give (Nr, Nθ, Nφ per field period); shift marks offset angular nodes; "
         "midpoint marks the radial midpoint rule."),
        ha="center", va="center", fontsize=7.5, color="#343A40",
    )
    fig.text(
        0.5, 0.045,
        "Fine-grid score spread (B, J, force): " + spread_text +
        "   |   measurement unresolved; diagnostic only",
        ha="center", va="center", fontsize=8.5, color="#343A40",
    )
    fig.savefig(figure_path, dpi=240, facecolor="white")
    plt.close(fig)

    manifest = {
        "schema": 1,
        "run_id": report["run_id"],
        "measurement_sha256": sha256_file(report_path),
        "figure_path": figure_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "figure_sha256": sha256_file(figure_path),
        "plot_script_sha256": sha256_file(Path(__file__)),
        "volume_rows": [row["grid_id"] for row in volume_rows],
        "targeted_rows": [row["grid_id"] for row in targeted_rows],
        "status": report["status"],
        "accepted": report["accepted"],
    }
    write_json(manifest_path, manifest)
    return figure_path, manifest_path


def main(argv=None):
    args = _parser().parse_args(argv)
    figure_path, manifest_path = generate(args.measurement_dir, args.output_dir, args.manifest_dir)
    print(json.dumps({"figure": figure_path.relative_to(ROOT).as_posix(),
                      "manifest": manifest_path.relative_to(ROOT).as_posix()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
