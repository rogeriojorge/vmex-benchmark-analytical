"""Plot measured VMEX and DESC errors for the integer-family controls."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def metrics(report, stage):
    row = report[stage]
    return (row["field_relative_l2"], row["current_relative_l2"],
            row["force_pressure_scale"])


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def panel(ax, records, title):
    x = np.arange(len(records))
    metrics_names = ("Relative B error", "Relative J error",
                     "Force RMS / exact |grad p| RMS")
    colors = ("#276FBF", "#D1495B", "#2A9D8F")
    markers = {"projection": "o", "solved": "s", "terminal": "^"}
    offsets = {"projection": -0.105, "solved": 0.105, "terminal": 0.105}
    for j, name in enumerate(metrics_names):
        values = []
        for i, record in enumerate(records):
            r = record["report"]
            for stage, label in (("projected", "projection"),
                                 ("solved", record.get("solved_label", "solved"))):
                if label == "terminal":
                    stage = "solved"
                value = metrics(r, stage)[j]
                values.append((i, label, value))
        axis = ax[j]
        for i, stage, value in values:
            marker = markers[stage]
            face = "white" if stage == "terminal" else colors[j]
            axis.scatter(i+offsets[stage], value, s=45, marker=marker,
                         facecolors=face, edgecolors=colors[j], linewidths=1.1,
                         zorder=3)
            if stage != "projection":
                projected = metrics(records[i]["report"], "projected")[j]
                axis.plot([i-offsets["projection"], i+offsets[stage]],
                          [projected, value], color=colors[j], alpha=.3,
                          linewidth=1, zorder=1)
        axis.set_yscale("log")
        axis.set_title(name, fontsize=10, pad=9)
        axis.grid(axis="y", which="both", alpha=.22)
        axis.set_xticks(x, [record["label"] for record in records],
                        rotation=30, ha="right", fontsize=8)
        axis.tick_params(axis="y", labelsize=8)
        if j == 0:
            axis.set_ylabel(title, fontsize=10, labelpad=14)


def main():
    vmex_proj = read("results/projection/integer_3d_vmex_ns129/native_scores.json")
    vmex_solved = read(
        "results/vmex/integer_3d_iota_ns129_niter3000_projected_ftol1e-10/native_scores.json")
    records = [
        {"label": "VMEX N129", "report": {"projected": vmex_proj,
                                             "solved": vmex_solved}},
    ]
    for resolution, chart in ((6, "base"), (6, "remapped"),
                              (8, "base"), (8, "remapped")):
        if chart == "remapped" and resolution == 8:
            path = ("results/desc/coordinate/integer_3d/"
                    "remapped_L8_maxiter120/desc_integer_remapped_L8.json")
            solved_label = "terminal"
        else:
            folder = f"{chart}_L{resolution}"
            path = (f"results/desc/coordinate/integer_3d/{folder}/"
                    f"desc_integer_{chart}_L{resolution}.json")
            solved_label = "solved"
        records.append({"label": f"DESC M{resolution} {chart}",
                        "report": read(path), "solved_label": solved_label,
                        "path": path})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), constrained_layout=True)
    panel(axes, records, "Integer 3-D: relative physical error")
    for axis in axes:
        axis.axvline(.5, color="#444444", linewidth=.7, alpha=.55)
    fig.suptitle("Matched held-out Cartesian scores against the exact integer 3-D field",
                 fontsize=13, weight="bold")
    fig.text(.5, -.015,
             "Circle: projection; square: solver output; open triangle: terminal iterate after the solver cap.",
             ha="center", fontsize=8)
    fig.savefig(ROOT / "figures/coordinate_solver_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    axisym_desc = read(
        "results/desc/coordinate/integer_axisymmetric/base_L6/"
        "desc_integer_base_L6.json")
    axisym_vmex = read("results/vmex/integer_axisymmetric_iota_ns129/native_scores.json")
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.7), constrained_layout=True)
    names = ("field_relative_l2", "current_relative_l2", "force_pressure_scale")
    titles = ("Relative B error", "Relative J error",
              "Force RMS / exact |grad p| RMS")
    for j, axis in enumerate(axes):
        vmex_value = axisym_vmex[names[j]]
        desc_projection = axisym_desc["projected"][names[j]]
        desc_solved = axisym_desc["solved"][names[j]]
        axis.scatter(-.12, vmex_value, s=48, marker="s", color="#276FBF",
                     edgecolor="white", linewidth=.6, zorder=3)
        axis.scatter(.88, desc_projection, s=48, marker="o", color="#D1495B",
                     edgecolor="white", linewidth=.6, zorder=3)
        axis.scatter(1.12, desc_solved, s=48, marker="s", color="#D1495B",
                     edgecolor="white", linewidth=.6, zorder=3)
        axis.plot([.88, 1.12], [desc_projection, desc_solved],
                  color="#D1495B", alpha=.4, linewidth=1)
        axis.set_yscale("log")
        axis.set_title(titles[j], fontsize=9)
        axis.set_xticks([0, 1], ["VMEX N129\nsolved", "DESC M6\nprojection / solve"],
                        fontsize=8)
        axis.grid(axis="y", which="both", alpha=.22)
        if j == 0:
            axis.set_ylabel("Axisymmetric integer control", fontsize=10)
    fig.suptitle("Axisymmetric control: same held-out scoring contract",
                 fontsize=12, weight="bold")
    fig.savefig(ROOT / "figures/coordinate_axisymmetric_control.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    paths = [ROOT / "benchmarks/plot_coordinate_comparison.py",
             ROOT / "results/projection/integer_3d_vmex_ns129/native_scores.json",
             ROOT / "results/vmex/integer_3d_iota_ns129_niter3000_projected_ftol1e-10/native_scores.json",
             ROOT / "results/vmex/integer_axisymmetric_iota_ns129/native_scores.json"]
    paths.extend(ROOT / record["path"] for record in records[1:])
    paths.append(ROOT / "results/desc/coordinate/integer_axisymmetric/base_L6/"
                 "desc_integer_base_L6.json")
    outputs = [ROOT / "figures/coordinate_solver_comparison.png",
               ROOT / "figures/coordinate_axisymmetric_control.png"]
    manifest = {
        "schema": 1,
        "evidence": "measured_native_physical_sample_scores",
        "sample_contract": "96 common physical Cartesian points per case; independent exact B, J, grad-p scorer",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "nonconverged_terminal": "DESC integer 3-D remapped L8, maxiter=120; shown as open triangle",
    }
    (ROOT / "results/desc/coordinate/figure_manifest.json").write_text(
        json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
