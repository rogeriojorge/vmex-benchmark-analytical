"""Plot saved axisymmetric JVP consistency and reconverged-branch checks."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evidence import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"results/vmex/response_runs/axisym-root-response-20260924T232219.753772Z"
REPORT = RUN/"response.json"
OUTPUT = ROOT/"figures/vmex_axisymmetric_response_consistency.png"
MANIFEST = ROOT/"results/vmex/axisymmetric_response_figure_manifest.json"
COLORS = {"c": "#0072B2", "delta": "#D55E00"}
LABELS = {"c": r"nonzero $c$ direction", "delta": r"null $\delta$ direction"}


def main():
    report = json.loads(REPORT.read_text())
    rung = next(row for row in report["rungs"] if row["ns"] == 65)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.45), sharey=True)
    for name in ("c", "delta"):
        response = rung["physical_responses"][name]
        frozen = response["frozen_path_fd"]
        branch = rung["physical_responses"]["_branch_fd"][name]
        axes[0].loglog([r["step"] for r in frozen],
                       [r["B_frozen_fd_minus_jvp_over_fixed_scale"] for r in frozen],
                       marker="o", linewidth=1.8, markersize=4.5,
                       color=COLORS[name], label=LABELS[name])
        axes[1].loglog([r["step"] for r in branch],
                       [r["B_fd_minus_jvp_over_fixed_scale"] for r in branch],
                       marker="s", linewidth=1.8, markersize=4.8,
                       color=COLORS[name], label=LABELS[name])
    axes[0].set_title("Frozen linear path")
    axes[1].set_title("Independently reconverged roots")
    axes[0].set_ylabel(r"$\|\mathrm{FD}-\mathrm{JVP}\|_2/(B_*/L_*)$")
    for axis in axes:
        axis.set_xlabel("Centered-difference step")
        axis.grid(True, which="both", color="#D8DEE5", linewidth=0.65)
        axis.legend(frameon=False, fontsize=8, loc="best")
        axis.invert_xaxis()
    fig.suptitle("Axisymmetric VMEX response at NS=65", fontsize=12, y=1.02)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    write_json(MANIFEST, {
        "schema": 1,
        "evidence": "saved_axisymmetric_response_comparison",
        "status": "diagnostic_unresolved_branch_response",
        "source_report": REPORT.relative_to(ROOT).as_posix(),
        "source_report_sha256": sha256_file(REPORT),
        "plot_script": Path(__file__).relative_to(ROOT).as_posix(),
        "plot_script_sha256": sha256_file(Path(__file__)),
        "figure": OUTPUT.relative_to(ROOT).as_posix(),
        "figure_sha256": sha256_file(OUTPUT),
        "ns": 65,
        "metric": "B centered-FD minus VMEX residual-level JVP L2, divided by B*/L* per parameter unit",
        "series": ["frozen_linear_path", "independently_reconverged_roots"],
    })
    print(json.dumps({"figure": OUTPUT.relative_to(ROOT).as_posix(),
                      "figure_sha256": sha256_file(OUTPUT),
                      "manifest": MANIFEST.relative_to(ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()
