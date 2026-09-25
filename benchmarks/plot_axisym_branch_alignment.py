"""Plot the NS65 m=1 radial-relabel candidate against the resolved branch."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT
from evidence import sha256_file, write_json

RUN = ROOT / "results/audit/axisym_radial_relabel/axisym-relabel-field-20260925T010115.952375Z"
REPORT = RUN / "radial_relabel_field_test.json"
ARRAYS = RUN / "radial_relabel_field_ns65.npz"
FIGURE = ROOT / "figures/vmex_axisym_m1_branch_alignment.png"
MANIFEST = ROOT / "results/audit/axisym_radial_relabel/branch_alignment_figure_manifest.json"


def main():
    report = json.loads(REPORT.read_text())
    arrays = np.load(ARRAYS, allow_pickle=False)
    if report["source"]["commit"] != "b5f5267efc0795c4a49a224e321e9b370975c14c":
        raise SystemExit("saved response uses an unexpected VMEX source pin")
    if sha256_file(ARRAYS) != report["arrays"]["sha256"]:
        raise SystemExit("response array hash does not match its report")
    names = [
        "Exact analytical delta response",
        "Residual tangent JVP",
        "Radial-relabel candidate FD",
        "Measured-state JVP",
        "Reconverged branch FD",
    ]
    branch = arrays["branch_B_fd"]
    response_values = [
        0.0,
        float(report["residual_tangent_B_jvp"]),
        float(np.linalg.norm(arrays["radial_relabel_candidate_centered_B_fd"])),
        float(report["measured_branch_state_B_jvp"]),
        float(report["branch_B_fd"]),
    ]
    differences = [
        float(np.linalg.norm(branch)),
        float(np.linalg.norm(arrays["residual_tangent_B_jvp"] - branch)),
        float(np.linalg.norm(arrays["radial_relabel_candidate_centered_B_fd"] - branch)),
        float(np.linalg.norm(arrays["measured_branch_state_B_jvp"] - branch)),
        0.0,
    ]
    colors = ("#505b66", "#9aa5b1", "#19758e", "#d18a28", "#183642")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.8), constrained_layout=True)
    ax = axes[0]
    bars = ax.barh(names, response_values, color=colors, height=0.60)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.215)
    ax.set_xlabel("$\|dB/d\delta\|_2$ (fixed 1 T response scale)")
    ax.set_title("Physical response magnitude", loc="left", fontsize=12)
    ax.grid(axis="x", color="#d6dce1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9, length=0)
    for bar, value in zip(bars, response_values):
        ax.annotate(f"{value:.3g}",
                    xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)

    ax = axes[1]
    error_labels = ["Exact null vs branch", "Residual tangent", "Radial candidate",
                    "Measured-state JVP", "Branch FD"]
    positive_errors = [max(value, 1e-8) for value in differences]
    bars = ax.barh(error_labels, positive_errors,
                   color=(colors[0], colors[1], colors[2], colors[3], colors[4]),
                   height=0.60)
    ax.set_xscale("log")
    ax.invert_yaxis()
    ax.set_xlim(1e-5, 0.3)
    ax.set_xlabel("$L_2$ difference from reconverged branch FD")
    ax.set_title("Agreement with the branch", loc="left", fontsize=12)
    ax.grid(axis="x", which="both", color="#d6dce1", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9, length=0)
    for bar, value in zip(bars, differences):
        label = "0" if value == 0 else f"{value:.3g}"
        ax.annotate(label,
                    xy=(max(bar.get_width(), 1e-5), bar.get_y()+bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)

    fig.suptitle("NS65 integer-axisymmetric $\delta$ response",
                 fontsize=16, fontweight="semibold", color="#183642")
    fig.text(0.5, -0.012,
             "96 fixed Cartesian points · h=1e-5 · branch roots anchored below 8.1e-14 · "
             "the radial candidate matches the branch but is not field-invariant",
             ha="center", fontsize=9, color="#4b5963")
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    manifest = {
        "schema": 1,
        "evidence": "saved_branch_alignment_figure",
        "figure": FIGURE.relative_to(ROOT).as_posix(),
        "figure_sha256": sha256_file(FIGURE),
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "report_sha256": sha256_file(REPORT),
        "arrays_sha256": sha256_file(ARRAYS),
        "source_pin": report["source"]["commit"],
        "sample_count": report["sample_count"],
        "analytical_delta_field_derivative": 0.0,
        "candidate_centered_field_change": response_values[2],
        "candidate_branch_difference": differences[2],
        "interpretation": "branch alignment is not a gauge or invariant-field transformation",
    }
    write_json(MANIFEST, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
