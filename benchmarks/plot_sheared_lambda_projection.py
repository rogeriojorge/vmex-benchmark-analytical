"""Plot matched physical errors for the saved sheared-A lambda projections."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT
from evidence import sha256_file, write_json

ZERO = ROOT / "results/projection/sheared_A/sheared-A-projection-20260924T235541.410475Z"
MAPPED = ROOT / "results/projection/sheared_A/sheared-A-straight-field-20260925T004606.885805Z"
FIGURE = ROOT / "figures/vmex_sheared_a_lambda_projection.png"
MANIFEST = ROOT / "results/projection/sheared_A/lambda_projection_figure_manifest.json"
FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin")


def main():
    zero_report_path = ZERO / "projection.json"
    mapped_report_path = MAPPED / "straight_field_projection.json"
    zero = json.loads(zero_report_path.read_text())
    mapped = json.loads(mapped_report_path.read_text())
    if zero["source"]["commit"] != mapped["source"]["commit"]:
        raise SystemExit("projection reports use different VMEX source pins")
    if zero["input_sha256"] != mapped["input_sha256"]:
        raise SystemExit("projection reports use different processed inputs")
    if not zero["native_score"] or not mapped["native_score"]:
        raise SystemExit("one projection has no physical score")
    if zero["native_score"]["sample_count"] != mapped["native_score"]["sample_count"]:
        raise SystemExit("projection sample counts differ")
    if zero["sample_validation"]["reference_samples"]["sha256"] != \
            mapped["artifacts"]["reference_samples"]["sha256"]:
        raise SystemExit("reference Cartesian sample grids differ")

    zero_state = np.load(ZERO / "seed_ns17.npz", allow_pickle=False)
    mapped_state = np.load(MAPPED / "seed_ns17.npz", allow_pickle=False)
    geometry_same = all(np.array_equal(zero_state[name], mapped_state[name])
                        for name in FIELDS)
    if not geometry_same:
        raise SystemExit("lambda projection changed the measured R/Z state")

    scores = [zero["native_score"], mapped["native_score"]]
    labels = [r"Exact $R/Z$, $\lambda=0$", "Exact field-line $\lambda$"]
    metrics = [
        ("field_relative_l2", r"$B$ relative $L_2$", "0.134", "7.07e-4"),
        ("current_relative_l2", r"$J$ relative $L_2$", "0.265", "5.13e-3"),
        ("gradp_relative_l2", r"$\nabla p$ relative $L_2$", "3.58e-4", "3.58e-4"),
        ("force_pressure_scale", "Force / pressure-gradient scale", "1.21", "2.44e-2"),
    ]
    colors = ("#8d99ae", "#176b87")
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.8), constrained_layout=True)
    for ax, (key, title, _, _) in zip(axes.flat, metrics):
        values = [float(score[key]) for score in scores]
        ax.set_xlim(min(values) * 0.55, max(values) * 2.2)
        bars = ax.barh(labels, values, color=colors, height=0.52)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, loc="left", pad=9)
        ax.grid(axis="x", which="both", color="#d6dce1", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9, length=0)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Relative error" if key != "force_pressure_scale"
                      else "RMS force / pressure-gradient RMS", fontsize=9)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3g}",
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=9, color="#24313a")

    fig.suptitle("Sheared-A NS17 exact-surface projections",
                 fontsize=16, fontweight="semibold", color="#183642")
    fig.text(0.5, -0.012,
             "96 identical Cartesian points · historical VMEX pin · no nonlinear solve · "
             "measurement not yet resolved",
             ha="center", fontsize=9, color="#4b5963")
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    record = {
        "schema": 1,
        "evidence": "matched_projection_figure",
        "figure": FIGURE.relative_to(ROOT).as_posix(),
        "figure_sha256": sha256_file(FIGURE),
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "source_report_sha256": {
            "lambda_zero": sha256_file(zero_report_path),
            "straight_field_lambda": sha256_file(mapped_report_path),
        },
        "same_source_pin": zero["source"]["commit"],
        "same_input_sha256": zero["input_sha256"],
        "same_reference_samples_sha256": zero["sample_validation"]["reference_samples"]["sha256"],
        "same_geometry_coefficients": geometry_same,
        "sample_count": zero["native_score"]["sample_count"],
        "metrics": {key: [float(row[key]) for row in scores]
                    for key, *_ in metrics},
        "interpretation": "projection comparison only; no nonlinear root or resolved-volume claim",
    }
    write_json(MANIFEST, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
