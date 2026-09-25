"""Compare sheared-A lambda projections with the bounded VMEX root."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT
from evidence import sha256_file, write_json

ZERO = ROOT / "results/projection/sheared_A/sheared-A-projection-20260924T235541.410475Z"
MAPPED = ROOT / "results/projection/sheared_A/sheared-A-straight-field-20260925T004606.885805Z"
RUN = ROOT / "results/vmex/runs/sheared-a-ns17-straight-field-terminal-20260925"
ZERO_TCON_RUN = ROOT / "results/vmex/runs/sheared-a-ns17-straight-field-tcon0-zero-terminal-20260925"
FIGURE = ROOT / "figures/vmex_sheared_a_lambda_recovery.png"
MANIFEST = ROOT / "results/projection/sheared_A/lambda_recovery_figure_manifest.json"
GEOMETRY_FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin")
METRICS = (
    ("field_relative_l2", r"$B$ relative $L_2$", "Relative error"),
    ("current_relative_l2", r"$J$ relative $L_2$", "Relative error"),
    ("gradp_relative_l2", r"$\nabla p$ relative $L_2$", "Relative error"),
    ("force_pressure_scale", "Force / pressure-gradient scale",
     "RMS force / pressure-gradient RMS"),
)


def main():
    zero_path, mapped_path, run_path, zero_tcon_path = (
        ZERO / "projection.json",
        MAPPED / "straight_field_projection.json",
        RUN / "forward.json",
        ZERO_TCON_RUN / "forward.json",
    )
    zero = json.loads(zero_path.read_text())
    mapped = json.loads(mapped_path.read_text())
    root = json.loads(run_path.read_text())
    zero_tcon_root = json.loads(zero_tcon_path.read_text())
    score = json.loads((RUN / "native_scores.json").read_text())
    zero_tcon_score = json.loads((ZERO_TCON_RUN / "native_scores.json").read_text())
    if not root["solver_converged"] or root["accepted"]:
        raise SystemExit("expected a converged, unaccepted diagnostic root")
    if zero_tcon_root["solver_converged"] or zero_tcon_root["accepted"]:
        raise SystemExit("expected the TCON0=0 diagnostic to remain capped and unaccepted")
    sources = (zero["source"]["commit"], mapped["source"]["commit"],
               root["source"]["commit"], zero_tcon_root["source"]["commit"])
    inputs = (zero["input_sha256"], mapped["input_sha256"], root["input_sha256"],
              zero_tcon_root["input_sha256"])
    if len(set(sources)) != 1 or len(set(inputs)) != 1:
        raise SystemExit("comparison mixes a VMEX source or input")
    if zero["native_score"]["sample_count"] != score["sample_count"] or \
            score["sample_count"] != zero_tcon_score["sample_count"]:
        raise SystemExit("matched physical sample counts differ")
    point_hashes = (zero["sample_validation"]["reference_samples"]["sha256"],
                    mapped["artifacts"]["reference_samples"]["sha256"],
                    root["artifacts"]["point_cloud"]["sha256"],
                    zero_tcon_root["artifacts"]["point_cloud"]["sha256"])
    if len(set(point_hashes)) != 1:
        raise SystemExit("the comparison did not use byte-identical Cartesian points")

    zero_state = np.load(ZERO / "seed_ns17.npz", allow_pickle=False)
    mapped_state = np.load(MAPPED / "seed_ns17.npz", allow_pickle=False)
    same_geometry = all(np.array_equal(zero_state[name], mapped_state[name])
                        for name in GEOMETRY_FIELDS)
    expected_seed = mapped["artifacts"]["seed"]["sha256"]
    if not same_geometry or root["seed_sha256"] != expected_seed or \
            zero_tcon_root["seed_sha256"] != expected_seed:
        raise SystemExit("lambda seed or exact R/Z geometry does not match provenance")

    records = [zero["native_score"], mapped["native_score"], score, zero_tcon_score]
    labels = [r"$\lambda=0$ geometry", "Field-line $\lambda$ projection",
              "TCON0=1: solver converged", "TCON0=0: 10,000-iteration cap"]
    colors = ("#9aa5b1", "#19758e", "#d18a28", "#b44c4c")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.3), constrained_layout=True)
    for ax, (key, title, xlabel) in zip(axes.flat, METRICS):
        values = [float(row[key]) for row in records]
        ax.set_xlim(min(values) * 0.55, max(values) * 2.2)
        bars = ax.barh(labels, values, color=colors, height=0.56)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12, loc="left", pad=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", which="both", color="#d6dce1", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9, length=0)
        ax.tick_params(axis="x", labelsize=8)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.3g}",
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points", va="center",
                        ha="left", fontsize=9, color="#24313a")
    fig.suptitle("Sheared-A exact projection and solver response",
                 fontsize=16, fontweight="semibold", color="#183642")
    fig.text(0.5, -0.012,
             "NS17 · 96 identical Cartesian points · TCON0=1 converged at 208; "
             "TCON0=0 capped at 10,000 · neither run was physically accepted",
             ha="center", fontsize=9, color="#4b5963")
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    record = {
        "schema": 1,
        "evidence": "matched_projection_and_solver_figure",
        "figure": FIGURE.relative_to(ROOT).as_posix(),
        "figure_sha256": sha256_file(FIGURE),
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "source_report_sha256": {
            "lambda_zero_projection": sha256_file(zero_path),
            "field_line_projection": sha256_file(mapped_path),
            "vmex_root": sha256_file(run_path),
            "vmex_score": sha256_file(RUN / "native_scores.json"),
            "tcon0_zero_root": sha256_file(zero_tcon_path),
            "tcon0_zero_score": sha256_file(ZERO_TCON_RUN / "native_scores.json"),
        },
        "source_pin": sources[0],
        "input_sha256": inputs[0],
        "point_cloud_sha256": point_hashes[0],
        "same_geometry_coefficients": same_geometry,
        "sample_count": score["sample_count"],
        "solver_status": {
            "tcon0_one": {"converged": root["solver_converged"],
                           "iterations": root["iterations"],
                           "accepted": root["accepted"],
                           "root_certified": root["root_certified"]},
            "tcon0_zero": {"converged": zero_tcon_root["solver_converged"],
                            "iterations": zero_tcon_root["iterations"],
                            "accepted": zero_tcon_root["accepted"],
                            "root_certified": zero_tcon_root["root_certified"]},
        },
        "measurement_resolved": False,
        "metrics": {key: [float(row[key]) for row in records]
                    for key, _, _ in METRICS},
        "interpretation": "the discrete solver root is not the most physically accurate sampled state",
    }
    write_json(MANIFEST, record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
