"""Plot saved NS33 warm trajectory and independent physical scores."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT


def main():
    short = ROOT / "results/vmex/integer_3d_short_warm_ns33_ftol1e-4"
    history = np.load(short / "history.npz")["fsq_history"]
    seed = json.loads((ROOT / "results/projection/integer_3d_vmex_ns33/native_scores.json").read_text())
    mid = json.loads((short / "probe.json").read_text())["physical_score"]
    loose = json.loads((ROOT / "results/vmex/integer_3d_iota_ns33_niter3000_projected_ftol1e-10/native_scores.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for col, name in enumerate(("R", "Z", "lambda")):
        axes[0].semilogy(np.arange(1, len(history)+1), history[:, col], label=name)
    axes[0].axhline(1e-4, color="0.4", linestyle="--", linewidth=1, label="stop tolerance")
    axes[0].set(xlabel="Warm iteration", ylabel="VMEX invariant force squared",
                title="NS33 discrete trajectory")
    axes[0].legend(fontsize=8)
    metrics = (("field_relative_l2", "B error"),
               ("current_relative_l2", "J error"),
               ("force_pressure_scale", "force / grad p"))
    states = (seed, mid, loose)
    labels = ("projected", "65 iterations", "311 iterations")
    for key, title in metrics:
        axes[1].semilogy(labels, [s[key] for s in states], marker="o", label=title)
    axes[1].set(ylabel="Independent physical error", title="NS33 sampled fields")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].legend(fontsize=8)
    output = ROOT / "figures/vmex_integer_3d_trajectory.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    manifest = dict(schema=1, evidence="figure_from_saved_data", figure=output.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    sources=["results/vmex/integer_3d_short_warm_ns33_ftol1e-4/history.npz",
                             "results/vmex/integer_3d_short_warm_ns33_ftol1e-4/probe.json",
                             "results/projection/integer_3d_vmex_ns33/native_scores.json",
                             "results/vmex/integer_3d_iota_ns33_niter3000_projected_ftol1e-10/native_scores.json"])
    (ROOT / "results/vmex/integer_3d_short_warm_ns33_ftol1e-4/figure_manifest.json").write_text(
        json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
