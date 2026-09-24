"""Plot measured constraint-switch residuals from saved data."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic import ROOT


def main():
    source = ROOT / "results/vmex/integer_3d_constraint_switch.json"
    rows = json.loads(source.read_text())["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), constrained_layout=True,
                             sharey=True)
    for axis, state in zip(axes, ("projected", "loose_root")):
        chosen = sorted((row for row in rows if row["state"] == state),
                        key=lambda row: row["ns"])
        ns = [row["ns"] for row in chosen]
        for component in ("fsqr", "fsqz"):
            for condition, style in (("default", "-"), ("zero_constraint", "--")):
                axis.semilogy(ns, [row[condition][component] for row in chosen],
                              marker="o", linestyle=style,
                              label=f"{component.upper()} {condition.replace('_', ' ')}")
        axis.set(xlabel="Radial surfaces NS", title=state.replace("_", " "))
        axis.set_xticks(ns)
        if state == "loose_root":
            axis.legend(fontsize=7)
    axes[0].set_ylabel("VMEX invariant force squared")
    output = ROOT / "figures/vmex_integer_constraint_switch.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    manifest = dict(schema=1, evidence="figure_from_saved_data",
                    figure=output.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    source=source.relative_to(ROOT).as_posix())
    (ROOT / "results/vmex/integer_3d_constraint_switch_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
