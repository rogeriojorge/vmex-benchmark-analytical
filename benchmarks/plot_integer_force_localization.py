"""Plot projected-state spectral-force marginals from saved JSON."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT


def main():
    source = ROOT / "results/vmex/integer_3d_force_localization_ns33.json"
    data = json.loads(source.read_text())["rows"]["projected"]
    keys = ("radial_fraction", "poloidal_mode_fraction", "toroidal_mode_fraction")
    titles = ("Radial row", "Poloidal mode m", "Toroidal mode |n|")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    for axis, key, title in zip(axes, keys, titles):
        for component in ("R", "Z"):
            values = np.asarray(data[component][key])
            axis.plot(np.arange(len(values)), values, marker="." if key != "radial_fraction" else None,
                      label=component)
        axis.set(xlabel=title, ylabel="Force fraction")
        axis.legend(fontsize=8)
    axes[0].set_title("Projected NS33 VMEX force")
    output = ROOT / "figures/vmex_integer_force_localization.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    manifest = dict(schema=1, evidence="figure_from_saved_data",
                    figure=output.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    source=source.relative_to(ROOT).as_posix())
    (ROOT / "results/vmex/integer_3d_force_localization_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
