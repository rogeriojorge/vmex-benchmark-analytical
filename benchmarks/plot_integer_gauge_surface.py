"""Plot continuum gauge finite-difference convergence and VMEX field floor."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic import ROOT


def main():
    continuum = ROOT / "results/projection/integer_3d_gauge_surface_continuum.json"
    projected = ROOT / "results/projection/integer_3d_gauge_surface_vmex.json"
    analytical_data = json.loads(continuum.read_text())
    projected_data = json.loads(projected.read_text())
    rows = analytical_data["finite_difference_convergence"]
    fig, ax = plt.subplots(figsize=(5.5,3.4), constrained_layout=True)
    ax.loglog([r["step"] for r in rows], [r["B_relative_l2"] for r in rows],
              marker="o", label="exact transformed chart")
    ax.axhline(projected_data["B_relative_l2"], color="C1", linestyle="--",
               label="finite VMEX state surface")
    ax.axhline(projected_data["unshifted_B_relative_l2"], color="C2", linestyle=":",
               label="unshifted VMEX surface")
    ax.set(xlabel="Central-difference step", ylabel="B relative L2 error",
           title="NS33 gauge surface at s=0.5")
    ax.legend(fontsize=8)
    output = ROOT / "figures/vmex_integer_gauge_surface.png"
    fig.savefig(output,dpi=180)
    plt.close(fig)
    manifest = dict(schema=1,evidence="figure_from_saved_data",
        figure=output.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        sources=[continuum.relative_to(ROOT).as_posix(),
                 projected.relative_to(ROOT).as_posix()])
    (ROOT / "results/projection/integer_3d_gauge_surface_figure_manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    main()
