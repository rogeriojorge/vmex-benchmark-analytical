"""Plot bounded gauge-scan residuals from saved JSON."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic import ROOT


def main():
    source = ROOT / "results/projection/integer_3d_gauge_scan_ns33.json"
    data = json.loads(source.read_text())
    baseline = json.loads((ROOT / "results/vmex/integer_3d_raw_residual_comparison.json").read_text())["rows"][0]["projected"]
    fig, axis = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)
    for m, n in ((2,0),(3,0),(2,1),(3,1)):
        rows = sorted((r for r in data["rows"] if r["m"] == m and r["n"] == n),
                      key=lambda r: r["amplitude"])
        axis.scatter([r["amplitude"] for r in rows],
                     [r["fsqr"]+r["fsqz"] for r in rows],
                     label=f"m={m}, n={n}")
    axis.axhline(baseline["fsqr"]+baseline["fsqz"],
                 linestyle="--", color="0.4", label="unshifted projection")
    axis.set(xlabel="Poloidal shift amplitude", ylabel="FSQR + FSQZ",
             title="NS33 projected gauge scan")
    axis.legend(fontsize=8)
    output = ROOT / "figures/vmex_integer_gauge_scan.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    manifest = dict(schema=1, evidence="figure_from_saved_data",
                    figure=output.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    sources=[source.relative_to(ROOT).as_posix(),
                             "results/vmex/integer_3d_raw_residual_comparison.json"])
    (ROOT / "results/projection/integer_3d_gauge_scan_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
