"""Plot asymmetric Solov'ev WOUT surface B errors from saved records."""
import hashlib
import json

import matplotlib.pyplot as plt

from analytic import ROOT

data = {}
sources = {}
for ns in (33, 65, 129):
    path = ROOT / f"results/vmex/solovev_asymmetric_iota_ns{ns}/wout_surface_scores.json"
    data[ns] = json.loads(path.read_text())
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

fig, ax = plt.subplots(figsize=(7, 4.5))
for s in (0.125, 0.5, 1.0):
    ax.semilogy(list(data), [next(r["field_relative_surface_l2"] for r in data[ns]["rows"]
                            if r["s"] == s) for ns in data], "o-", label=f"s={s:g}")
ax.set(xticks=list(data), xlabel="VMEX radial surfaces NS",
       ylabel="Area-weighted B relative L2 on surface",
       title="Asymmetric Solov'ev: native WOUT surface field")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
path = ROOT / "figures/vmex_lasym_surface_B.png"
fig.savefig(path, dpi=170)
plt.close(fig)
(ROOT / "results/vmex/lasym_figure_manifest.json").write_text(json.dumps(dict(
    schema=1, figure=str(path.relative_to(ROOT)), generator="benchmarks/plot_lasym_surfaces.py",
    evidence="analytic_recovery", scope="surface B only; WOUT half-mesh field spectra, not continuous J/force",
    sources_sha256=sources, figure_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    indent=2)+"\n")
