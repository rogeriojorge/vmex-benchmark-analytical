"""Plot measured integer 3-D projection and loose-root physical errors."""
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT

levels = (33, 65, 129)
projected = [json.loads((ROOT / f"results/projection/integer_3d_vmex_ns{ns}/native_scores.json").read_text())
             for ns in levels]
solved = [json.loads((ROOT / f"results/vmex/integer_3d_iota_ns{ns}_niter3000_projected_ftol1e-10/native_scores.json").read_text())
          for ns in levels]
quantities = (("field_relative_l2", "Cartesian B relative L2", 1e-5),
              ("current_relative_l2", "Cartesian J relative L2", 1e-3),
              ("force_pressure_scale", "Force / exact grad p RMS", 1e-3))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), layout="constrained")
for ax, (key, label, target) in zip(axes, quantities):
    ax.semilogy(levels, [r[key] for r in projected], "o-", label="exact projected state", color="#176b87")
    ax.semilogy(levels, [r[key] for r in solved], "s-", label="loose VMEX root", color="#bf5b31")
    ax.axhline(target, color="#555555", linestyle="--", linewidth=1, label="initial target")
    ax.set_xticks(levels, labels=[str(x) for x in levels])
    ax.set_xlabel("NS")
    ax.set_ylabel(label)
    ax.grid(True, which="both", alpha=0.25)
axes[0].legend(fontsize=7)
fig.suptitle("Integer 3-D: projection and separate loose-root evidence")
path = ROOT / "figures/vmex_integer_3d_projection.png"
fig.savefig(path, dpi=170)
plt.close(fig)
record = dict(schema=1, evidence="measured_figure", source_files=[
    f"results/projection/integer_3d_vmex_ns{ns}/native_scores.json" for ns in levels] + [
    f"results/vmex/integer_3d_iota_ns{ns}_niter3000_projected_ftol1e-10/native_scores.json" for ns in levels],
    figure="figures/vmex_integer_3d_projection.png", sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    ns=levels, roots_ftol=1e-10, root_status="exploratory_not_accepted")
(ROOT / "results/projection/integer_3d_figure_manifest.json").write_text(
    json.dumps(record, indent=2, allow_nan=False)+"\n")
print(record)
