"""Plot the measured axisymmetric recovery ladder from saved score records."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT

RUNS = [(33, "integer_axisymmetric_iota_ns33"),
        (65, "integer_axisymmetric_iota_ns65"),
        (129, "integer_axisymmetric_iota_ns129")]
KEYS = (("field_relative_l2", "B relative L2"),
        ("current_relative_l2", "J relative L2"),
        ("force_pressure_scale", "force / exact pressure gradient"))

data = []
sources = {}
for ns, name in RUNS:
    path = ROOT / "results/vmex" / name / "forward.json"
    record = json.loads(path.read_text())
    if not record["converged"] or not record["continuous_field_scored"]:
        raise ValueError(f"Unscored or unconverged rung: {name}")
    data.append((ns, record["native_score"]))
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

fig, ax = plt.subplots(figsize=(7, 4.5))
for key, label in KEYS:
    ax.semilogy([ns for ns, _ in data], [q[key] for _, q in data], "o-", label=label)
ax.set(xticks=[ns for ns, _ in data], xlabel="VMEX radial surfaces NS",
       ylabel="Whole-volume sampled relative error",
       title="Axisymmetric integer equilibrium: cold VMEX recovery")
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
out = ROOT / "figures/vmex_axisymmetric_recovery.png"
fig.savefig(out, dpi=170)
plt.close(fig)
(ROOT / "results/vmex/figure_manifest.json").write_text(json.dumps({
    "schema": 1, "figure": str(out.relative_to(ROOT)),
    "generator": "benchmarks/plot_vmex_recovery.py",
    "evidence": "analytic_recovery", "status": "partial_case_ladder",
    "scorer": "benchmarks/score_samples.py", "source_sha256": sources,
    "figure_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    "units": "dimensionless relative L2; 3x8x4 Gauss/angular physical-volume samples",
    "scope": "prescribed-iota axisymmetric integer case only; no projection baseline",
}, indent=2)+"\n")
