"""Score LASYM WOUT surface fields without a fitted high-order lift.

The WOUT route evaluates each native full-mesh surface and adjacent half-mesh
field spectra. It is a surface B check, not a continuous-volume J/force score.
"""
import json
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex

from analytic import ROOT, cases, field, label_at_s

if len(sys.argv) != 3:
    raise SystemExit("Usage: python benchmarks/score_wout_surfaces.py WOUT CASE")
path = Path(sys.argv[1])
case = cases()[sys.argv[2]]
wout = vmex.read_wout(path)
ns = int(wout.ns)
rows = []
for j in sorted(set([max(1, ns//8), ns//4, ns//2, 3*ns//4, ns-1])):
    surf = vmex.surface_field_data_from_wout(wout, ntheta=16, nphi=8, s_index=j)
    xyz = np.moveaxis(np.asarray(surf.gamma), 0, -1).reshape(-1, 3)
    B = np.moveaxis(np.asarray(surf.B_total), 0, -1).reshape(-1, 3)
    area = np.linalg.norm(np.moveaxis(np.asarray(surf.area_vector), 0, -1), axis=-1).ravel()
    exact_B, label = field(case, xyz)
    exact_B, label = np.asarray(exact_B), np.asarray(label)
    err = B-exact_B
    rms = lambda q: np.sqrt(np.sum(area*np.sum(q*q, axis=-1))/np.sum(area))
    rows.append(dict(index=j, s=j/(ns-1), field_relative_surface_l2=float(rms(err)/rms(exact_B)),
                     field_max_abs_T=float(np.max(np.linalg.norm(err, axis=-1))),
                     label_max_over_edge=float(np.max(np.abs(label-label_at_s(case, j/(ns-1))))/case.edge)))
out = path.parent / "wout_surface_scores.json"
gate = 1e-5
out.write_text(json.dumps(dict(schema=1, evidence="analytic_recovery",
    status="passed" if max(r["field_relative_surface_l2"] for r in rows) <= gate else "failed",
    case=case.name, path="WOUT native-surface Fourier tables", field_samples_per_surface=128,
    surface_B_relative_gate=gate, no_volume_current_or_force=True, rows=rows),
    indent=2, allow_nan=False)+"\n")
for row in rows:
    print(row)
