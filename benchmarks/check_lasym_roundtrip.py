"""Compare LASYM WOUT surface fields before and after state reconstruction."""
import json
from pathlib import Path
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex

from analytic import ROOT, cases, field

if len(sys.argv) != 4:
    raise SystemExit("Usage: python benchmarks/check_lasym_roundtrip.py INPUT WOUT CASE")
input_path, wout_path = Path(sys.argv[1]), Path(sys.argv[2])
case = cases()[sys.argv[3]]
inp = vmex.VmecInput.from_file(input_path)
wout = vmex.read_wout(wout_path)
if not (inp.lasym and wout.lasym):
    raise SystemExit("This check requires a genuine LASYM input and output")
start = perf_counter()
state = vmex.state_from_wout(wout, inp=inp)
recreated = vmex.wout_from_state(inp=inp, state=state, fsqr=0.0, fsqz=0.0, fsql=0.0)
ns = int(wout.ns)
rows = []
for j in sorted(set((max(1, ns//8), ns//4, ns//2, 3*ns//4, ns-1))):
    original = vmex.surface_field_data_from_wout(wout, s_index=j, ntheta=16, nphi=8)
    rebuilt = vmex.surface_field_data_from_wout(recreated, s_index=j, ntheta=16, nphi=8)
    xyz0 = np.moveaxis(np.asarray(original.gamma), 0, -1).reshape(-1, 3)
    xyz1 = np.moveaxis(np.asarray(rebuilt.gamma), 0, -1).reshape(-1, 3)
    B0 = np.moveaxis(np.asarray(original.B_total), 0, -1).reshape(-1, 3)
    B1 = np.moveaxis(np.asarray(rebuilt.B_total), 0, -1).reshape(-1, 3)
    exact_B = np.asarray(field(case, xyz0)[0])
    rows.append(dict(index=j, geometry_max_abs_m=float(np.max(np.linalg.norm(xyz1-xyz0, axis=-1))),
                     field_reconstruction_relative_l2=float(np.linalg.norm(B1-B0)/np.linalg.norm(B0)),
                     original_exact_field_relative_l2=float(np.linalg.norm(B0-exact_B)/np.linalg.norm(exact_B))))
status = "passed" if (max(r["geometry_max_abs_m"] for r in rows) < 1e-9 and
                      max(r["field_reconstruction_relative_l2"] for r in rows) < 1e-7) else "failed"
out = wout_path.parent / "lasym_roundtrip.json"
out.write_text(json.dumps(dict(schema=1, evidence="integration_consistency",
    status=status, case=case.name, ns=ns, path="WOUT -> native restart state -> WOUT -> surface B",
    elapsed_seconds=perf_counter()-start, no_live_volume_field=True, rows=rows),
    indent=2, allow_nan=False)+"\n")
for row in rows:
    print(row)
if status != "passed":
    raise SystemExit("LASYM WOUT-state round trip failed")
