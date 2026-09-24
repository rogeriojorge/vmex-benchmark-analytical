"""Run one generated fixed-boundary input; this is a forward smoke, not a certificate.

Usage: python benchmarks/run_vmex.py inputs/input.integer_3d_iota
The source API was reviewed, but this script was not executed in the handoff
container because VMEX was not installed. Physical scoring is a separate step.
"""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex

from analytic import ROOT

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
path = Path(sys.argv[1]).resolve()
manifest = path.parent / "manifest.json"
if manifest.exists():
    rows = json.loads(manifest.read_text())["records"]
    row = next((r for r in rows if r["file"] == path.name), None)
    if row is not None and (row["boundary_max_error_m"] > 1e-6 or
                            row["symmetric_omitted_coeff_m"] > 1e-10):
        raise SystemExit("Input representation unresolved. Refine and regenerate its manifest first.")
inp = vmex.VmecInput.from_file(path)
if inp.lfreeb:
    raise SystemExit("Use the coupled free-boundary protocol in plan P6; this runner is fixed-boundary only.")
out = ROOT / "results/vmex" / path.name.removeprefix("input.")
out.mkdir(parents=True, exist_ok=True)
start = perf_counter()
result = vmex.solve_multigrid(inp, verbose=True)
jax.block_until_ready(result.state)
seconds = perf_counter()-start
wout = vmex.wout_from_result(inp, result)
vmex.write_wout(out / "wout.nc", wout)
repo = Path(vmex.__file__).resolve().parent
proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True)
record = dict(schema=1, evidence="vmex_forward_smoke", full_benchmark_pass=False,
              vmex_commit=proc.stdout.strip() if proc.returncode == 0 else None,
              vmex_version=importlib.metadata.version("vmex"),
              input=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
              python=sys.version, platform=platform.platform(), devices=[str(d) for d in jax.devices()],
              solve_seconds_including_first_compile=seconds, converged=bool(result.converged),
              fsqr=float(np.asarray(result.fsqr)), fsqz=float(np.asarray(result.fsqz)),
              fsql=float(np.asarray(result.fsql)),
              scalars={k: float(np.asarray(getattr(wout, k))) for k in ("wb", "volume_p", "betatotal")},
              continuous_field_scored=False, input_sign_recovery_certified=False)
(out / "forward.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
if not result.converged:
    raise SystemExit("VMEX did not converge. Saved output is diagnostic, not an accepted equilibrium.")
