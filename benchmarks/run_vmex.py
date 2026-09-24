"""Run one fixed-boundary input and save native Cartesian physical samples.

Usage: python benchmarks/run_vmex.py inputs/input.integer_axisymmetric_iota [NS]
An optional NS runs one cold radial level. A converged solve is scored but is
not accepted without the plan's representation and convergence gates.
"""
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import hashlib

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex

from analytic import ROOT
from analytic import cases
from native_samples import sample_native, sample_lifted_lasym
from score_samples import score

if len(sys.argv) not in (2, 3):
    raise SystemExit(__doc__)
path = Path(sys.argv[1]).resolve()
ns_override = int(sys.argv[2]) if len(sys.argv) == 3 else None
if ns_override is not None and ns_override < 3:
    raise SystemExit("NS override must be at least 3")
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
if ns_override is not None:
    inp = replace(inp, ns_array=[ns_override], ftol_array=[1e-14], niter_array=[30000])
out_name = path.name.removeprefix("input.")
if ns_override is not None:
    out_name += f"_ns{ns_override}"
out = ROOT / "results/vmex" / out_name
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
              ns_override=ns_override,
              python=sys.version, platform=platform.platform(), devices=[str(d) for d in jax.devices()],
              solve_seconds_including_first_compile=seconds, converged=bool(result.converged),
              fsqr=float(np.asarray(result.fsqr)), fsqz=float(np.asarray(result.fsqz)),
              fsql=float(np.asarray(result.fsql)),
              scalars={k: float(np.asarray(getattr(wout, k))) for k in ("wb", "volume_p", "betatotal")},
              continuous_field_scored=False, input_sign_recovery_certified=False)
(out / "forward.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
if not result.converged:
    raise SystemExit("VMEX did not converge. Saved output is diagnostic, not an accepted equilibrium.")
case_name = next((r["case"] for r in rows if r["file"] == path.name), None) if manifest.exists() else None
if case_name is None:
    raise SystemExit("A manifest case is required for native physical sampling.")
sample_start = perf_counter()
try:
    if inp.lasym:
        samples = sample_lifted_lasym(inp, result.state, cases()[case_name])
        field_path = "continuous_fitted_state_lasym"
    else:
        samples = sample_native(inp, result.state, cases()[case_name])
        field_path = "native_clebsch_cartesian"
except (ValueError, NotImplementedError) as exc:
    record.update(postprocess_status="failed", postprocess_reason=f"{type(exc).__name__}: {exc}")
    (out / "forward.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    raise SystemExit("VMEX solved, but physical postprocessing failed; inspect forward.json") from exc
sample_path = out / "native_samples.npz"
np.savez_compressed(sample_path, **samples)
scores = score(samples)
(out / "native_scores.json").write_text(json.dumps(scores, indent=2, allow_nan=False)+"\n")
record.update(continuous_field_scored=True, native_sample_sha256=hashlib.sha256(
    sample_path.read_bytes()).hexdigest(), native_score=scores,
    field_path=field_path,
    native_sample_and_score_seconds=perf_counter()-sample_start)
(out / "forward.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
