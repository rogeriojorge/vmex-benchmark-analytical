"""Run one fixed-boundary input and save native Cartesian physical samples.

Usage: python benchmarks/run_vmex.py INPUT [NS [NITER [SEED_NPZ]]]
Optional NS and NITER run one bounded radial level. SEED_NPZ is a projected
SpectralState at that NS. A converged solve is scored but is
not accepted without the plan's representation and convergence gates.
BENCH_FTOL overrides the stage tolerance; BENCH_TCON0 overrides TCON0, with
the literal value 'default' using the input deck. TCON0 runs also preserve and
score a capped terminal state when the solver does not converge.
"""
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter
import hashlib
import os

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import scipy
import vmex
from vmex.core.errors import VmecConvergenceError
from vmex.core.solver import SpectralState

from analytic import ROOT
from analytic import cases
from native_samples import sample_native, sample_lifted_lasym
from score_samples import score

if len(sys.argv) not in (2, 3, 4, 5):
    raise SystemExit(__doc__)
path = Path(sys.argv[1]).resolve()
ns_override = int(sys.argv[2]) if len(sys.argv) == 3 else None
if len(sys.argv) >= 4:
    ns_override = int(sys.argv[2])
niter_override = int(sys.argv[3]) if len(sys.argv) >= 4 else 30000
ftol_override = float(os.environ.get("BENCH_FTOL", "1e-14"))
tcon_text = os.environ.get("BENCH_TCON0")
tcon_override = None if tcon_text is None or tcon_text.lower() == "default" else float(tcon_text)
seed_path = Path(sys.argv[4]).resolve() if len(sys.argv) == 5 else None
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
if niter_override < 1:
    raise SystemExit("NITER must be positive")
if not np.isfinite(ftol_override) or ftol_override <= 0:
    raise SystemExit("BENCH_FTOL must be finite and positive")
if tcon_override is not None and (not np.isfinite(tcon_override) or tcon_override < 0):
    raise SystemExit("BENCH_TCON0 must be finite and nonnegative, or 'default'")
peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
peak_rss_mib = peak_rss / (1024**2 if sys.platform == "darwin" else 1024)
if ns_override is not None:
    inp = replace(inp, ns_array=[ns_override], ftol_array=[ftol_override], niter_array=[niter_override])
seed = None
seed_sha = None
if seed_path is not None:
    with np.load(seed_path) as arrays:
        seed = SpectralState(**{name: arrays[name] for name in
            ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    seed_sha = hashlib.sha256(seed_path.read_bytes()).hexdigest()
out_name = path.name.removeprefix("input.")
if ns_override is not None:
    out_name += f"_ns{ns_override}"
if len(sys.argv) >= 4:
    out_name += f"_niter{niter_override}"
if seed is not None:
    out_name += "_projected"
if ftol_override != 1e-14:
    out_name += f"_ftol{ftol_override:g}"
if tcon_text is not None:
    out_name += "_tcon0default" if tcon_override is None else f"_tcon0{tcon_override:g}"
out = ROOT / "results/vmex" / out_name
out.mkdir(parents=True, exist_ok=True)
start = perf_counter()
try:
    result = vmex.solve_multigrid(
        inp,
        ns_array=[ns_override] if ns_override is not None else None,
        ftol_array=[ftol_override] if ns_override is not None else None,
        niter_array=[niter_override] if ns_override is not None else None,
        initial_state=seed, verbose=True, tcon0=tcon_override,
        raise_on_max_iterations=tcon_text is None)
except VmecConvergenceError as exc:
    repo = Path(vmex.__file__).resolve().parent
    proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, capture_output=True)
    failure = dict(schema=1, evidence="vmex_forward_smoke", status="failed",
                   full_benchmark_pass=False, vmex_commit=proc.stdout.strip() if proc.returncode == 0 else None,
              input=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
              ns_override=ns_override, niter_limit=niter_override,
                   ftol=ftol_override, tcon0_requested=tcon_override,
                   tcon0_effective=float(inp.tcon0) if tcon_override is None else tcon_override,
                   initialization="projected" if seed is not None else "cold",
                   seed_sha256=seed_sha,
                   solve_seconds_including_first_compile=perf_counter()-start,
                   peak_rss_mib=peak_rss_mib,
                   python=sys.version, platform=platform.platform(),
                   devices=[str(d) for d in jax.devices()],
                   jax_version=jax.__version__, numpy_version=np.__version__,
                   scipy_version=scipy.__version__,
                   exception_type=type(exc).__name__, reason=str(exc),
                   continuous_field_scored=False, memory_unmeasured=False)
    (out / "forward.json").write_text(json.dumps(failure, indent=2, allow_nan=False)+"\n")
    raise SystemExit("VMEX did not converge; inspect forward.json") from exc
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
              ns_override=ns_override, niter_limit=niter_override, ftol=ftol_override,
              tcon0_requested=tcon_override,
              tcon0_effective=float(inp.tcon0) if tcon_override is None else tcon_override,
              initialization="projected" if seed is not None else "cold",
              seed_sha256=seed_sha,
              python=sys.version, platform=platform.platform(), devices=[str(d) for d in jax.devices()],
              jax_version=jax.__version__, numpy_version=np.__version__,
              scipy_version=scipy.__version__,
              solve_seconds_including_first_compile=seconds, converged=bool(result.converged),
              iterations=int(result.iterations), ier_flag=int(result.ier_flag),
              peak_rss_mib=peak_rss_mib,
              fsqr=float(np.asarray(result.fsqr)), fsqz=float(np.asarray(result.fsqz)),
              fsql=float(np.asarray(result.fsql)),
              scalars={k: float(np.asarray(getattr(wout, k))) for k in ("wb", "volume_p", "betatotal")},
              continuous_field_scored=False, input_sign_recovery_certified=False)
(out / "forward.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
case_name = next((r["case"] for r in rows if r["file"] == path.name), None) if manifest.exists() else None
if case_name is None:
    raise SystemExit("A manifest case is required for native physical sampling.")
sample_start = perf_counter()
try:
    if inp.lasym:
        samples = sample_lifted_lasym(inp, result.state, cases()[case_name])
        field_path = "continuous_fitted_state_lasym"
    else:
        samples = sample_native(inp, result.state, cases()[case_name], chunk_size=32)
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
    native_sample_and_score_seconds=perf_counter()-sample_start,
    peak_rss_mib=max(peak_rss_mib,
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)),
    terminal_state_scored=not bool(result.converged),
    accepted=False)
(out / "forward.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
