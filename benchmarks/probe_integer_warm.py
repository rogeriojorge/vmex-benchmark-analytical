"""Record one bounded, deliberately loose warm trajectory for diagnosis."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex
from vmex.core.errors import VmecConvergenceError
from vmex.core.solver import SpectralState

from analytic import ROOT, cases
from native_samples import sample_native
from score_samples import score


def main():
    ns, ftol, limit = 33, 1e-4, 500
    inp = vmex.VmecInput.from_file(ROOT / "inputs/input.integer_3d_iota")
    inp = replace(inp, ns_array=[ns], ftol_array=[ftol], niter_array=[limit])
    seed_path = ROOT / f"results/projection/integer_3d_vmex_ns{ns}/seed.npz"
    source = Path(vmex.__file__).resolve().parents[1]
    source_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    with np.load(seed_path) as arrays:
        seed = SpectralState(**{name: arrays[name] for name in
            ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    out = ROOT / "results/vmex/integer_3d_short_warm_ns33_ftol1e-4"
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    try:
        result = vmex.solve(inp, initial_state=seed, ftol=ftol,
                            max_iterations=limit, mode="cli", verbose=False)
    except VmecConvergenceError as exc:
        record = dict(schema=1, evidence="bounded_warm_trajectory", status="failed",
            vmex_commit=source_commit,
            ns=ns, ftol=ftol, iteration_limit=limit, reason=str(exc),
            final_fsq=exc.fsq, elapsed_seconds=perf_counter()-start,
            seed_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
            physical_field_scored=False)
        (out / "probe.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        return
    solve_seconds = perf_counter()-start
    state_path = out / "state.npz"
    np.savez_compressed(state_path, **{name: np.asarray(getattr(result.state, name)) for name in
        ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    np.savez_compressed(out / "history.npz", fsq_history=result.fsq_history)
    sample_start = perf_counter()
    samples = sample_native(inp, result.state, cases()["integer_3d"])
    np.savez_compressed(out / "native_samples.npz", **samples)
    scores = score(samples)
    sample_seconds = perf_counter()-sample_start
    record = dict(schema=1, evidence="bounded_warm_trajectory", status="diagnostic_only",
        vmex_commit=source_commit,
        ns=ns, ftol=ftol, iteration_limit=limit, iterations=result.iterations,
        fsqr=result.fsqr, fsqz=result.fsqz, fsql=result.fsql,
        seed_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
        solve_seconds_including_first_compile=solve_seconds,
        sample_and_score_seconds=sample_seconds, memory_unmeasured=True,
        physical_score=scores, no_acceptance_inferred=True)
    (out / "probe.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python benchmarks/probe_integer_warm.py")
    main()
