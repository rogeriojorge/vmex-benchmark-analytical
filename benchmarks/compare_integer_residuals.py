"""Compare VMEX invariant forces for projected and exploratory solved states.

The same deck, radial grid, input profiles, and fixed boundary are used for
each pair. WOUT reconstruction is checked separately from physical scoring.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex
from vmex.core.solver import (SpectralState, evaluate_forces, hot_restart_state,
                              prepare_runtime, resolution_from_input, runtime_with_baselines)

from analytic import ROOT


def read_seed(path):
    with np.load(path) as arrays:
        return SpectralState(**{name: arrays[name] for name in
            ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})


STATE_NAMES = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def score(state, runtime):
    # Match solve(initial_state=...): boundary transfer precedes rebinding the
    # m=1 constraint baselines to this specific state.
    prepared = hot_restart_state(runtime, state)
    prepared_runtime = runtime_with_baselines(runtime, prepared)
    _, residual, diagnostic = evaluate_forces(prepared, prepared_runtime)
    values = {key: float(np.asarray(getattr(residual, key))) for key in
              ("fsqr", "fsqz", "fsql", "fedge", "gcr2", "gcz2", "gcl2")}
    values.update(wb=float(np.asarray(diagnostic.wb)), wp=float(np.asarray(diagnostic.wp)),
                  jacobian_sign_changed=bool(np.asarray(diagnostic.jacobian_sign_changed)))
    return prepared, values


def main():
    rows = []
    source_commit = None
    for ns in (33, 65, 129):
        inp = vmex.VmecInput.from_file(ROOT / "inputs/input.integer_3d_iota")
        inp = replace(inp, ns_array=[ns], ftol_array=[1e-10], niter_array=[3000])
        runtime = prepare_runtime(inp, resolution_from_input(inp, ns=ns))
        seed = read_seed(ROOT / f"results/projection/integer_3d_vmex_ns{ns}/seed.npz")
        wout_path = ROOT / f"results/vmex/integer_3d_iota_ns{ns}_niter3000_projected_ftol1e-10/wout.nc"
        root_path = wout_path.with_name("wout_restarted_state.npz")
        if wout_path.exists() and os.environ.get("BENCH_USE_SAVED_STATE") != "1":
            returned = vmex.state_from_wout(wout_path, inp=inp)
            if root_path.exists():
                saved = read_seed(root_path)
                if any(not np.array_equal(np.asarray(getattr(saved, name)),
                                          np.asarray(getattr(returned, name))) for name in STATE_NAMES):
                    raise ValueError("Saved restarted state differs from local WOUT")
            else:
                np.savez_compressed(root_path, **{name: np.asarray(getattr(returned, name))
                                                  for name in STATE_NAMES})
        else:
            returned = read_seed(root_path)
        seed_prepared, seed_score = score(seed, runtime)
        root_prepared, root_score = score(returned, runtime)
        forward = json.loads((wout_path.parent / "forward.json").read_text())
        if source_commit is None:
            source_commit = forward["vmex_commit"]
        elif forward["vmex_commit"] != source_commit:
            raise ValueError("Compared states use different VMEX source pins")
        restart_minus_solver = {key: root_score[key] - forward[key]
                                for key in ("fsqr", "fsqz", "fsql")}
        component_shifts = {name: float(np.linalg.norm(
            np.asarray(getattr(root_prepared, name))-np.asarray(getattr(seed_prepared, name))))
            for name in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")}
        rows.append(dict(ns=ns, projected=seed_score, wout_restarted=root_score,
                         restarted_state_sha256=hashlib.sha256(root_path.read_bytes()).hexdigest(),
                         projected_edge_adjustment_l2=float(np.linalg.norm(
                             np.asarray(seed_prepared.R_cos[-1])-np.asarray(seed.R_cos[-1]))+
                             np.linalg.norm(np.asarray(seed_prepared.Z_sin[-1])-np.asarray(seed.Z_sin[-1]))),
                         wout_restart_minus_solver_residual=restart_minus_solver,
                         state_component_l2_shifts=component_shifts))
        print(f"NS={ns}: projected={seed_score}; WOUT restart={root_score}", flush=True)
    out = ROOT / "results/vmex/integer_3d_raw_residual_comparison.json"
    out.write_text(json.dumps(dict(schema=1, evidence="vmex_invariant_force_comparison",
        status="diagnostic_only", vmex_commit=source_commit,
        common_deck="inputs/input.integer_3d_iota",
        common_ftol=1e-10, common_niter=3000, no_acceptance_inferred=True,
        warm_start_baselines_rebound_per_state=True,
        wout_restart_roundtrip_checked=True, memory_unmeasured=True, rows=rows),
        indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python benchmarks/compare_integer_residuals.py")
    main()
