"""Isolate the VMEX spectral-condensation contribution to integer 3-D FSQ."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex
from vmex.core.solver import (SpectralState, evaluate_forces, hot_restart_state,
                              prepare_runtime, resolution_from_input, runtime_with_baselines)

from analytic import ROOT


def load_state(path):
    with np.load(path) as arrays:
        return SpectralState(**{name: arrays[name] for name in
            ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})


def evaluate(inp, state, *, tcon0):
    ns = state.R_cos.shape[0]
    rt = prepare_runtime(inp, resolution_from_input(inp, ns=ns), tcon0=tcon0)
    prepared = hot_restart_state(rt, state)
    rt = runtime_with_baselines(rt, prepared)
    _, residual, diagnostic = evaluate_forces(prepared, rt)
    tcon = np.asarray(diagnostic.cache.tcon)
    return dict(fsqr=float(np.asarray(residual.fsqr)),
                fsqz=float(np.asarray(residual.fsqz)),
                fsql=float(np.asarray(residual.fsql)),
                tcon_min=float(tcon.min()), tcon_max=float(tcon.max()),
                jacobian_sign_changed=bool(np.asarray(diagnostic.jacobian_sign_changed)))


def main():
    source = Path(vmex.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                     text=True).strip()
    rows = []
    for ns in (33, 65, 129):
        inp = vmex.VmecInput.from_file(ROOT / "inputs/input.integer_3d_iota")
        inp = replace(inp, ns_array=[ns], ftol_array=[1e-10], niter_array=[3000])
        base = ROOT / f"results/vmex/integer_3d_iota_ns{ns}_niter3000_projected_ftol1e-10"
        paths = {"projected": ROOT / f"results/projection/integer_3d_vmex_ns{ns}/seed.npz",
                 "loose_root": base / "wout_restarted_state.npz"}
        if ns == 33:
            paths["iteration_65"] = ROOT / "results/vmex/integer_3d_short_warm_ns33_ftol1e-4/state.npz"
        for name, path in paths.items():
            state = load_state(path)
            default = evaluate(inp, state, tcon0=None)
            no_constraint = evaluate(inp, state, tcon0=0.0)
            rows.append(dict(ns=ns, state=name, default=default,
                             zero_constraint=no_constraint,
                             R_default_over_zero_constraint=default["fsqr"]/no_constraint["fsqr"],
                             Z_default_over_zero_constraint=default["fsqz"]/no_constraint["fsqz"]))
            print(ns, name, default["fsqr"], no_constraint["fsqr"],
                  default["fsqz"], no_constraint["fsqz"])
    out = ROOT / "results/vmex/integer_3d_constraint_switch.json"
    out.write_text(json.dumps(dict(schema=1, evidence="vmex_constraint_switch_diagnostic",
        status="diagnostic_only", vmex_commit=commit,
        input="inputs/input.integer_3d_iota", no_physical_acceptance_inferred=True,
        method="same state and deck, warm-start edge/baselines; vary only tcon0 from deck default to zero",
        rows=rows), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
