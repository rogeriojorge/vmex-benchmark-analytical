"""Localize VMEX's NS33 invariant force on saved integer 3-D states.

This diagnostic uses the pinned solver's private force pipeline because the
public evaluator returns scalar residuals rather than spectral force blocks.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex
from vmex.core import solver
from vmex.core.solver import SpectralState

from analytic import ROOT

BLOCKS = {
    "R": ("force_R_cc", "force_R_ss", "force_R_sc", "force_R_cs"),
    "Z": ("force_Z_sc", "force_Z_cs", "force_Z_cc", "force_Z_ss"),
    "lambda": ("force_lambda_sc", "force_lambda_cs", "force_lambda_cc", "force_lambda_ss"),
}


def state_from_npz(path):
    with np.load(path) as arrays:
        return SpectralState(**{name: arrays[name] for name in
            ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})


def analyze(state, inp):
    rt = solver.prepare_runtime(inp, solver.resolution_from_input(inp, ns=33))
    state = solver.hot_restart_state(rt, state)
    rt = solver.runtime_with_baselines(rt, state)
    _, residual, diagnostic = solver.evaluate_forces(state, rt)
    geometry, jacobian, metrics, fields, _ = solver._field_chain_lane(state, rt)
    (R_cos, R_sin, Z_cos, Z_sin), _ = solver._geometry(state, rt)
    scaled, _, _ = solver._force_pipeline(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields,
        R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin,
        cache=diagnostic.cache, rt=rt, iteration=1, fsqz_previous=1.0)
    result = {}
    for component, names in BLOCKS.items():
        present = [np.asarray(getattr(scaled, name)) for name in names
                   if getattr(scaled, name) is not None]
        squared = np.sum(np.stack([block**2 for block in present]), axis=0)
        if component != "lambda":
            squared[-1] = 0.0  # fixed-boundary edge excluded from FSQR/FSQZ
        total = float(np.sum(squared))
        expected = float(np.asarray(getattr(residual,
            {"R":"gcr2", "Z":"gcz2", "lambda":"gcl2"}[component])))
        if not np.isclose(total, expected, rtol=1e-9, atol=1e-14):
            raise ValueError(f"{component} force blocks do not reproduce invariant residual")
        by_radial = squared.sum(axis=(1, 2))
        by_m = squared.sum(axis=(0, 2))
        by_n = squared.sum(axis=(0, 1))
        top = np.unravel_index(np.argmax(squared), squared.shape)
        result[component] = dict(total_sumsq=total,
            radial_fraction=(by_radial/total).tolist(),
            poloidal_mode_fraction=(by_m/total).tolist(),
            toroidal_mode_fraction=(by_n/total).tolist(),
            top_cell=dict(radial_index=int(top[0]), poloidal_m=int(top[1]),
                          toroidal_n_abs=int(top[2]), fraction=float(squared[top]/total)))
    return result


def main():
    source = Path(vmex.__file__).resolve().parents[1]
    source_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    inp = vmex.VmecInput.from_file(ROOT / "inputs/input.integer_3d_iota")
    inp = replace(inp, ns_array=[33], ftol_array=[1e-10], niter_array=[3000])
    paths = {
        "projected": ROOT / "results/projection/integer_3d_vmex_ns33/seed.npz",
        "iteration_65": ROOT / "results/vmex/integer_3d_short_warm_ns33_ftol1e-4/state.npz",
        "loose_root": ROOT / "results/vmex/integer_3d_iota_ns33_niter3000_projected_ftol1e-10/wout_restarted_state.npz",
    }
    rows = {name: analyze(state_from_npz(path), inp) for name, path in paths.items()}
    out = ROOT / "results/vmex/integer_3d_force_localization_ns33.json"
    out.write_text(json.dumps(dict(schema=1, evidence="vmex_spectral_force_localization",
        status="diagnostic_only", vmex_commit=source_commit,
        input="inputs/input.integer_3d_iota", ns=33,
        warm_start_baselines_rebound_per_state=True, rows=rows),
        indent=2, allow_nan=False)+"\n")
    for name, components in rows.items():
        print(name, {key: (value["top_cell"], value["radial_fraction"][:4])
                     for key, value in components.items()})


if __name__ == "__main__":
    main()
