"""Scan small physical-field-preserving poloidal gauge shifts at NS33.

For theta_new = theta_old + u(theta_new,s,phi), the exact Clebsch field is
unchanged when physical lambda_new = -phip(s)*u. The trial is projected back
to VMEX's finite Fourier/radial representation before any force evaluation.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex
from vmex.core.fourier import mode_table, trig_tables
from vmex.core.residuals import m1_physical_to_constrained
from vmex.core.solver import (SpectralState, evaluate_forces, hot_restart_state,
                              prepare_runtime, resolution_from_input, runtime_with_baselines)
from vmex.core.transforms import physical_to_internal_scale

from analytic import ROOT, cases, surface
from native_samples import sample_native
from score_samples import score


def project_shift(case, rt, *, m, n, amplitude, grid=64):
    modes = mode_table(rt.resolution.mpol, rt.resolution.ntor)
    scale = physical_to_internal_scale(modes, trig_tables(rt.resolution))
    theta = 2*np.pi*np.arange(grid)[:, None]/grid
    phi = 2*np.pi*np.arange(grid)[None, :]/(grid*case.nfp)
    s = np.linspace(0, 1, rt.resolution.ns)
    coefficients = np.zeros((3, len(s), modes.mnmax))
    phipf = np.asarray(rt.setup.phipf)
    lamscale = float(np.asarray(rt.setup.lamscale))
    for j, sj in enumerate(s):
        u = amplitude*sj*(1-sj)*np.sin(m*theta-n*case.nfp*phi)
        xyz = surface(case, case.edge*sj, theta-u, phi)
        R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
        lam_internal = -phipf[j]*u/lamscale
        spectra = [np.fft.fft2(a)/(grid*grid) for a in (R, Z, lam_internal)]
        for k, (mk, nk) in enumerate(zip(modes.m, modes.n)):
            factor = 1 if mk == nk == 0 else 2
            coefficients[0,j,k] = factor*spectra[0][mk,-nk].real
            coefficients[1,j,k] = -factor*spectra[1][mk,-nk].imag
            coefficients[2,j,k] = -factor*spectra[2][mk,-nk].imag
    R_cos, Z_sin, L_sin = (coefficients[i]*scale[None,:] for i in range(3))
    R_cos, Z_sin, _, _ = m1_physical_to_constrained(
        R_cos, Z_sin, modes=modes, lthreed=True, lasym=False, lconm1=True)
    zeros = np.zeros_like(R_cos)
    return SpectralState(R_cos=R_cos, R_sin=zeros, Z_cos=zeros,
                         Z_sin=Z_sin, L_cos=zeros, L_sin=L_sin)


def force(state, rt):
    prepared = hot_restart_state(rt, state)
    prepared_rt = runtime_with_baselines(rt, prepared)
    _, residual, diagnostics = evaluate_forces(prepared, prepared_rt)
    return dict(fsqr=float(np.asarray(residual.fsqr)),
                fsqz=float(np.asarray(residual.fsqz)),
                fsql=float(np.asarray(residual.fsql)),
                jacobian_sign_changed=bool(np.asarray(diagnostics.jacobian_sign_changed)))


def main():
    start = perf_counter()
    source = Path(vmex.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                     text=True).strip()
    case = cases()["integer_3d"]
    inp = vmex.VmecInput.from_file(ROOT / "inputs/input.integer_3d_iota")
    inp = replace(inp, ns_array=[33], ftol_array=[1e-10], niter_array=[3000])
    rt = prepare_runtime(inp, resolution_from_input(inp, ns=33))
    rows = []
    for m, n in ((2,0),(3,0),(2,1),(3,1)):
        for amplitude in (-0.2,-0.1,0.1,0.2):
            state = project_shift(case, rt, m=m, n=n, amplitude=amplitude)
            result = force(state, rt)
            rows.append(dict(m=m,n=n,amplitude=amplitude,**result))
            print(m,n,amplitude,result,flush=True)
    best = min(rows, key=lambda row: row["fsqr"]+row["fsqz"])
    selected = project_shift(case, rt, m=best["m"], n=best["n"],
                             amplitude=best["amplitude"])
    detail = ROOT / "results/projection/integer_3d_gauge_scan_ns33"
    detail.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(detail / "selected_state.npz", **{
        name: np.asarray(getattr(selected, name)) for name in
        ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    sample_start = perf_counter()
    samples = sample_native(inp, selected, case, runtime=rt)
    np.savez_compressed(detail / "selected_native_samples.npz", **samples)
    selected_score = score(samples)
    with np.load(ROOT / "results/projection/integer_3d_vmex_ns33/native_samples.npz") as baseline:
        if not np.array_equal(samples["xyz"], baseline["xyz"]):
            raise ValueError("Selected and baseline states have different held-out points")
        weighted_difference = {name: float(np.sqrt(np.sum(
            baseline["weights"][:,None]*(samples[name]-baseline[name])**2)/
            np.sum(baseline["weights"][:,None]*baseline[name]**2)))
            for name in ("B", "J", "gradp")}
    sample_seconds = perf_counter()-sample_start
    out = ROOT / "results/projection/integer_3d_gauge_scan_ns33.json"
    out.write_text(json.dumps(dict(schema=1, evidence="projected_gauge_scan",
        status="diagnostic_only", vmex_commit=commit, case="integer_3d", ns=33,
        physical_lambda_formula="-phipf(s) * u(theta_new,s,phi)",
        shift_formula="amplitude*s*(1-s)*sin(m*theta_new-n*nfp*phi)",
        selected_by="minimum fsqr+fsqz among scanned shifts",
        selected=dict(**best, native_score=selected_score,
                      relative_to_projected_native_samples=weighted_difference),
        scan_and_sample_seconds=perf_counter()-start,
        native_sample_seconds=sample_seconds, memory_unmeasured=True,
        physical_field_after_finite_projection_scored=True, rows=rows),
        indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
