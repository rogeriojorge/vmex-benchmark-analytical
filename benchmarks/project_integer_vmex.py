"""Project an exact case's geometry into a VMEX state and score it before solving.

Surfaces are placed at the exact normalized toroidal flux (``label_at_s``). lambda
is zero, which is the exact field only where the analytic angle is already a
straight-field-line angle (the integer family); for other families the seed is
geometry-only and is recorded as such.
"""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex
from vmex.core.fourier import mode_table, trig_tables
from vmex.core.residuals import m1_physical_to_constrained
from vmex.core.solver import SpectralState, resolution_from_input
from vmex.core.transforms import physical_to_internal_scale

from analytic import ROOT, cases, field, label_at_s, surface

ARGS = sys.argv[1:]
CASE = ARGS.pop(ARGS.index("--case")+1) if "--case" in ARGS else "integer_3d"
SOLVE = "--solve" in ARGS  # also run VMEX from the projection and write its WOUT
FTOL = float(ARGS.pop(ARGS.index("--ftol")+1)) if "--ftol" in ARGS else 1e-12
ARGS = [a for a in ARGS if a not in ("--case", "--solve", "--ftol")]
NS = int(ARGS[0]) if ARGS else 33
if len(ARGS) > 1 or NS < 5:
    raise SystemExit("Usage: python benchmarks/project_integer_vmex.py [NS>=5] [--case NAME] [--solve [--ftol F]]")
GRID = 64


def project(inp, case, ns=NS, grid=GRID):
    """Fit full-mesh R/Z on the exact flux surfaces; lambda = 0."""
    res = resolution_from_input(inp, ns=ns)
    modes = mode_table(res.mpol, res.ntor)
    trig = trig_tables(res)
    scale = physical_to_internal_scale(modes, trig)
    theta = 2*np.pi*np.arange(grid)[:, None]/grid
    phi = 2*np.pi*np.arange(grid)[None, :]/(grid*case.nfp)
    s = np.linspace(0, 1, ns)
    coefficients = np.zeros((4, ns, modes.mnmax))  # R cos, Z sin, R sin, Z cos
    for j, sj in enumerate(s):
        xyz = surface(case, float(label_at_s(case, sj)), theta, phi)
        R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
        spectra = [np.fft.fft2(a)/(grid*grid) for a in (R, Z)]
        for k, (m, n) in enumerate(zip(modes.m, modes.n)):
            factor = 1 if m == n == 0 else 2
            coefficients[:, j, k] = factor*np.array((spectra[0][m, -n].real, -spectra[1][m, -n].imag,
                                                     -spectra[0][m, -n].imag, spectra[1][m, -n].real))
    R_cos, Z_sin, R_sin, Z_cos = (coefficients[i]*scale[None, :] for i in range(4))
    zeros = np.zeros_like(R_cos)
    if not case.lasym:
        R_sin = Z_cos = zeros
    R_cos, Z_sin, R_sin, Z_cos = m1_physical_to_constrained(
        R_cos, Z_sin, R_sin, Z_cos, modes=modes, lthreed=bool(res.lthreed),
        lasym=bool(case.lasym), lconm1=True)
    R_sin, Z_cos = (zeros if a is None else a for a in (R_sin, Z_cos))
    return SpectralState(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos,
                         Z_sin=Z_sin, L_cos=zeros, L_sin=zeros)


def main():
    case = cases()[CASE]
    inp = vmex.VmecInput.from_file(ROOT / "inputs" / f"input.{CASE}_iota")
    inp = replace(inp, ns_array=[NS], ftol_array=[1e-14], niter_array=[3000])
    start = perf_counter()
    state = project(inp, case)
    # These residual placeholders only permit in-memory surface conversion.
    # This projected state has not passed a nonlinear VMEX solve.
    wout = vmex.wout_from_state(inp=inp, state=state, fsqr=0.0, fsqz=0.0, fsql=0.0)
    rows = []
    for j in sorted(set((max(1, NS//8), NS//4, NS//2, 3*NS//4, NS-1))):
        data = vmex.surface_field_data_from_wout(wout, s_index=j, ntheta=16, nphi=8)
        xyz = np.moveaxis(np.asarray(data.gamma), 0, -1).reshape(-1, 3)
        B = np.moveaxis(np.asarray(data.B_total), 0, -1).reshape(-1, 3)
        exact = np.asarray(field(case, xyz)[0])
        rows.append(dict(index=j, B_relative_surface_l2=float(np.linalg.norm(B-exact)/np.linalg.norm(exact)),
                         label_max_over_edge=float(np.max(abs(np.asarray(field(case, xyz)[1])
                                                         - float(label_at_s(case, j/(NS-1)))))/case.edge)))
    source = Path(vmex.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    out = ROOT / f"results/projection/{CASE}_vmex_ns{NS}"
    out.mkdir(parents=True, exist_ok=True)
    max_B = max(r["B_relative_surface_l2"] for r in rows)
    status = "passed" if max_B < 1e-5 else "failed"
    (out / "projection.json").write_text(json.dumps(dict(schema=2, evidence="analytic_projection",
        status=status, vmex_commit=commit, case=CASE, ns=NS, angular_fit_grid=GRID,
        no_nonlinear_solve=True, flux_map="label_at_s",
        seed=("exact_field_lambda_zero" if case.family == "integer" else "geometry_only_lambda_zero"),
        field_path="projected state -> WOUT surface field", elapsed_seconds=perf_counter()-start,
        max_B_relative_surface_l2=max_B, rows=rows), indent=2, allow_nan=False)+"\n")
    np.savez_compressed(out / "seed.npz", **{name: np.asarray(getattr(state, name)) for name in
        ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    if SOLVE:
        from vmex.core.errors import VmecConvergenceError
        from vmex.core.solver import solve
        try:  # tcon0 = 0: no spectral-condensation force at the exact root
            result = solve(inp, resolution_from_input(inp, ns=NS), initial_state=state, ftol=FTOL,
                           max_iterations=30000, mode="jit", tcon0=0.0)
        except VmecConvergenceError as error:
            raise SystemExit(f"seeded solve unconverged: fsq {error.fsq} after {error.iteration}")
        vmex.write_wout(out / f"wout_{CASE}_seeded_ns{NS}.nc", vmex.wout_from_result(inp, result))
        print({"seeded_solve": dict(iterations=int(result.iterations), fsq=[float(result.fsqr),
               float(result.fsqz), float(result.fsql)])})
    print({"status": status, "max_B_relative_surface_l2": max_B, "rows": rows})
    if status != "passed":
        print("Projection does not meet the initial surface B target")


if __name__ == "__main__":
    main()
