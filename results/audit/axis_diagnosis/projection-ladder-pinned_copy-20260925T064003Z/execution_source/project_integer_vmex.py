"""Project exact integer geometry into a VMEX state and score it before solving."""
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
from vmex.core.solver import SpectralState, prepare_runtime, resolution_from_input
from vmex.core.transforms import physical_to_internal_scale

from analytic import ROOT, cases, field, surface

CASE = "integer_3d"
NS = int(sys.argv[1]) if len(sys.argv) >= 2 else 33
VOLUME = len(sys.argv) == 3 and sys.argv[2] == "--volume"
if len(sys.argv) not in (1, 2, 3) or (len(sys.argv) == 3 and not VOLUME) or NS < 5:
    raise SystemExit("Usage: python benchmarks/project_integer_vmex.py [NS>=5 [--volume]]")
GRID = 64


def project(inp, case, ns=NS, grid=GRID):
    """Fit actual full-mesh R/Z; lambda=0 is certified separately here."""
    res = resolution_from_input(inp, ns=ns)
    modes = mode_table(res.mpol, res.ntor)
    trig = trig_tables(res)
    scale = physical_to_internal_scale(modes, trig)
    theta = 2*np.pi*np.arange(grid)[:, None]/grid
    phi = 2*np.pi*np.arange(grid)[None, :]/(grid*case.nfp)
    s = np.linspace(0, 1, ns)
    coefficients = np.zeros((2, ns, modes.mnmax))
    for j, sj in enumerate(s):
        xyz = surface(case, case.edge*sj, theta, phi)
        R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
        spectra = [np.fft.fft2(a)/(grid*grid) for a in (R, Z)]
        for k, (m, n) in enumerate(zip(modes.m, modes.n)):
            factor = 1 if m == n == 0 else 2
            coefficients[0, j, k] = factor*spectra[0][m, -n].real
            coefficients[1, j, k] = -factor*spectra[1][m, -n].imag
    R_cos, Z_sin = (coefficients[i]*scale[None, :] for i in range(2))
    R_cos, Z_sin, _, _ = m1_physical_to_constrained(
        R_cos, Z_sin, modes=modes, lthreed=bool(res.lthreed),
        lasym=False, lconm1=True)
    zeros = np.zeros_like(R_cos)
    return SpectralState(R_cos=R_cos, R_sin=zeros, Z_cos=zeros,
                         Z_sin=Z_sin, L_cos=zeros, L_sin=zeros)


def main():
    case = cases()[CASE]
    inp = vmex.VmecInput.from_file(ROOT / "inputs" / f"input.{CASE}_iota")
    inp = replace(inp, ns_array=[NS], ftol_array=[1e-14], niter_array=[3000])
    start = perf_counter()
    state = project(inp, case)
    runtime = prepare_runtime(inp, resolution_from_input(inp, ns=NS))
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
                         label_max_over_edge=float(np.max(abs(np.asarray(field(case, xyz)[1])-case.edge*j/(NS-1)))/case.edge)))
    source = Path(vmex.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    out = ROOT / f"results/projection/integer_3d_vmex_ns{NS}"
    out.mkdir(parents=True, exist_ok=True)
    max_B = max(r["B_relative_surface_l2"] for r in rows)
    status = "passed" if max_B < 1e-5 else "failed"
    (out / "projection.json").write_text(json.dumps(dict(schema=1, evidence="analytic_projection",
        status=status, vmex_commit=commit, case=CASE, ns=NS, angular_fit_grid=GRID,
        no_nonlinear_solve=True, lambda_zero_surface_certificate="results/projection/integer_3d_surface.json",
        field_path="projected state -> WOUT surface field", elapsed_seconds=perf_counter()-start,
        max_B_relative_surface_l2=max_B, rows=rows), indent=2, allow_nan=False)+"\n")
    np.savez_compressed(out / "seed.npz", **{name: np.asarray(getattr(state, name)) for name in
        ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    if VOLUME:
        from native_samples import sample_native
        from score_samples import score
        samples = sample_native(inp, state, case, runtime=runtime)
        np.savez_compressed(out / "native_samples.npz", **samples)
        scores = score(samples)
        (out / "native_scores.json").write_text(json.dumps(scores, indent=2, allow_nan=False)+"\n")
        print({"projected_native_score": scores})
    print({"status": status, "max_B_relative_surface_l2": max_B, "rows": rows})
    if status != "passed":
        print("Projection does not meet the initial surface B target")


if __name__ == "__main__":
    main()
