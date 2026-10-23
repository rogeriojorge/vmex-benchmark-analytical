"""Project an exact case's geometry (and, for sheared cases, lambda) into a VMEX state before solving.

Surfaces are placed at the exact normalized toroidal flux (``label_at_s``).  The lambda seed is
one of

* ``exact_field_lambda_zero``: the integer family, where the analytic angle is already a
  straight-field-line angle, so lambda = 0 is the exact field;
* ``full_flux_lambda``: sheared (and any non-integer) families, lambda from the complete exact
  field through both signed flux densities (``flux_lambda.exact_surface_lambda``), not from the
  field-line direction alone;
* ``geometry_only_lambda_zero``: lambda = 0 where the exact field needs it, kept as a negative
  control (``--lambda zero``).

``project`` is importable; the command line wrapper writes a schema-2 ``projection.json``.
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
from vmex.core.solver import SpectralState, prepare_runtime, resolution_from_input
from vmex.core.transforms import physical_to_internal_scale

from analytic import ROOT, cases, field, label_at_s, surface
from flux_lambda import fourier_coefficients, seed_type

GRID = 64          # angular fit grid for R and Z
LAMBDA_GRID = 65   # odd grid for the full-flux lambda


def with_resolution(inp, ns, mpol=None, ntor=None):
    """The deck at one radial grid and, optionally, another Fourier resolution (arrays padded or cropped)."""
    mpol, ntor = mpol or int(inp.mpol), ntor or int(inp.ntor)
    changes = dict(ns_array=[ns], ftol_array=[1e-14], niter_array=[3000], mpol=mpol, ntor=ntor)
    for name in ("rbc", "zbs", "rbs", "zbc"):
        old = np.asarray(getattr(inp, name))
        new = np.zeros((2*ntor+1, mpol))
        m, n = min(mpol, old.shape[1]), min(ntor, int(inp.ntor))
        new[ntor-n:ntor+n+1, :m] = old[int(inp.ntor)-n:int(inp.ntor)+n+1, :m]
        changes[name] = new
    for name in ("raxis_c", "zaxis_s", "raxis_s", "zaxis_c"):
        old = np.asarray(getattr(inp, name))
        changes[name] = np.pad(old, (0, max(0, ntor+1-len(old))))[:ntor+1]
    return replace(inp, **changes)


def lambda_rows(case, ns, n=LAMBDA_GRID):
    """Physical full-flux lambda on every interior full-mesh surface, and its certificates.

    Row 0 (the axis) is left zero; the caller applies VMEC's axis closure.
    """
    from flux_lambda import exact_surface_lambda
    lam = np.zeros((ns, n, n))
    certificates = [None]
    for j, sj in enumerate(np.linspace(0, 1, ns)):
        if j == 0:
            continue
        lam[j], certificate, _ = exact_surface_lambda(case, float(sj), n)
        certificates.append(certificate)
    return lam, certificates


def project(inp, case, ns, grid=GRID, lambda_mode="auto", lambda_grid=LAMBDA_GRID):
    """Fit full-mesh R/Z on the exact flux surfaces and add the chosen lambda seed.

    Returns ``(state, info)``; ``info`` records the seed type and lambda certificates.
    """
    kind = seed_type(case, lambda_mode)
    resolution = resolution_from_input(inp, ns=ns)
    modes = mode_table(resolution.mpol, resolution.ntor)
    trig = trig_tables(resolution)
    scale = np.asarray(physical_to_internal_scale(modes, trig))
    theta = 2*np.pi*np.arange(grid)[:, None]/grid
    phi = 2*np.pi*np.arange(grid)[None, :]/(grid*case.nfp)
    s = np.linspace(0, 1, ns)
    coefficients = np.zeros((4, ns, modes.mnmax))  # R cos, Z sin, R sin, Z cos
    for j, sj in enumerate(s):
        xyz = surface(case, float(label_at_s(case, sj)), theta, phi)
        R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
        (Rc, Rs), (Zc, Zs) = (fourier_coefficients(a, modes) for a in (R, Z))
        coefficients[:, j] = np.array((Rc, Zs, Rs, Zc))
    R_cos, Z_sin, R_sin, Z_cos = (coefficients[i]*scale[None, :] for i in range(4))
    zeros = np.zeros_like(R_cos)
    if not case.lasym:
        R_sin = Z_cos = zeros
    R_cos, Z_sin, R_sin, Z_cos = m1_physical_to_constrained(
        R_cos, Z_sin, R_sin, Z_cos, modes=modes, lthreed=bool(resolution.lthreed),
        lasym=bool(case.lasym), lconm1=True)
    R_sin, Z_cos = (zeros if a is None else a for a in (R_sin, Z_cos))
    L_cos = L_sin = zeros
    info = {"seed": kind, "lambda_grid": None, "lambda_certificates": None}
    if kind == "full_flux_lambda":
        runtime = prepare_runtime(inp, resolution)
        phipf = np.asarray(runtime.setup.phipf, dtype=float)
        lamscale = float(np.asarray(runtime.setup.lamscale))
        lam, certificates = lambda_rows(case, ns, lambda_grid)
        cos = np.zeros((ns, modes.mnmax))
        sin = np.zeros((ns, modes.mnmax))
        for j in range(1, ns):
            cos[j], sin[j] = fourier_coefficients(lam[j], modes)
        m, n = np.asarray(modes.m, dtype=int), np.asarray(modes.n, dtype=int)
        axis_copy = (m == 0) & (n > 0)  # VMEC's symmetric lambda axis closure
        if ns > 1:
            sin[0, axis_copy], cos[0, axis_copy] = sin[1, axis_copy], cos[1, axis_copy]
        discarded = float(np.linalg.norm(cos)) if not case.lasym else 0.0
        if not case.lasym:
            cos = np.zeros_like(cos)
        L_cos, L_sin = (c*phipf[:, None]/lamscale*scale[None, :] for c in (cos, sin))
        keys = ("gradient_reconstruction_relative", "closedness_scaled_rms", "period_mean_scaled_max",
                "minimum_straight_angle_derivative")
        rows = [c for c in certificates if c is not None]
        info.update(lambda_grid=lambda_grid, discarded_cosine_l2=discarded,
                    lambda_physical_sine_l2=float(np.linalg.norm(sin)),
                    lambda_certificates={k: (min if k.startswith("minimum") else max)(c[k] for c in rows)
                                         for k in keys})
    state = SpectralState(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin, L_cos=L_cos, L_sin=L_sin)
    return state, info


def surface_rows(inp, case, state, ns):
    """Surface B error of the projected state through its WOUT, at five native surfaces."""
    # These residual placeholders only permit in-memory surface conversion.
    # This projected state has not passed a nonlinear VMEX solve.
    wout = vmex.wout_from_state(inp=inp, state=state, fsqr=0.0, fsqz=0.0, fsql=0.0)
    rows = []
    for j in sorted(set((max(1, ns//8), ns//4, ns//2, 3*ns//4, ns-1))):
        data = vmex.surface_field_data_from_wout(wout, s_index=j, ntheta=16, nphi=8)
        xyz = np.moveaxis(np.asarray(data.gamma), 0, -1).reshape(-1, 3)
        B = np.moveaxis(np.asarray(data.B_total), 0, -1).reshape(-1, 3)
        exact = np.asarray(field(case, xyz)[0])
        label = np.asarray(field(case, xyz)[1])
        rows.append(dict(index=j, B_relative_surface_l2=float(np.linalg.norm(B-exact)/np.linalg.norm(exact)),
                         label_max_over_edge=float(np.max(abs(label-float(label_at_s(case, j/(ns-1)))))
                                                   / case.edge)))
    return wout, rows


def parse(argv):
    args, options = list(argv), {}
    for flag, default, kind in (("--case", "integer_3d", str), ("--ftol", 1e-12, float),
                                ("--mpol", None, int), ("--ntor", None, int), ("--grid", GRID, int),
                                ("--lambda", "auto", str)):
        options[flag[2:]] = kind(args.pop(args.index(flag)+1)) if flag in args else default
        if flag in args:
            args.remove(flag)
    options["solve"] = "--solve" in args
    args = [a for a in args if a != "--solve"]
    options["ns"] = int(args[0]) if args else 33
    if len(args) > 1 or options["ns"] < 5:
        raise SystemExit("Usage: python benchmarks/project_integer_vmex.py [NS>=5] [--case NAME] "
                         "[--mpol M] [--ntor N] [--grid G] [--lambda auto|zero|full_flux] "
                         "[--solve [--ftol F]]")
    return options


def main(argv=None):
    opts = parse(sys.argv[1:] if argv is None else argv)
    CASE, NS = opts["case"], opts["ns"]
    case = cases()[CASE]
    inp = vmex.VmecInput.from_file(ROOT / "inputs" / f"input.{CASE}_iota")
    override = {k: opts[k] for k in ("mpol", "ntor") if opts[k] is not None}
    inp = with_resolution(inp, NS, **override)
    start = perf_counter()
    state, info = project(inp, case, NS, grid=opts["grid"], lambda_mode=opts["lambda"])
    wout, rows = surface_rows(inp, case, state, NS)
    source = Path(vmex.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    tag = "".join(f"_{k[0]}{v}" for k, v in override.items())
    tag += "" if opts["lambda"] == "auto" else f"_lambda_{opts['lambda']}"
    out = ROOT / f"results/projection/{CASE}_vmex_ns{NS}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    max_B = max(r["B_relative_surface_l2"] for r in rows)
    status = "passed" if max_B < 1e-5 else "failed"
    (out / "projection.json").write_text(json.dumps(dict(schema=2, evidence="analytic_projection",
        status=status, vmex_commit=commit, case=CASE, ns=NS, mpol=int(inp.mpol), ntor=int(inp.ntor),
        angular_fit_grid=opts["grid"], no_nonlinear_solve=True, flux_map="label_at_s",
        seed=info["seed"], lambda_grid=info["lambda_grid"], lambda_certificates=info["lambda_certificates"],
        field_path="projected state -> WOUT surface field", elapsed_seconds=perf_counter()-start,
        max_B_relative_surface_l2=max_B, rows=rows), indent=2, allow_nan=False)+"\n")
    np.savez_compressed(out / "seed.npz", **{name: np.asarray(getattr(state, name)) for name in
        ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    if opts["solve"]:
        from vmex.core.errors import VmecConvergenceError
        from vmex.core.solver import solve
        try:  # tcon0 = 0: no spectral-condensation force at the exact root
            result = solve(inp, resolution_from_input(inp, ns=NS), initial_state=state, ftol=opts["ftol"],
                           max_iterations=30000, mode="jit", tcon0=0.0)
        except VmecConvergenceError as error:
            raise SystemExit(f"seeded solve unconverged: fsq {error.fsq} after {error.iteration}")
        vmex.write_wout(out / f"wout_{CASE}_seeded_ns{NS}.nc", vmex.wout_from_result(inp, result))
        print({"seeded_solve": dict(iterations=int(result.iterations), fsq=[float(result.fsqr),
               float(result.fsqz), float(result.fsql)])})
    print({"status": status, "seed": info["seed"], "max_B_relative_surface_l2": max_B, "rows": rows})
    if status != "passed":
        print("Projection does not meet the initial surface B target")


if __name__ == "__main__":
    main()
