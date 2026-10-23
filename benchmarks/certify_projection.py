"""Certify the exact-field projection of a sheared case: reference, lambda, state and quadrature.

Four independent layers, each reported separately:

* ``reference``: the exact chart's own volume quadrature against the closed-form volume, mean
  field energy and force balance of ``analytic.shear_averages`` (reference uncertainty);
* ``lambda``: B rebuilt from the full-flux lambda and the chart tangents,
  ``B = [(T - P lam_phi) x_theta + P (1 + lam_theta) x_phi]/J``, against the exact B on a held-out
  shifted grid that the lambda was not fitted on;
* ``state``: the projected VMEX state's continuous interior field (B, J = curl B/mu0, and the
  force J x B - grad p against the exact grad p) at exact-chart volume points with exact Jacobian
  weights, for a ladder of quadrature rules;
* ``uncertainty``: the spread of every metric across the quadrature ladder (finer rule, shifted
  angular grid, independent radial panel rule), compared with a tenth of the gate.

Volume points are (label, theta, t) of the exact chart; the weight is the absolute determinant
of the chart's automatic derivative, so no solver geometry enters the measure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from analytic import ROOT, cases, differential_fields, flux, label_at_s, shear_chart, volume, shear_averages

GATES = {"B": 1e-5, "J": 1e-3, "force": 1e-3}  # relative L2 on the volume: B, J, |J x B - grad p|/|grad p|
LEVELS = {  # name: (radial nodes, theta points, t points, rule, angular shift, panels)
    "q1": (12, 25, 25, "gauss", 0.0, 1), "q2": (16, 33, 33, "gauss", 0.0, 1),
    "q3": (24, 49, 49, "gauss", 0.0, 1), "q4": (32, 65, 65, "gauss", 0.0, 1),
    "q4_shift": (32, 65, 65, "gauss", 0.5, 1), "q4_panels": (8, 65, 65, "gauss", 0.0, 4),
}
FINEST, PREVIOUS, SHIFTED, PANELS = "q4", "q3", "q4_shift", "q4_panels"
REGIONS = {"axis": (0.0, 0.02), "bulk": (0.02, 0.9), "edge": (0.9, 1.0)}


def radial_nodes(edge, nodes, rule, panels):
    """Gauss-Legendre in the analytic label, on ``panels`` equal panels of ``nodes`` points each."""
    x, w = leggauss(nodes)
    lo = edge*np.arange(panels)/panels
    width = edge/panels
    return (lo[:, None]+(x[None, :]+1)/2*width).ravel(), np.tile(w/2*width, panels)


def chart_jacobian(case, label, theta, t):
    """|det d x / d (label, theta, t)| of the exact sheared chart, by forward-mode AD."""
    eps, S, lam, _ = case.parameters

    def chart(q):
        return shear_chart(jnp.sqrt(2*q[0]), q[1], q[2], eps, S, lam, case.tau, xp=jnp)

    def one(q):
        return jnp.abs(jnp.linalg.det(jax.jacfwd(chart)(q)))
    return np.asarray(jax.jit(jax.vmap(one))(jnp.stack((label, theta, t), -1)))


def chunked(function, array, size=32768):
    return np.concatenate([np.asarray(function(array[i:i+size])) for i in range(0, len(array), size)])


def volume_rule(case, name):
    """Points, weights, exact fields and radial bookkeeping for one quadrature level."""
    if case.family != "sheared":
        raise ValueError("the chart-Jacobian volume rule is implemented for the sheared family")
    nr, nth, nt, rule, shift, panels = LEVELS[name]
    r, wr = radial_nodes(case.edge, nr, rule, panels)
    th = 2*np.pi*(np.arange(nth)+shift)/nth
    t = 2*np.pi*(np.arange(nt)+shift)/nt
    L, T, A = (a.ravel() for a in np.meshgrid(r, th, t, indexing="ij"))
    shape = (len(r), nth, nt)
    weights = chart_jacobian(case, jnp.asarray(L), jnp.asarray(T), jnp.asarray(A))
    weights = weights*np.broadcast_to(wr[:, None, None], shape).ravel()*(2*np.pi/nth)*(2*np.pi/nt)
    xyz = np.asarray(shear_chart(np.sqrt(2*L), T, A, *case.parameters[:-1], case.tau))
    B, curl, gp, _, _ = (np.concatenate(p) for p in zip(*[
        differential_fields(case, xyz[i:i+16384]) for i in range(0, len(xyz), 16384)]))
    edge_flux = float(flux(case, case.edge))
    s = np.broadcast_to((np.asarray([float(flux(case, x)) for x in r])/edge_flux)[:, None, None], shape).ravel()
    return dict(name=name, xyz=xyz, w=weights, B=B, J=curl, gp=gp, s=s)


def relative(w, a, b):
    return float(np.sqrt(np.sum(w*np.sum((a-b)**2, -1))/np.sum(w*np.sum(b*b, -1))))


def metrics(rule, B, J, valid=None):
    """Relative L2 errors over the whole volume and over axis/bulk/edge shells of the same rule.

    ``valid`` marks points the native field could evaluate; a point outside the projected boundary
    is excluded from the sums but reported (count and volume fraction), never silently dropped.
    """
    valid = np.ones(len(rule["w"]), bool) if valid is None else valid
    B, J = np.where(valid[:, None], B, rule["B"]), np.where(valid[:, None], J, rule["J"])

    def block(mask):
        w = np.where(mask & valid, rule["w"], 0.0)
        force = np.cross(J, B)-rule["gp"]
        return {"B": relative(w, B, rule["B"]), "J": relative(w, J, rule["J"]),
                "force": float(np.sqrt(np.sum(w*np.sum(force**2, -1))/np.sum(w*np.sum(rule["gp"]**2, -1)))),
                "B_max_abs_over_Bmax": float(np.max(np.where(mask[:, None], np.abs(B-rule["B"]), 0))
                                             / np.max(np.abs(rule["B"])))}
    out = {"all": block(np.ones(len(rule["w"]), bool))}
    for region, (lo, hi) in REGIONS.items():
        out[region] = block((rule["s"] >= lo) & (rule["s"] < hi))
    for region, mask in (("all", np.ones(len(valid), bool)),
                         *((r, (rule["s"] >= lo) & (rule["s"] < hi)) for r, (lo, hi) in REGIONS.items())):
        out[region]["outside_boundary_points"] = int(np.sum(mask & ~valid))
        out[region]["outside_boundary_volume_fraction"] = float(np.sum(rule["w"][mask & ~valid])/np.sum(rule["w"][mask]))
    return out


def reference_certificate(case, rules):
    """The exact chart quadrature against the closed forms; also the exact force residual."""
    expected = shear_averages(*case.parameters, case.tau)
    rows = {}
    for name, r in rules.items():
        V = float(np.sum(r["w"]))
        energy = float(np.sum(r["w"]*np.sum(r["B"]**2, -1))/V)
        force = np.cross(r["J"], r["B"])-r["gp"]
        rows[name] = {"volume_relative_error": V/float(volume(case))-1,
                      "mean_B2_relative_error": energy/expected["mean_B2"]-1,
                      "exact_force_over_gradp": float(np.sqrt(np.sum(r["w"]*np.sum(force**2, -1))
                                                              / np.sum(r["w"]*np.sum(r["gp"]**2, -1))))}
    return rows


def native_field(inp, state):
    import vmex
    return vmex.VmecInteriorField.from_state(inp, state)


def native_BJ(field, xyz):
    """Native B, J = curl B (mu0 = 1 units) and a validity mask (False: outside the projected boundary).

    An eager call that stalls inside the plasma raises in vmex rather than returning NaN, so a
    non-finite value here means the exact-chart point lies outside the projected last surface.
    """
    B = chunked(lambda p: field.B(jnp.asarray(p)), xyz)
    g = chunked(lambda p: field.gradB(jnp.asarray(p)), xyz)
    valid = np.isfinite(B).all(-1) & np.isfinite(g).all((-1, -2))
    J = np.stack((g[:, 2, 1]-g[:, 1, 2], g[:, 0, 2]-g[:, 2, 0], g[:, 1, 0]-g[:, 0, 1]), -1)
    return B, J, valid


def spectral_derivative(values, axis, period):
    n = values.shape[axis]
    wave = 2*np.pi*np.fft.fftfreq(n, d=period/n)
    shape = [1, 1]
    shape[axis] = n
    return np.fft.ifft(np.fft.fft(values, axis=axis)*(1j*wave.reshape(shape)), axis=axis).real


def lambda_reconstruction(case, s, n=65, n_check=97, shift=0.37, radial_step=6.25e-6):
    """Rebuild B from the full-flux lambda on a shifted, finer grid and compare with the exact B.

    lambda is recovered on an n x n grid and interpolated spectrally to the check grid, which is
    offset by ``shift`` cells from any fitted grid; the chart tangents there come from the exact
    surface independently.
    """
    from flux_lambda import exact_surface_lambda
    from analytic import field, iota, surface
    lam, certificate, _ = exact_surface_lambda(case, s, n)
    nfp = case.nfp
    label = float(label_at_s(case, s))
    th = 2*np.pi*(np.arange(n_check)[:, None]+shift)/n_check
    ph = 2*np.pi*(np.arange(n_check)[None, :]+shift)/(n_check*nfp)
    spectrum = np.fft.fft2(lam)/lam.size
    m = np.fft.fftfreq(n, d=1/n)
    kz = nfp*np.fft.fftfreq(n, d=1/n)

    def evaluate(coefficient):
        et = np.exp(1j*m[None, :]*th)                     # (check, n) over theta
        ep = np.exp(1j*kz[None, :]*ph[0][:, None])         # (check, n) over phi
        return (et @ coefficient @ ep.T).real
    lam_theta = evaluate(1j*m[:, None]*spectrum)
    lam_phi = evaluate(1j*kz[None, :]*spectrum)

    def position(value):
        return np.asarray(surface(case, float(label_at_s(case, value)), th, ph))
    xyz = position(s)
    R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
    shape = xyz.shape
    eR = np.broadcast_to(np.stack((np.cos(ph), np.sin(ph), np.zeros_like(ph)), -1), shape)
    ephi = np.broadcast_to(np.stack((-np.sin(ph), np.cos(ph), np.zeros_like(ph)), -1), shape)
    ez = np.zeros(shape)
    ez[..., 2] = 1
    xt = spectral_derivative(R, 0, 2*np.pi)[..., None]*eR+spectral_derivative(Z, 0, 2*np.pi)[..., None]*ez
    xp = (spectral_derivative(R, 1, 2*np.pi/nfp)[..., None]*eR + R[..., None]*ephi
          + spectral_derivative(Z, 1, 2*np.pi/nfp)[..., None]*ez)
    xs = (position(s+radial_step)-position(s-radial_step))/(2*radial_step) if s+radial_step <= 1 else \
        (3*xyz-4*position(s-radial_step)+position(s-2*radial_step))/(2*radial_step)
    jac = np.sum(xs*np.cross(xt, xp), -1)
    P, T = certificate["toroidal_signed"], certificate["poloidal_signed"]
    rebuilt = ((T-P*lam_phi)[..., None]*xt+(P*(1+lam_theta))[..., None]*xp)/jac[..., None]
    exact = np.asarray(field(case, xyz)[0])
    err = np.linalg.norm(rebuilt-exact)/np.linalg.norm(exact)
    return {"s": s, "grid": n, "check_grid": n_check, "check_shift_cells": shift,
            "B_rebuilt_relative_l2": float(err),
            "B_rebuilt_max_over_max": float(np.max(np.abs(rebuilt-exact))/np.max(np.abs(exact))),
            "lambda_max_abs": float(np.max(np.abs(lam))),
            "min_1_plus_lambda_theta": float(np.min(1+lam_theta)),
            "iota_check": float(T/P-iota(case, label)),
            "certificate": {k: v for k, v in certificate.items() if not isinstance(v, str)}}


def uncertainty(scores):
    """Per-metric quadrature uncertainty: max of ladder step, angular shift and independent radial rule."""
    out = {}
    for region in ("all", *REGIONS):
        out[region] = {}
        for key in ("B", "J", "force"):
            ref = scores[FINEST][region][key]
            parts = {"ladder_step": abs(scores[PREVIOUS][region][key]-ref),
                     "angular_shift": abs(scores[SHIFTED][region][key]-ref),
                     "radial_panels": abs(scores[PANELS][region][key]-ref)}
            out[region][key] = {**parts, "value": ref, "uncertainty": max(parts.values()),
                                "gate": GATES[key], "uncertainty_below_tenth_gate": max(parts.values()) < GATES[key]/10,
                                "value_below_gate": ref < GATES[key]}
    return out


def certify_state(case, inp, state, rules):
    field = native_field(inp, state)
    scores, times = {}, {}
    for name, rule in rules.items():
        start = perf_counter()
        B, J, valid = native_BJ(field, rule["xyz"])
        scores[name] = metrics(rule, B, J, valid)
        times[name] = perf_counter()-start
    return scores, times


def parse_config(text):
    ns, mpol, ntor, grid = (int(v) for v in text.split(":"))
    return ns, mpol, ntor, grid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="sheared_A")
    parser.add_argument("--config", action="append", default=[], metavar="NS:MPOL:NTOR:GRID",
                        help="projection resolution to certify (repeatable); mpol/ntor 0 keeps the deck's")
    parser.add_argument("--levels", nargs="*", default=list(LEVELS), choices=list(LEVELS))
    parser.add_argument("--lambda-mode", default="auto", choices=("auto", "zero", "full_flux"))
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    import vmex
    from project_integer_vmex import project, with_resolution
    case = cases()[args.case]
    out = args.out or ROOT/"results/projection_certificate"/args.case
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    rules = {name: volume_rule(case, name) for name in args.levels}
    print("reference rules built in %.1f s" % (perf_counter()-start), flush=True)
    reference = reference_certificate(case, rules)
    for name, row in reference.items():
        print(f"  {name}: V {row['volume_relative_error']:+.2e}  <B2> {row['mean_B2_relative_error']:+.2e}"
              f"  exact force/|grad p| {row['exact_force_over_gradp']:.2e}", flush=True)
    target = out/"reference.json"
    if not target.exists():
        recon = [lambda_reconstruction(case, s) for s in (0.05, 0.25, 0.5, 0.75, 0.95)]
        ladder = [lambda_reconstruction(case, s, n=n, n_check=129) for s in (0.25, 0.5, 0.95) for n in (33, 49, 65, 97)]
        target.write_text(json.dumps({"schema": 1, "evidence": "reference_quadrature_and_lambda_reconstruction",
            "case": args.case, "quadrature_vs_closed_form": reference, "lambda_reconstruction": recon,
            "lambda_grid_ladder": [{k: r[k] for k in ("s", "grid", "check_grid", "B_rebuilt_relative_l2")} for r in ladder],
            "levels": {k: list(v) for k, v in LEVELS.items()}}, indent=2, allow_nan=False)+"\n")
        print("lambda reconstruction B error:", [f"{r['B_rebuilt_relative_l2']:.1e}" for r in recon], flush=True)
    if args.reference_only:
        return
    source = Path(vmex.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    for text in args.config:
        ns, mpol, ntor, grid = parse_config(text)
        inp = with_resolution(vmex.VmecInput.from_file(ROOT/"inputs"/f"input.{args.case}_iota"),
                              ns, mpol or None, ntor or None)
        path = out/f"ns{ns}_m{int(inp.mpol)}_n{int(inp.ntor)}_g{grid}_{args.lambda_mode}.json"
        if path.exists():
            print(f"{path.name} exists; skipping", flush=True)
            continue
        t0 = perf_counter()
        state, info = project(inp, case, ns, grid=grid, lambda_mode=args.lambda_mode,
                              lambda_grid=grid+1-grid % 2)  # odd lambda grid (65 for the default 64)
        t1 = perf_counter()
        scores, times = certify_state(case, inp, state, rules)
        record = {"schema": 1, "evidence": "projection_certificate", "case": args.case, "vmex_commit": commit,
                  "ns": ns, "mpol": int(inp.mpol), "ntor": int(inp.ntor), "angular_fit_grid": grid,
                  "seed": info["seed"], "lambda_grid": info["lambda_grid"],
                  "lambda_certificates": info["lambda_certificates"], "gates": GATES,
                  "projection_seconds": t1-t0, "scoring_seconds": times, "scores": scores}
        if all(k in scores for k in (FINEST, PREVIOUS, SHIFTED, PANELS)):
            record["uncertainty"] = uncertainty(scores)
        path.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        a = scores[args.levels[-1]]["all"] if FINEST not in scores else scores[FINEST]["all"]
        print(f"{path.name}: B {a['B']:.2e}  J {a['J']:.2e}  force {a['force']:.2e}  "
              f"({t1-t0:.0f}s project, {sum(times.values()):.0f}s score)", flush=True)


if __name__ == "__main__":
    main()
