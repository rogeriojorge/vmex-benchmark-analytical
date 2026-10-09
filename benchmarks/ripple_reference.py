"""Effective ripple of the exact sheared equilibria: real space versus NEO_JAX.

Reference: Nemov et al., Phys. Plasmas 6, 4622 (1999), eq. (29), evaluated on one
field line traced with the exact field.  |grad psi| and the geodesic curvature
k_G = (b x n).(b.grad)b come from the analytic B, psi and dB/dx.  psi is the
analytic pressure label; its normalization cancels in eq. (29), so it maps onto
NEO's toroidal flux with no factor.  Eq. (29) scales as (R0 B0)^2.  NEO (ref_swi=2)
takes B0 = Boozer B_00 of the surface, the Boozer-angle mean of B, which with the
Boozer Jacobian ~ 1/B^2 is <B^3>/<B^2> (field-line average with weight dl/B), and
R0 = Boozer R_00 of the first surface in the file (neo_jax.driver.compute_reference).
We store eps^{3/2}/R0^2 and multiply by NEO's rt0 when comparing.

    python benchmarks/ripple_reference.py reference   # real-space convergence ladder
    python benchmarks/ripple_reference.py neo [NS]    # VMEX -> booz_xform_jax -> NEO_JAX
"""
from dataclasses import replace
import json
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import ROOT, cases, field, flux, iota, label_at_s, surface

CASES = ("sheared_issan", "sheared_issan_tau", "sheared_issan_tau_mirror")
S_VALUES = (0.25, 0.5, 0.8)
LADDER = ((400, 1024, 400), (800, 1024, 400), (1600, 1024, 400), (1600, 2048, 400), (1600, 2048, 800))
OUT = ROOT / "results" / "ripple"


def trace(case, label, turns, steps):
    """RK4 in cylindrical phi; returns samples at uniform phi."""
    rhs = lambda x: (lambda B: B*(x[0]**2+x[1]**2)/(x[0]*B[1]-x[1]*B[0]))(field(case, x)[0])
    h = 2*np.pi/steps

    def step(x, _):
        k1 = rhs(x); k2 = rhs(x+h/2*k1); k3 = rhs(x+h/2*k2); k4 = rhs(x+h*k3)
        x = x+h/6*(k1+2*k2+2*k3+k4)
        return x, x
    x0 = jnp.asarray(surface(case, label, 0., 0.))
    xs = jax.jit(lambda x: jax.lax.scan(step, x, None, length=turns*steps)[1])(x0)
    return np.concatenate([np.asarray(x0)[None], np.asarray(xs)]), h


def local(case, xs):
    def one(x):
        B, psi = field(case, x)
        grad = jax.grad(lambda y: field(case, y)[1])(x)
        J = jax.jacfwd(lambda y: field(case, y)[0])(x)
        Bm = jnp.linalg.norm(B)
        b, n = B/Bm, grad/jnp.linalg.norm(grad)
        kG = jnp.dot(jnp.cross(b, n), J@b)/Bm
        dl = Bm*(x[0]**2+x[1]**2)/jnp.abs(x[0]*B[1]-x[1]*B[0])
        return jnp.stack((Bm, dl/Bm, jnp.linalg.norm(grad), kG, psi))
    f = jax.jit(jax.vmap(one))
    return np.concatenate([np.asarray(f(c)) for c in np.array_split(xs, max(1, len(xs)//65536))])


def sqrt_cell(a0, a1, h):
    """Exact integral over a cell of sqrt(max(a,0)) for linear a."""
    p = lambda a: np.maximum(a, 0.)**1.5
    d = a1-a0
    small = np.abs(d) < 1e-14
    return np.where(small, h*np.sqrt(np.maximum((a0+a1)/2, 0.)),
                    h*2/3*(p(a1)-p(a0))/np.where(small, 1., d))


def nemov(q, h, npitch):
    """eps_eff^{3/2}/R0^2 from eq. (29); wells touching the line ends are dropped."""
    B, w, gpsi, kG = (q[:, i] for i in range(4))
    cell = lambda v: (v[1:]+v[:-1])/2
    B0 = np.sum(cell(w*B**3))/np.sum(cell(w*B**2))
    beta, wc, gc, hc = B/B0, cell(w), cell(w*gpsi), cell(w*gpsi*kG)
    L, mean_grad = np.sum(wc)*h, np.sum(gc)*h
    lo, hi = beta.min(), beta.max()
    bp = lo+(hi-lo)*(np.arange(npitch)+0.5)/npitch
    total = 0.
    for b in bp:
        root = sqrt_cell(b-beta[:-1], b-beta[1:], h)
        inside = np.maximum(b-beta[:-1], b-beta[1:]) > 0
        well = np.cumsum(np.r_[True, inside[1:] != inside[:-1]])-1
        Hc = root*hc*(4*cell(B0/B)-1/b)/b
        Ic = root*wc/np.sqrt(b)
        H, I = (np.bincount(well, v) for v in (Hc, Ic))
        keep = np.bincount(well, inside) > 0
        keep[[0, -1]] = False
        total += np.sum(H[keep]**2/I[keep])
    return np.pi/(8*np.sqrt(2))*L/mean_grad**2*total*(hi-lo)/npitch


def reference():
    out = {}
    for name in CASES:
        case = cases()[name]
        for s in S_VALUES:
            label = float(label_at_s(case, s))
            rows, lines = [], {}
            for turns, steps, npitch in LADDER:
                t0 = perf_counter()
                if steps not in lines:
                    xs, h = trace(case, label, max(t for t, *_ in LADDER), steps)
                    lines[steps] = local(case, xs), h
                q, h = lines[steps]
                q = q[:turns*steps+1]
                rows.append(dict(turns=turns, steps=steps, npitch=npitch,
                                 eps32_over_R0sq=float(nemov(q, h, npitch)),
                                 label_drift=float(np.max(np.abs(q[:, 4]-label))/label),
                                 seconds=perf_counter()-t0))
            best = rows[-1]["eps32_over_R0sq"]
            spread = max(abs(r["eps32_over_R0sq"]-best) for r in rows[1:-1])  # length, step, pitch
            out[f"{name}/s={s}"] = dict(case=name, s=s, label=label, iota=float(iota(case, label)),
                                         eps32_over_R0sq=best, convergence_abs=spread, ladder=rows)
            print(name, s, best, spread/best, flush=True)
    return out


def neo(ns):
    import vmex
    from build_inputs import boundary_coefficients, fit_profiles
    from neo_jax import NeoConfig
    from neo_jax.driver import compute_reference
    from neo_jax.io import booz_xform_to_boozerdata
    from neo_jax.pipeline import booz_xform_from_vmec_wout, run_boozer_to_neo
    out = {}
    for name in CASES:
        case = cases()[name]
        rows, ntor, *_ = boundary_coefficients(case)
        polp, poli, *_ = fit_profiles(case)
        mu0 = 4e-7*np.pi
        lines = ["&INDATA", "LFREEB=F", f"LASYM={'T' if case.lasym else 'F'}", f"NFP={case.nfp}",
                 "MPOL=13", f"NTOR={ntor}", f"NS_ARRAY=17 {ns}", "FTOL_ARRAY=1e-12 1e-15",
                 "NITER_ARRAY=20000 60000", "NSTEP=200", "DELT=0.5", "GAMMA=0",
                 "PMASS_TYPE='power_series'", f"PHIEDGE={float(flux(case, case.edge)):.17e}",
                 "AM="+" ".join(f"{x/mu0:.17e}" for x in polp.coef), "NCURR=0",
                 "PIOTA_TYPE='power_series'", "AI="+" ".join(f"{x:.17e}" for x in poli.coef)]
        for m, n, rc, rs, zc, zs in rows:
            for key, value in (("RBC", rc), ("ZBS", zs), ("RBS", rs), ("ZBC", zc)):
                if (case.lasym or key in ("RBC", "ZBS")) and abs(value) > 2e-15:
                    lines.append(f"{key}({int(n)},{int(m)})={value:.17e}")
        OUT.mkdir(parents=True, exist_ok=True)
        deck = OUT / f"input.{name}_ns{ns}"
        deck.write_text("\n".join(lines+["/"])+"\n")
        inp = vmex.VmecInput.from_file(deck)
        t0, path = perf_counter(), OUT / f"wout_{name}_ns{ns}.nc"
        if not path.exists():
            vmex.write_wout(path, vmex.wout_from_result(inp, vmex.solve_multigrid(inp)))
        wout = vmex.read_wout(path)
        booz = booz_xform_from_vmec_wout(wout, mboz=48, nboz=48, surfaces=list(S_VALUES))
        config = NeoConfig(theta_n=200, phi_n=200, npart=150, acc_req=1e-3, nstep_max=5000,
                           nstep_min=1000, multra=2, rational_surface_policy="approximate")
        res = run_boozer_to_neo(booz, config=config, use_jax=True)
        rt0 = compute_reference(booz_xform_to_boozerdata(booz))["rt0"]
        for i, s in enumerate(S_VALUES):
            out[f"{name}/s={s}"] = dict(case=name, s=float(np.asarray(booz["s_b"])[i]), ns=ns,
                                         eps32=float(res[i].epsilon_effective), rt0=rt0,
                                         fsqr=float(wout.fsqr), seconds=perf_counter()-t0)
            print(name, s, out[f"{name}/s={s}"], flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "reference":
        key, result = "reference", reference()
    else:
        ns = int(sys.argv[2]) if len(sys.argv) > 2 else 65
        key, result = f"neo_vmex_ns{ns}", neo(ns)
    path = OUT / "ripple_reference.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**data, key: result}, indent=2, allow_nan=False)+"\n")
