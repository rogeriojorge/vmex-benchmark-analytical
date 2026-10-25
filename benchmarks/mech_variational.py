"""Stage 6: replace the force on the first interior shell's m = 1 dofs by the exact gradient of the discrete energy.

``mech_gradcheck`` shows the VMEX/VMEC force is +(NS-1) dW/dx (W = wb - wp, state independent) for every R/Z dof except those that touch the
axis extrapolation: the m = 1 dofs of the first interior shell, where it is neither proportional to dW/dx nor of the
same sign.  Here the force on exactly those dofs (``rows='m1_js1'``; wider sets ``m1_js<=K`` and ``all_js1`` are
options) is replaced by scale*dW/dx (AD of the same discrete energy), every other row keeps the VMEX kernel, and the
nonlinear system is solved by dense Newton from the exact seed (constraint at the exact baseline, so F_con(x*) = 0):

    G_var(x) = G(x) with rows R replaced by scale * grad W(x)[R].

The result is compared with the VMEX root of the unmodified system started at the same seed (``mech_solve`` record):
if the first-shell inconsistency were responsible for the near-axis layer, the layer (J error in cells 0-3) and the
error would drop; if only the energy's own first-cell truncation matters, they would not.

The scale (F = scale dW/dx, scale = NS - 1 in these units) is not assumed: it is fitted (median of F_MHD/(dW/dx) over the
R/Z dofs of shells >= 3 at a perturbed state, where the gradient is large) and recorded.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech_core as M  # noqa: E402
import mech_linear as L  # noqa: E402
import mech_solve as MS  # noqa: E402

ROOT = M.ROOT


def make_W(seed, rt, channels):
    import vmex.core.solver as S

    def W(x):
        st = L.unpack(x, seed, channels)
        s = rt.setup.s_full
        setup = rt.setup
        _, geometry = S._geometry(st, rt)
        jacobian = S.half_mesh_jacobian(geometry, s=s)
        metrics = S.metric_elements(geometry, s=s)
        fields = S.magnetic_fields(
            geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig, s=s, phips=setup.phips,
            phipf=setup.phipf, chips=setup.chips, signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
            ncurr=setup.ncurr, enclosed_current=setup.icurv)
        e = S.energies_and_force_norms(jacobian=jacobian, metrics=metrics, fields=fields, trig=rt.trig, s=s, signgs=setup.signgs)
        return e.wb-e.wp
    return W


def row_mask(rows, ns, m, channels):
    """Bool mask over the packed equations: R/Z rows replaced by the variational gradient."""
    js = np.arange(ns)[:, None]
    mm = m[None, :]
    if rows == "m1_js1":
        pick = (js == 1) & (mm == 1)
    elif rows.startswith("m1_js<="):
        pick = (js >= 1) & (js <= int(rows.split("=")[1])) & (mm == 1)
    elif rows == "all_js1":
        pick = (js == 1) & (mm >= 1)
    elif rows == "none":
        pick = (js == 1) & (mm == 99)
    else:
        raise ValueError(rows)
    pick = np.broadcast_to(pick, (ns, len(m)))
    return np.concatenate([(pick if c[0] in "RZ" else np.zeros_like(pick)).ravel() for c in channels])


def newton(G, x0, active_mask, iters=12, tol=1e-13, log=print):
    x = x0
    hist = []
    for k in range(iters):
        g = np.asarray(G(x))
        J = L.dense_jacobian(G, x)
        scale = np.abs(J).max()
        cols = (np.abs(J).max(0) > 1e-12*scale) & active_mask
        rows_ = (np.abs(J).max(1) > 1e-12*scale) & active_mask
        Ja = J[np.ix_(rows_, cols)]
        if Ja.shape[0] != Ja.shape[1]:
            raise RuntimeError(f"active system not square {Ja.shape}")
        step = np.zeros(x.shape[0])
        step[cols] = np.linalg.solve(Ja, -g[rows_])
        gnorm = float(np.linalg.norm(g[rows_]))
        hist.append(dict(iteration=k, G_norm=gnorm, step_norm=float(np.linalg.norm(step))))
        log(f"  newton {k}: |G| {gnorm:.3e} |step| {np.linalg.norm(step):.3e}", flush=True)
        x = x+jnp.asarray(step)
        if gnorm < tol*max(1.0, hist[0]["G_norm"]) and k > 0:
            break
    return x, hist


def run(case, ns, rows, baseline="exact", tcon0=1.0, out=None, score=True, fit_scale=True):
    import vmex
    t0 = time.perf_counter()
    inp, rt, seed, _ = M.exact_seed(case, ns, tcon0)
    rt = M.with_baseline(rt, seed, baseline)
    cache, _ = M.fresh_cache(seed, rt)
    channels = L.CHANNELS_SYM
    m = np.asarray(rt.modes.m)
    mn = len(m)
    G = L.make_G(seed, rt, cache, channels)
    W = make_W(seed, rt, channels)
    gradW = jax.grad(W)
    x0 = L.pack(seed, channels)
    # fit the scale -F/(dW/dx) on well-resolved R/Z dofs (shells >= 3) of a perturbed state
    rng = np.random.default_rng(11)
    xp = x0+jnp.asarray(1e-3*rng.standard_normal(x0.shape)*L.evolve_mask(ns, m, channels))
    G_off = L.make_G(seed, rt, replace(cache, tcon=cache.tcon*0.0), channels)   # pure MHD force: the constraint has no energy
    gp, Fp = np.asarray(gradW(xp)), np.asarray(G_off(xp))
    js = np.tile(np.repeat(np.arange(ns), mn), len(channels))
    isrz = np.concatenate([np.full(ns*mn, c[0] in "RZ") for c in channels])
    sel = isrz & (js >= 3) & (js <= ns-3) & L.evolve_mask(ns, m, channels) & (np.abs(gp) > 1e-3*np.abs(gp).max())
    ratios = Fp[sel]/gp[sel]      # F = +scale dW/dx (scale = NS - 1 in these units)
    scale = float(np.median(ratios)) if fit_scale else float(ns-1)
    spread = float(np.max(np.abs(ratios-scale))/abs(scale))
    pick = row_mask(rows, ns, m, channels)

    def Gvar(x):
        g = G(x)
        gw = gradW(x)
        return jnp.where(jnp.asarray(pick), scale*gw, g)
    rec = {"case": case, "ns": ns, "rows": rows, "baseline": baseline, "tcon0": tcon0, "fitted_scale": scale, "fit_spread": spread,
           "n_replaced": int(pick.sum()),
           "G_at_seed": float(np.linalg.norm(np.asarray(G(x0)))), "Gvar_at_seed": float(np.linalg.norm(np.asarray(Gvar(x0))))}
    active = L.evolve_mask(ns, m, channels)
    xs, hist = newton(Gvar, x0, active)
    rec["newton"] = hist
    rec["root_distance"] = float(np.linalg.norm(np.asarray(xs-x0)))
    if score:
        rule = MS.rule_for(case, ns)
        st = L.unpack(xs, seed, channels)
        field = vmex.VmecInteriorField.from_state(inp, st)
        sc = MS.ss.score_state(field, {"m4": rule})
        rec["scores"] = {k: sc["scores"]["m4"]["all"][k] for k in ("B", "J", "force")}
        rec["scores"].update({f"{r}_{k}": sc["scores"]["m4"][r][k] for r in ("axis", "bulk") for k in ("B", "J")})
        rec["profile"] = MS.cell_profiles(field, rule, ns)
    rec["seconds"] = time.perf_counter()-t0
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out/f"{case}-ns{ns}-{rows.replace('<=', 'le')}-{baseline}.json").write_text(json.dumps(rec, indent=1, allow_nan=False)+"\n")
    sc = rec.get("scores", {})
    print(case, ns, rows, "scale %.4f spread %.1e" % (scale, spread), "B %.2e J %.2e" % (sc.get("B", np.nan), sc.get("J", np.nan)),
          "peak J cell %.1e" % max(rec["profile"]["J_local"]) if score else "", flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="solovev_symmetric")
    p.add_argument("--ns", type=int, nargs="+", default=[17])
    p.add_argument("--rows", nargs="+", default=["none", "m1_js1"])
    p.add_argument("--baseline", default="exact")
    p.add_argument("--out", type=Path, default=ROOT/"results/mechanism/variational")
    a = p.parse_args(argv)
    for ns in a.ns:
        for r in a.rows:
            run(a.case, ns, r, a.baseline, out=a.out)


if __name__ == "__main__":
    main()
