"""Force residual of an exact (projected) Lopez D state as VMEC sees it, and where it lives.

The state is the exact vacuum equilibrium projected onto the (mpol, ntor, ns) VMEC representation
(``lopez_recovery.vmex_state``): one funct3d pass, no update. VMEX's ``_evaluate`` is the VMEC2000
funct3d/residue chain; ``_force_pipeline`` is wrapped only to keep the scalxc-scaled force that feeds
``fsqr/fsqz/fsql`` (the quantity VMEC thresholds), which is then resolved by

* radial shell (js),
* poloidal number m,
* the gauge-invariant resonance distance d = n - c m (field-line phase m theta - n phi advances like
  (m c - n) phi along a field line of transform c, so d = 0 are the exactly resonant harmonics; d is
  unchanged by theta' = theta - k phi, n' = n - k m, c' = c - k),
* the largest single modes.

Fractions are of the sum of squares of the scaled blocks (the sum that fsqr + fsqz + fsql normalize).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def spectral_report(force, rt, c, top=12):
    """Distribution of sum(block^2) over shell, m, d and mode, from a scaled SpectralForce."""
    from vmex.core.solver import _force_to_state
    ns = int(np.asarray(force.force_R_cc).shape[0])
    st = _force_to_state(force, rt)
    m = np.asarray(rt.modes.m)
    n = np.asarray(rt.modes.n)
    weight = np.where(n == 0, 1.0, 2.0)   # signed packing halves cc/ss into (cc +- ss)/2: restore the block sum
    comp = {"R": (st.R_cos, st.R_sin), "Z": (st.Z_cos, st.Z_sin), "L": (st.L_cos, st.L_sin)}
    energy = {k: weight*(np.asarray(a)**2 + np.asarray(b)**2) for k, (a, b) in comp.items()}   # (ns, mnmax)
    # blocks keep js = 0..ns-1; the radial force rows that VMEC evolves exclude the boundary row
    total = sum(e.sum() for e in energy.values())
    tot_all = float(total)
    e_sum = sum(energy.values())
    d = n - c*m
    dmax = int(np.abs(d).max())
    by_d = np.array([e_sum[:, d == k].sum() for k in range(-dmax, dmax + 1)])/tot_all
    by_m = np.array([e_sum[:, m == k].sum() for k in range(m.max() + 1)])/tot_all
    by_js = e_sum.sum(1)/tot_all
    modes = e_sum.sum(0)/tot_all
    order = np.argsort(modes)[::-1][:top]
    nmodes = {int(k): int((d == k).sum()) for k in (-1, 0, 1)}
    uniform_d0 = (d == 0).sum()/len(d)
    resonant = (d == 0) & (m >= 1)
    return dict(
        total_block_sum=tot_all,
        by_species={k: float(v.sum()/tot_all) for k, v in energy.items()},
        fraction_d0=float(by_d[dmax]), fraction_abs_d_le1=float(by_d[dmax - 1:dmax + 2].sum()),
        uniform_mode_fraction_d0=float(uniform_d0), n_modes_d=nmodes,
        fraction_d0_m_ge_1=float(e_sum[:, resonant].sum()/tot_all),
        d_range=[-dmax, dmax], by_d=[float(x) for x in by_d],
        by_m=[float(x) for x in by_m], by_js=[float(x) for x in by_js],
        fraction_js_le_ns_over_4=float(by_js[: ns//4 + 1].sum()),
        fraction_n_signed_abs_gt_ntor_half=float(e_sum[:, np.abs(n) > np.abs(n).max()//2].sum()/tot_all),
        top_modes=[dict(m=int(m[k]), n=int(n[k]), d=int(d[k]), fraction=float(modes[k])) for k in order])


def residual_of_state(inp, state, ns, c, fsqz_previous=1.0, iteration=100):
    """(report dict) of one funct3d pass at ``state``; ``fsqz_previous`` >= 1e-6 keeps the constrained
    m=1 Z force (VMEC zeroes it once fsqz < 1e-6)."""
    import jax.numpy as jnp
    import vmex.core.solver as solver
    rt = solver.prepare_runtime(inp, solver.resolution_from_input(inp, ns=ns))
    captured = {}
    original = solver._force_pipeline

    def wrapped(*a, **k):
        out = original(*a, **k)
        captured["scaled"], captured["preconditioned"] = out[0], out[1]
        return out
    solver._force_pipeline = wrapped
    try:
        cache = solver._zero_cache(rt)
        t0 = time.perf_counter()
        res = solver._evaluate(state, cache, jnp.asarray(iteration), jnp.asarray(iteration),
                               jnp.asarray(float(fsqz_previous)), rt, collect_health=True)
        seconds = time.perf_counter() - t0
    finally:
        solver._force_pipeline = original
    r, p = res.residuals, res.pre
    out = dict(fsqr=float(r.fsqr), fsqz=float(r.fsqz), fsql=float(r.fsql),
               fsq_scaled_sum=float(r.fsqr + r.fsqz + r.fsql),
               fsqr1=float(p.fsqr1), fsqz1=float(p.fsqz1), fsql1=float(p.fsql1),
               wb=float(res.wb), wp=float(res.wp), r00=float(res.r00), z00=float(res.z00),
               jacobian_sign_changed=bool(res.jacobian_sign_changed), seconds=seconds,
               fsqz_previous=float(fsqz_previous))
    out["spectrum"] = spectral_report(captured["scaled"], rt, c)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mpol", type=int, default=10)
    p.add_argument("--ntor", type=int, default=72)
    p.add_argument("--ns", type=int, nargs="+", default=[17])
    p.add_argument("--kind", default="eta", help="eta | alpha | shift:dl:db")
    p.add_argument("--c", type=int, default=None, help="integer transform of the angle gauge (default: recorded)")
    p.add_argument("--ai", type=float, default=None, help="evaluate with constant iota = ai instead of c")
    p.add_argument("--closure", choices=("current", "iota"), default="iota")
    p.add_argument("--tcon0", type=float, default=None,
                   help="m=1/spectral-condensation constraint strength (deck default 1); 0 switches the constraint force off")
    p.add_argument("--gate", choices=("open", "closed"), default="open",
                   help="open: constrained m=1 Z force kept; closed: zeroed (VMEC below fsqz = 1e-6)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    from dataclasses import replace
    import lopez_recovery as lr
    rows = []
    for ns in a.ns:
        inp = lr.deck_for(a.mpol, a.ntor, ns, a.kind, 1 if a.closure == "current" else 0, directory=a.out.parent, c=a.c)
        c = lr.gauge_with(a.c)[2]
        if a.tcon0 is not None:
            inp = replace(inp, tcon0=a.tcon0)
        if a.ai is not None:
            ai = np.zeros(21)
            ai[0] = a.ai
            inp = replace(inp, ai=ai)
        t0 = time.perf_counter()
        state = lr.vmex_state(inp, ns, a.kind, c=a.c)
        rec = residual_of_state(inp, state, ns, c, fsqz_previous=1.0 if a.gate == "open" else 0.0)
        rec.update(ns=ns, mpol=a.mpol, ntor=a.ntor, kind=a.kind, c=c, iota_evaluated=a.ai if a.ai is not None else c,
                   closure=a.closure, gate=a.gate, tcon0=float(inp.tcon0), seed_seconds=time.perf_counter() - t0 - rec["seconds"])
        rows.append(rec)
        s = rec["spectrum"]
        print(f"ns={ns} {a.kind} c={c} iota={rec['iota_evaluated']} fsqr={rec['fsqr']:.3e} fsqz={rec['fsqz']:.3e} "
              f"fsql={rec['fsql']:.3e} d0={s['fraction_d0']:.3f} (uniform {s['uniform_mode_fraction_d0']:.4f}) "
              f"js<=ns/4 {s['fraction_js_le_ns_over_4']:.3f}", flush=True)
    a.out.write_text(json.dumps(rows, indent=1, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
