"""Stage 6: spectrum of the preconditioned force Jacobian at the exact seed (is the exact state a stable root of the VMEC flow?).

VMEC relaxes x' = P(x) F(x) (damped, preconditioned).  For F = -dW/dx (up to a positive diagonal scaling) an equilibrium is a
stable fixed point iff every eigenvalue of dG/dx, G = P F, has negative real part.  A positive real eigenvalue is a direction
along which the exact state is a saddle of the discrete energy: the flow leaves it exponentially (growth rate = the eigenvalue
in units of the iteration), whatever the solver tolerance.  This tool builds the dense Jacobian of G (frozen preconditioner
cache, exact AD, the same map ``solver._preconditioned_force_signed`` that the 2D preconditioner differentiates), removes
dofs/equations that are not evolved, and reports

* the number of eigenvalues with Re > 0 (and > 1e-8 of the largest |Re|), the largest real parts and their imaginary parts,
* for the leading unstable pair the share of the eigenvector in each channel, in m and in the radial third of the grid,
* the smallest |Re| eigenvalues of the stable set (near-null / gauge directions).

Dense geev: only for small systems (Solov'ev) or one-off on the office box (integer 3-D, NS 17: ~15k dofs).
"""
from __future__ import annotations

import argparse
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

ROOT = M.ROOT


def make_Gpre(seed, rt, cache, channels, fsqz_previous=1.0, iteration=100):
    import vmex.core.solver as S

    @jax.jit
    def Gpre(x):
        gc = S._preconditioned_force_signed(L.unpack(x, seed, channels), cache, rt, iteration=jnp.asarray(iteration+1),
                                            fsqz_previous=jnp.asarray(float(fsqz_previous)))
        return L.pack(gc, channels)
    return Gpre


def run(case, ns, baseline="vmec", tcon0=1.0, deck=None, mpol=None, ntor=None, chunk=32, out=None, polar=False, ai0=None):
    t0 = time.perf_counter()
    changes = {}
    if ai0 is not None:    # operator with another constant iota evaluated at the SAME exact state (resonance test)
        ai = np.zeros(21)
        ai[0] = ai0
        changes["ai"] = ai
    inp, rt, seed, _ = M.exact_seed(case, ns, tcon0, deck=deck, mpol=mpol, ntor=ntor, polar=polar, **changes)
    rt = M.with_baseline(rt, seed, baseline)
    cache, _ = M.fresh_cache(seed, rt)
    channels = L.CHANNELS_ALL if bool(inp.lasym) else L.CHANNELS_SYM
    m = np.asarray(rt.modes.m)
    mn = len(m)
    G = make_Gpre(seed, rt, cache, channels)
    x0 = L.pack(seed, channels)
    J = L.dense_jacobian(G, x0, chunk=chunk)
    t_jac = time.perf_counter()-t0
    mask = L.evolve_mask(ns, m, channels)
    scale = np.abs(J).max()
    cols = (np.abs(J).max(0) > 1e-12*scale) & mask
    rows = (np.abs(J).max(1) > 1e-12*scale) & mask
    keep = cols & rows
    if cols.sum() != rows.sum() or not np.array_equal(cols, rows):
        # structurally empty parity rows/cols differ only through rounding: use the intersection (square, same index set)
        pass
    Ja = J[np.ix_(keep, keep)]
    w, v = np.linalg.eig(Ja)
    order = np.argsort(-w.real)
    idx = np.where(keep)[0]
    rec = {"case": case, "ns": ns, "baseline": baseline, "tcon0": tcon0, "polar": polar, "ai0_override": ai0, "n_active": int(keep.sum()), "seconds_jacobian": t_jac,
           "max_abs_real": float(np.abs(w.real).max()), "n_positive_real": int((w.real > 0).sum()),
           "n_positive_real_above_1e-8": int((w.real > 1e-8*np.abs(w.real).max()).sum()),
           "largest_real": [[float(w[i].real), float(w[i].imag)] for i in order[:8]],
           "smallest_abs_real_stable": [float(x) for x in np.sort(np.abs(w.real[w.real < 0]))[:6]]}
    lead = []
    for k in order[:2]:
        vec = np.zeros(J.shape[0], complex)
        vec[idx] = v[:, k]
        e = np.abs(vec)**2
        e /= e.sum()
        ch = {c: float(e[i*ns*mn:(i+1)*ns*mn].sum()) for i, c in enumerate(channels)}
        em = np.zeros(m.max()+1)
        er = np.zeros(3)
        for i in range(len(channels)):
            blk = e[i*ns*mn:(i+1)*ns*mn].reshape(ns, mn)
            for q in range(m.max()+1):
                em[q] += blk[:, m == q].sum()
            for t in range(3):
                er[t] += blk[t*ns//3:(t+1)*ns//3 if t < 2 else ns].sum()
        lead.append({"eigenvalue": [float(w[k].real), float(w[k].imag)], "channel_share": ch, "m_share": [float(x) for x in em],
                     "radial_thirds_share": [float(x) for x in er]})
    rec["leading"] = lead
    rec["seconds"] = time.perf_counter()-t0
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, allow_nan=False)+"\n")
    print(case, f"ns={ns}", baseline, f"n={rec['n_active']}", f"Re>0: {rec['n_positive_real']} (above 1e-8 max: {rec['n_positive_real_above_1e-8']})",
          "largest", " ".join("%.2e%+.1ej" % tuple(x) for x in rec["largest_real"][:4]), "max|Re| %.2e" % rec["max_abs_real"],
          "%.0fs" % rec["seconds"], flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="solovev_symmetric")
    p.add_argument("--ns", type=int, nargs="+", default=[17])
    p.add_argument("--baseline", nargs="+", default=["vmec"])
    p.add_argument("--tcon0", type=float, nargs="+", default=[1.0])
    p.add_argument("--mpol", type=int, default=None)
    p.add_argument("--ntor", type=int, default=None)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--out", type=Path, default=ROOT/"results/mechanism/eigen")
    p.add_argument("--polar", action="store_true")
    p.add_argument("--ai0", type=float, default=None, help="evaluate the operator with constant iota = ai0 at the same exact state")
    a = p.parse_args(argv)
    for ns in a.ns:
        for b in a.baseline:
            for t in a.tcon0:
                run(a.case, ns, b, t, mpol=a.mpol, ntor=a.ntor, chunk=a.chunk, polar=a.polar, ai0=a.ai0,
                    out=a.out/f"{a.case}-ns{ns}-tcon{t:g}-{b.replace(':', '')}{'-polar' if a.polar else ''}{'-ai0_%g' % a.ai0 if a.ai0 is not None else ''}.json")


if __name__ == "__main__":
    main()
