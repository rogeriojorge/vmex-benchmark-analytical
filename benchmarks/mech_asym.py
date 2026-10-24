"""Stage 6, asymmetric plateau: where the iteration-1 residual of an asymmetric exact state lives and what it depends on.

Every configuration is one funct3d pass at the exact (hot-restarted) seed with the constraint force off
(``tcon0 = 0``; ``--default`` also records the default constraint), split into

* the stellarator-even channels (R_cos, Z_sin, L_sin) and the odd channels (R_sin, Z_cos, L_cos): for symmetric
  data run through the full (``LASYM``) basis the odd channels must vanish to roundoff, which separates a
  code-path defect from an effect of the asymmetric data;
* modes m, radial shells (``rz_by_js``), and the single mode carrying the largest share.

Scans: NS, tilt ``chi`` (0 = symmetric data in the full basis), poloidal resolution ``mpol`` and the angular grid
``ntheta`` at fixed radial grid, so that a plateau that is flat in NS can be assigned to the radial grid, the
angular grid, the data, or the operator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech_core as M  # noqa: E402
import mech_decks as D  # noqa: E402

ROOT = M.ROOT
EVEN = ("R_cos", "Z_sin", "L_sin")
ODD = ("R_sin", "Z_cos", "L_cos")


def channel_report(force, rt, res, cache):
    """fsq-normalised sums of squares per channel, per m (even/odd), top modes."""
    a, w = M.block_arrays(force, rt)
    m = np.asarray(rt.modes.m)
    n = np.asarray(rt.modes.n)
    ns = a["R_cos"].shape[0]
    r1 = float(res.residuals.fsqr)/max(float(res.residuals.gcr2), 1e-300)
    rl = float(cache.fnormL)
    out = {"fsqr": float(res.residuals.fsqr), "fsqz": float(res.residuals.fsqz), "fsql": float(res.residuals.fsql)}
    tot = {}
    top = []
    for k, arr in a.items():
        jsmax = ns-1 if k[0] in "RZ" else ns
        e = w*arr[:jsmax]**2
        scale = r1 if k[0] in "RZ" else rl
        tot[k] = float(e.sum()*scale)
        out[f"{k}_by_m"] = [float(e[:, m == q].sum()*scale) for q in range(int(m.max())+1)]
        out[f"{k}_by_js"] = [float(x) for x in e.sum(1)*scale]
        for q in range(e.shape[1]):
            top.append((float(e[:, q].sum()*scale), k, int(m[q]), int(n[q])))
    out["channel_sum"] = tot
    out["even_sum"] = float(sum(tot[k] for k in EVEN if k[0] in "RZ"))
    out["odd_sum"] = float(sum(tot[k] for k in ODD if k[0] in "RZ"))
    out["top_modes"] = [dict(value=v, channel=k, m=mm, n=nn) for v, k, mm, nn in sorted(top, reverse=True)[:6]]
    return out


def one(label, case, deck, ns, mpol=None, ntor=None, ntheta=None, default=False, polar=False):
    t = time.perf_counter()
    changes = {"ntheta": ntheta} if ntheta else {}
    rec = {"label": label, "case": case.name, "ns": ns, "mpol": mpol, "ntor": ntor, "ntheta": ntheta, "polar": polar}
    for tcon0 in ((0.0, 1.0) if default else (0.0,)):
        inp, rt, seed, info = M.exact_seed(case, ns, tcon0, deck=deck, mpol=mpol, ntor=ntor, polar=polar, **changes)
        force, res, cache = M.scaled_force(seed, rt, 1.0)
        rec[f"tcon0_{tcon0:g}"] = channel_report(force, rt, res, cache)
        rec["mpol_used"], rec["ntheta_used"] = int(inp.mpol), int(rt.resolution.ntheta)
    rec["seconds"] = time.perf_counter()-t
    e = rec["tcon0_0"]
    print(f"{label:34s} ns={ns:3d} fsqr {e['fsqr']:.2e} fsqz {e['fsqz']:.2e} fsql {e['fsql']:.2e} "
          f"even {e['even_sum']:.2e} odd {e['odd_sum']:.2e} top {e['top_modes'][0]['channel']} m{e['top_modes'][0]['m']} "
          f"({e['top_modes'][0]['value']:.2e})", flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT/"results/mechanism/asym_residual.json")
    p.add_argument("--scratch", type=Path, default=Path("/tmp/mech_decks"))
    p.add_argument("--default", action="store_true", help="also record the default constraint")
    p.add_argument("--only", nargs="*", help="run only these labels")
    p.add_argument("--named", nargs="*", help="committed cases of cases.json (their committed decks) instead of the Solov'ev scans")
    p.add_argument("--ns", type=int, nargs="+", default=[17, 33], help="radial grids for --named")
    p.add_argument("--polar", action="store_true", help="re-parametrise the exact seed into the m = 1 polar gauge (all configurations)")
    a = p.parse_args(argv)
    if a.named:
        from analytic import cases
        rows = [one(f"{name} ns{ns}"+(" polar" if a.polar else ""), cases()[name], None, ns, default=a.default, polar=a.polar) for name in a.named for ns in a.ns]
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1, allow_nan=False)+"\n")
        return
    configs = []
    sym_full = D.variant("solovev_asymmetric", "solovev_symmetric_full_basis", chi=0.0)
    cases_chi = {c: D.variant("solovev_asymmetric", f"solovev_chi{c:g}", chi=c) for c in (0.05, 0.1, 0.2)}
    base = D.variant("solovev_asymmetric", "solovev_asymmetric", chi=0.3)
    decks = {}

    def deck(case):
        if case.name not in decks:
            decks[case.name], _ = D.write_deck(case, a.scratch/f"input.{case.name}_iota")
        return decks[case.name]
    for ns in (17, 33, 65, 129):
        configs.append((f"asym chi0.3 ns{ns}", base, ns, {}))
    for ns in (17, 33, 65):
        configs.append((f"symdata full-basis ns{ns}", sym_full, ns, {}))
    for c, case in cases_chi.items():
        configs.append((f"chi{c:g} ns33", case, 33, {}))
    for mpol in (9, 17, 21):
        configs.append((f"asym mpol{mpol} ns33", base, 33, dict(mpol=mpol)))
    for ntheta in (24, 48, 64):
        configs.append((f"asym ntheta{ntheta} ns33", base, 33, dict(ntheta=ntheta)))
    rows = []
    for label, case, ns, kw in configs:
        if a.only and label not in a.only:
            continue
        rows.append(one(label+(" polar" if a.polar else ""), case, deck(case), ns, default=a.default, polar=a.polar, **kw))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
