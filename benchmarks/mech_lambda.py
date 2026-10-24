"""Stage 6, asymmetric plateau: is the lambda (cos part) of the exact seed the source of the flat Z_00 / R_00 force?

R, Z are held at the exact surfaces and lambda is re-solved by Newton from the seed (the lambda-channel equations of the VMEX
force at tcon0 = 0; dofs: L_cos, L_sin on shells >= 1).  Reported: how far lambda moves (relative to the seed, per parity
and m), the residual of every channel before and after, and the B error of the re-solved lambda against the exact field
(a lambda that zeroes the force but breaks B would be a different, wrong-field state).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech_asym as A  # noqa: E402
import mech_core as M  # noqa: E402
import mech_linear as L  # noqa: E402
import mech_solve as MS  # noqa: E402

ROOT = M.ROOT


def run(case, ns, out=None):
    import vmex
    inp, rt, seed, _ = M.exact_seed(case, ns, 0.0)
    rt = M.with_baseline(rt, seed, "vmec")
    cache, res0 = M.fresh_cache(seed, rt)
    channels = L.CHANNELS_ALL
    m = np.asarray(rt.modes.m)
    mn = len(m)
    G = L.make_G(seed, rt, cache, channels)
    x0 = L.pack(seed, channels)
    lam_idx = np.concatenate([np.full(ns*mn, c.startswith("L")) for c in channels])
    active = L.evolve_mask(ns, m, channels) & lam_idx

    def G_lam(xl):
        x = x0.at[jnp.asarray(np.where(lam_idx)[0])].set(xl)
        return G(x)[jnp.asarray(np.where(lam_idx)[0])]
    xl = x0[jnp.asarray(np.where(lam_idx)[0])]
    sub_active = active[lam_idx]
    for k in range(8):
        g = np.asarray(G_lam(xl))
        J = L.dense_jacobian(G_lam, xl)
        scale = np.abs(J).max()
        cols = (np.abs(J).max(0) > 1e-12*scale) & sub_active
        rows = (np.abs(J).max(1) > 1e-12*scale) & sub_active
        step = np.zeros(xl.shape[0])
        step[cols] = np.linalg.solve(J[np.ix_(rows, cols)], -g[rows])
        xl = xl+jnp.asarray(step)
        if np.linalg.norm(g[rows]) < 1e-13 and k > 0:
            break
    x1 = x0.at[jnp.asarray(np.where(lam_idx)[0])].set(xl)
    st1 = L.unpack(x1, seed, channels)
    out_rec = {"case": case, "ns": ns}
    for tag, st in (("seed_lambda", seed), ("resolved_lambda", st1)):
        force, res, c2 = M.scaled_force(st, rt, 1.0, cache=cache)
        out_rec[tag] = A.channel_report(force, rt, res, cache)
    d = {}
    for c in ("L_cos", "L_sin"):
        a, b = np.asarray(getattr(seed, c)), np.asarray(getattr(st1, c))
        d[c] = {"seed_norm": float(np.linalg.norm(a)), "change_norm": float(np.linalg.norm(b-a)),
                "change_by_m": [float(np.linalg.norm((b-a)[:, m == q])) for q in range(int(m.max())+1)]}
    out_rec["lambda_change"] = d
    rule = MS.rule_for(case, ns)
    for tag, st in (("seed_lambda", seed), ("resolved_lambda", st1)):
        sc = MS.ss.score_state(vmex.VmecInteriorField.from_state(inp, st), {"m4": rule})
        out_rec[f"{tag}_score"] = {k: sc["scores"]["m4"]["all"][k] for k in ("B", "J", "force")}
    f = lambda r: (r["fsqr"], r["fsqz"], r["fsql"])  # noqa: E731
    print(case, ns, "fsq(r,z,l) seed-lambda %.2e %.2e %.2e -> resolved %.2e %.2e %.2e" % (*f(out_rec["seed_lambda"]), *f(out_rec["resolved_lambda"])),
          "| lambda change L_cos %.2e (seed %.2e) L_sin %.2e (seed %.2e)" % (d["L_cos"]["change_norm"], d["L_cos"]["seed_norm"], d["L_sin"]["change_norm"], d["L_sin"]["seed_norm"]),
          "| B %.2e -> %.2e" % (out_rec["seed_lambda_score"]["B"], out_rec["resolved_lambda_score"]["B"]), flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(out_rec, indent=1, allow_nan=False)+"\n")
    return out_rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="solovev_asymmetric")
    p.add_argument("--ns", type=int, nargs="+", default=[17, 33])
    p.add_argument("--out", type=Path, default=ROOT/"results/mechanism/lambda")
    a = p.parse_args(argv)
    for ns in a.ns:
        run(a.case, ns, a.out/f"{a.case}-ns{ns}.json")


if __name__ == "__main__":
    main()
