"""Stage 6: the VMEX force operator evaluated at WOUT states written by another code (or by VMEX itself).

A solver that reports fsq = 1e-14 has a root of ITS discrete operator.  Feeding its WOUT state to VMEX's
``funct3d`` pass (same deck, same tcon0, preconditioner cache refreshed at that state) gives the residual that VMEX's
operator assigns to it: ~1e-14 if the two operators agree on that root, larger by the operator difference otherwise.
A VMEX WOUT is the control (it must return its own final fsq).  The residual is also split by channel (even/odd
stellarator parity), mode m and shell so a difference can be assigned to a term.

``python mech_crosscode.py --case solovev_asymmetric --wouts DIR`` expects ``DIR/wout_<case>-cold-ns<NS>-<code>.nc``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech_core as M  # noqa: E402
import mech_asym as A  # noqa: E402

ROOT = M.ROOT


def state_distance_wout(a, b, rt):
    out = {}
    for k in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
        x, y = np.asarray(getattr(a, k)), np.asarray(getattr(b, k))
        out[k] = float(np.linalg.norm(x-y)/max(np.linalg.norm(y), 1e-300))
    return out


def residual_at_wout(case, ns, wout_path, tcon0=1.0):
    import vmex
    from vmex.core.restart import state_from_wout
    inp = M.case_deck(case, ns, tcon0)
    inp_rt, rt, seed, _ = M.exact_seed(case, ns, tcon0)
    w = vmex.read_wout(wout_path)
    st = state_from_wout(w, inp=inp)
    rt = M.with_baseline(rt, st, "vmec")
    cache, res = M.fresh_cache(st, rt)
    force, res2, cache = M.scaled_force(st, rt, 1.0, cache=cache)
    rec = {"wout": Path(wout_path).name, "sha256": hashlib.sha256(Path(wout_path).read_bytes()).hexdigest(),
           "wout_fsq": [float(w.fsqr), float(w.fsqz), float(w.fsql)], "tcon0": tcon0,
           "vmex_fsq": [float(res2.residuals.fsqr), float(res2.residuals.fsqz), float(res2.residuals.fsql)],
           "channels": A.channel_report(force, rt, res2, cache), "distance_to_exact_seed": state_distance_wout(st, seed, rt)}
    off, res_off, _ = M.scaled_force(st, rt, 0.0, cache=cache)
    rec["vmex_fsq_tcon0_off"] = [float(res_off.residuals.fsqr), float(res_off.residuals.fsqz), float(res_off.residuals.fsql)]
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--wouts", type=Path, required=True)
    p.add_argument("--ns", type=int, nargs="+", default=[17, 33, 65, 129])
    p.add_argument("--codes", nargs="+", default=["vmec2000", "vmex"])
    p.add_argument("--protocol", default="cold")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    rows = []
    for ns in a.ns:
        for code in a.codes:
            path = a.wouts/f"wout_{a.case}-{a.protocol}-ns{ns}-{code}.nc"
            if not path.exists():
                continue
            r = residual_at_wout(a.case, ns, path)
            r.update(case=a.case, ns=ns, code=code, protocol=a.protocol)
            rows.append(r)
            ch = r["channels"]
            print(f"{a.case} ns={ns:3d} {code:9s} wout fsq {max(r['wout_fsq']):.1e} -> VMEX fsq r/z/l "
                  f"{r['vmex_fsq'][0]:.2e} {r['vmex_fsq'][1]:.2e} {r['vmex_fsq'][2]:.2e}  even {ch['even_sum']:.2e} odd {ch['odd_sum']:.2e}", flush=True)
    out = a.out or ROOT/f"results/mechanism/crosscode_{a.case}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
