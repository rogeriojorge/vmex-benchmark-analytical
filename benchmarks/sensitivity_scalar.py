"""Scalar volume observable from saved arrays: the volume mean of |B| over the fixed region s <= s_max.

The Eulerian derivative over a fixed physical region is d<|B|>/dp = <B . dB / |B|>, evaluated with the
same quadrature weights on the exact field, the VMEX tangent (JVP) and the reconverged-solve finite
differences.  Metrics follow the pre-registered gates: relative error for a nonzero parameter, and
(a*/B*) |d<|B|>/dp| for the null parameter delta (a* = delta0, B* = 1 T).  No solve is run.

Usage: python sensitivity_scalar.py [--dir results/sensitivity]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analytic import ROOT, cases
import derivative_map as dm


def mean_derivative(B, dB, w):
    absB = np.linalg.norm(B, axis=-1)
    return float(np.sum(w*np.sum(B*dB, -1)/absB)/np.sum(w))


def analyse(directory):
    rows = []
    for path in sorted(Path(directory).glob("*_ns*.json")):
        rec = json.loads(path.read_text())
        d = np.load(path.with_suffix(".npz"))
        edge = cases()[rec["case"]].edge
        tag = path.stem.split(f"_ns{rec['ns']}", 1)[1]
        root_ok = max(rec["root"].values()) < dm.ROOT_GATE
        for obs, (xyz_key, w_key, B_key) in {"volume": ("volume_xyz", "volume_weights", "B0_volume")}.items():
            w, B = d[w_key], d[B_key]
            for name in rec["params"]:
                exact = mean_derivative(d["exact_B_volume"], d[f"exact_volume_{name}"], w)
                jvp = mean_derivative(B, d[f"jvp_volume_{name}"], w)
                fds = {}
                for fd in rec["params"][name]["fd"]:
                    key = f"fd_volume_{name}_h{fd['relative_step']:g}"
                    if key in d:
                        fds[fd["relative_step"]] = mean_derivative(B, d[key], w)
                best_h = min(fds, key=lambda h: abs(fds[h]-jvp)) if fds else None
                row = {"case": rec["case"], "tag": tag, "ns": rec["ns"], "param": name, "root_accepted": bool(root_ok),
                       "exact": exact, "jvp": jvp, "fd_best_h": best_h,
                       "fd": None if best_h is None else fds[best_h]}
                for label in ("jvp", "fd"):
                    v = row[label]
                    if v is None:
                        continue
                    row[f"{label}_metric"] = (edge*abs(v-exact) if name == "delta" else abs(v-exact)/abs(exact))
                thr = dm.GATES["null_scaled"] if name == "delta" else dm.GATES["nonzero_relative"]
                row["gate_pass"] = bool(root_ok and row["jvp_metric"] <= thr)
                rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT/"results/sensitivity")
    args = ap.parse_args()
    rows = analyse(args.dir)
    (args.dir/"scalar_mean_absB.json").write_text(json.dumps(rows, indent=1)+"\n")
    lines = ["| case | NS | param | root | exact d<|B|>/dp | JVP | JVP metric | gate | FD (best h) | FD metric |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fd = "-" if r["fd"] is None else f"{r['fd']:.6e} ({r['fd_best_h']:g})"
        fdm = "-" if r["fd"] is None else f"{r['fd_metric']:.3e}"
        lines.append(f"| {r['case']}{r['tag']} | {r['ns']} | {r['param']} | {'ok' if r['root_accepted'] else 'REJECTED'} | "
                     f"{r['exact']:.6e} | {r['jvp']:.6e} | {r['jvp_metric']:.3e} | {'PASS' if r['gate_pass'] else 'FAIL'} | {fd} | {fdm} |")
    (args.dir/"SCALAR.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
