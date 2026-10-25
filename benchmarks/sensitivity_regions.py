"""Region-resolved response errors from saved arrays: where in the volume the response error lives.

Reads ``<case>_ns<NS>.npz`` written by sensitivity_vmex.py (JVP, exact, FD arrays and the exact-chart
s of every point) and reports, for the cloud and the volume rule, the metric of the whole set and of
the axis (s < S_AXIS) and bulk (s >= S_AXIS) subsets, plus the shell-RMS profile in s.  The split is a
post-hoc diagnostic (chosen after the NS 33/65 records showed the error localised at small s); the
pre-registered gates apply to the full sets.  Sub-sets are never dropped from the full-set metric.

Usage: python sensitivity_regions.py [--dir results/sensitivity] [--case integer_axisymmetric]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analytic import ROOT, cases
import derivative_map as dm

S_AXIS = 0.1
ROOT_GATE = 1e-9  # |F| (preconditioned and raw) at the returned state; a derivative of a non-root is not accepted
SHELLS = (0.0, 0.002, 0.02, 0.07, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0)


def metric(diff, exact, name, weights, edge):
    rms = dm.weighted_rms(diff, weights)
    if name == "delta":
        return edge*rms
    return rms/dm.weighted_rms(exact, weights)


def regions(npz, name, obs, source, edge):
    d = np.load(npz)
    coords = d["cloud_coords_s_theta_phi" if obs == "cloud" else f"{obs}_coords_s_theta_phi"]
    s = coords[:, 0]
    w = None if obs == "cloud" else d[f"{obs}_weights"]
    approx, exact = d[f"{source}_{obs}_{name}"] if source == "jvp" else d[source], d[f"exact_{obs}_{name}"]
    out = {}
    for label, mask in (("all", np.ones(len(s), bool)), ("axis", s < S_AXIS), ("bulk", s >= S_AXIS)):
        if mask.any():
            out[label] = metric((approx-exact)[mask], exact[mask], name, None if w is None else w[mask], edge)
    shells = []
    for lo, hi in zip(SHELLS[:-1], SHELLS[1:]):
        mask = (s >= lo) & (s < hi+(hi == 1.0)*1e-12)
        if mask.any():
            shells.append({"s_lo": lo, "s_hi": hi, "points": int(mask.sum()),
                           "rms_error_T": dm.weighted_rms((approx-exact)[mask], None if w is None else w[mask])})
    out["shells"] = shells
    return out


def analyse(directory, case_filter=None):
    rows = []
    for path in sorted(Path(directory).glob("*_ns*.json")):
        rec = json.loads(path.read_text())
        if case_filter and rec["case"] not in case_filter:
            continue
        npz = path.with_suffix(".npz")
        edge = cases()[rec["case"]].edge
        ok = rec["root"]["raw"] < ROOT_GATE and rec["root"]["preconditioned"] < ROOT_GATE
        entry = {"case": rec["case"], "root_accepted": bool(ok), "ns": rec["ns"], "tag": path.stem.split(f"_ns{rec['ns']}")[1], "params": {}}
        for name in rec["params"]:
            entry["params"][name] = {obs: regions(npz, name, obs, "jvp", edge) for obs in rec["params"][name]["jvp"]}
        rows.append(entry)
    return sorted(rows, key=lambda r: (r["case"], r["tag"], r["ns"]))


def markdown(rows):
    out = ["| case | NS | param | set | all | axis (s<%.2g) | bulk (s>=%.2g) | gate (all) | gate (bulk) |" % (S_AXIS, S_AXIS),
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for name, obs in r["params"].items():
            kind = "delta" if name == "delta" else "c"
            thr = dm.GATES["null_scaled"] if kind == "delta" else dm.GATES["nonzero_relative"]
            for o, v in obs.items():
                axis = f"{v['axis']:.3e}" if "axis" in v else "-"
                verdict = lambda x: "REJECTED (root)" if not r["root_accepted"] else ("PASS" if x <= thr else "FAIL")
                out.append(f"| {r['case']}{r['tag']} | {r['ns']} | {name} | {o} | {v['all']:.3e} | {axis} | {v['bulk']:.3e} | "
                           f"{verdict(v['all'])} | {verdict(v['bulk'])} |")
    out += ["", "Shell RMS of the response error (tesla per unit parameter, volume rule):", "",
            "| case | NS | param | " + " | ".join(f"s {lo:g}-{hi:g}" for lo, hi in zip(SHELLS[:-1], SHELLS[1:])) + " |",
            "|---|---|---|" + "---|"*(len(SHELLS)-1)]
    for r in rows:
        for name, obs in r["params"].items():
            cells = {(s["s_lo"], s["s_hi"]): s["rms_error_T"] for s in obs["volume"]["shells"]}
            out.append(f"| {r['case']}{r['tag']} | {r['ns']} | {name} | " + " | ".join(
                f"{cells[(lo, hi)]:.2e}" if (lo, hi) in cells else "-" for lo, hi in zip(SHELLS[:-1], SHELLS[1:])) + " |")
    return "\n".join(out)+"\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT/"results/sensitivity")
    ap.add_argument("--case", nargs="*")
    args = ap.parse_args()
    rows = analyse(args.dir, args.case)
    (args.dir/"regions.json").write_text(json.dumps(rows, indent=1)+"\n")
    text = markdown(rows)
    (args.dir/"REGIONS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
