"""Print the radial and angular projection ladders from stored ``projection_certificate`` records."""
import json
from pathlib import Path
import sys

from analytic import ROOT


def load(case):
    rows = [json.loads(p.read_text()) for p in sorted((ROOT/"results/projection_certificate"/case).glob("ns*.json"))]
    return sorted(rows, key=lambda r: (r["ns"], r["mpol"], r["ntor"], r["angular_fit_grid"]))


def table(rows, region="all"):
    from certify_projection import FINEST, GATES
    lines = [f"{'NS':>4} {'mpol':>4} {'ntor':>4} {'fit':>4} {'seed':<22} {'B':>9} {'J':>9} {'force':>9}  "
             f"{'u(B)':>8} {'u(J)':>8} {'u(F)':>8}  gates met / u<gate/10"]
    for r in rows:
        s = r["scores"][FINEST][region]
        u = r.get("uncertainty", {}).get(region, {})
        met = "".join("Y" if s[k] < GATES[k] else "n" for k in ("B", "J", "force"))
        unc = "".join("Y" if u[k]["uncertainty_below_tenth_gate"] else "n" for k in ("B", "J", "force")) if u else "--"
        us = [f"{u[k]['uncertainty']:8.1e}" if u else f"{'--':>8}" for k in ("B", "J", "force")]
        out = s.get("outside_boundary_points", 0)
        lines.append(f"{r['ns']:4d} {r['mpol']:4d} {r['ntor']:4d} {r['angular_fit_grid']:4d} {r['seed']:<22} "
                     f"{s['B']:9.2e} {s['J']:9.2e} {s['force']:9.2e}  {us[0]} {us[1]} {us[2]}  {met} / {unc}"
                     + (f"  [{out} pts outside boundary, vol frac {s['outside_boundary_volume_fraction']:.1e}]" if out else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "sheared_A"
    rows = load(case)
    for region in ("all", "axis", "bulk", "edge"):
        print(f"\n{case}: {region}\n{table(rows, region)}")
