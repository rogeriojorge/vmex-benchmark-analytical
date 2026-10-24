"""Aggregate the iota-resonance runs (results/iotaF/solves/*): convergence behaviour and field/energy scores.

Convergence is read from the solver stdout table (columns ITER FSQR FSQZ FSQL RAX ZAX DELT WMHD):
``fsq_min`` is the smallest FSQR+FSQZ+FSQL printed, ``delt_final`` the final time step (a falling DELT is the
signature of repeated residual-growth restarts), ``converged`` the receipt flag.
"""
import json
from pathlib import Path
import re
import sys


def parse_log(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.match(r"\s*(\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+\S+\s+\S+\s+([0-9.]+E[+-]\d+)", line)
        if m:
            rows.append((int(m[1]), float(m[2]) + float(m[3]) + float(m[4]), float(m[5])))
    return rows


def collect(root):
    out = []
    for r in sorted(Path(root).glob("*/receipt.json")):
        d = r.parent
        rec = json.loads(r.read_text())
        row = dict(run=d.name, protocol=rec.get("protocol"), closure=rec.get("closure"), ns=rec.get("ns"), kind=rec.get("kind"),
                   gauge_c=rec.get("gauge_c"), ai=rec.get("ai"), tcon0=rec.get("ladder", {}).get("tcon0"),
                   status=rec.get("status"), converged=rec.get("converged"), fsq_final=rec.get("fsq_final"),
                   wall_seconds=rec.get("wall_seconds"), iterations=rec.get("iterations_final_stage"))
        log = d/"stdout.log"
        if not log.exists():
            log = Path(root).parents[2]/"logs"/f"{d.name}.log"
        if log.exists():
            rows = parse_log(log)
            if rows:
                row.update(fsq_min=min(x[1] for x in rows), fsq_last=rows[-1][1], delt_final=rows[-1][2], first_fsq=rows[0][1],
                           printed_rows=len(rows))
        for s in d.glob("lopez_score_v1.json"):
            sc = json.loads(s.read_text())
            v = sc["vacuum"]
            row.update(B_vol=sc["B_relative_volume_l2"], B_s05=sc["B_relative_surface"]["value"], axis_over_minor=sc["axis_offset_over_minor"],
                       flux_label=sc["flux_label_error_max"], signed_dW=v["energy_b2_integral"]/v["energy_target_circulation_phi"] - 1,
                       iota_err=sc["iota_max_abs_error"])
        out.append(row)
    return out


if __name__ == "__main__":
    rows = collect(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(rows, indent=1, allow_nan=False) + "\n")
    for r in rows:
        print("%-38s conv %-5s fsq_min %s fsq_last %s delt %s B_vol %s dW %s" % (
            r["run"], r["converged"], "%.1e" % r["fsq_min"] if "fsq_min" in r else "-", "%.1e" % r["fsq_last"] if "fsq_last" in r else "-",
            "%.2f" % r["delt_final"] if "delt_final" in r else "-", "%.2e" % r["B_vol"] if "B_vol" in r else "-",
            "%+.1e" % r["signed_dW"] if "signed_dW" in r else "-"))
