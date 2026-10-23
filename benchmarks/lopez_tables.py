"""Collect Lopez D run records (receipt + lopez_score) into one summary and markdown tables.

Layout read: <root>/solves/<run>/ (lopez_solve.py) and <root>/solves_vmec2000/<run>/, <root>/runs/<run>/
(cross_code_run.py), each with receipt.json, lopez_score_v1.json and the solver log stdout.log.
The projection tables come from projection_oracle_*.json and projection_vmex_eta.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
LINE = re.compile(r"^\s+(\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)\s+([0-9.]+E[+-]\d+)")


def fsq_history(log):
    rows = []
    for line in Path(log).read_text(errors="replace").splitlines():
        m = LINE.match(line)
        if m:
            rows.append((int(m.group(1)), *(float(m.group(i)) for i in (2, 3, 4))))
    return rows


def run_row(d):
    receipt = json.loads((d/"receipt.json").read_text())
    score_path = d/"lopez_score_v1.json"
    if not score_path.exists():
        return dict(run=d.name, status=receipt.get("status"), note="no score")
    s = json.loads(score_path.read_text())
    v = s["vacuum"]
    hist = fsq_history(d/"stdout.log") if (d/"stdout.log").exists() else []
    peak = [max(h[1:]) for h in hist]
    code = receipt.get("code", "vmex")
    protocol = receipt.get("protocol", "cold" if "-ns" in d.name else "?")
    ladder = receipt.get("ladder", {})
    fsq = receipt.get("fsq_final") or s["fsq"]
    return dict(
        run=d.name, code=code, protocol=protocol, closure=receipt.get("closure") or ("current" if "current" in d.name else "iota"),
        ns=s["ns"], ftol=(ladder.get("ftol_array") or [None])[-1], niter=ladder.get("niter_per_stage"),
        converged=receipt.get("converged"), fsq_final_max=max(fsq) if all(x is not None for x in fsq) else None,
        fsq_logged_min=min(peak) if peak else None, iterations_logged=hist[-1][0] if hist else None,
        B_s05=s["B_relative_surface"]["value"], B_axis=s["B_relative_near_axis"]["value"],
        B_vol=s["B_relative_volume_l2"], flux=s["flux_label_error_max"], iota=s["iota_max_abs_error"],
        axis_over_minor=s["axis_offset_over_minor"], curl_s05=v["curl_over_B_over_a_rms_at_report"],
        curl_outer=v["curl_over_B_over_a_rms_outer_s_ge_0.25"], Iencl=v["enclosed_current_over_aB_max"],
        circ=v["circulation_rel_error_max"], W=v["energy_rel_error"], Phi=v["toroidal_flux_rel_error"],
        wall_s=receipt.get("wall_seconds"), wout_sha256=(receipt.get("wout") or {}).get("sha256"))


def collect(root):
    rows = []
    for sub in ("solves", "solves_vmec2000", "runs"):
        for d in sorted((root/sub).glob("*")) if (root/sub).exists() else []:
            if (d/"receipt.json").exists():
                rows.append(run_row(d))
    return rows


def fmt(v, kind="g"):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, str)):
        return str(v)
    return ("%.2e" % v) if kind == "e" or abs(v) < 1e-2 or abs(v) >= 1e3 else ("%.3f" % v)


def markdown(rows):
    cols = [("run", "run"), ("code", "code"), ("protocol", "protocol"), ("closure", "closure"), ("ns", "NS"),
            ("converged", "conv"), ("fsq_final_max", "fsq final"), ("fsq_logged_min", "fsq min"),
            ("B_s05", "B s=.5"), ("B_axis", "B axis"), ("B_vol", "B vol"), ("flux", "flux"), ("iota", "iota"),
            ("axis_over_minor", "axis/a"), ("curl_s05", "curl s=.5"), ("Iencl", "I encl"), ("circ", "circ"),
            ("W", "W"), ("wall_s", "wall s")]
    out = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|"*len(cols)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(k)) for k, _ in cols) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/"results/lopez"
    rows = collect(root)
    (root/"summary.json").write_text(json.dumps(rows, indent=1, allow_nan=False) + "\n")
    print(markdown(rows))
