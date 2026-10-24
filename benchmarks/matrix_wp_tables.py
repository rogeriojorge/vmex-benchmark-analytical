"""Work-precision tables from the repeated timing runs (``results/matrix/wp_runs``).

Every configuration is run several times; the WOUT of every repeat is hashed (VMEX: bitwise-identical repeats; VMEC2000, whose WOUT carries a timestamp: identical final residuals and
iterations; one repeat is scored, repeat 0).  VMEX repeat 0 starts from an empty compilation cache (its wall
time includes XLA compilation), later repeats reuse the persistent cache.  VMEC2000 has no compile step.
The box is shared: the 1-minute load average before and after each run is reported with the spread, and
a configuration whose repeats differ by more than 25 % is flagged as noisy.

Output: ``results/matrix/wp_summary.json`` and ``results/matrix/WP_TABLES.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from matrix_tables import ROOT, floor_for, log_iterations


def load_runs(runs_dir):
    out = []
    for path in sorted(Path(runs_dir).glob("*/receipt.json")):
        d = path.parent
        rec = json.loads(path.read_text())
        tag = rec["run_id"].split("-")[1][len(rec["protocol"]):]
        score_path = d/"volume_score_v1.json"
        score = json.loads(score_path.read_text()) if score_path.exists() else None
        last, total = log_iterations(d/"stdout.log")
        out.append({"id": rec["run_id"], "case": rec["case"], "code": rec["code"], "protocol": rec["protocol"], "ns": rec["ns"],
                    "tag": tag, "kind": "rep" if tag.startswith("_rep") else "ftol", "rep": int(tag[4:]) if tag.startswith("_rep") else None,
                    "started": rec.get("started_utc"), "ftol": rec["ftol"], "ladder_ftols": rec["ladder"]["ftol_array"], "wall": rec.get("wall_seconds"),
                    "process": rec.get("process_seconds"), "cpu": rec.get("cpu_seconds"), "iters_last": last, "iters_total": total,
                    "fsq_max": max(rec["fsq_final"]) if rec.get("fsq_final") else None, "wout_sha256": (rec.get("wout") or {}).get("sha256"), "fsq_final": rec.get("fsq_final"),
                    "load": [rec.get("load1_start"), rec.get("load1_end")], "omp": rec.get("omp_threads"),
                    "score": score["headline"] if score and "headline" in score else None,
                    "mpol": score["mpol"] if score else None, "ntor": score["ntor"] if score else None})
    return out


def groups(runs):
    g = {}
    for r in runs:
        key = (r["case"], r["code"], r["protocol"], r["ns"], r["kind"], r["ftol"] if r["kind"] == "ftol" else None)
        g.setdefault(key, []).append(r)
    return g


def summarise(runs):
    # the first VMEX run of each sweep group has an empty compilation cache; later ones reuse it
    first_sweep = {}
    for r in sorted((r for r in runs if r["kind"] == "ftol" and r["code"] == "vmex"), key=lambda r: r["started"]):
        first_sweep.setdefault((r["case"], r["protocol"], r["ns"]), r["id"])
    rows = []
    for (case, code, proto, ns, kind, sweep_ftol), rs in sorted(groups(runs).items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][4], kv[0][3], kv[0][5] or 0)):
        rs = sorted(rs, key=lambda r: r["rep"] or 0)
        first = rs[0]
        walls = [r["wall"] for r in rs if r["wall"] is not None]
        if kind == "ftol":
            cold = code == "vmex" and first_sweep[(case, proto, ns)] == first["id"]
            warm = [] if cold else walls
        else:
            cold = False
            warm = [r["wall"] for r in rs[1:] if r["wall"] is not None] if code == "vmex" else walls
        # VMEC2000 stamps its WOUT, so a byte hash differs between identical solves: compare residuals as well
        identical = (len({r["wout_sha256"] for r in rs}) == 1 if code == "vmex" else
                     len({(tuple(r["fsq_final"]), r["iters_total"]) for r in rs}) == 1)
        scored = next((r["score"] for r in rs if r["score"]), None)
        floor = floor_for(case, ns, first["mpol"], first["ntor"]) if first["mpol"] else None
        row = {"case": case, "code": code, "protocol": proto, "ns": ns, "kind": kind, "ftol": first["ftol"], "n": len(rs),
               "cold_cache_sweep_point": cold,
               "wall_first": first["wall"], "wall_median": statistics.median(walls) if walls else None,
               "wall_min": min(walls) if walls else None, "wall_max": max(walls) if walls else None,
               "wall_warm_median": statistics.median(warm) if warm else None,
               "noisy": bool(walls and (max(walls)-min(walls)) > 0.25*statistics.median(walls)),
               "iters_last": first["iters_last"], "iters_total": first["iters_total"], "fsq_max": first["fsq_max"],
               "wout_identical_across_repeats": identical, "load1_range": [min(x for r in rs for x in r["load"] if x is not None),
                                                                           max(x for r in rs for x in r["load"] if x is not None)],
               "B": scored and scored["B"], "J": scored and scored["J"], "force": scored and scored["force"],
               "floor_B": floor and floor["B"]}
        rows.append(row)
    return rows


def fmt(x, spec="%.2e"):
    return "-" if x is None else spec % x


def timing_table(rows):
    lines = ["| case | code | protocol | NS | ftol | repeats | wall first s | warm median s | min-max s | iters (total) | fsq | B | load1 | note |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["kind"] != "rep":
            continue
        note = ("noisy; " if r["noisy"] else "") + ("" if r["wout_identical_across_repeats"] else "repeats differ; ")
        lines.append("| %s | %s | %s | %d | %.0e | %d | %s | %s | %s-%s | %s | %s | %s | %s-%s | %s |" % (
            r["case"], r["code"], r["protocol"], r["ns"], r["ftol"], r["n"], fmt(r["wall_first"], "%.1f"), fmt(r["wall_warm_median"], "%.1f"),
            fmt(r["wall_min"], "%.1f"), fmt(r["wall_max"], "%.1f"), r["iters_total"] or "-", fmt(r["fsq_max"], "%.1e"), fmt(r["B"]),
            fmt(r["load1_range"][0], "%.0f"), fmt(r["load1_range"][1], "%.0f"), note.strip("; ")))
    return "\n".join(lines)


def sweep_table(rows):
    lines = ["| case | code | NS | protocol | ftol (all stages) | wall s | iters (total) | fsq reached | B | B/floor |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["kind"] != "ftol":
            continue
        ratio = r["B"]/r["floor_B"] if r["B"] and r["floor_B"] else None
        lines.append("| %s | %s | %d | %s | %.0e | %s | %s | %s | %s | %s |" % (
            r["case"], r["code"], r["ns"], r["protocol"], r["ftol"], fmt(r["wall_first"], "%.1f")+(" (cold cache)" if r["cold_cache_sweep_point"] else ""), r["iters_total"] or "-",
            fmt(r["fsq_max"], "%.1e"), fmt(r["B"]), fmt(ratio, "%.0f")))
    return "\n".join(lines)


def matched_table(rows, targets=(1e-2, 1e-3, 1e-4, 1e-5)):
    """Cheapest configuration reaching each B target, per case and code (warm wall, cold wall in parentheses for VMEX)."""
    lines = ["| case | code | B target | cheapest configuration | warm wall s (first) | iterations | B | fsq |", "|---|---|---|---|---|---|---|---|"]
    for case in sorted({r["case"] for r in rows}):
        for code in ("vmex", "vmec2000"):
            pool = [r for r in rows if r["case"] == case and r["code"] == code and r["B"] is not None and r["wall_warm_median"] is not None]
            for t in targets:
                ok = sorted((r for r in pool if r["B"] <= t), key=lambda r: r["wall_warm_median"])
                if ok:
                    b = ok[0]
                    what = f"{b['protocol']} NS{b['ns']} ftol {b['ftol']:.0e}" + (" (sweep)" if b["kind"] == "ftol" else "")
                    lines.append(f"| {case} | {code} | {t:.0e} | {what} | {b['wall_warm_median']:.1f} ({b['wall_first']:.1f}) | {b['iters_total'] or '-'} | {b['B']:.2e} | {b['fsq_max']:.1e} |")
                else:
                    best = min((r["B"] for r in pool), default=None)
                    lines.append(f"| {case} | {code} | {t:.0e} | none reached (best B {fmt(best)}) | - | - | - | - |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=ROOT/"results/matrix/wp_runs")
    p.add_argument("--out", type=Path, default=ROOT/"results/matrix")
    a = p.parse_args(argv)
    rows = summarise(load_runs(a.runs))
    (a.out/"wp_summary.json").write_text(json.dumps({"schema": 1, "rows": rows}, indent=1, allow_nan=False)+"\n")
    text = ["## Timing: repeated runs per configuration", "", timing_table(rows), "", "## Tolerance sweep (every ladder stage at the swept ftol)", "",
            sweep_table(rows), "", "## Cheapest configuration reaching each whole-volume B target", "", matched_table(rows), ""]
    (a.out/"WP_TABLES.md").write_text("\n".join(text))
    print(f"{len(rows)} configurations")


if __name__ == "__main__":
    main()
