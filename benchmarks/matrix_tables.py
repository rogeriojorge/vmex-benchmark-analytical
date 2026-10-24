"""Aggregate the Stage 4 matrix: one row per run, projection floors, ratios, work-precision.

Reads ``results/matrix/runs/*/{receipt.json, volume_score_v1.json, stdout.log}`` and the projection
floors in ``results/projection_certificate/<case>/ns*_m*_n*_g64_auto_mesh.json`` (the same mesh-aligned rule,
whole volume).  Writes ``results/matrix/summary.json`` and ``results/matrix/TABLES.md``.

Protocol labels: cold (single stage, deck initial guess), continued (NS ladder), exact-seed
(projected exact state; ``_tcon0`` marks the spectral-condensation force switched off).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"cold": "cold", "ladder": "continued", "seed": "exact-seed", "restart": "param-restart"}
GATES = {"B": 1e-5, "J": 1e-3, "force": 1e-3}
ITER = re.compile(r"^\s*(\d+)\s+[0-9.]+E[+-]\d+\s+[0-9.]+E[+-]\d+\s+[0-9.]+E[+-]\d+")


STAGE = re.compile(r"^\s*NS =\s+(\d+)\s")


def log_iterations(path):
    """(iterations of the last stage, total over stages) from the stage banners and ITER tables, or (None, None)."""
    if not Path(path).exists():
        return None, None
    stages = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if STAGE.match(line):
            stages.append(0)
            continue
        m = ITER.match(line)
        if m and stages:
            stages[-1] = max(stages[-1], int(m.group(1)))
    stages = [n for n in stages if n]
    return (stages[-1], sum(stages)) if stages else (None, None)


def floor_for(case, ns, mpol, ntor, root=ROOT):
    path = root/"results/projection_certificate"/case/f"ns{ns}_m{mpol}_n{ntor}_g64_auto_mesh.json"
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    a = r["scores"]["m4"]["all"]
    u = r.get("uncertainty", {}).get("all", {})
    return {"B": a["B"], "J": a["J"], "force": a["force"], "seed": r["seed"],
            "uncertainty": {k: u[k]["uncertainty"] for k in u} if u else None, "vmex_commit": r["vmex_commit"]}


def gates_passed(B, J, F):
    return "".join(k for k, v, g in (("B", B, GATES["B"]), ("J", J, GATES["J"]), ("F", F, GATES["force"]))
                   if v is not None and v <= g) or "-"


def row_for(run_dir, root=ROOT):
    rec = json.loads((run_dir/"receipt.json").read_text())
    score_path = run_dir/"volume_score_v1.json"
    sc = json.loads(score_path.read_text()) if score_path.exists() else None
    last_iter, total_iter = log_iterations(run_dir/"stdout.log")
    row = {"run_id": rec["run_id"], "case": rec["case"], "code": rec["code"], "protocol": rec["protocol"],
           "tag": rec["run_id"].split("-")[1][len(rec["protocol"]):], "ns": rec["ns"], "status": rec["status"],
           "ftol": rec["ftol"], "fsq_final": rec.get("fsq_final"),
           "fsq_max": max(rec["fsq_final"]) if rec.get("fsq_final") else None,
           "converged": rec.get("fsq_leq_ftol"), "ier_flag": rec.get("ier_flag"),
           "iterations_final_stage": rec.get("iterations_final_stage", last_iter), "iterations_total": total_iter,
           "wall_seconds": rec.get("wall_seconds"), "process_seconds": rec.get("process_seconds"),
           "seed_seconds": (rec.get("seed") or {}).get("seconds"), "load1": [rec.get("load1_start"), rec.get("load1_end")],
           "host": rec.get("host"), "omp_threads": rec.get("omp_threads"), "error": rec.get("error"),
           "fsq_first_iteration": rec.get("fsq_first_iteration")}
    if sc and "headline" in sc:
        h = sc["headline"]
        row.update(mpol=sc["mpol"], ntor=sc["ntor"], B=h["B"], J=h["J"], force=h["force"],
                   uncertainty=h.get("uncertainty"), invalid_volume_fraction=h["invalid_volume_fraction"],
                   B_regions={r: sc["scores"][h["level"]][r]["B"] for r in ("axis", "bulk", "edge")},
                   gates=gates_passed(h["B"], h["J"], h["force"]))
        floor = floor_for(rec["case"], rec["ns"], sc["mpol"], sc["ntor"], root)
        row["floor"] = floor
        if floor:
            row["ratio"] = {k: (row[k]/floor[k] if row[k] is not None and floor[k] else None) for k in ("B", "J", "force")}
        s = sc.get("surface")
        if s:
            row["surface"] = {"flux_label_error_max": s["flux_label_error_max"], "axis_offset_over_minor": s["axis_offset_over_minor"],
                              "iota_max_abs_error": s["iota_max_abs_error"], "B_surface_s": s["B_relative_surface"]["value"]}
    elif sc:
        row["native_error"] = sc.get("native_error")
    return row


def fmt(x, spec="%.1e"):
    return "-" if x is None else spec % x


def matrix_table(rows):
    lines = ["| case | code | protocol | NS | conv | fsq max | iters (last/total) | wall s | B | J | force | gates | floor B | B/floor | invalid vol | flux lbl | axis/a |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        it = f"{r['iterations_final_stage'] or '-'}/{r['iterations_total'] or '-'}"
        conv = {True: "yes", False: "no", None: "?"}[r["converged"]]
        s = r.get("surface") or {}
        lines.append("| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["case"], r["code"], LABEL[r["protocol"]]+r["tag"], r["ns"], conv, fmt(r["fsq_max"]), it, fmt(r["wall_seconds"], "%.1f"),
            fmt(r.get("B")), fmt(r.get("J")), fmt(r.get("force")), r.get("gates", "-"),
            fmt((r.get("floor") or {}).get("B")), fmt((r.get("ratio") or {}).get("B"), "%.0f"),
            fmt(r.get("invalid_volume_fraction")), fmt(s.get("flux_label_error_max")), fmt(s.get("axis_offset_over_minor"))))
    return "\n".join(lines)


def work_precision(rows, targets=(1e-3, 1e-4, 1e-5)):
    """Cheapest cold/continued run per code and case reaching each whole-volume B target.

    A run that exhausted its iteration cap counts (its B is what it is); its fsq is shown so the reader sees
    it was not converged.
    """
    lines = ["| case | code | B target | cheapest run | wall s | iterations (total) | B | fsq max |", "|---|---|---|---|---|---|---|---|"]
    for case in sorted({r["case"] for r in rows}):
        for code in ("vmex", "vmec2000"):
            pool = [r for r in rows if r["case"] == case and r["code"] == code and r["protocol"] in ("cold", "ladder")
                    and r.get("B") is not None and r["wall_seconds"] is not None and not r["tag"]]
            for t in targets:
                ok = sorted((r for r in pool if r["B"] <= t), key=lambda r: r["wall_seconds"])
                if ok:
                    b = ok[0]
                    lines.append(f"| {case} | {code} | {t:.0e} | {LABEL[b['protocol']]} NS{b['ns']} | {b['wall_seconds']:.1f} | "
                                 f"{b['iterations_total'] or '-'} | {b['B']:.2e} | {b['fsq_max']:.1e}{'' if b['converged'] else ' (not converged)'} |")
                else:
                    lines.append(f"| {case} | {code} | {t:.0e} | none reached | - | - | {fmt(min((r['B'] for r in pool), default=None))} (best) | - |")
    return "\n".join(lines)


def seed_residual_table(rows):
    """Force residual (max of fsqr, fsqz, fsql) of the projected exact state at iteration 1, per seeded run."""
    lines = ["| case | NS | default constraint | tcon0 = 0 | lambda = 0 (default constraint) |", "|---|---|---|---|---|"]
    by = {}
    for r in rows:
        if r["protocol"] == "seed" and r.get("fsq_first_iteration"):
            by[(r["case"], r["ns"], r["tag"])] = max(r["fsq_first_iteration"])
    for case, ns in sorted({(c, n) for c, n, _ in by}):
        lines.append("| %s | %d | %s | %s | %s |" % (case, ns, fmt(by.get((case, ns, ""))), fmt(by.get((case, ns, "_tcon0"))),
                                                     fmt(by.get((case, ns, "_lambda0")))))
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=ROOT/"results/matrix/runs")
    p.add_argument("--out", type=Path, default=ROOT/"results/matrix")
    a = p.parse_args(argv)
    rows = [row_for(d) for d in sorted(a.runs.glob("*/receipt.json")) for d in [d.parent]]
    order = {"cold": 0, "ladder": 1, "seed": 2, "restart": 3}
    rows.sort(key=lambda r: (r["case"], r["code"], order[r["protocol"]], r["tag"], r["ns"]))
    (a.out/"summary.json").write_text(json.dumps({"schema": 1, "rows": rows}, indent=1, allow_nan=False)+"\n")
    text = ["## Recovery matrix", "", matrix_table(rows), "", "## Work-precision (cheapest converged run per B target)", "",
            work_precision(rows), "", "## Residual of the projected exact state at iteration 1 (max of fsqr, fsqz, fsql)", "",
            seed_residual_table(rows), ""]
    (a.out/"TABLES.md").write_text("\n".join(text))
    print(f"{len(rows)} runs aggregated")


if __name__ == "__main__":
    main()
