"""Collect cross-code runs (VMEX, VMEC2000, VMEC++) and DESC records into one table.

Reads only saved receipts and score files; never re-runs or edits them.
Writes ``results/cross_code/summary.json``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    rows = []
    for receipt_path in sorted((ROOT/"results/cross_code/runs").glob("*/receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        score_path = receipt_path.parent/"exact_score.json"
        case, ns = receipt["run_id"].rsplit("-ns", 1)
        ns = int(ns.split("-")[0])
        row = {"code": receipt["code"], "case": case.removesuffix("_iota"), "ns": ns,
               "status": receipt["status"], "wall_seconds": receipt["wall_seconds"],
               "code_identity": receipt["code_identity"], "receipt_sha256": sha(receipt_path)}
        if score_path.exists():
            s = json.loads(score_path.read_text())
            converged = max(s["fsq"]) <= 1e-12
            routes = {r["j"]: r for r in s["routes"]}
            mid = routes[min(routes, key=lambda j: abs(routes[j]["s"]-0.5))]
            row.update(converged=converged, fsq_max=max(s["fsq"]), axis_offset_m=s["axis_offset_m"],
                       axis_offset_over_minor=s["axis_offset_over_minor"],
                       flux_label_error_max=s["flux_label_error_max"], B_wout_mid=s["B_relative_mid"],
                       B_wout_first_surface=s["B_relative_near_axis"], iota_error=s["iota_max_abs_error"],
                       J_wout_mid=mid.get("J_wout_relative_l2"), J_native_mid=mid.get("J_native_relative_l2"),
                       B_native_mid=mid.get("B_native_relative_l2"),
                       J_wout_first_surface=routes[min(routes)].get("J_wout_relative_l2"),
                       J_native_first_surface=routes[min(routes)].get("J_native_relative_l2"),
                       score_sha256=sha(score_path))
        else:
            row.update(converged=False)
        rows.append(row)
    desc_rows = []
    for rec in sorted((ROOT/"results/desc/general").glob("*/record.json")):
        r = json.loads(rec.read_text())
        v, a = r["solved"]["volume"]["gauss"], r["solved"]["rings"]["0.0001"]
        desc_rows.append({"code": "desc", "case": r["case"], "M": r["resolution"]["M"],
                          "solver_success": r["solver"]["success"], "iterations": r["solver"]["iterations"],
                          "B_volume": v["B_relative_l2"], "J_volume": v["J_relative_l2"],
                          "force_volume": v["force_over_gradp"], "B_axis": a["B_relative_l2"],
                          "J_axis": a["J_relative_l2"],
                          "B_mid": r["solved"]["rings"]["0.7"]["B_relative_l2"],
                          "J_mid": r["solved"]["rings"]["0.7"]["J_relative_l2"], "record_sha256": sha(rec)})
    for rec in sorted((ROOT/"results/desc/axis_volume").glob("integer_3d_base_L*.json")):
        r = json.loads(rec.read_text())
        v, a = r["volume"]["gauss"], r["rings"]["0.0001"]
        desc_rows.append({"code": "desc", "case": "integer_3d", "M": r["desc_resolution"]["M"],
                          "solver_success": True, "B_volume": v["B_relative_l2"], "J_volume": v["J_relative_l2"],
                          "force_volume": v["force_over_gradp"], "B_axis": a["B_relative_l2"],
                          "J_axis": a["J_relative_l2"], "flux_label_error_axis": a["flux_label_error_max"],
                          "record_sha256": sha(rec)})
    out = ROOT/"results/cross_code/summary.json"
    out.write_text(json.dumps({"schema": 1, "vmec_family": rows, "desc": desc_rows}, indent=2)+"\n")
    for r in rows:
        print(r["case"], r["ns"], r["code"], r["status"], r.get("converged"),
              "flux %.2e" % r["flux_label_error_max"] if "flux_label_error_max" in r else "")
    print(len(rows), "vmec-family rows,", len(desc_rows), "desc rows")


if __name__ == "__main__":
    main()
