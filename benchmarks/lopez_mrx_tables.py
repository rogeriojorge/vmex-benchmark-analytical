"""Collect the MRX records on Lopez D into results/lopez_mrx/TABLES.md."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"results/lopez_mrx"


def load(pattern):
    return [(p, json.loads(p.read_text())) for p in sorted(RES.glob(pattern))]


def fmt(x, f="%.2e"):
    return "-" if x is None else f % x


def ladder_table():
    rows = ["| chart | p | splines (r, theta, zeta) | dofs | B vol L2 | B s=0.5 | B near axis | pointwise curl | energy err | circ err | enclosed I | one-turn defect | rotation dev | RSS GB | wall s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for pattern, chart in (("ladder/label_tw-1_p*.json", "label (twist 1)"), ("cyl/cyl_10_72_*.json", "VMEC-like (10,72)")):
        for p, r in sorted(load(pattern), key=lambda x: (x[1]["args"]["p"], x[1]["args"]["ns"])):
            a, s, c = r["args"], r["score"], r["closure"]
            rows.append("| %s | %d | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.1f | %.0f |" % (
                chart, a["p"], "x".join(map(str, a["ns"])), r["dofs_2form"], fmt(s["B_rel_volume_l2"]), fmt(s["B_rel_surface_l2"]),
                fmt(s["B_rel_near_axis_l2"]), fmt(s["pointwise_curl_over_B_over_a"]), fmt(s["energy_rel_error"]),
                fmt(s["circulation_rel_error_max"]), fmt(s["enclosed_current_over_aB_max"]), fmt(c["return_defect_max"]),
                fmt(c["rotation_dev_max"]), r["max_rss_gb"], r["total_s"]))
    return "\n".join(rows)


def truncation_table():
    rows = ["| deck (mpol, ntor) | boundary truncation error | splines | B vol L2 | B s=0.5 | one-turn defect | rotation dev |", "|---|---|---|---|---|---|---|"]
    for p, r in sorted(load("cyl/cyl_*_6_8_96.json"), key=lambda x: x[1]["args"]["ntor"]):
        a, s, c = r["args"], r["score"], r["closure"]
        rows.append("| (%d, %d) | %s | %s | %s | %s | %s | %s |" % (a["mpol"], a["ntor"], fmt(r["boundary_truncation_error"]),
                    "x".join(map(str, a["ns"])), fmt(s["B_rel_volume_l2"]), fmt(s["B_rel_surface_l2"]),
                    fmt(c["return_defect_max"]), fmt(c["rotation_dev_max"])))
    return "\n".join(rows)


def ham_table():
    rows = ["| resolution | (m, n) | eps | one-turn defect | iota - 1 range | O | X | O positions (u, v) | B shape change vs undeformed exact |", "|---|---|---|---|---|---|---|---|---|"]
    items = []
    for p, r in load("ham/*.json"):
        if p.name.startswith("isl_"):
            continue
        a = r["args"]
        items.append((tuple(a["ns"]), a["m"], a["n"], a["eps"], r))
    for ns, m, n, eps, r in sorted(items, key=lambda x: (x[0], x[1], x[2], x[3])):
        h = r["effective_hamiltonian"]
        resolved = r["one_turn"]["defect_max"] >= 1e-4      # below this the fixed points are numerical noise of the tracer / discretisation
        o = ", ".join("(%.2f, %.2f)" % (q["u"], q["v"]) for q in h["points"] if q["kind"] == "O") if resolved else "not resolved"
        rows.append("| %s | (%d, %d) | %g | %s | [%s, %s] | %s | %s | %s | %s |" % (
            "x".join(map(str, ns)), m, n, eps, fmt(r["one_turn"]["defect_max"]), fmt(r["one_turn"]["rotation_dev_min"]),
            fmt(r["one_turn"]["rotation_dev_max"]), h["n_O"] if resolved else "-", h["n_X"] if resolved else "-", o,
            fmt(r["score"].get("B_rel_volume_l2_scaled", r["score"]["B_rel_volume_l2"]))))
    return "\n".join(rows)


def islands_table():
    rows = ["| resolution | (m, n) | eps | chain at r | width | note |", "|---|---|---|---|---|---|"]
    for p, r in load("ham/isl_*.json"):
        a = r["args"]
        for c in r.get("islands", []):
            note = "ray window saturated (lower bound)" if c["ray"] >= 0.39 else ""
            rows.append("| %s | (%d, %d) | %g | %.3f | %.3f | %s |" % ("x".join(map(str, a["ns"])), a["m"], a["n"], a["eps"], c["r_chain"], c["width"], note))
    return "\n".join(rows)


def cylham_table():
    rows = ["| deck (mpol, ntor) | splines | boundary truncation error | B vol L2 | one-turn defect | O | X | island regions (radial extent) |", "|---|---|---|---|---|---|---|---|"]
    for p, r in sorted(load("cylham/cyl_*.json"), key=lambda x: x[1]["args"]["ntor"]):
        a, h = r["args"], r.get("effective_hamiltonian")
        if h is None:
            continue
        ok = r["closure"]["return_defect_max"] >= 1e-3
        rows.append("| (%d, %d) | %s | %s | %s | %s | %s | %s | %s |" % (
            a["mpol"], a["ntor"], "x".join(map(str, a["ns"])), fmt(r["boundary_truncation_error"]), fmt(r["score"]["B_rel_volume_l2"]),
            fmt(r["closure"]["return_defect_max"]), h["n_O"] if ok else "-", h["n_X"] if ok else "-",
            ", ".join("%.2f" % g["radial_extent"] for g in h["regions"]) if ok else "not resolved (defect < 1e-3)"))
    return "\n".join(rows)


def relax_table():
    rows = ["| shear eps | seed | start iota range | relaxed iota range | start islands | relaxed islands (r, width, O, X) | residual first -> last | wall s |", "|---|---|---|---|---|---|---|---|"]
    for p, r in load("relax/eps*_seed*.json"):
        a, st = r["args"], r["stages"]
        isl = "; ".join("(%.2f, %.3f, %d, %d)" % (c["r_chain"], c["width"], c["n_O"], c["n_X"]) for c in st[1]["islands"]) or "none"
        isl0 = "; ".join("(%.2f, %.3f)" % (c["r_chain"], c["width"]) for c in st[0]["islands"]) or "none"
        rows.append("| %g | %g | [%.3f, %.3f] | [%.3f, %.3f] | %s | %s | %.2e -> %.2e | %.0f |" % (
            a["eps"], a["seed"], st[0]["iota_min"], st[0]["iota_max"], st[1]["iota_min"], st[1]["iota_max"], isl0, isl,
            r["relax"]["resid"][0], r["relax"]["resid"][-1], r["total_s"]))
    return "\n".join(rows)


def wout_table():
    path = RES/"wout_lines.json"
    if not path.exists():
        return ""
    rows = ["| state | s | rotation mean | spread over starts | WOUT iotaf | exact normal fraction (rms) | exact label spread (max) |", "|---|---|---|---|---|---|---|"]
    for name, r in json.loads(path.read_text()).items():
        ex = {round(e["s"], 6): e for e in r["exact_on_surfaces"]}
        for l in r["lines"]:
            e = ex[round(l["s"], 6)]
            rows.append("| %s | %.3f | %.5f | %s | %.3f | %s | %s |" % (name, l["s"], l["rotation_mean"], fmt(l["rotation_spread"]), l["iotaf"],
                                                                      fmt(e["normal_fraction_rms"]), fmt(e["label_spread_max"])))
    return "\n".join(rows)


def main():
    text = "# MRX on Lopez D (generated by lopez_mrx_tables.py)\n\n## Vacuum solve against the exact field: resolution ladder\n\n" + ladder_table()
    text += "\n\n## VMEC-style chart: effect of the deck truncation\n\n" + truncation_table()
    text += "\n\n## Boundary deformation: closed lines, rotational transform and fixed points of the one-turn map\n\n" + ham_table()
    text += "\n\n## MRX island-chain finder (``islands``, m_max = 1) on the deformed domain\n\n" + islands_table()
    text += "\n\n## VMEC-deck domains: structure of the true vacuum field (one-turn map)\n\n" + cylham_table()
    text += "\n\n## Added current: sheared start field through iota = 1, ideal Newton relaxation\n\n" + relax_table()
    text += "\n\n## VMEC states: closed lines of their own field and the exact field on their surfaces\n\n" + wout_table() + "\n"
    (RES/"TABLES.md").write_text(text)
    print(text[:3000])


if __name__ == "__main__":
    main()
