"""Collect results/sensitivity/*.json into one table, gate verdicts and a continuum-limit fit.

Gates are the pre-registered ones in derivative_map.GATES.  Reads records only; runs no solve.
Usage: python sensitivity_table.py [--dir results/sensitivity] [--case integer_axisymmetric ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analytic import ROOT, cases
import derivative_map as dm


def load(directory, case=None):
    records = []
    for p in sorted(Path(directory).glob("*_ns*.json")):
        r = json.loads(p.read_text())
        r["tag"] = p.stem.split(f"_ns{r['ns']}", 1)[1]  # the file name is the authority
        records.append(r)
    records = [r for r in records if case is None or r["case"] in case]
    return sorted(records, key=lambda r: (r["case"], r.get("tag", ""), r["ns"]))


def kind(name):
    return "delta" if name == "delta" else "c"


def metric(score, name):
    return score["null_scaled_rms"] if name == "delta" else score["relative_error"]


def fd_plateau(entry, obs, name):
    """FD ladder: (h, |FD-JVP|_rms, FD error vs exact) rows, and the plateau (the best |FD-JVP|)."""
    rows = [(f["relative_step"], f["scores"][obs]["minus_jvp_rms_T"], metric(f["scores"][obs], name),
             f["branch_residual"]["raw"]["relative_residual"]) for f in entry["fd"] if obs in f["scores"]]
    best = min(rows, key=lambda r: r[1]) if rows else None
    return rows, best


def order_fit(records, name, obs):
    """Observed order p of error ~ NS^-p from the finest pairs (and the pairwise orders)."""
    records = [r for r in records if obs in r["params"][name]["jvp"] and max(r["root"].values()) < dm.ROOT_GATE]
    ns = np.array([r["ns"] for r in records], float)
    err = np.array([metric(r["params"][name]["jvp"][obs], name) for r in records])
    orders = [float(np.log(err[i]/err[i+1])/np.log((ns[i+1]-1)/(ns[i]-1))) for i in range(len(err)-1)]
    return {"ns": ns.astype(int).tolist(), "error": err.tolist(), "pairwise_order": orders}


def table(records):
    out, summary = [], {}
    for case in sorted({(r["case"], r.get("tag", "")) for r in records}):
        rs = [r for r in records if (r["case"], r.get("tag", "")) == case]
        case, tag = case
        names = list(rs[0]["params"])
        out += [f"## {case}{tag}", "",
                f"VMEX {rs[0]['vmex_commit'][:8]} ({rs[0]['vmex_version']}), tcon0 {rs[0]['tcon0']:g}, seed {rs[0]['seed']}, "
                f"ftol {rs[0]['ftol']:g}, JAX {rs[0]['jax']}; volume region s <= {rs[0]['s_max']}.", "",
                "| NS | observable | param | root pre / raw | tangent certificate (resid/tol) / independent rel. resid | transpose | JVP metric | gate | FD-JVP in metric units, best h | FD metric at best |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            for name in names:
                e = r["params"][name]
                for obs in e["jvp"]:
                    jv = e["jvp"][obs]
                    rows, best = fd_plateau(e, obs, name)
                    j = names.index(name)
                    lib = r["tangent_report"]["residual_norm"][j]/r["tangent_report"]["tolerance"][j]
                    tr = r["transpose"][obs]["max_mismatch_over_bound"] if obs in r["transpose"] else float("nan")
                    scale = cases()[case].edge if name == "delta" else 1.0/jv["rms_exact_T"]
                    fdtxt = f"{best[1]*scale:.2e} ({best[0]:g})" if best else "-"
                    fdm = f"{metric(next(f for f in e['fd'] if f['relative_step'] == best[0])['scores'][obs], name):.3e}" if best else "-"
                    out.append(f"| {r['ns']} | {obs} | {name} | {r['root']['preconditioned']:.1e} / {r['root']['raw']:.1e} | "
                               f"{lib:.1e} / {e['linear']['raw']['relative_residual']:.1e} | {tr:.1e} | "
                               f"{metric(jv, name):.3e} | {('REJECTED (root)' if max(r['root'].values()) >= dm.ROOT_GATE else 'PASS' if jv['gate_pass'] else 'FAIL')} | {fdtxt} | {fdm} |")
        out.append("")
        for name in names:
            for obs in rs[-1]["params"][name]["jvp"]:
                if len([r for r in rs if obs in r["params"][name]["jvp"]]) > 1:
                    summary[f"{case}{tag}/{name}/{obs}"] = order_fit(rs, name, obs)
        for r in rs:
            if "dense_control" in r:
                d = r["dense_control"]
                out += [f"Dense control NS {r['ns']}: {d['active_dofs']} active dofs, sigma_min {d['sigma_min']:.2e}, "
                        f"cond {d['condition']:.2e}, forward-vs-reverse Jacobian {d['forward_vs_reverse_jacobian_rel']:.1e}, "
                        f"tangent rel. error {d['tangent_rel_error']}, reverse rel. error {d['reverse_rel_error']}.", ""]
    # volume_fine series: the no-FD "_fine" records and any untagged record that carries the fine rule
    for case in sorted({r["case"] for r in records}):
        rs = [r for r in records if r["case"] == case and r.get("tag", "") in ("", "_fine")
              and any("volume_fine" in v["jvp"] for v in r["params"].values())]
        rs = sorted({r["ns"]: r for r in rs}.values(), key=lambda r: r["ns"])
        for name in (rs[0]["params"] if rs else []):
            if len(rs) > 1:
                summary[f"{case}/{name}/volume_fine(combined tags)"] = order_fit(rs, name, "volume_fine")
    out += ["## Continuum limit (JVP metric versus NS)", "", "| series | NS | metric | pairwise order |", "|---|---|---|---|"]
    for key, v in summary.items():
        out.append(f"| {key} | {v['ns']} | {', '.join(f'{x:.3e}' for x in v['error'])} | {', '.join(f'{x:.2f}' for x in v['pairwise_order'])} |")
    return "\n".join(out)+"\n", summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT/"results/sensitivity")
    ap.add_argument("--case", nargs="*")
    args = ap.parse_args()
    text, summary = table(load(args.dir, args.case))
    (args.dir/"TABLES.md").write_text(text)
    (args.dir/"orders.json").write_text(json.dumps(summary, indent=1)+"\n")
    print(text)


if __name__ == "__main__":
    main()
