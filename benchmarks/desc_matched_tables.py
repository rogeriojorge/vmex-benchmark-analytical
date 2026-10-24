"""Tables from the ``desc_matched.py`` records: factor isolation, matched M ladders, work-precision.

Reads ``results/desc/matched/<tag>/record.json`` (schema 3) and writes ``tables.md`` and ``index.json``
beside them.  Tags: ``factor_<M>_<name>`` (sheared A factor isolation), ``ladder_<case>_<protocol>_M<M>``
(matched ladders, protocol ``cold`` or ``exact``) and ``repeat<k>_...`` (timing repeats of a ladder tag).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/"results/desc/matched"
LEVEL = "q2"  # preferred; a record scored only at another level (factor runs: q1) uses its coarsest


def level_of(r):
    return LEVEL if LEVEL in r["chart"] else sorted(r["chart"])[0]


def load(directory):
    records = {}
    for f in sorted(directory.glob("*/record.json")):
        records[f.parent.name] = json.loads(f.read_text())
    return records


PROTOCOL_ORDER = ("strict", "loose", "exact", "pub")


# sheared-A factor runs that are exactly a ladder protocol (scored at q1, not q2)
# (the factor runs of the loose and pub protocols duplicate ladder runs and serve as extra timing repeats)
FACTOR_AS_LADDER = {"factor_8_base": "strict", "factor_10_base": "strict"}


def protocol_of(tag):
    m = re.fullmatch(r"(?:repeat\d+_)?ladder_(.+)_(strict|loose|exact|pub)_M\d+", tag)
    return m.group(2) if m else None


def row(tag, r):
    s, c, nat = r["solver"], r["chart"][level_of(r)]["all"], r["native"]["volume"]["gauss"]
    p, res = r["protocol"], r["resolution"]
    return {"tag": tag, "protocol": protocol_of(tag), "case": r["case"], "M": res["M"], "N": res["N"], "L": res["L"], "sym": res["sym"],
            "seed": p["seed"], "closure": p["closure"], "optimizer": p["optimizer"], "grid": p["grid"], "tol": p["tol"],
            "maxiter": p["maxiter"], "success": s["success"], "iterations": s["iterations"], "nfev": s["nfev"],
            "message": s["message"], "solve_s": s["solve_seconds"], "cpu_s": s.get("solve_cpu_seconds"), "total_s": r["timings"]["total"],
            "seed_s": r["timings"]["seed"], "build_s": r["timings"]["build"], "cost0": s["initial_cost"],
            "cost": s["final_cost"], "opt": s["final_optimality"],
            "B": c["B"], "J": c["J"], "force": c["force"], "B_native": nat["B_relative_l2"],
            "J_native": nat["J_relative_l2"], "force_native": nat["force_over_gradp"],
            "level": level_of(r), "outside": c["outside_boundary_points"], "load_start": r["environment_start"]["loadavg"][0],
            "load_end": r["environment_end"]["loadavg"][0], "solve_grid_nodes": r["solve_grid"]["num_nodes"],
            "nested": r.get("nested_final")}


def fmt(x, spec=".2e"):
    return "-" if x is None else format(x, spec)


def factor_table(rows):
    lines = []
    for case, M in sorted({(r["case"], r["M"]) for r in rows}):
        sub = [r for r in rows if r["M"] == M and r["case"] == case]
        lines += [f"\n### {case}, M = {M}\n",
                  "| run | seed | closure | optimizer | grid | N | L | tol | status | its | nfev | cost0 | cost | solve s | B | J | force | B native |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(sub, key=lambda x: x["tag"]):
            status = "converged" if r["success"] else ("cap" if r["iterations"] >= r["maxiter"] - 1 or r["nfev"] >= r["maxiter"] else "stopped")
            lines.append(f"| {re.sub(r"^(issan_)?factor_\d+_", "", r["tag"])} | {r['seed']} | {r['closure']} | {r['optimizer']} | {r['grid']} | {r['N']} | {r['L']} | {r['tol']} | "
                         f"{status} | {r['iterations']} | {r['nfev']} | {fmt(r['cost0'])} | {fmt(r['cost'])} | {r['solve_s']:.0f} | "
                         f"{r['B']:.2e} | {r['J']:.2e} | {r['force']:.2e} | {r['B_native']:.2e} |")
    return "\n".join(lines)


def ladder_table(rows):
    lines = ["\n| case | protocol | M | N | L | status | its | nfev | solve s | solve cpu s | total s | B | J | force | B native | J native | outside pts |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["case"], PROTOCOL_ORDER.index(x["protocol"]), x["M"])):
        status = "converged" if r["success"] else "cap/stopped"
        lines.append(f"| {r['case']} | {r['protocol']} | {r['M']} | {r['N']} | {r['L']} | {status} | {r['iterations']} | {r['nfev']} | "
                     f"{r['solve_s']:.0f} | {fmt(r['cpu_s'], '.0f')} | {r['total_s']:.0f} | {r['B']:.2e} | {r['J']:.2e} | {r['force']:.2e} | "
                     f"{r['B_native']:.2e} | {r['J_native']:.2e} | {r['outside']} |")
    return "\n".join(lines)


def repeat_table(records):
    groups = {}
    for tag, r in records.items():
        m = re.fullmatch(r"(?:repeat\d+_)?(ladder_.*)", tag)
        if m and (tag.startswith("repeat") or any(t.startswith("repeat") and t.endswith(m.group(1)) for t in records)):
            groups.setdefault(m.group(1), []).append((tag, r))
    lines = ["\n| tag | runs | solve s (each) | median | spread (max/min) | B (each) | load at start (each) |", "|---|---|---|---|---|---|---|"]
    for key, items in sorted(groups.items()):
        t = sorted(r["solver"]["solve_seconds"] for _, r in items)
        lines.append(f"| {key} | {len(items)} | {', '.join(f'{x:.1f}' for x in t)} | {t[len(t)//2]:.1f} | {t[-1]/t[0]:.2f} | "
                     f"{', '.join(f'{r['chart'][level_of(r)]['all']['B']:.2e}' for _, r in items)} | "
                     f"{', '.join(f'{r['environment_start']['loadavg'][0]:.1f}' for _, r in items)} |")
    return "\n".join(lines)


def work_precision_figure(ladder, path):
    """B error against solve wall time for every ladder run, one panel per case; lines join an M ladder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = sorted({r["case"] for r in ladder})
    cols = 3
    fig, axes = plt.subplots((len(names)+cols-1)//cols, cols, figsize=(4.6*cols, 3.6*((len(names)+cols-1)//cols)), squeeze=False)
    style = {"strict": ("tab:red", "o"), "loose": ("tab:blue", "s"), "exact": ("tab:green", "^"), "pub": ("tab:purple", "D")}
    for ax, name in zip(axes.ravel(), names):
        for proto in PROTOCOL_ORDER:
            pts = sorted((r["M"], r["solve_s"], r["B"], r["success"]) for r in ladder if r["case"] == name and r["protocol"] == proto)
            if not pts:
                continue
            color, marker = style[proto]
            ax.plot([p[1] for p in pts], [p[2] for p in pts], marker=marker, color=color, lw=1, ms=5, label=proto)
            for M, t, b, ok in pts:
                ax.annotate(str(M), (t, b), textcoords="offset points", xytext=(3, 3), fontsize=6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("solve wall time (s), loaded shared host")
        ax.set_ylabel("whole-volume B error")
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    records = load(args.dir)
    rows = [row(t, r) for t, r in records.items()]
    factor = [r for r in rows if re.match(r"(issan_)?factor_", r["tag"])]
    ladder = [r for r in rows if r["tag"].startswith("ladder_")]
    ladder += [{**r, "protocol": p} for r in rows for t, p in FACTOR_AS_LADDER.items() if r["tag"] == t]
    text = ["# DESC matched runs: generated tables", "", "Scores: exact-chart whole-volume relative L2 (level q2); `B native` is DESC's own-grid score."]
    if factor:
        text += ["\n## Factor isolation", factor_table(factor)]
    if ladder:
        text += ["\n## Matched M ladders (work-precision rows)", ladder_table(ladder)]
    rep = repeat_table(records)
    if "repeat" in "".join(records):
        text += ["\n## Timing repeats", rep]
    if ladder:
        work_precision_figure(ladder, ROOT/"figures/desc_work_precision.png")
    (args.dir/"tables.md").write_text("\n".join(text)+"\n")
    (args.dir/"index.json").write_text(json.dumps(rows, indent=1, allow_nan=False, default=float)+"\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
