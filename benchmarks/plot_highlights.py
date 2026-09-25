"""README highlight figures, drawn only from saved records.

Writes figures/highlight_*.png and figures/highlight_manifest.json (input and
output hashes).  Never edits or re-runs a record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"figures"
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"
CODE = {  # fixed categorical order; marker shape carries identity too
    "vmex": ("VMEX", "#2a78d6", "o"),
    "vmec2000": ("VMEC2000", "#eb6834", "s"),
    "vmecpp": ("VMEC++", "#1baf7a", "^"),
    "desc": ("DESC", "#eda100", "D"),
}
INPUTS = {}
NAMES = {"integer_axisymmetric": "integer axisymmetric", "integer_3d": "integer 3-D",
         "solovev_symmetric": "Solov'ev", "solovev_asymmetric": "asymmetric Solov'ev", "sheared_A": "sheared A"}


def load(rel):
    path = ROOT/rel
    INPUTS[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def save(fig, name):
    path = FIG/name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def fig_fixes():
    pin = load(sorted(p.relative_to(ROOT).as_posix() for p in
                      (ROOT/"results/audit/axis_diagnosis").glob("projection-ladder-pinned_copy-*/ladder.json"))[-1])
    lin = load(sorted(p.relative_to(ROOT).as_posix() for p in
                      (ROOT/"results/audit/axis_diagnosis").glob("projection-ladder-linear_extrapolation-*/ladder.json"))[-1])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, label in ((axes[0], "B_relative_l2", "B error on the axis"),
                           (axes[1], "J_relative_l2_ij", "J error at s = 1e-4")):
        s_key = "1e-08" if key == "B_relative_l2" else "0.0001"
        for rec, tag, color, marker in ((pin, "before #452", "#e34948", "s"), (lin, "after #452", "#2a78d6", "o")):
            ns = [r["ns"] for r in rec["rows"]]
            val = [r["by_s"][s_key][key] for r in rec["rows"]]
            ax.loglog(ns, val, marker=marker, color=color, lw=2, ms=8, label=tag)
            ax.annotate(tag, (ns[-1], val[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK, fontsize=9, va="center")
        style(ax, "radial surfaces NS", "relative error", label)
        ax.set_xticks([33, 65, 129, 257], ["33", "65", "129", "257"])
    axes[0].text(0.03, 0.05, "exact field projected onto VMEX\n(no solve), integer 3-D case",
                 transform=axes[0].transAxes, color=MUTED, fontsize=8)
    # Refinement traces: residual after each attempted step.
    base = load(sorted(p.relative_to(ROOT).as_posix() for p in
                       (ROOT/"results/audit/refinement_observation").glob("ns65-tcon0-base-*/observation.json"))[0])
    new = load(sorted(p.relative_to(ROOT).as_posix() for p in
                      (ROOT/"results/audit/refinement_observation_variants").glob("*/observation.json"))[0])
    ax = axes[2]
    # Draw "after" first so the shared first pass of "before" stays visible on top.
    for rec, tag, color, marker in ((new, "after #453", "#2a78d6", "o"), (base, "before #453", "#e34948", "s")):
        obs = rec["observations"]["raw_host"]
        start = obs["residual_before_refinement_operator"]["preconditioned"]["norm"]
        steps = [t["trial_preconditioned_residual"] for t in obs["trace"] if "trial_preconditioned_residual" in t]
        best = np.minimum.accumulate([start]+steps)
        ax.semilogy(range(len(best)), best, marker=marker, color=color, lw=2 if tag.startswith("after") else 1.5,
                    ms=8 if tag.startswith("after") else 5, label=tag)
        ax.annotate(f"{tag}: {best[-1]:.0e}", (len(best)-1, best[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    ax.axhline(1e-11, color=MUTED, lw=1, ls="--")
    ax.text(0.2, 1.6e-11, "refine_tol", color=MUTED, fontsize=8)
    style(ax, "Newton step", "best residual so far", "Refinement for derivatives (NS65)")
    ax.set_xlim(-0.5, 15)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.text(0.98, 0.97, "same host state,\none refinement call each", transform=ax.transAxes, color=MUTED, fontsize=8, ha="right", va="top")
    fig.tight_layout()
    return save(fig, "highlight_upstream_fixes.png")


def fig_parity(summary):
    rows = summary["vmec_family"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for case, ls in (("integer_axisymmetric", "-"), ("solovev_symmetric", ":")):
        for code, offset in (("vmex", 1.0), ("vmec2000", 1.0), ("vmecpp", 1.0)):
            pts = sorted((r["ns"], r["flux_label_error_max"]) for r in rows
                         if r["case"] == case and r["code"] == code and r.get("converged"))
            if not pts:
                continue
            name, color, marker = CODE[code]
            ax.loglog(*zip(*pts), ls=ls, marker=marker, color=color, lw=1.5, ms=9 if code == "vmex" else 6,
                      mfc="none" if code == "vmex" else color, label=f"{name}")
        ax.annotate(NAMES[case], pts[-1], xytext=(6, 0), textcoords="offset points",
                    color=INK, fontsize=9, va="center")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], frameon=False, fontsize=9, loc="upper right")
    style(ax, "radial surfaces NS", "max flux-label error", "Same deck, same answer")
    ax.set_xticks([33, 65, 129, 257], ["33", "65", "129", "257"])
    # Hard cases from a cold start at the finest NS run, versus DESC.
    ax = axes[1]
    cases = [("integer_3d", 129), ("sheared_A", 65)]
    width = 0.2
    for i, (case, ns) in enumerate(cases):
        for k, code in enumerate(("vmex", "vmec2000", "vmecpp")):
            name, color, marker = CODE[code]
            match = [r for r in rows if r["case"] == case and r["code"] == code and r["ns"] == ns]
            x = i + (k-1)*width
            if match and "flux_label_error_max" in match[0]:
                ax.bar(x, match[0]["flux_label_error_max"], width*0.9, color=color)
            else:
                ax.text(x, 2e-3, "no\noutput", ha="center", color=MUTED, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    ax.set_xticks(range(len(cases)), [f"{NAMES[c]}\nNS{n}, cold start" for c, n in cases])
    style(ax, "", "max flux-label error", "Hard cases from a cold start")
    ax.title.set_position((0, 1.16))
    for code in ("vmex", "vmec2000", "vmecpp"):
        ax.bar(np.nan, np.nan, color=CODE[code][1], label=CODE[code][0])
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3)
    fig.tight_layout()
    return save(fig, "highlight_code_parity.png")


def fig_routes():
    rec = load("results/cross_code/runs/integer_axisymmetric_iota-ns129-vmec2000/exact_score.json")
    s = [r["s"] for r in rec["routes"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key_w, key_n, title in ((axes[0], "J_wout_relative_l2", "J_native_relative_l2", "Current density J"),
                                    (axes[1], None, "B_native_relative_l2", "Magnetic field B")):
        if key_w:
            ax.semilogy(s, [r[key_w] for r in rec["routes"]], marker="s", color="#eb6834", lw=2, ms=7,
                        label="WOUT output (what VMEC2000 and VMEC++ provide)")
        else:
            wout_B = {f["j"]: f["B_relative_surface_l2"] for f in rec["fields"]}
            ax.semilogy(s, [wout_B[r["j"]] for r in rec["routes"]], marker="s", color="#eb6834", lw=2, ms=7,
                        label="WOUT output")
        ax.semilogy(s, [r[key_n] for r in rec["routes"]], marker="o", color="#2a78d6", lw=2, ms=7,
                    label="VMEX native field")
        style(ax, "normalized toroidal flux s", "relative error", title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("One equilibrium (VMEC2000 output, integer axisymmetric, NS129), two ways to read it",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, "highlight_field_routes.png")


def fig_hard_cases(summary):
    desc = {(r["case"], r["M"]): r for r in summary["desc"]}
    vmec = summary["vmec_family"]
    c2 = load("results/audit/measurement_c2_axisfix/integer3d-ns257-tcon0-0-full-axisfix-20260925/measurement.json")
    gp = ROOT/"results/audit/measurement_c2_axisfix/integer3d-ns257-tcon0-0-full-axisfix-20260925/grid_checkpoints/full_midpoint.json"
    INPUTS[gp.relative_to(ROOT).as_posix()] = hashlib.sha256(gp.read_bytes()).hexdigest()
    mid = json.loads(gp.read_text())["route_A"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    # (a) integer 3-D volume J: DESC vs M, VMEX seeded roots
    ax = axes[0]
    ms = sorted(m for (c, m) in desc if c == "integer_3d")
    ax.semilogy(ms, [desc[("integer_3d", m)]["J_volume"] for m in ms], marker="D", color=CODE["desc"][1], lw=2, ms=8)
    ax.annotate("DESC", (ms[1], desc[("integer_3d", ms[1])]["J_volume"]), xytext=(-34, -16),
                textcoords="offset points", color=INK, fontsize=9)
    ax.axhline(mid["current_relative_l2"], color=CODE["vmex"][1], lw=2, ls="--")
    ax.text(ms[-1], mid["current_relative_l2"]*1.3, "VMEX NS257 (seeded, TCON0=0)", color=INK, fontsize=8, ha="right")
    ax.axhline(1e-3, color=MUTED, lw=1, ls=":")
    ax.text(ms[-1], 1.25e-3, "J gate 1e-3", color=MUTED, fontsize=8, ha="right")
    style(ax, "DESC poloidal/toroidal resolution M", "volume J error", "Integer 3-D")
    # (b) sheared A and (c) asymmetric Solov'ev: B and J per code
    for ax, case, dm, vm_ns, title in ((axes[1], "sheared_A", 10, 65, "Sheared A"),
                                       (axes[2], "solovev_asymmetric", 10, 65, "Up-down asymmetric Solov'ev")):
        labels, vals, colors, notes = [], [], [], []
        for code in ("vmex", "vmec2000", "vmecpp"):
            r = [x for x in vmec if x["case"] == case and x["code"] == code and x["ns"] == vm_ns]
            labels.append(f"{CODE[code][0]}\nNS{vm_ns}")
            colors.append(CODE[code][1])
            if r and "B_wout_mid" in r[0]:
                vals.append(r[0]["B_wout_mid"])
                notes.append("" if r[0]["converged"] else "not converged")
            else:
                vals.append(np.nan)
                crash = case == "solovev_asymmetric" and code == "vmecpp"
                notes.append("crashed\n(LASYM)" if crash else "no output\n(did not\nconverge)")
        d = desc.get((case, dm))
        labels.append(f"DESC\nM{dm}")
        colors.append(CODE["desc"][1])
        vals.append(d["B_mid"] if d else np.nan)  # rho = 0.7, s = 0.49: same kind of point as the WOUT mid surface
        notes.append("" if d and d["solver_success"] else "iteration cap")
        x = np.arange(len(labels))
        ax.bar(x, vals, 0.6, color=colors)
        for xi, v, n in zip(x, vals, notes):
            if n:
                ax.text(xi, (v*1.5 if np.isfinite(v) else 3e-10), n, ha="center", color=MUTED, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1)
        ax.set_xticks(x, labels, fontsize=8)
        style(ax, "", "B error at mid radius (s near 0.5)", title)
    fig.tight_layout()
    return save(fig, "highlight_hard_cases.png")


def main():
    summary = load("results/cross_code/summary.json")
    outputs = [fig_fixes(), fig_parity(summary), fig_routes(), fig_hard_cases(summary)]
    manifest = {"schema": 1, "generator": "benchmarks/plot_highlights.py",
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs": INPUTS,
                "outputs": {n: hashlib.sha256((FIG/n).read_bytes()).hexdigest() for n in outputs}}
    (FIG/"highlight_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(outputs)


if __name__ == "__main__":
    main()
