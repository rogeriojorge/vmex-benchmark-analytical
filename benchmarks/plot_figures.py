"""Draw every README figure from saved records (never re-runs or edits them).

Writes figures/*.png and figures/manifest.json with input and output hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from analytic import cases, label_at_s, surface

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"figures"
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"
BEFORE, AFTER = "#e34948", "#2a78d6"
CODE = {"vmex": ("VMEX", "#2a78d6", "o"), "vmec2000": ("VMEC2000", "#eb6834", "s"),
        "vmecpp": ("VMEC++", "#1baf7a", "^"), "desc": ("DESC", "#eda100", "D")}
NAMES = {"integer_axisymmetric": "integer axisymmetric", "integer_3d": "integer 3-D",
         "solovev_symmetric": "Solov'ev", "solovev_asymmetric": "asymmetric Solov'ev", "sheared_A": "sheared A"}
INPUTS: dict[str, str] = {}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left"})


def load(rel):
    path = ROOT/rel
    INPUTS[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def style(ax, xlabel="", ylabel="", title=None):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def ns_axis(ax, values):
    ax.set_xscale("log")
    ax.set_xticks(values, [str(v) for v in values])


def label_end(ax, x, y, text, dx=6):
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", color=INK, fontsize=9, va="center")


def save(fig, name):
    fig.savefig(FIG/name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return name


def fig_cases():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    theta = np.linspace(0, 2*np.pi, 241)
    for ax, name, phis in ((axes[0], "integer_3d", (0.0, np.pi/4)), (axes[1], "sheared_A", (0.0, np.pi/4)),
                           (axes[2], "solovev_asymmetric", (0.0,))):
        case = cases()[name]
        for k, phi in enumerate(phis):
            for s in (0.1, 0.4, 0.7, 1.0):
                xyz = np.asarray(surface(case, float(label_at_s(case, s)), theta, np.full_like(theta, phi)))
                R, Z = np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]
                ax.plot(R, Z, color=(AFTER if k == 0 else "#eb6834"), lw=1.6 if s == 1.0 else 0.8)
        ax.set_aspect("equal")
        style(ax, "R", "Z" if ax is axes[0] else "", NAMES[name])
    fig.tight_layout()
    return save(fig, "exact_cases.png")


def fig_parity(summary):
    rows = summary["vmec_family"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for case, ls in (("integer_axisymmetric", "-"), ("solovev_symmetric", ":")):
        last = None
        for code in ("vmex", "vmec2000", "vmecpp"):
            pts = sorted((r["ns"], r["flux_label_error_max"]) for r in rows
                         if r["case"] == case and r["code"] == code and r.get("converged"))
            name, color, marker = CODE[code]
            ax.plot(*zip(*pts), ls=ls, marker=marker, color=color, lw=1.5, ms=10 if code == "vmex" else 6,
                    mfc="none" if code == "vmex" else color, label=name if case == "integer_axisymmetric" else None)
            last = pts[-1]
        label_end(ax, *last, NAMES[case])
    ax.set_yscale("log")
    ns_axis(ax, [33, 65, 129, 257])
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax, "radial surfaces NS", "flux-surface error", "Converged cases: identical answers")
    ax = axes[1]
    width = 0.24
    for i, (case, ns) in enumerate((("integer_3d", 129), ("sheared_A", 65))):
        for k, code in enumerate(("vmex", "vmec2000", "vmecpp")):
            match = [r for r in rows if r["case"] == case and r["code"] == code and r["ns"] == ns]
            x = i+(k-1)*width
            if match and "flux_label_error_max" in match[0]:
                ax.bar(x, match[0]["flux_label_error_max"], width*0.9, color=CODE[code][1])
            else:
                ax.text(x, 1.5e-3, "no\noutput", ha="center", color=MUTED, fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1)
    ax.set_xticks([0, 1], ["integer 3-D, NS129", "sheared A, NS65"])
    style(ax, "", "flux-surface error", "Hard cases, cold start: all fail")
    fig.tight_layout()
    return save(fig, "code_parity.png")


def fig_readout():
    rec = load("results/cross_code/runs/integer_axisymmetric_iota-ns129-vmec2000/exact_score.json")
    routes = rec["routes"]
    s = [r["s"] for r in routes]
    wout_B = {f["j"]: f["B_relative_surface_l2"] for f in rec["fields"]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, wout, native, title in (
            (axes[0], [r["J_wout_relative_l2"] for r in routes], [r["J_native_relative_l2"] for r in routes], "Current J"),
            (axes[1], [wout_B[r["j"]] for r in routes], [r["B_native_relative_l2"] for r in routes], "Field B")):
        ax.semilogy(s, wout, marker="s", color=CODE["vmec2000"][1], lw=2, ms=6, label="WOUT file")
        ax.semilogy(s, native, marker="o", color=AFTER, lw=2, ms=6, label="VMEX native field")
        style(ax, "normalized toroidal flux s", "relative error", title)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    return save(fig, "field_readout.png")


def fig_axis_row():
    before, after = load("results/fixes/axis_row_before_452.json"), load("results/fixes/axis_row_after_452.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, key, s_key, title in ((axes[0], "B_relative_l2", "1e-08", "B on the axis"),
                                  (axes[1], "J_relative_l2_ij", "0.0001", "J next to the axis (s = 1e-4)")):
        for rec, tag, color, marker in ((before, "before #452", BEFORE, "s"), (after, "after #452", AFTER, "o")):
            ns = [r["ns"] for r in rec["rows"]]
            val = [r["by_s"][s_key][key] for r in rec["rows"]]
            ax.plot(ns, val, marker=marker, color=color, lw=2, ms=7)
            label_end(ax, ns[-1], val[-1], tag)
        ax.set_yscale("log")
        ns_axis(ax, [33, 65, 129, 257])
        style(ax, "radial surfaces NS", "relative error" if ax is axes[0] else "", title)
    fig.tight_layout()
    return save(fig, "fix_axis_row.png")


def fig_refinement():
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for rel, tag, color, marker, lw, ms in (("results/fixes/refinement_after_453.json", "after #453", AFTER, "o", 2, 7),
                                           ("results/fixes/refinement_before_453.json", "before #453", BEFORE, "s", 1.5, 5)):
        obs = load(rel)["observations"]["raw_host"]
        start = obs["residual_before_refinement_operator"]["preconditioned"]["norm"]
        steps = [t["trial_preconditioned_residual"] for t in obs["trace"] if "trial_preconditioned_residual" in t]
        best = np.minimum.accumulate([start]+steps)
        ax.semilogy(range(len(best)), best, marker=marker, color=color, lw=lw, ms=ms)
        label_end(ax, len(best)-1, best[-1], f"{tag}: {best[-1]:.0e}")
    ax.axhline(1e-11, color=MUTED, lw=1, ls="--")
    ax.text(0.2, 1.7e-11, "refine_tol", color=MUTED, fontsize=8)
    ax.set_xlim(-0.5, 14.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    style(ax, "Newton step", "best residual", "One refinement call, NS65")
    fig.tight_layout()
    return save(fig, "fix_refinement.png")


def fig_integer_3d(summary):
    desc = sorted((r["M"], r["J_volume"], r["solver_success"]) for r in summary["desc"] if r["case"] == "integer_3d")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    ax = axes[0]
    ax.semilogy([d[0] for d in desc], [d[1] for d in desc], marker="D", color=CODE["desc"][1], lw=2, ms=7)
    ax.axhline(1e-3, color=MUTED, lw=1, ls=":")
    ax.text(desc[-1][0], 1.25e-3, "J target 1e-3", color=MUTED, fontsize=8, ha="right")
    ax.set_xticks([d[0] for d in desc])
    style(ax, "DESC resolution M = N", "volume J error", "DESC")
    ax = axes[1]
    pts = []
    for ns in (129, 257):
        rec = load(f"results/vmex_seeded/integer_3d_ns{ns}.json")
        pts.append((ns, rec["grids"]["full_midpoint"]["current_relative_l2"]))
    ax.plot(*zip(*pts), marker="o", color=CODE["vmex"][1], lw=2, ms=7)
    ns_axis(ax, [129, 257])
    ax.axhline(1e-3, color=MUTED, lw=1, ls=":")
    style(ax, "VMEX radial surfaces NS", "", "VMEX (started from the exact state)")
    fig.tight_layout()
    return save(fig, "integer_3d_accuracy.png")


def fig_hard_cases(summary):
    desc = {(r["case"], r["M"]): r for r in summary["desc"]}
    rows = summary["vmec_family"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, case, ns, m in ((axes[0], "sheared_A", 65, 10), (axes[1], "solovev_asymmetric", 65, 10)):
        labels, vals, colors, notes = [], [], [], []
        for code in ("vmex", "vmec2000", "vmecpp"):
            r = [x for x in rows if x["case"] == case and x["code"] == code and x["ns"] == ns]
            labels.append(f"{CODE[code][0]}\nNS{ns}")
            colors.append(CODE[code][1])
            if r and "B_wout_mid" in r[0]:
                vals.append(r[0]["B_wout_mid"])
                notes.append("" if r[0]["converged"] else "not\nconverged")
            else:
                vals.append(np.nan)
                notes.append("no output" if code == "vmecpp" and case == "sheared_A" else "no LASYM")
        d = desc[(case, m)]
        labels.append(f"DESC\nM{m}")
        colors.append(CODE["desc"][1])
        vals.append(d["B_mid"])
        notes.append("" if d["solver_success"] else "iteration\ncap")
        x = np.arange(len(labels))
        ax.bar(x, vals, 0.6, color=colors)
        for xi, v, n in zip(x, vals, notes):
            if n:
                ax.text(xi, v*1.6 if np.isfinite(v) else 3e-10, n, ha="center", color=MUTED, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1)
        ax.set_xticks(x, labels, fontsize=8)
        style(ax, "", "B error at mid radius" if ax is axes[0] else "", NAMES[case])
    fig.tight_layout()
    return save(fig, "hard_cases.png")


def main():
    summary = load("results/cross_code/summary.json")
    outputs = [fig_cases(), fig_parity(summary), fig_readout(), fig_axis_row(), fig_refinement(),
               fig_integer_3d(summary), fig_hard_cases(summary)]
    manifest = {"generator": "benchmarks/plot_figures.py",
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "inputs": INPUTS,
                "outputs": {n: hashlib.sha256((FIG/n).read_bytes()).hexdigest() for n in outputs}}
    (FIG/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(outputs)


if __name__ == "__main__":
    main()
