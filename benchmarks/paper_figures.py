"""Draw the six paper figures from archived records, through results/index.json only.

Every number plotted is read from a file listed in the index; each file is hashed on first use and must match the
index.  Nothing is recomputed, nothing under results/ is written.  Outputs: figures/paper/<name>.{png,pdf} and
figures/paper/manifest.json (index hash, records used per figure, input and output hashes, generator hash).

    python benchmarks/paper_figures.py [--out figures/paper] [--only fig1 fig4]

Style follows benchmarks/plot_figures.py (ink, grid, minimal spines); colours are the Okabe-Ito set and every series
also differs in marker or line style, so the figures read in greyscale and under common colour-vision deficiencies.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT/"results/index.json"
INK, MUTED, GRID = "#1f1f1e", "#6b6b68", "#e4e4e1"
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.titlelocation": "left", "legend.fontsize": 7.5,
                     "pdf.fonttype": 42, "axes.labelsize": 9})


class Index:
    """The result index: files are only reachable through it, and verified against their recorded hash."""

    def __init__(self, path: Path = INDEX):
        self.path = path
        self.data = json.loads(path.read_text())
        self.hashes = {f["path"]: f["sha256"] for r in self.data["records"].values() for f in r["files"]}
        self.owner = {f["path"]: k for k, r in self.data["records"].items() for f in r["files"]}
        self.checked: dict[str, str] = {}
        self.touched: set[str] = set()

    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def used_keys(self) -> list[str]:
        return sorted({self.owner[p] for p in self.touched})

    def paths(self, key: str, pattern: str = "*") -> list[str]:
        return [f["path"] for f in self.data["records"][key]["files"] if fnmatch.fnmatch(f["path"], pattern)]

    def read(self, rel: str) -> bytes:
        if rel not in self.hashes:
            raise KeyError(f"{rel} is not listed in the result index")
        self.touched.add(rel)
        if rel not in self.checked:
            blob = (ROOT/rel).read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if digest != self.hashes[rel]:
                raise ValueError(f"{rel}: sha256 {digest} differs from the index ({self.hashes[rel]})")
            self.checked[rel] = digest
        return (ROOT/rel).read_bytes()

    def load(self, rel: str):
        return json.loads(self.read(rel))

    def records(self, key: str, pattern: str = "*") -> list[dict]:
        return [self.load(p) for p in self.paths(key, pattern)]


# ---- style ------------------------------------------------------------------------------------------------------
def style(ax, xlabel="", ylabel="", title=None):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, color=INK)
    ax.grid(True, color=GRID, lw=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if ax.get_yscale() == "log":
        lo, hi = ax.get_ylim()
        if hi/lo < 60:
            ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(10, 0.4)))
            ax.tick_params(axis="y", which="minor", labelsize=6.5, labelcolor=MUTED)


def tag(ax, letter):
    ax.text(-0.02, 1.07, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom", ha="right")


def ns_axis(ax, values=(17, 33, 65, 129)):
    ax.set_xscale("log")
    ax.set_xticks(list(values), [str(v) for v in values])


def save(fig, out: Path, name: str):
    fig.savefig(out/f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig.savefig(out/f"{name}.pdf", bbox_inches="tight", facecolor="white", metadata={"CreationDate": None, "Producer": None, "Creator": None})
    plt.close(fig)


def gate(ax, value, label, x=None, color=MUTED):
    ax.axhline(value, color=color, ls=(0, (4, 3)), lw=0.9)
    ax.annotate(label, xy=(0.99 if x is None else x, value), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=7, color=color)


# ---- figure 1 ---------------------------------------------------------------------------------------------------
def cert_B(rec):
    level = "q4" if "q4" in rec["scores"] else "m4"
    return rec["scores"][level]["all"]["B"]


def fig1(idx: Index, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    # (a) NS ladder at fixed (mpol, ntor)
    ax = axes[0, 0]
    series = [("sheared A (13,12)", "proj.cert.sheared_A", 13, 12, 64, BLUE, "o", "-"),
              ("sheared Issan (17,24)", "proj.cert.sheared_issan", 17, 24, 96, GREEN, "s", "-"),
              ("integer 3-D (13,12)", "proj.cert.integer_3d", 13, 12, 64, VERM, "^", "-"),
              ("integer 3-D, tau = 0.5 (13,12)", "proj.cert.integer_3d_tau", 13, 12, 64, ORANGE, "v", "--"),
              ("asymmetric Solov'ev (13,0)", "proj.cert.solovev_asymmetric", 13, 0, 64, PURPLE, "D", "-.")]
    for label, key, m, n, g, color, marker, ls in series:
        pts = {}
        for r in idx.records(key):
            if r["mpol"] == m and r["ntor"] == n and r["angular_fit_grid"] == g and "q4" in r["scores"]:
                pts[r["ns"]] = cert_B(r)
        ns = sorted(pts)
        ax.plot(ns, [pts[k] for k in ns], marker=marker, ls=ls, color=color, lw=1.4, ms=5.5, label=label)
    base = np.array([17.0, 129.0])
    ax.plot(base, 2.3e-5*(base/17.0)**-2, color=MUTED, lw=0.8, ls=":")
    ax.annotate("NS$^{-2}$", (60, 2.0e-6), color=MUTED, fontsize=7.5)
    gate(ax, 1e-5, "gate B = 1e-5")
    ax.set_yscale("log")
    ns_axis(ax)
    ax.legend(frameon=False, loc="lower left")
    style(ax, "radial surfaces NS", "B error, relative L2, whole volume", "Projection certificate against NS (fixed mpol, ntor)")
    tag(ax, "a")
    # (b) angular ladder
    ax = axes[0, 1]
    angular = [("sheared A, NS 65", "proj.cert.sheared_A", 65, BLUE, "o", "-"), ("sheared A, NS 129", "proj.cert.sheared_A", 129, SKY, "o", "--"),
               ("sheared Issan, NS 33", "proj.cert.sheared_issan", 33, GREEN, "s", "-")]
    for label, key, ns, color, marker, ls in angular:
        pts = {}
        for r in idx.records(key):
            if r["ns"] == ns and "q4" in r["scores"] and r["angular_fit_grid"] in (64, 96):
                pts[r["mpol"]] = cert_B(r)
        ms = sorted(pts)
        ax.plot(ms, [pts[m] for m in ms], marker=marker, ls=ls, color=color, lw=1.4, ms=5.5, label=label)
    gate(ax, 1e-5, "gate B = 1e-5")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="lower left")
    style(ax, "poloidal modes mpol (ntor 6-16 for A, 18-30 for Issan)", "B error, relative L2, whole volume",
          "Projection certificate against angular resolution")
    tag(ax, "b")
    # (c) Lopez D projection oracle
    ax = axes[1, 0]
    eta = idx.load("results/lopez/projection_oracle_eta.json")
    alpha = idx.load("results/lopez/projection_oracle_alpha.json")
    readout = idx.load("results/lopez/projection_vmex_eta.json")
    ladder = [(4, 16), (6, 24), (8, 40), (10, 56), (10, 72), (12, 96)]
    colors = {17: SKY, 33: BLUE, 65: "#00304f"}
    for ns in (17, 33, 65):
        y = [next(r["B_vol"] for r in eta if r["ns"] == ns and (r["mpol"], r["ntor"]) == mn) for mn in ladder]
        ax.plot(range(len(ladder)), y, marker="o", color=colors[ns], lw=1.4, ms=5, label=f"exact state, NS {ns}")
    for ns, mk in ((17, "x"), (33, "x"), (65, "x")):
        row = next(r for r in readout if r["ns"] == ns and r["closure"] == "current")
        ax.plot([4], [row["B_vol"]], marker=mk, color=VERM, ms=7, mew=1.6, ls="none", label="VMEX WOUT read-out (10,72)" if ns == 17 else None)
        ax.annotate(f"NS {ns}", (4, row["B_vol"]), xytext=(5, 0), textcoords="offset points", fontsize=7, color=VERM, va="center")
    for ns, mk in ((17, "D"), (33, "D"), (65, "D")):
        pts = [next(r["B_vol"] for r in alpha if r["ns"] == ns and (r["mpol"], r["ntor"]) == mn) for mn in ladder[4:]]
        ax.plot([4, 5], pts, marker=mk, ls="none", mfc="none", color=PURPLE, ms=6, label="alpha-angle gauge (direction-only)" if ns == 17 else None)
    ax.set_xticks(range(len(ladder)), [f"{m},{n}" for m, n in ladder])
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="lower left")
    style(ax, "(mpol, ntor)", "B error, relative L2, whole volume", "Lopez D: finite-basis projection floor")
    tag(ax, "c")
    # (d) MRX ladder
    ax = axes[1, 1]
    rows = [r for r in idx.records("lopez_mrx.ladder", "*/ladder/label_tw-1_p*.json")]
    for p, color, marker in ((2, SKY, "s"), (3, BLUE, "o"), (4, VERM, "^")):
        pts = sorted((r["args"]["ns"][0], r["score"]["B_rel_volume_l2"]) for r in rows if r["args"]["p"] == p and r["args"]["twist"] == -1 and not r["args"]["pert"])
        ax.plot(*zip(*pts), marker=marker, color=color, lw=1.4, ms=5.5, label=f"exact-flux chart, p = {p}")
    deck = sorted((r["args"]["ns"][0], r["score"]["B_rel_volume_l2"]) for r in idx.records("lopez_mrx.deck", "*/cyl/cyl_10_72_*.json"))
    ax.plot(*zip(*deck), ls="none", marker="*", color=INK, ms=9, label="VMEC-deck chart (10,72), p = 3")
    oracle = next(r["B_vol"] for r in eta if r["ns"] == 65 and (r["mpol"], r["ntor"]) == (10, 72))
    ax.axhline(oracle, color=MUTED, ls=":", lw=1)
    ax.annotate("VMEC projection floor (10,72), NS 65", xy=(0.99, oracle), xycoords=("axes fraction", "data"), ha="right", va="top", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([6, 8, 12, 16], ["6", "8", "12", "16"])
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="lower left")
    style(ax, "radial splines (angular splines scale with them)", "B error, relative L2, whole volume", "Lopez D: MRX vacuum solve")
    tag(ax, "d")
    fig.tight_layout()
    save(fig, out, "fig1_references_projection")


# ---- figure 2 ---------------------------------------------------------------------------------------------------
CASES = [("solovev_symmetric", "Solov'ev"), ("solovev_asymmetric", "asymmetric Solov'ev"), ("integer_3d", "integer 3-D"),
         ("integer_3d_tau", "integer 3-D, tau = 0.5"), ("sheared_A", "sheared A")]
PROTO = {"cold": ("cold start", BLUE), "continued": ("NS ladder", GREEN), "exact-seed": ("exact seed", VERM),
         "exact-seed_tcon0": ("exact seed, tcon0 = 0", PURPLE)}
CODE_MARK = {"vmex": "o", "vmec2000": "s"}


def proto_of(row):
    if row["protocol"] == "cold":
        return "cold"
    if row["protocol"] == "ladder":
        return "continued"
    if row["protocol"] == "seed" and row["tag"] == "":
        return "exact-seed"
    if row["protocol"] == "seed" and row["tag"] == "_tcon0":
        return "exact-seed_tcon0"
    return None


def mech_rows(idx: Index):
    rows = []
    for r in idx.records("mech.runs"):
        rid = r["run_id"]
        if "defect" in rid:
            continue
        rows.append(dict(case=r["case"], ns=r["ns"], tcon0=r["tcon0"], baseline=r["baseline"], polar=rid.endswith("-polar"),
                         B=r["headline"]["B"], conv=r["solve"]["converged"], id=rid, rec=r))
    return rows


def fig2(idx: Index, out: Path):
    fig = plt.figure(figsize=(12.5, 10.2))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.05], hspace=0.55, wspace=0.28)
    matrix = idx.load("results/matrix/summary.json")["rows"]
    mech = mech_rows(idx)
    first = None
    for j, (case, title) in enumerate(CASES):
        ax = fig.add_subplot(gs[0, j], sharey=first)
        first = first or ax
        sub = [r for r in matrix if r["case"] == case and proto_of(r)]
        floor = sorted({(r["ns"], r["floor"]["B"]) for r in sub if r.get("floor")})
        ax.plot(*zip(*floor), color=INK, ls=(0, (4, 3)), lw=1.2, label="projection floor")
        for proto, (plabel, color) in PROTO.items():
            for code, marker in CODE_MARK.items():
                pts = sorted((r["ns"], r["B"], bool(r["converged"])) for r in sub if proto_of(r) == proto and r["code"] == code)
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=0.8, alpha=0.6, ls="-" if code == "vmex" else ":")
                for ns, b, ok in pts:
                    ax.plot(ns, b, marker=marker, color=color, ms=5.5 if code == "vmex" else 5, mfc=color if ok else "white", mew=1.2, ls="none")
        if case in ("solovev_asymmetric", "integer_3d_tau"):
            pol = sorted((m["ns"], m["B"]) for m in mech if m["case"] == case and m["polar"] and m["baseline"] == "vmec" and m["tcon0"] == 1.0)
            if pol:
                ax.plot(*zip(*pol), color=SKY, marker="*", ms=9, lw=0.8, ls="-", mec=INK, mew=0.4)
        ax.set_yscale("log")
        ns_axis(ax)
        style(ax, "NS" if j == 2 else "", "B error, VMEX and VMEC2000" if j == 0 else "", title)
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
    tag(fig.axes[0], "a")
    # DESC
    desc = idx.load("results/desc/matched/index.json")
    dproto = {"strict": ("strict (cold)", BLUE, "o"), "loose": ("loose (cold)", GREEN, "s"), "exact": ("exact lambda seed", VERM, "^"), "pub": ("published", PURPLE, "D")}
    first = None
    for j, (case, title) in enumerate(CASES):
        ax = fig.add_subplot(gs[1, j], sharey=first)
        first = first or ax
        for proto, (plabel, color, marker) in dproto.items():
            pts = sorted((r["M"], r["B"], r["success"]) for r in desc if r["case"] == case and r["protocol"] == proto
                         and r["tag"].startswith("ladder_"))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=0.8, alpha=0.6)
            for m, b, ok in pts:
                ax.plot(m, b, marker=marker, color=color, ms=5.5, mfc=color if ok else "white", mew=1.2, ls="none")
        ax.set_yscale("log")
        ax.set_xticks([4, 6, 8, 10, 12])
        style(ax, "DESC resolution M" if j == 2 else "", "B error, DESC" if j == 0 else "", "")
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
    tag(fig.axes[len(CASES)], "b")
    # Lopez D
    ax = fig.add_subplot(gs[2, 0:2])
    early = {r["run"]: r for r in idx.load("results/lopez/summary.json")}
    final = {r["run"]: r for r in idx.load("results/iotaF/collect_final.json")}
    lm = {r["run_id"]: r for r in idx.records("mech.lopez")}

    def line(ax, runs, label, color, marker, ls="-", hollow=False):
        pts = [(ns, v) for ns, v in runs]
        ax.plot(*zip(*pts), color=color, marker=marker, ls=ls, lw=1.3, ms=6, mfc="white" if hollow else color, mew=1.2, label=label)

    line(ax, [(early[f"cold_iota_ns{n}_strict"]["ns"], early[f"cold_iota_ns{n}_strict"]["B_vol"]) for n in (17, 33, 65)],
         "VMEX cold, iota closure", BLUE, "o")
    line(ax, [(early[f"lopez_D_iota-ns{n}-vmec2000"]["ns"], early[f"lopez_D_iota-ns{n}-vmec2000"]["B_vol"]) for n in (17, 33, 65)],
         "VMEC2000 cold, iota closure", VERM, "s")
    line(ax, [(final["seed_iota_ns17_polarBase"]["ns"], final["seed_iota_ns17_polarBase"]["B_vol"]),
              (final["seed_iota_ns33_polarBase"]["ns"], final["seed_iota_ns33_polarBase"]["B_vol"])],
         "VMEX, polar-gauge exact state", GREEN, "^")
    for ident, label, marker in (("lopez_D-ns17-m10n72-polar_eta-tcon1-exact", "polar seed, exact baseline", "v"),):
        ax.plot([lm[ident]["ns"]], [lm[ident]["score"]["B_vol"]], marker=marker, color=GREEN, mfc="white", mew=1.4, ms=7, ls="none", label=label)
    eta = idx.load("results/lopez/projection_oracle_eta.json")
    read = [r for r in idx.load("results/lopez/projection_vmex_eta.json") if r["closure"] == "current"]
    line(ax, [(r["ns"], r["B_vol"]) for r in sorted(read, key=lambda r: r["ns"])], "projection, read through a WOUT", SKY, "x", ls="--")
    line(ax, [(r["ns"], r["B_vol"]) for r in eta if (r["mpol"], r["ntor"]) == (10, 72)], "projection floor, (10,72)", INK, "+", ls=":")
    for key, pattern, label, color, ls in (("lopez_mrx.ladder", "*/ladder/label_tw-1_p3_16_24_64.json", "MRX, exact-flux chart, p = 3", PURPLE, "-"),
                                           ("lopez_mrx.deck", "*/cyl/cyl_10_72_6_8_96.json", "MRX, VMEC-deck chart (10,72)", ORANGE, "--")):
        rec = idx.records(key, pattern)[0]
        ax.axhline(rec["score"]["B_rel_volume_l2"], color=color, ls=ls, lw=1.4, label=label)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([17, 33, 65], ["17", "33", "65"])
    lopez_handles = ax.get_legend_handles_labels()
    style(ax, "radial surfaces NS (nested codes); MRX lines are NS-free", "B error (whole volume)", "Lopez D recovery (iota = 1, vacuum)")
    tag(ax, "c")
    # legend panel
    axl = fig.add_subplot(gs[2, 2:5])
    axl.axis("off")
    h1 = [Line2D([], [], color=c, marker="o", ls="-", lw=1, label=l) for l, c in PROTO.values()]
    h1 += [Line2D([], [], color=SKY, marker="*", mec=INK, mew=0.4, ms=9, ls="-", lw=0.8, label="exact seed, polar gauge"),
           Line2D([], [], color=INK, ls=(0, (4, 3)), label="projection floor")]
    h2 = [Line2D([], [], color=MUTED, marker="o", ls="none", label="VMEX"), Line2D([], [], color=MUTED, marker="s", ls="none", label="VMEC2000"),
          Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none", label="open: not converged")]
    h3 = [Line2D([], [], color=c, marker=m, ls="-", lw=1, label=l) for l, c, m in dproto.values()]
    for handles, title, anchor in ((h1, "row a: protocol", (0.0, 1.0)), (h3 + h2, "row b: DESC protocol; markers", (0.31, 1.0)),
                                   (lopez_handles[0], "row c: Lopez D series", (0.66, 1.0))):
        labels = [h.get_label() for h in handles]
        leg = axl.legend(handles=handles, labels=labels, title=title, loc="upper left", frameon=False, bbox_to_anchor=anchor, borderaxespad=0,
                         alignment="left")
        axl.add_artist(leg)
    fig.text(0.5, 0.03, "B error: relative L2 against the exact field on exact-chart quadrature points, whole volume (one norm for every code). "
                        "Exact seeds of asymmetric cases use the unrotated seed unless marked polar.", ha="center", fontsize=7, color=MUTED)
    save(fig, out, "fig2_recovery_matrix")


# ---- figure 3 ---------------------------------------------------------------------------------------------------
def fig3(idx: Index, out: Path):
    wp = idx.load("results/matrix/wp_summary.json")["passes"]
    desc = idx.load("results/desc/matched/index.json")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    # group the two passes of every configuration
    groups: dict[tuple, list] = {}
    for rows in wp.values():
        for r in rows:
            groups.setdefault((r["case"], r["code"], r["protocol"], r["ns"], r["kind"], r["ftol"]), []).append(r)
    code_color = {"vmex": BLUE, "vmec2000": VERM}
    panels = [("solovev_symmetric", "Solov'ev"), ("solovev_asymmetric", "asymmetric Solov'ev"), ("integer_3d", "integer 3-D")]
    dproto = {"strict": ("strict", "o"), "loose": ("loose", "s"), "exact": ("exact seed", "^"), "pub": ("published", "D")}
    for ax, (case, title), letter in zip(axes.ravel()[:3], panels, "abc"):
        for code, color in code_color.items():
            sweep, fixed = [], []
            for (c, cd, proto, ns, kind, ftol), rs in sorted(groups.items(), key=lambda kv: -kv[0][5]):
                if c != case or cd != code:
                    continue
                xs = [r["wall_first"] if r["wall_warm_median"] is None else r["wall_warm_median"] for r in rs]
                lo, hi = min(r["wall_min"] for r in rs), max(r["wall_max"] for r in rs)
                noisy = any(r["noisy"] for r in rs)
                point = (float(np.mean(xs)), rs[0]["B"], lo, hi, noisy, ns)
                (sweep if kind == "ftol" else fixed).append(point)
            for pts, marker, ls in ((sweep, "s", "-"), (fixed, "o", "none")):
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=0.9, ls=ls, alpha=0.8)
                for x, y, lo, hi, noisy, ns in pts:
                    ax.errorbar([x], [y], xerr=[[x-lo], [hi-x]], color=color, marker=marker, ms=5.5, mfc="white" if noisy else color, mew=1.1,
                                lw=0.7, capsize=1.5, ls="none")
                    if marker == "o":
                        ax.annotate(f"{ns}", (x, y), xytext=(3, 4), textcoords="offset points", fontsize=6.5, color=color)
        for r in desc:
            if r["case"] == case and r["protocol"] in dproto and r["tag"].startswith("ladder_"):
                ax.plot(r["solve_s"], r["B"], marker=dproto[r["protocol"]][1], color=GREEN, ms=5, ls="none", mfc=GREEN if r["success"] else "white", mew=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        style(ax, "wall time per solve (s)", "B error (whole volume)", title)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
        tag(ax, letter)
    ax = axes[1, 1]
    case_mark = {"integer_3d_tau": "v", "sheared_A": "D", "sheared_issan": "P"}
    cmap = {"integer_3d_tau": VERM, "sheared_A": BLUE, "sheared_issan": PURPLE}
    for r in desc:
        if r["protocol"] in dproto and r["case"] in case_mark and r["tag"].startswith("ladder_"):
            ax.plot(r["solve_s"], r["B"], marker=dproto[r["protocol"]][1], color=cmap[r["case"]], ms=5.5, ls="none",
                    mfc=cmap[r["case"]] if r["success"] else "white", mew=1.1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, "DESC solve time (s)", "B error (whole volume)", "DESC only: integer 3-D tau = 0.5, sheared A, sheared Issan")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks([10, 20, 50, 100, 200, 500])
    tag(ax, "d")
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ls="none", label="VMEX, fixed configuration (label: NS)"),
                               Line2D([], [], color=BLUE, marker="s", ls="-", lw=0.9, label="VMEX, tolerance sweep"),
                               Line2D([], [], color=VERM, marker="o", ls="none", label="VMEC2000, fixed configuration"),
                               Line2D([], [], color=VERM, marker="s", ls="-", lw=0.9, label="VMEC2000, tolerance sweep"),
                               Line2D([], [], color=GREEN, marker="o", ls="none", label="DESC, M ladders (marker: protocol)"),
                               Line2D([], [], color=MUTED, marker="o", mfc="white", ls="none", label="open: noisy (VMEX/VMEC2000) or iteration cap (DESC)")],
                      frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    axes[1, 1].legend(handles=[Line2D([], [], color=c, marker="o", ls="none", label=k.replace("_", " ")) for k, c in cmap.items()] +
                      [Line2D([], [], color=MUTED, marker=m, ls="none", label=l) for l, m in (v for v in dproto.values())], frameon=False, ncol=2, loc="upper right")
    loads = [r["load1_range"] for rows in wp.values() for r in rows]
    fig.text(0.5, -0.115, "Shared 36-core office box, 4 OpenMP threads, CPU. One-minute load during the VMEX/VMEC2000 runs: "
                          f"{min(l[0] for l in loads):.0f} to {max(l[1] for l in loads):.0f}; DESC runs 7 to 60. Bars: min-max of 2-3 repeats and both passes; "
                          "VMEX includes XLA compilation in the first repeat; time ratios are not resolved where load differs.",
             ha="center", fontsize=7, color=MUTED)
    fig.tight_layout()
    save(fig, out, "fig3_work_precision")


# ---- figure 4 ---------------------------------------------------------------------------------------------------
def fig4(idx: Index, out: Path):
    regions = idx.load("results/sensitivity/regions.json")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fine = {r["ns"]: r for r in regions if r["case"] == "integer_axisymmetric" and r["tag"] in ("", "_fine") and
            "volume_fine" in r["params"]["c"]}
    base = {r["ns"]: r for r in regions if r["case"] == "integer_axisymmetric" and r["tag"] == ""}
    nss = [33, 65, 129, 257]
    for ax, name, gate_value, label, letter in ((axes[0], "c", 1e-3, "nonzero response, c: relative RMS error of dB", "a"),
                                                 (axes[1], "delta", 1e-5, "null response, delta: edge x RMS(dB) (T)", "b")):
        def get(ns, obs, part):
            src = fine.get(ns) if obs == "volume_fine" else base[ns]
            return src["params"][name][obs][part]
        series = [("whole volume (s <= 0.9), fine rule", "volume_fine", "all", VERM, "o", "-"), ("whole 80-point cloud", "cloud", "all", ORANGE, "s", "--"),
                  ("bulk (s >= 0.1), fine rule", "volume_fine", "bulk", BLUE, "o", "-"), ("bulk, 80-point cloud", "cloud", "bulk", SKY, "s", "--")]
        for lab, obs, part, color, marker, ls in series:
            xs = [n for n in nss if (n in fine if obs == "volume_fine" else n in base)]
            ax.plot(xs, [get(n, obs, part) for n in xs], marker=marker, color=color, ls=ls, lw=1.4, ms=5.5, label=lab)
        gate(ax, gate_value, f"pre-registered gate {gate_value:g}")
        ax.set_yscale("log")
        ns_axis(ax, nss)
        if letter == "a":
            ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.3))
        style(ax, "radial surfaces NS", label, "Axisymmetric integer family")
        tag(ax, letter)
    # 3-D
    ax = axes[2]
    tagged = {r["ns"]: r for r in idx.records("sens.3d", "*/integer_3d_ns*.json") if r["ns"] in (17, 33) and r["tag"] == ""}
    names = [("c", "relative"), ("a", "relative"), ("delta", "null")]
    width = 0.19
    for i, (name, kind) in enumerate(names):
        for k, ns in enumerate((17, 33)):
            e = tagged[ns]["params"][name]
            field = "relative_error" if kind == "relative" else "null_scaled_rms"
            jvp = e["jvp"]["volume"][field]
            fd = min(f["scores"]["volume"][field] for f in e["fd"])
            x = i + (k - 0.5)*2*width
            ax.bar(x - width/2, jvp, width, color=VERM if k else "#e9a98a", label=None)
            ax.bar(x + width/2, fd, width, color=BLUE if k else "#8fbfe0", label=None)
        ax.annotate("", (i, 1), alpha=0)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 2e1)
    ax.set_xticks(range(3), ["c (relative)", "a (relative)", "delta (null, T)"])
    ax.axhline(1e-3, color=MUTED, ls=(0, (4, 3)), lw=0.9)
    ax.annotate("gate c, a: 1e-3 (delta: 1e-5)", xy=(0.01, 1e-3), xycoords=("axes fraction", "data"), ha="left", va="top", fontsize=7, color=MUTED)
    ax.legend(handles=[matplotlib.patches.Patch(color="#e9a98a", label="tangent, NS 17"), matplotlib.patches.Patch(color=VERM, label="tangent, NS 33"),
                       matplotlib.patches.Patch(color="#8fbfe0", label="reconverged FD, best step, NS 17"),
                       matplotlib.patches.Patch(color=BLUE, label="reconverged FD, best step, NS 33")], frameon=False, loc="upper right", ncol=1)
    style(ax, "", "response error against the exact derivative (whole volume)", "3-D integer family, accepted roots")
    tag(ax, "c")
    fig.tight_layout()
    save(fig, out, "fig4_sensitivities")


# ---- figure 5 ---------------------------------------------------------------------------------------------------
def fig5(idx: Index, out: Path):
    rows = mech_rows(idx)
    matrix = idx.load("results/matrix/summary.json")["rows"]
    sym = [r for r in rows if r["case"] == "solovev_symmetric"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    ax = axes[0]
    nss = [17, 33, 65, 129]

    def series(baseline, tcon):
        d = {r["ns"]: r["B"] for r in sym if r["baseline"] == baseline and abs(r["tcon0"]-tcon) < 1e-12 and r["id"].count("-") == 3}
        return [(n, d[n]) for n in nss if n in d]
    ax.plot(*zip(*series("vmec", 1.0)), color=VERM, marker="o", lw=1.5, ms=6, label="default baseline (tcon0 = 1)")
    ax.plot(*zip(*series("exact", 1.0)), color=BLUE, marker="s", lw=1.5, ms=6, label="exact baseline (tcon0 = 1)")
    ax.plot(*zip(*series("vmec", 0.0)), color=GREEN, marker="^", lw=1.5, ms=6, ls="--", label="tcon0 = 0")
    floor = sorted({(r["ns"], r["floor"]["B"]) for r in matrix if r["case"] == "solovev_symmetric" and r.get("floor")})
    ax.plot(*zip(*floor), color=INK, ls=":", lw=1.2, label="projection floor")
    base = np.array([17.0, 129.0])
    ax.plot(base, 6.5e-5*(base/17.0)**-1.7, color=MUTED, lw=0.8, ls="-.")
    ax.annotate("NS$^{-1.7}$", (60, 1.1e-5), color=MUTED, fontsize=7.5)
    ax.set_yscale("log")
    ns_axis(ax, nss)
    ax.legend(frameon=False, loc="lower left")
    style(ax, "radial surfaces NS", "B error (whole volume)", "Solved error against NS, exact seed")
    tag(ax, "a")
    # dose response
    ax = axes[1]
    for ns, ls, marker in ((65, "-", "o"), (129, "--", "s")):
        for baseline, color in (("vmec", VERM), ("exact", BLUE)):
            pts = sorted((r["tcon0"], r["B"]) for r in sym if r["ns"] == ns and r["baseline"] == baseline and r["id"].count("-") == 3)
            if baseline == "exact" and pts[0][0] > 0:
                pts.insert(0, (0.0, next(r["B"] for r in sym if r["ns"] == ns and r["baseline"] == "vmec" and r["tcon0"] == 0.0 and r["id"].count("-") == 3)))
            ax.plot([max(t, 3e-3) for t, _ in pts], [b for _, b in pts], color=color, ls=ls, marker=marker, lw=1.4, ms=5.5,
                    label=f"{'default' if baseline == 'vmec' else 'exact'} baseline, NS {ns}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0], ["0", "0.01", "0.03", "0.1", "0.3", "1"])
    ax.legend(frameon=False, loc="upper left")
    style(ax, "constraint strength tcon0 (0 drawn at 0.003)", "B error (whole volume)", "Dose response")
    tag(ax, "b")
    # linear attribution
    ax = axes[2]
    lin = {(r["ns"], r["baseline"]): r for r in idx.records("mech.linear", "*/solovev_symmetric-ns*-tcon1-*.json") if r["case"] == "solovev_symmetric" and r["tcon0"] == 1.0}
    for baseline, color, name in (("vmec", VERM, "default baseline"), ("exact", BLUE, "exact baseline")):
        ns = [n for n in nss if (n, baseline) in lin]
        solved = {r["ns"]: r["B"] for r in sym if r["baseline"] == baseline and abs(r["tcon0"]-1.0) < 1e-12 and r["id"].count("-") == 3}
        ax.plot(ns, [solved[n] for n in ns], color=color, lw=1.6, marker="o", ms=6, label=f"{name}: nonlinear solve")
        ax.plot(ns, [lin[(n, baseline)]["scores"]["total"]["B"] for n in ns], color=color, lw=0, marker="x", ms=7, mew=1.6, label=f"{name}: linear prediction")
        ax.plot(ns, [lin[(n, baseline)]["scores"]["con"]["B"] for n in ns], color=color, lw=1.0, ls="--", marker="v", ms=4.5, mfc="white", label=f"{name}: constraint part")
    mhd = [lin[(n, "exact")]["scores"]["mhd"]["B"] for n in nss]
    ax.plot(nss, mhd, color=GREEN, lw=1.2, ls="-.", marker="D", ms=4.5, label="MHD part (same for both)")
    ax.set_yscale("log")
    ns_axis(ax, nss)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    style(ax, "radial surfaces NS", "B error of x* + delta (whole volume)", "Linear attribution of the first-order error")
    tag(ax, "c")
    fig.tight_layout()
    save(fig, out, "fig5_constraint_bias")


# ---- figure 6 ---------------------------------------------------------------------------------------------------
def fig6(idx: Index, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    # (a) iteration-1 residual
    ax = axes[0]
    own = idx.load("results/mechanism/asym_residual.json")
    pol = idx.load("results/mechanism/asym_residual_polar.json")
    named = idx.load("results/mechanism/asym_residual_named.json")
    named_p = idx.load("results/mechanism/asym_residual_named_polar.json")

    def fsq(r):
        return r["tcon0_0"]["fsqr"] + r["tcon0_0"]["fsqz"]
    sel = lambda rows, pre: sorted((r["ns"], fsq(r)) for r in rows if re.fullmatch(pre, r["label"]))
    ax.plot(*zip(*sel(own, r"asym chi0\.3 ns\d+")), color=VERM, marker="o", lw=1.5, ms=6, label="asymmetric Solov'ev, own angle")
    ax.plot(*zip(*sel(pol, r"asym chi0\.3 ns\d+ polar")), color=BLUE, marker="s", lw=1.5, ms=6, label="asymmetric Solov'ev, polar gauge")
    ax.plot(*zip(*sel(own, r"symdata full-basis ns\d+")), color=GREEN, marker="^", lw=1.2, ms=5.5, ls="--", label="symmetric data, full basis")
    ax.plot(*zip(*sel(named, r"integer_3d_tau ns\d+")), color=ORANGE, marker="v", lw=1.2, ms=6, ls="-.", label="integer 3-D tau = 0.5, own angle")
    ax.plot(*zip(*sel(named_p, r"integer_3d_tau ns\d+ polar")), color=SKY, marker="D", lw=1.2, ms=5, ls="-.", label="integer 3-D tau = 0.5, polar gauge")
    ax.set_yscale("log")
    ns_axis(ax, [17, 33, 65, 129])
    ax.legend(frameon=False, loc="center right", fontsize=7)
    style(ax, "radial surfaces NS", "iteration-1 residual fsqr + fsqz, tcon0 = 0", "Residual of the exact seed")
    tag(ax, "a")
    # (b) solved B
    ax = axes[1]
    rows = mech_rows(idx)

    def solved(case, polar, tcon, baseline="vmec"):
        d = sorted((r["ns"], r["B"]) for r in rows if r["case"] == case and r["polar"] == polar and abs(r["tcon0"]-tcon) < 1e-12
                   and r["baseline"] == baseline and r["id"].replace("-polar", "").count("-") == 3)
        return d
    ax.plot(*zip(*solved("solovev_asymmetric", False, 0.0)), color=VERM, marker="o", lw=1.5, ms=6, label="asymmetric, own angle, tcon0 = 0")
    ax.plot(*zip(*solved("solovev_asymmetric", True, 0.0)), color=BLUE, marker="s", lw=1.5, ms=6, label="asymmetric, polar gauge, tcon0 = 0")
    ax.plot(*zip(*solved("solovev_symmetric", False, 0.0)), color=GREEN, marker="^", lw=1.2, ms=5.5, ls="--", label="symmetric, tcon0 = 0")
    ax.plot(*zip(*solved("solovev_asymmetric", False, 1.0)), color=VERM, marker="o", lw=1.0, ms=5, mfc="white", ls=":", label="asymmetric, own angle, default")
    ax.plot(*zip(*solved("solovev_asymmetric", True, 1.0)), color=BLUE, marker="s", lw=1.0, ms=5, mfc="white", ls=":", label="asymmetric, polar gauge, default")
    ax.set_yscale("log")
    ns_axis(ax, [17, 33, 65, 129])
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    style(ax, "radial surfaces NS", "B error (whole volume)", "Solved error, exact seed")
    tag(ax, "b")
    # (c) eigenvalue counts
    ax = axes[2]
    eig = idx.records("mech.eigen")
    groups = [("integer 3-D", "integer_3d", (9, 13, 17), False), ("sheared A", "sheared_A", (9, 13, 17), False),
              ("asym. Solov'ev, own angle", "solovev_asymmetric", (17, 33), False), ("asym. Solov'ev, polar", "solovev_asymmetric", (17, 33), True)]
    pos = 0
    ticks, labels = [], []
    w = 0.38
    for title, case, nss, polar in groups:
        for ns in nss:
            for off, baseline, color in ((-w/2, "vmec", VERM), (w/2, "exact", BLUE)):
                if case == "solovev_asymmetric" and baseline == "exact":
                    continue
                match = [r for r in eig if r["case"] == case and r["ns"] == ns and r["baseline"] == baseline and r["tcon0"] == 1.0
                         and bool(r.get("polar", False)) == polar and r.get("ai0_override") is None]
                if not match:
                    continue
                count = match[0]["n_positive_real"]
                ax.bar(pos + (off if case != "solovev_asymmetric" else 0), count, w if case != "solovev_asymmetric" else 2*w*0.7, color=color)
                ax.annotate(str(count), (pos + (off if case != "solovev_asymmetric" else 0), count), ha="center", va="bottom", fontsize=7, color=INK,
                            xytext=(0, 2), textcoords="offset points")
            ticks.append(pos)
            labels.append(f"NS {ns}")
            pos += 1
        pos += 0.8
    ax.set_xticks(ticks, labels, rotation=60, fontsize=7)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, 600)
    ax.legend(handles=[matplotlib.patches.Patch(color=VERM, label="default baseline (tcon0 = 1)"), matplotlib.patches.Patch(color=BLUE, label="exact baseline")],
              frameon=False, loc="upper right")
    y0 = -0.38
    spans = [(0, 2, "integer 3-D"), (3.8, 5.8, "sheared A"), (7.6, 8.4, "asym. Solov'ev\n(own angle)"), (10.6, 11.4, "asym. Solov'ev\n(polar)")]
    for lo, hi, text in spans:
        ax.annotate(text, xy=((lo+hi)/2, y0), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=6.5, color=INK)
    style(ax, "", "eigenvalues with positive real part", "Flow Jacobian at the exact seed")
    ax.xaxis.set_tick_params(pad=1)
    tag(ax, "c")
    fig.tight_layout()
    save(fig, out, "fig6_seed_gauge_stability")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6}
NAMES = {"fig1": "fig1_references_projection", "fig2": "fig2_recovery_matrix", "fig3": "fig3_work_precision",
         "fig4": "fig4_sensitivities", "fig5": "fig5_constraint_bias", "fig6": "fig6_seed_gauge_stability"}


def build(out: Path, only=None, index_path: Path = INDEX):
    out.mkdir(parents=True, exist_ok=True)
    idx = Index(index_path)
    manifest = {"generator": "benchmarks/paper_figures.py", "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "index": "results/index.json", "index_sha256": idx.sha256(), "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                "figures": {}}
    for key, fn in FIGS.items():
        if only and key not in only:
            continue
        idx.touched = set()
        fn(idx, out)
        name = NAMES[key]
        declared = sorted({k for p in idx.data["figures"][name]["panels"].values() for k in p["records"]})
        manifest["figures"][name] = {
            "records_used": idx.used_keys(), "records_declared": declared,
            "inputs": {p: idx.checked[p] for p in sorted(idx.touched)},
            "outputs": {f"{name}.{ext}": hashlib.sha256((out/f"{name}.{ext}").read_bytes()).hexdigest() for ext in ("png", "pdf")}}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=1)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT/"figures/paper")
    parser.add_argument("--only", nargs="*", choices=sorted(FIGS))
    args = parser.parse_args(argv)
    manifest = build(args.out, args.only)
    for name, info in manifest["figures"].items():
        print(f"{name}: {len(info['inputs'])} record files, records {', '.join(info['records_used'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
