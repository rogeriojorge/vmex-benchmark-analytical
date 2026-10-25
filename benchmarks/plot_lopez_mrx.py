"""Figures of the MRX study of Lopez D (small PNG; data are the JSON/NPZ records in results/lopez_mrx).

    python plot_lopez_mrx.py ladder|perturb|closure|sections|all
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"results/lopez_mrx"
FIG = ROOT/"figures/lopez_mrx"


def load(pattern):
    return [json.loads(p.read_text()) for p in sorted(RES.glob(pattern))]


def fig_ladder():
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for p, mk in ((2, "s"), (3, "o"), (4, "^")):
        rows = sorted(load("ladder/label_tw-1_p%d_*.json" % p), key=lambda r: r["args"]["ns"][0])
        if not rows:
            continue
        n = np.array([r["args"]["ns"][0] for r in rows], float)
        ax[0].loglog(n, [r["score"]["B_rel_volume_l2"] for r in rows], mk + "-", label="exact-flux-surface chart, p=%d" % p)
        ax[1].loglog(n, [r["closure"]["return_defect_max"] for r in rows], mk + "-", label="p=%d" % p)
    deck = [r for r in load("cyl/cyl_10_72_*.json") if r["closure"]["rotation_dev_max"] < 1]
    ax[0].loglog([r["args"]["ns"][0] + 0.0 for r in deck], [r["score"]["B_rel_volume_l2"] for r in deck], "k*", ms=9, label="VMEC deck chart (10,72), p=3")
    ax[1].loglog([r["args"]["ns"][0] for r in deck], [r["closure"]["return_defect_max"] for r in deck], "k*", ms=9, label="VMEC deck chart (10,72), p=3")
    ax[0].axhline(8e-3, color="r", ls="--", lw=1); ax[0].text(6.1, 9e-3, "VMEC cold-start plateau (0.8%)", color="r", fontsize=8)
    ax[0].axhline(1.4e-3, color="m", ls=":", lw=1); ax[0].text(6.1, 1.55e-3, "VMEC polar-seed state (1.4e-3)", color="m", fontsize=8)
    ax[0].axhline(6e-5, color="g", ls="-.", lw=1); ax[0].text(6.1, 6.6e-5, "VMEC projection floor at (10,72)", color="g", fontsize=8)
    ax[0].set_xlabel("radial splines n_r (angular splines scale with it)"); ax[0].set_ylabel("B error vs exact, volume L2")
    ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_title("fixed-boundary vacuum solve vs exact field")
    ax[1].set_xlabel("radial splines n_r"); ax[1].set_ylabel("max one-turn return displacement P(y)-y"); ax[1].legend(fontsize=7)
    ax[1].set_title("closed lines (iota = 1)")
    tr = sorted(load("cyl/cyl_*_6_8_96.json"), key=lambda r: r["boundary_truncation_error"])
    e = np.array([r["boundary_truncation_error"] for r in tr])
    ax[2].loglog(e, [r["score"]["B_rel_volume_l2"] for r in tr], "ko-", label="B error (volume L2)")
    ax[2].loglog(e, [r["closure"]["return_defect_max"] for r in tr], "rs-", label="one-turn defect")
    ax[2].loglog(e, e/0.0097*0.5, "k:", lw=0.8, label="0.5 x truncation / minor radius")
    for r, x in zip(tr, e):
        ax[2].annotate("(%d,%d)" % (r["args"]["mpol"], r["args"]["ntor"]), (x, r["score"]["B_rel_volume_l2"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[2].set_xlabel("max boundary truncation error of the deck (length)"); ax[2].legend(fontsize=7)
    ax[2].set_title("VMEC deck (mpol, ntor) truncation, MRX 6x8x96")
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG/"ladder.png", dpi=110); plt.close(fig)


def _ham_cases():
    cases = {}
    for p in sorted((RES/"ham").glob("e*_m*n*_ns*.json")):
        r = json.loads(p.read_text())
        a = r["args"]
        cases.setdefault((a["m"], a["n"], tuple(a["ns"])), []).append(r)
    return cases


def fig_perturb():
    cases = _ham_cases()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {}
    for (m, n, ns), rows in sorted(cases.items()):
        rows.sort(key=lambda r: r["args"]["eps"])
        eps = [r["args"]["eps"] for r in rows]
        col = colors.setdefault((m, n), "C%d" % len(colors))
        ls_ = "-" if ns[0] == 12 else "--"
        lab = "(m,n)=(%d,%d)" % (m, n) if ns[0] == 12 else None
        ax[0].loglog(eps, [r["one_turn"]["defect_max"] for r in rows], "o" + ls_, color=col, label=lab, ms=4)
        ax[1].loglog(eps, [max(abs(r["one_turn"]["rotation_dev_min"]), abs(r["one_turn"]["rotation_dev_max"])) for r in rows], "o" + ls_, color=col, label=lab, ms=4)
        ax[2].loglog(eps, [r["score"].get("B_rel_volume_l2_scaled", np.nan) for r in rows], "o" + ls_, color=col, label=lab, ms=4)
    for k, (key, rows) in enumerate(sorted((k, v) for k, v in cases.items() if False)):
        pass
    for p in sorted((RES/"ham").glob("base_ns*.json")):
        r = json.loads(p.read_text())
        ls_ = "-" if r["args"]["ns"][0] == 12 else "--"
        ax[0].axhline(r["one_turn"]["defect_max"], color="k", ls=":" if ls_ == "--" else "-.", lw=0.8)
        ax[1].axhline(max(abs(r["one_turn"]["rotation_dev_min"]), abs(r["one_turn"]["rotation_dev_max"])), color="k", ls=":" if ls_ == "--" else "-.", lw=0.8)
        ax[2].axhline(r["score"].get("B_rel_volume_l2_scaled", np.nan), color="k", ls=":" if ls_ == "--" else "-.", lw=0.8)
    ax[0].set_ylabel("max |P(y) - y| (logical radius = 1)"); ax[1].set_ylabel("max |iota - 1|")
    ax[2].set_ylabel("B shape change vs undeformed exact")
    for a_ in ax:
        a_.set_xlabel("boundary deformation eps"); a_.legend(fontsize=8)
    ax[0].set_title("one-turn return displacement"); ax[1].set_title("rotational transform departs from 1")
    ax[2].set_title("field-shape change of the deformed domain")
    fig.text(0.5, 0.005, "solid: MRX 12x20x48, dashed: 8x12x24; horizontal lines: undeformed domain (discretisation floor); (m,n) = (2,0) is a union of exact lines (control)", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG/"perturb_scaling.png", dpi=110); plt.close(fig)


def fig_ham(tags, out="perturb_hamiltonian.png", directory="ham"):
    """Effective Hamiltonian of the one-turn map (contours), O/X fixed points and the strobe plot for chosen runs."""
    fig, ax = plt.subplots(2, len(tags), figsize=(3.9*len(tags), 7.6))
    ax = np.atleast_2d(ax)
    for j, tag in enumerate(tags):
        z = np.load(RES/directory/f"{tag}.npz")
        r = json.loads((RES/directory/f"{tag}.json").read_text())
        u, H = z["hgrid_u"], z["hgrid_H"]
        a = ax[0, j]
        a.contour(u, u, H.T, 30, cmap="viridis", linewidths=0.8)
        for q in r["effective_hamiltonian"]["points"]:
            a.plot(q["u"], q["v"], "ro" if q["kind"] == "O" else "bx", ms=5)
        th = np.linspace(0, 2*np.pi, 200); a.plot(np.cos(th), np.sin(th), "k-", lw=0.6)
        a.set_aspect("equal"); a.set_title("%s\nH of P-id; O %d X %d, D_max %.1e" % (tag.split("_ns")[0], r["effective_hamiltonian"]["n_O"], r["effective_hamiltonian"]["n_X"], r["one_turn"]["defect_max"]), fontsize=8)
        b = ax[1, j]
        for line in z["strobe"]:
            b.plot(line[:, 0], line[:, 1], ".", ms=1.2)
        b.plot(np.cos(th), np.sin(th), "k-", lw=0.6); b.set_aspect("equal"); b.set_title("strobe, %d turns" % r["strobe"]["turns"], fontsize=8)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG/out, dpi=100); plt.close(fig)


def fig_sections(names, out="perturb_sections.png"):
    fig, ax = plt.subplots(2, len(names), figsize=(4.2*len(names), 8))
    ax = np.atleast_2d(ax)
    for j, name in enumerate(names):
        z = np.load(RES/f"perturb/{name}.npz")
        r = json.loads((RES/f"perturb/{name}.json").read_text())
        a = ax[0, j]
        pts = z["strobe"]
        for line in pts:
            a.plot(line[:, 0], line[:, 1], ".", ms=1.5)
        th = np.linspace(0, 2*np.pi, 200); a.plot(np.cos(th), np.sin(th), "k-", lw=0.6)
        fx = z["fixed"]; fk = z["fixed_kind"]
        if len(fx):
            a.plot(fx[fk == "O", 0], fx[fk == "O", 1], "ro", ms=5, mfc="none"); a.plot(fx[fk == "X", 0], fx[fk == "X", 1], "bx", ms=6)
        a.set_aspect("equal"); a.set_title("%s: strobe, %d turns\nO %d X %d" % (name.split("_ns")[0], r["strobe"]["turns"], r["n_O"], r["n_X"]), fontsize=9)
        b = ax[1, j]
        prof = r["rotation_profile"]
        b.fill_between(prof["r"], prof["dev_min"], prof["dev_max"], alpha=.4)
        b.plot(prof["r"], prof["dev_min"], "k.-", lw=.6); b.plot(prof["r"], prof["dev_max"], "k.-", lw=.6)
        b.axhline(0, color="r", lw=.8); b.set_xlabel("logical radius"); b.set_ylabel("iota - 1 (min..max over angle)")
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG/out, dpi=100); plt.close(fig)


def fig_vmec_sections(phis=(0.0, 1.7, 3.6), out="vmec_sections.png"):
    """(R, Z) cuts of the exact flux surfaces and of settled VMEC states at fixed geometric phi."""
    import lopez_surfaces as ls
    import score_lopez as sl
    W = Path.home()/"local/vmex-bench-work/mrxJ-wouts"
    states = [("cold, iota closure (NS 17)", W/"cold_iota_ns17.nc", "tab:red"),
              ("cold, iota closure (NS 65)", W/"cold_iota_ns65.nc", "tab:orange"),
              ("exact seed, polar gauge (NS 17)", W/"seed_polar_iota_ns17.nc", "tab:blue")]
    sigma, c = ls.gauge(ls.OPT_FRAME)
    fig, ax = plt.subplots(1, len(phis), figsize=(4.6*len(phis), 4.6))
    th = np.linspace(0, 2*np.pi, 181)
    for k, phi in enumerate(phis):
        a = ax[k]
        for s in (0.25, 0.5, 1.0):
            R, Z = ls.boundary_rz(th, np.full_like(th, phi), ls.OPT_FRAME, sigma, c, s=s)
            a.plot(R, Z, "k-", lw=1.0, label="exact" if s == 0.25 else None)
        for name, path, col in states:
            if not path.exists():
                continue
            w = sl.read_wout(path)
            grid = (64, 256)
            f = sl.evaluate_surfaces(w, "full", grid)
            iz = int(round(phi/(2*np.pi)*grid[1])) % grid[1]
            for s in (0.25, 0.5, 1.0):
                j = int(round(s*(w.ns - 1)))
                a.plot(np.append(f.R[j, :, iz], f.R[j, 0, iz]), np.append(f.Z[j, :, iz], f.Z[j, 0, iz]), "-", color=col, lw=0.8,
                       label=name if s == 0.25 else None)
        a.set_aspect("equal"); a.set_title("phi = %.1f rad: s = 0.25, 0.5, 1" % phi, fontsize=9)
        a.set_xlabel("R"); a.set_ylabel("Z")
        if k == 0:
            a.legend(fontsize=7)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG/out, dpi=100); plt.close(fig)


def fig_closure():
    """Closed-line statistics of the VMEC states (circle-map spread, normal fraction) against the MRX defects."""
    rec = json.loads((RES/"wout_lines.json").read_text())
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for name, r in rec.items():
        s = [x["s"] for x in r["lines"]]
        ax[0].semilogy(s, [max(x["rotation_spread"], 1e-9) for x in r["lines"]], "o-", label=name)
        ax[1].semilogy([x["s"] for x in r["exact_on_surfaces"]], [x["normal_fraction_rms"] for x in r["exact_on_surfaces"]], "o-", label=name)
        ax[2].plot([x["s"] for x in r["exact_on_surfaces"]], [x["label_spread_max"] for x in r["exact_on_surfaces"]], "o-", label=name)
    lad = load("ladder/label_tw-1_p3_*.json")
    if lad:
        ax[0].axhspan(min(r["closure"]["return_defect_max"] for r in lad), max(r["closure"]["return_defect_max"] for r in lad),
                      color="gray", alpha=.3, label="MRX one-turn defect (ladder range)")
    ax[0].set_ylabel("spread over starts of the one-turn rotation"); ax[0].set_title("closed lines of each state's own field")
    ax[1].set_ylabel("||B_exact . n|| / ||B_exact|| on the surface"); ax[1].set_title("is the surface a union of exact lines?")
    ax[2].set_ylabel("max |exact flux label - s| on the surface"); ax[2].set_title("same surface, exact flux label")
    for a in ax:
        a.set_xlabel("s"); a.legend(fontsize=6)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG/"closure_vmec_vs_mrx.png", dpi=105); plt.close(fig)


def fig_cylham(out="deck_hamiltonian.png"):
    """Effective Hamiltonian of the one-turn map of the true vacuum field in the VMEC-deck domains."""
    tags = ["6_24", "8_40", "10_56", "10_72"]
    fig, ax = plt.subplots(1, len(tags), figsize=(3.9*len(tags), 4.2))
    for a, t in zip(ax, tags):
        z = np.load(RES/f"cylham/cyl_{t}_6_8_96_ham.npz")
        r = json.loads((RES/f"cylham/cyl_{t}_6_8_96.json").read_text())
        h = r["effective_hamiltonian"]
        a.contour(z["hgrid_u"], z["hgrid_u"], z["hgrid_H"].T, 30, cmap="viridis", linewidths=0.8)
        for q in h["points"]:
            a.plot(q["u"], q["v"], "ro" if q["kind"] == "O" else "bx", ms=5)
        th = np.linspace(0, 2*np.pi, 200); a.plot(np.cos(th), np.sin(th), "k-", lw=0.6); a.set_aspect("equal")
        a.set_title("deck (%s): boundary err %.1e\nB err %.1e, one-turn defect %.1e\nO %d X %d" % (
            t.replace("_", ","), r["boundary_truncation_error"], r["score"]["B_rel_volume_l2"], r["closure"]["return_defect_max"], h["n_O"], h["n_X"]), fontsize=8)
    fig.tight_layout(); FIG.mkdir(parents=True, exist_ok=True); fig.savefig(FIG/out, dpi=100); plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("ladder", "all"):
        fig_ladder()
    if what in ("perturb", "all"):
        fig_perturb()
        fig_ham(["e0p1_m2n1_ns12_20_48", "e0p1_m3n1_ns12_20_48", "e0p05_m2n2_ns12_20_48", "e0p1_m2n0_ns12_20_48"])
    if what in ("deck", "all"):
        fig_cylham()
    if what in ("closure", "all"):
        fig_closure()
    if what in ("sections", "all"):
        fig_vmec_sections()
