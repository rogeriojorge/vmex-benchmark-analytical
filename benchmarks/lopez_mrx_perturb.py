"""Island formation of the Lopez D vacuum field under a small boundary deformation, in MRX.

The deformation scales the label radius by 1 + eps rho^m cos(2 pi (m t - n zeta + phase)) (t the label
angle, n counting source-phase periods): a boundary that is no longer a union of field lines unless n = 0
(n = 0 is a control: the exact field is still the vacuum field of the deformed domain). The harmonic
2-form of the deformed solid torus is computed (one linear solve) and the one-turn return map P of its
field lines on the zeta = 0 cut is analysed in the logical (u, v) chart:

* D = P - id on a polar grid (max / rms, in units of the logical radius) and the rotation per turn of every
  line about the axis (iota; the chart is twisted so unperturbed lines rotate once);
* the fixed points of P (zeros of D) by Newton, O / X by Greene's residue, and their count (a map of the
  disk isotopic to the identity has #O - #X = 1 inside);
* a stroboscopic Poincare plot (points at integer turns) of lines over many turns.

    python lopez_mrx_perturb.py --eps 1e-3 --m 1 --n 1 --ns 10 16 32 --out results/lopez_mrx/perturb/x
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lopez_mrx as lm  # noqa: E402


def one_turn_field(seq, dof, n_r=10, n_th=24, steps=48):
    seeds = lm.disk_seeds(n_r, n_th, r_min=0.05, r_max=0.95, offset=0.0)
    ys, ok = lm.trace_lines(seq, dof, seeds, 1, steps)
    return seeds, ys[:, 0], ys[:, steps]


def fixed_points_scan(seq, dof, n_r=7, n_th=18, steps=96, iters=25, r_max=0.93):
    from mrx.diagnostics.islands import fixed_points
    r = np.linspace(0.1, r_max, n_r)
    th = (np.arange(n_th) + 0.31)/n_th
    g = np.stack(np.broadcast_arrays(r[:, None], th[None, :]), -1).reshape(-1, 2)
    fp = fixed_points(seq, dof, 1, g, steps_per_period=steps, iters=iters, step_cap=0.03)
    pts = []
    for k in range(len(g)):
        if fp["defect"][k] < 1e-10 and fp["r"][k] < r_max + 0.02:
            if all(np.hypot(*(fp["uv"][k] - p["uv"])) > 2e-3 for p in pts):
                pts.append(dict(uv=np.asarray(fp["uv"][k]), r=float(fp["r"][k]), theta=float(fp["theta"][k]),
                                residue=float(fp["residue"][k]), kind=str(fp["kind"][k])))
    return pts


def strobe(seq, dof, turns, steps, n_lines=24, r_max=0.9):
    seeds = lm.disk_seeds(n_lines//4, 4, r_min=0.1, r_max=r_max, offset=0.13)
    ys, ok = lm.trace_lines(seq, dof, seeds, turns, steps)
    return seeds, ys[:, ::steps], ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--twist", type=int, default=-1)
    ap.add_argument("--ns", type=int, nargs=3, default=(10, 16, 32))
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--turns", type=int, default=400)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--islands", action="store_true", help="MRX island-chain widths (Poincare of --lines lines, --periods turns)")
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--lines", type=int, default=80)
    ap.add_argument("--periods", type=int, default=200)
    ap.add_argument("--out", required=True, help="path prefix; writes .json and .npz")
    a = ap.parse_args(argv)
    t0 = time.time()
    pert = ((a.eps, a.m, a.n, a.phase),) if a.eps else ()
    seq = lm.build_sequence(a.ns, a.p, lm.LabelMap(pert=pert, twist=a.twist))
    dof, spread = lm.vacuum_field(seq)
    rec = dict(args=vars(a), flux_cut_spread=spread, built_s=time.time() - t0)
    rec["score"] = lm.score(seq, dof)
    seeds, y0, y1 = one_turn_field(seq, dof)
    d = np.linalg.norm(y1 - y0, axis=-1)
    exp_rot = float(-a.twist)
    pr = lm.disk_seeds(8, 8)
    ys, ok = lm.trace_lines(seq, dof, np.concatenate([[[0.01, 0.0]], pr]), 20, 48)
    rot = lm.rotation_per_turn(ys[1:], 48, lm.axis_track(ys[0], 48))
    rec["rotation_profile"] = dict(r=sorted(set(np.round(pr[:, 0], 6).tolist())),
                                   dev_min=[float(rot[np.isclose(pr[:, 0], r)].min() - exp_rot) for r in sorted(set(pr[:, 0]))],
                                   dev_max=[float(rot[np.isclose(pr[:, 0], r)].max() - exp_rot) for r in sorted(set(pr[:, 0]))])
    rec["one_turn"] = dict(defect_max=float(d.max()), defect_rms=float(np.sqrt(np.mean(d**2))),
                           rotation_dev_min=float((rot - exp_rot).min()), rotation_dev_max=float((rot - exp_rot).max()))
    fps = fixed_points_scan(seq, dof)
    rec["fixed_points"] = [dict(r=p["r"], theta=p["theta"], residue=p["residue"], kind=p["kind"]) for p in fps]
    rec["n_O"] = sum(p["kind"] == "O" for p in fps)
    rec["n_X"] = sum(p["kind"] == "X" for p in fps)
    u, v, Dg = lm.one_turn_grid(seq, dof, n=a.grid)
    H = lm.effective_hamiltonian(u, v, Dg)
    cps = lm.critical_points(u, v, Dg)
    rec["effective_hamiltonian"] = dict(grid=a.grid, n_O=sum(p[2] == "O" for p in cps), n_X=sum(p[2] == "X" for p in cps),
                                        points=[dict(u=p[0], v=p[1], kind=p[2]) for p in cps],
                                        regions=lm.island_regions(u, v, H, cps),
                                        D_over_J_grad_H=float(np.nanmax(np.abs(np.nan_to_num(Dg))))) 
    if a.islands:
        from mrx.diagnostics.islands import islands
        from mrx.diagnostics.poincare import poincare
        res = poincare(seq, dof, lines=a.lines, periods=a.periods, planes=2)
        chains = islands(seq, dof, res, m_max=1, periods=a.periods)
        shown = res["shown"]
        rec["poincare"] = dict(lines=a.lines, periods=a.periods, chaotic=int((res["keep"] & res["chaotic"]).sum()),
                               lost=int((~res["keep"]).sum()),
                               iota_min=float(res["iota"][shown].min()) if shown.any() else None,
                               iota_max=float(res["iota"][shown].max()) if shown.any() else None, drift=float(res["drift"]))
        rec["islands"] = [dict(r_chain=c["r_chain"], width=c["width"], ray=c["ray"], ray_lo=c["ray_lo"], ray_hi=c["ray_hi"],
                               residue=c["residue"], n_O=len(c["O"]), n_X=len(c["X"]), n_locked=c["n_locked"]) for c in chains]
    ss, pts, ok2 = strobe(seq, dof, a.turns, a.steps)
    rad = np.linalg.norm(pts, axis=-1)
    rec["strobe"] = dict(turns=a.turns, lost=int((~ok2).sum()), radial_excursion_max=float((rad.max(1) - rad.min(1)).max()))
    rec["total_s"] = time.time() - t0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    np.savez_compressed(str(out) + ".npz", hgrid_u=u, hgrid_D=Dg, hgrid_H=H, seeds=seeds, y0=y0, y1=y1, strobe_seeds=ss, strobe=pts,
                        fixed=np.array([p["uv"] for p in fps]).reshape(-1, 2),
                        fixed_kind=np.array([p["kind"] for p in fps]))
    s = rec["score"]
    if a.islands:
        print("   islands:", [(round(c["r_chain"], 3), round(c["width"], 3)) for c in rec["islands"]], rec["poincare"])
    print("eps %.1e (m,n)=(%d,%d) ns %s | Bvol %.2e | D max %.2e rot dev [%.2e, %.2e] | O %d X %d | strobe exc %.2e | %.0fs" % (
        a.eps, a.m, a.n, a.ns, s["B_rel_volume_l2"], rec["one_turn"]["defect_max"], rec["one_turn"]["rotation_dev_min"],
        rec["one_turn"]["rotation_dev_max"], rec["n_O"], rec["n_X"], rec["strobe"]["radial_excursion_max"], rec["total_s"]), flush=True)


if __name__ == "__main__":
    main()
