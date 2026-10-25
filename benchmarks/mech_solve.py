"""Stage 6 solve-and-score driver: exact-seed solves with the spectral-condensation constraint taken apart.

One run = (case, NS, tcon0, baseline, optional extra): project the exact state, solve to ``ftol`` with VMEX
``caf09ee7`` (same entry point as the Stage 4 matrix: ``solve_multigrid`` with ``initial_state``), score the
solved state with the certificate norms (``score_solved``), and record

* the iteration-1 residual and its split into F_MHD and F_con (``mech_core.parts_report``),
* the solved-minus-seed state distance per channel, mode m and radius (state space, no quadrature),
* radially resolved B and J errors per radial cell (the near-axis layer), for the solved state and for the
  seed itself (the projection floor on the same cells),
* the headline B, J, force errors.

The constraint baseline (``--baseline``) is ``vmec`` (rcon0 = s * edge, the VMEC rule), ``exact`` (the real-space
rcon, zcon of the exact seed: F_con vanishes at the exact state) or ``blend:a``.  ``--tcon0`` scales the whole
constraint (0 = off).  ``--lasym-fix`` options for the asymmetric studies live in ``mech_asym.py``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import certify_projection as cp  # noqa: E402
import mech_core as M  # noqa: E402
import score_solved as ss  # noqa: E402
from analytic import cases  # noqa: E402

ROOT = M.ROOT
SCHEMA = 1
_RULES = {}


def rule_for(case_name, ns, level="m4"):
    key = (case_name, ns, level)
    if key not in _RULES:
        if len(_RULES) > 6:
            _RULES.clear()
        _RULES[key] = cp.volume_rule(cases()[case_name], level, ns)
    return _RULES[key]


def cell_profiles(field, rule, ns):
    """Per radial cell: relative L2 error of B and J (local norm) and of J in units of the global RMS |J|.

    Cell index j covers s in [j, j+1]/(NS-1); the mesh rule has the same cells, so the points of cell j are
    those with floor(s*(NS-1)) = j.
    """
    evaluate = ss.make_evaluator(field)
    B, J, valid = ss.native_BJ_safe(evaluate, rule["xyz"])
    B, J = np.where(valid[:, None], B, rule["B"]), np.where(valid[:, None], J, rule["J"])   # invalid points carry zero weight; no NaN may leak through 0*NaN
    cell = np.minimum((rule["s"]*(ns-1)).astype(int), ns-2)
    w = rule["w"]
    Jg = np.sqrt(np.sum(w*np.sum(rule["J"]**2, -1))/np.sum(w))
    Bg = np.sqrt(np.sum(w*np.sum(rule["B"]**2, -1))/np.sum(w))
    rows = {"cell": [], "s_mid": [], "B_local": [], "J_local": [], "J_over_global": [], "B_over_global": [], "invalid": []}
    for j in range(ns-1):
        k = cell == j
        ok = k & valid
        wj = np.where(ok, w, 0.0)
        eb = np.sum(wj*np.sum((B-rule["B"])**2, -1))
        ej = np.sum(wj*np.sum((J-rule["J"])**2, -1))
        nb = np.sum(wj*np.sum(rule["B"]**2, -1))
        nj = np.sum(wj*np.sum(rule["J"]**2, -1))
        vol = np.sum(wj)
        rows["cell"].append(j)
        rows["s_mid"].append((j+0.5)/(ns-1))
        rows["B_local"].append(float(np.sqrt(eb/nb)) if nb > 0 else None)
        rows["J_local"].append(float(np.sqrt(ej/nj)) if nj > 0 else None)
        rows["J_over_global"].append(float(np.sqrt(ej/vol)/Jg) if vol > 0 else None)
        rows["B_over_global"].append(float(np.sqrt(eb/vol)/Bg) if vol > 0 else None)
        rows["invalid"].append(int(np.sum(k & ~valid)))
    return rows


def state_distance(a, b, rt):
    """Relative distance of two SpectralStates per channel, with the m-resolved and radial profile."""
    m = np.asarray(rt.modes.m)
    out = {}
    for k in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
        x, y = np.asarray(getattr(a, k)), np.asarray(getattr(b, k))
        d = x-y
        nrm = max(np.linalg.norm(y), 1e-300)
        out[k] = {"abs": float(np.linalg.norm(d)), "rel_to_seed": float(np.linalg.norm(d)/nrm),
                  "by_m_abs": [float(np.linalg.norm(d[:, m == q])) for q in range(int(m.max())+1)],
                  "max_by_radius": [float(v) for v in np.abs(d).max(1)]}
    return out


def solve_state(inp, seed, baseline, tcon0, ftol, niter, mode="cli", defect=None):
    """Solve from ``seed`` with the chosen constraint baseline; returns (result, seconds)."""
    from vmex.core.multigrid import solve_multigrid
    with M.use_baseline_state(baseline, seed, inp, tcon0), M.use_defect(defect):
        t = time.perf_counter()
        result = solve_multigrid(inp, initial_state=seed, mode=mode, verbose=False, raise_on_max_iterations=False)
        sec = time.perf_counter()-t
    return result, sec


def run_one(case, ns, tcon0, baseline="vmec", ftol=1e-14, niter=20000, out_parent=None, tag="", defect_kind=None,
            keep_wout=True, score=True, parts=True, profile=True, polar=False):
    import vmex
    name = f"{case}-ns{ns}-tcon{tcon0:g}-{baseline.replace(':', '')}{'-polar' if polar else ''}{'-defect_'+defect_kind.replace(':', '') if defect_kind else ''}{tag}"
    rec = {"schema": SCHEMA, "run_id": name, "case": case, "ns": ns, "tcon0": tcon0, "baseline": baseline, "ftol": ftol,
           "niter": niter, "started_utc": datetime.now(timezone.utc).isoformat(), "host": os.uname().nodename.split(".")[0],
           "jax": jax.__version__, "vmex_version": vmex.__version__, "vmex_file": str(vmex.__file__),
           "load1_start": os.getloadavg()[0]}
    from evidence import source_metadata
    rec["code_identity"] = source_metadata(vmex.__file__, "uwplasma/vmex", None)
    t = time.perf_counter()
    inp, rt, seed, info = M.exact_seed(case, ns, tcon0, polar=polar)
    rec["polar_seed"] = bool(polar)
    inp = M.replace(inp, ftol_array=np.asarray([ftol]), niter_array=np.asarray([niter]), tcon0=float(tcon0))
    rec["seed"] = {"type": info["seed"], "seconds": time.perf_counter()-t}
    if parts:
        # iteration-1 residual of the seed under the chosen baseline, constraint on/off and split
        rt_b = M.with_baseline(rt, seed, baseline)
        rec["iteration1"] = M.parts_report(seed, rt_b)
    defect = None
    if defect_kind:
        rec["defect"] = defect_kind
        defect = M.defect_forces(seed, M.with_baseline(rt, seed, baseline), defect_kind)
    result, sec = solve_state(inp, seed, baseline, tcon0, ftol, niter, defect=defect)
    rec["solve"] = {"seconds": sec, "iterations": int(result.iterations), "converged": bool(result.converged),
                    "fsq_final": [float(result.fsqr), float(result.fsqz), float(result.fsql)],
                    "ier_flag": int(result.ier_flag), "jacobian_resets": int(result.jacobian_resets)}
    rec["fsq_history"] = [[int(i+1), float(v)] for i, v in enumerate(np.max(np.asarray(result.fsq_history)[:, :3], axis=1))
                          if i % 25 == 0 or i == len(result.fsq_history)-1]
    rec["distance_to_seed"] = state_distance(result.state, seed, rt)
    outdir = (out_parent or ROOT/"results/mechanism/runs")/name
    outdir.mkdir(parents=True, exist_ok=True)
    wpath = outdir/f"wout_{name}.nc"
    w = vmex.wout_from_result(inp, result)
    vmex.write_wout(wpath, w)
    rec["wout"] = {"path": wpath.name, "sha256": ss.sha256(wpath)}
    deck = outdir/f"input.{name}"
    inp.to_indata(deck)
    if score:
        rule = rule_for(case, ns)
        field = ss.field_from_wout(vmex.read_wout(wpath), deck)
        sc = ss.score_state(field, {"m4": rule, "m3": rule_for(case, ns, "m3"), "m4_shift": rule_for(case, ns, "m4_shift")})
        rec["headline"] = ss.headline(sc)
        rec["regions"] = {k: sc["scores"]["m4"][k] for k in ("axis", "bulk", "edge")}
        if profile:
            rec["profile_solved"] = cell_profiles(field, rule, ns)
            seed_field = vmex.VmecInteriorField.from_state(inp, seed)
            rec["profile_seed"] = cell_profiles(seed_field, rule, ns)
            sd = ss.score_state(seed_field, {"m4": rule})
            rec["seed_floor"] = {k: sd["scores"]["m4"]["all"][k] for k in ("B", "J", "force")}
    rec["finished_utc"] = datetime.now(timezone.utc).isoformat()
    rec["load1_end"] = os.getloadavg()[0]
    (outdir/"record.json").write_text(json.dumps(rec, indent=1, allow_nan=False)+"\n")
    if not keep_wout:
        wpath.unlink()
    h = rec.get("headline", {})
    print(name, "it", rec["solve"]["iterations"], "fsq", ["%.1e" % v for v in rec["solve"]["fsq_final"]],
          "B %.2e J %.2e F %.2e" % (h.get("B", np.nan), h.get("J", np.nan), h.get("force", np.nan)), "%.0fs" % sec, flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--ns", type=int, nargs="+", required=True)
    p.add_argument("--tcon0", type=float, nargs="+", default=[1.0])
    p.add_argument("--baseline", nargs="+", default=["vmec"])
    p.add_argument("--ftol", type=float, default=1e-14)
    p.add_argument("--niter", type=int, default=20000)
    p.add_argument("--tag", default="")
    p.add_argument("--defect", default=None, help="con | mhd | mhdaxis:K | mhdbulk:K | all (constant forcing that cancels that part of F(x*))")
    p.add_argument("--out-parent", type=Path, default=None)
    p.add_argument("--no-wout", action="store_true")
    p.add_argument("--polar", action="store_true", help="exact seed re-parametrised into the m = 1 polar gauge (mech_core.polar_rotation)")
    a = p.parse_args(argv)
    for ns in a.ns:
        for b in a.baseline:
            for t in a.tcon0:
                run_one(a.case, ns, t, b, a.ftol, a.niter, a.out_parent, a.tag, a.defect, keep_wout=not a.no_wout, polar=a.polar)


if __name__ == "__main__":
    main()
