"""Stage 6: Lopez D polar-gauge exact seed with the constraint baseline of the exact state.

The iotaF track found that the polar-gauge exact seed (``lopez_polar``, kind ``polar:eta``) stays on the exact state while
fsq limit-cycles 1e-8..1e-4, and that the spectral-condensation force is O(1) at that state.  This runner puts the
baselines ``rcon0, zcon0`` of the exact polar state (``mech_core.use_baseline_state``, baseline ``exact``) so that the
constraint force vanishes there, and compares with the default baseline under the same deck, seed, NS, cap and ``ftol``:

* the iteration-1 residual and its split into F_MHD and F_con (``mech_core.parts_report``);
* the fsq trajectory (thinned history), its minimum and the spread over the last third of the run (limit cycle measure);
* the Lopez scorer (axis offset, B errors, energy) of the last iterate.

``python mech_lopez.py --ns 17 --niter 6000 --baseline vmec exact --out DIR``.
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
import mech_core as M  # noqa: E402

ROOT = M.ROOT


def run(ns, mpol, ntor, niter, ftol, baseline, tcon0, out, kind="polar:eta", closure=0, scale_lambda=None):
    import vmex
    import vmex.core.solver as S
    from dataclasses import replace
    from vmex.core.multigrid import solve_multigrid
    import lopez_recovery as lr
    import score_lopez as sl
    from evidence import source_metadata
    out.mkdir(parents=True, exist_ok=True)
    name = f"lopez_D-ns{ns}-m{mpol}n{ntor}-{kind.replace(':', '_')}-tcon{tcon0:g}-{baseline.replace(':', '')}"
    inp = lr.deck_for(mpol, ntor, ns, kind, closure, directory=out)
    inp = replace(inp, ns_array=np.asarray([ns]), ftol_array=np.asarray([ftol]), niter_array=np.asarray([niter]), tcon0=float(tcon0))
    seed = lr.vmex_state(inp, ns, kind)
    rec = {"schema": 1, "run_id": name, "ns": ns, "mpol": mpol, "ntor": ntor, "kind": kind, "closure": closure, "tcon0": tcon0,
           "baseline": baseline, "niter": niter, "ftol": ftol, "started_utc": datetime.now(timezone.utc).isoformat(),
           "host": os.uname().nodename.split(".")[0], "jax": jax.__version__, "load1_start": os.getloadavg()[0],
           "code_identity": source_metadata(vmex.__file__, "uwplasma/vmex", None), "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    resolution = S.resolution_from_input(inp, ns=ns)
    rt = S.prepare_runtime(inp, resolution, tcon0=float(tcon0))
    seed = S.hot_restart_state(rt, seed)
    rt = S.runtime_with_baselines(rt, seed)
    rec["iteration1"] = M.parts_report(seed, M.with_baseline(rt, seed, baseline))
    t = time.perf_counter()
    with M.use_baseline_state(baseline, seed, inp, tcon0):
        result = solve_multigrid(inp, initial_state=seed, mode="cli", verbose=False, raise_on_max_iterations=False)
    rec["wall_seconds"] = time.perf_counter()-t
    hist = np.asarray(result.fsq_history)
    f = np.max(hist[:, :3], axis=1)
    third = f[2*len(f)//3:]
    rec["fsq_final"] = [float(result.fsqr), float(result.fsqz), float(result.fsql)]
    rec["iterations"] = int(result.iterations)
    rec["converged"] = bool(result.converged)
    rec["fsq_min"] = float(f.min())
    rec["fsq_min_iteration"] = int(f.argmin()+1)
    rec["last_third"] = {"min": float(third.min()), "median": float(np.median(third)), "max": float(third.max()),
                         "log10_spread": float(np.log10(third.max()/third.min()))}
    rec["history"] = [[int(i+1), float(f[i])] for i in range(0, len(f), 25)] + [[len(f), float(f[-1])]]
    rec["delt_last"] = float(hist[-1, 10]) if hist.shape[1] > 10 else None
    w = vmex.wout_from_result(inp, result)
    wpath = out/f"wout_{name}.nc"
    vmex.write_wout(wpath, w)
    try:
        score = sl.score_wout(w, sl.lopez_case(iota=lr.gauge_with(None)[2]), keep_profile=False)
        rec["score"] = lr.compact(score)
    except Exception as exc:
        rec["score_error"] = f"{type(exc).__name__}: {exc}"
    rec["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out/f"{name}.json").write_text(json.dumps(rec, indent=1, allow_nan=False)+"\n")
    print(name, "it", rec["iterations"], "fsq_min %.2e" % rec["fsq_min"], "last third min/max %.1e/%.1e" % (third.min(), third.max()),
          "iter1 con/mhd rz %.2e/%.2e" % (rec["iteration1"]["fsq_rz_con"], rec["iteration1"]["fsq_rz_mhd"]),
          "B_vol %.2e axis %.2e" % (rec.get("score", {}).get("B_vol", np.nan), rec.get("score", {}).get("axis", np.nan)),
          "%.0fs" % rec["wall_seconds"], flush=True)
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ns", type=int, default=17)
    p.add_argument("--mpol", type=int, default=10)
    p.add_argument("--ntor", type=int, default=72)
    p.add_argument("--niter", type=int, default=6000)
    p.add_argument("--ftol", type=float, default=1e-12)
    p.add_argument("--tcon0", type=float, nargs="+", default=[1.0])
    p.add_argument("--baseline", nargs="+", default=["vmec", "exact"])
    p.add_argument("--kind", default="polar:eta")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    for b in a.baseline:
        for t in a.tcon0:
            run(a.ns, a.mpol, a.ntor, a.niter, a.ftol, b, t, a.out, a.kind)


if __name__ == "__main__":
    main()
