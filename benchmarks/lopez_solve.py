"""Lopez D fixed-boundary VMEX solves: cold ladder, exact seed, direction-only seed.

Protocols (kept separate, never mixed in one record):

* ``cold``   : the committed deck, radial ladder 17 -> ... -> NS, VMEX's own axis guess and
               linear interpolation of the boundary. Nothing of the interior is supplied.
* ``seed``   : single radial stage NS started from the exact state projected onto the VMEC
               representation (exact geometry, lambda = 0, ``--kind eta``).
* ``alpha``  : the same with the ordinary ellipse angle (direction-only gauge, lambda = 0),
               deck boundary written in that angle too. Negative control.

The final stage may exhaust NITER (``--niter``); the last iterate is written to a WOUT regardless
(the library's ``raise_on_max_iterations=False`` path) and its true status is in the receipt
(``converged``, ``ier_flag``, final ``fsq``). Every run writes receipt.json first, then updates it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
LADDER = (17, 33, 65, 129)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", choices=("cold", "seed", "alpha"), required=True)
    p.add_argument("--closure", choices=("current", "iota"), required=True)
    p.add_argument("--ns", type=int, required=True)
    p.add_argument("--mpol", type=int, default=10)
    p.add_argument("--ntor", type=int, default=72)
    p.add_argument("--ftol", type=float, default=1e-12)
    p.add_argument("--niter", type=int, default=3000, help="iteration cap per stage")
    p.add_argument("--tcon0", type=float, default=None)
    p.add_argument("--tag", default="", help="suffix of the run id (e.g. loose)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    from dataclasses import replace
    import vmex
    from vmex.core.multigrid import solve_multigrid
    import lopez_recovery as lr
    import score_lopez as sl
    from evidence import source_metadata

    a.out.mkdir(parents=True, exist_ok=False)
    run_id = f"lopez_D_{a.closure}-ns{a.ns}-{a.protocol}{a.tag}-vmex"
    closure = 1 if a.closure == "current" else 0
    if a.protocol == "cold":
        inp = vmex.VmecInput.from_file(ROOT/"inputs"/f"input.lopez_D_{a.closure}")
        stages = [n for n in LADDER if n < a.ns] + [a.ns]
        state = None
        deck_text_path = ROOT/"inputs"/f"input.lopez_D_{a.closure}"
    else:
        kind = "eta" if a.protocol == "seed" else "alpha"
        inp = lr.deck_for(a.mpol, a.ntor, a.ns, kind, closure, directory=a.out)
        stages = [a.ns]
        deck_text_path = next(a.out.glob("input.lopez_*"))
        t_seed = time.perf_counter()
        state = lr.vmex_state(inp, a.ns, kind)
        seed_seconds = time.perf_counter() - t_seed
    if a.tcon0 is not None:
        inp = replace(inp, tcon0=a.tcon0)
    ftols = [1e-12]*(len(stages) - 1) + [a.ftol]
    inp = replace(inp, ns_array=np.asarray(stages), ftol_array=np.asarray(ftols),
                  niter_array=np.asarray([a.niter]*len(stages)))
    name = f"{run_id}"
    inp.to_indata(a.out/f"input.{name}")
    receipt = dict(schema=1, run_id=run_id, protocol=a.protocol, closure=a.closure, ns=a.ns, mpol=a.mpol, ntor=a.ntor,
                   ladder=dict(ns_array=stages, ftol_array=ftols, niter_per_stage=a.niter, tcon0=a.tcon0),
                   deck_sha256=sha256(a.out/f"input.{name}"), source_deck=deck_text_path.name,
                   source_deck_sha256=sha256(deck_text_path), code="vmex",
                   code_identity=source_metadata(vmex.__file__, "uwplasma/vmex", None),
                   started_utc=datetime.now(timezone.utc).isoformat(), status="running",
                   host=os.uname().nodename.split(".")[0], argv=sys.argv, jax=jax.__version__,
                   runner_sha256=sha256(__file__), omp_threads=os.environ.get("OMP_NUM_THREADS"))
    (a.out/"receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    t0 = time.perf_counter()
    err = None
    try:
        result = solve_multigrid(inp, initial_state=state, mode="cli", verbose=True,
                                 raise_on_max_iterations=False)
    except Exception as exc:  # recorded, not hidden
        err = f"{type(exc).__name__}: {exc}"
        result = None
    receipt["wall_seconds"] = time.perf_counter() - t0
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if result is None:
        receipt.update(status="failed", error=err, wout=None)
    else:
        w = vmex.wout_from_result(inp, result)
        path = a.out/f"wout_{name}.nc"
        vmex.write_wout(path, w)
        receipt.update(status="wout_written", wout=dict(path=path.name, sha256=sha256(path)),
                       converged=bool(getattr(result, "converged", False)), ier_flag=int(getattr(result, "ier_flag", -1)),
                       iterations_final_stage=int(result.iterations),
                       fsq_final=[float(result.fsqr), float(result.fsqz), float(result.fsql)])
        if state is not None:
            receipt["seed_seconds"] = seed_seconds
        try:
            score = sl.score_wout(w, sl.lopez_case())
            score.update(wout_sha256=receipt["wout"]["sha256"], scorer_sha256=sha256(Path(sl.__file__)),
                         run_id=run_id, code="vmex", wall_seconds=receipt["wall_seconds"])
            (a.out/f"lopez_score_v{sl.SCHEMA}.json").write_text(json.dumps(score, indent=1, allow_nan=False) + "\n")
        except Exception as exc:  # e.g. a non-finite state; the WOUT is kept for inspection
            receipt["score_error"] = f"{type(exc).__name__}: {exc}"
    (a.out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(run_id, receipt["status"], "%.0fs" % receipt["wall_seconds"], receipt.get("fsq_final"), flush=True)


if __name__ == "__main__":
    main()
