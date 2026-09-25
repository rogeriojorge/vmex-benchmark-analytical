"""Run one exact-solution deck through VMEX, VMEC2000 or VMEC++ with identical input.

The deck is generated once per (case, NS) from ``inputs/<deck>`` with a fixed
radial ladder, tolerances and iteration caps, written with VMEX's namelist
writer, and the same bytes are given to every code.  Each run gets its own
directory with the deck, WOUT, log and a receipt naming the exact code
(commit / binary hash / package version).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LADDER = (17, 33, 65, 129, 257)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit(path):
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    return {"commit": out.stdout.strip() or None, "tracked_tree_clean": dirty.stdout.strip() == ""}


def make_deck(deck, ns, ftol, niter, out):
    import vmex
    inp = vmex.VmecInput.from_file(ROOT/"inputs"/deck)
    stages = [n for n in LADDER if n < ns] + [ns]
    ftols = [1e-12]*(len(stages)-1) + [ftol]
    inp = replace(inp, ns_array=np.asarray(stages), ftol_array=np.asarray(ftols),
                  niter_array=np.asarray([niter]*len(stages)))
    inp.to_indata(out)
    return {"ns_array": stages, "ftol_array": ftols, "niter_per_stage": niter}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", required=True, help="file name under inputs/")
    parser.add_argument("--ns", type=int, required=True)
    parser.add_argument("--code", choices=("vmex", "vmec2000", "vmecpp"), required=True)
    parser.add_argument("--ftol", type=float, default=1e-14)
    parser.add_argument("--niter", type=int, default=30000)
    parser.add_argument("--vmex-path", type=Path, help="VMEX checkout to put on PYTHONPATH")
    parser.add_argument("--vmec2000", default=shutil.which("xvmec2000"))
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/cross_code/runs")
    args = parser.parse_args(argv)
    case = args.deck.removeprefix("input.")
    run_id = f"{case}-ns{args.ns}-{args.code}"
    out = args.output_parent/run_id
    out.mkdir(parents=True, exist_ok=False)
    name = f"{case}_ns{args.ns}"
    deck_path = out/f"input.{name}"
    ladder = make_deck(args.deck, args.ns, args.ftol, args.niter, deck_path)
    receipt = {"schema": 1, "run_id": run_id, "case_deck": args.deck, "deck_sha256": sha256(deck_path),
               "ladder": ladder, "code": args.code, "started_utc": datetime.now(timezone.utc).isoformat(),
               "runner_sha256": sha256(__file__), "host": os.uname().nodename.split(".")[0]}
    env = dict(os.environ)
    if args.code == "vmex":
        env["PYTHONPATH"] = str(args.vmex_path)
        cmd = [sys.executable, "-m", "vmex", deck_path.name]
        receipt["code_identity"] = {"repository": "uwplasma/vmex", **git_commit(args.vmex_path)}
    elif args.code == "vmec2000":
        cmd = [args.vmec2000, deck_path.name]
        binary = Path(args.vmec2000).resolve()
        receipt["code_identity"] = {"repository": "PrincetonUniversity/STELLOPT (VMEC2000)",
                                    "binary_sha256": sha256(binary), **git_commit(binary.parent)}
    else:
        import vmecpp
        import importlib.metadata
        cmd = [sys.executable, "-c",
               "import vmecpp,sys; o=vmecpp.run(vmecpp.VmecInput.from_file(sys.argv[1]), verbose=True); "
               "o.wout.save(sys.argv[2])", deck_path.name, f"wout_{name}.nc"]
        receipt["code_identity"] = {"repository": "proximafusion/vmecpp",
                                    "version": importlib.metadata.version("vmecpp")}
    receipt["command"] = cmd
    t0 = time.perf_counter()
    with open(out/"stdout.log", "w") as log:
        proc = subprocess.run(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt["wall_seconds"] = time.perf_counter()-t0
    receipt["returncode"] = proc.returncode
    wout = out/f"wout_{name}.nc"
    receipt["wout"] = {"path": wout.name, "sha256": sha256(wout)} if wout.exists() else None
    receipt["status"] = "wout_written" if wout.exists() else "failed_no_wout"
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(run_id, receipt["status"], "%.1fs" % receipt["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
