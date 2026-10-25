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

from evidence import source_metadata

ROOT = Path(__file__).resolve().parents[1]
LADDER = (17, 33, 65, 129, 257)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imported_source(module, env):
    """Where ``module`` is actually imported from under ``env``; None when it is not importable."""
    out = subprocess.run([sys.executable, "-c", f"import {module}; print({module}.__file__)"],
                         env=env, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def write_receipt(out, receipt):
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")


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
    receipt = {"schema": 2, "run_id": run_id, "case_deck": args.deck, "deck_sha256": sha256(deck_path),
               "ladder": ladder, "code": args.code, "started_utc": datetime.now(timezone.utc).isoformat(),
               "status": "running", "runner_sha256": sha256(__file__), "host": os.uname().nodename.split(".")[0],
               "benchmark": {**source_metadata(__file__, "rogeriojorge/vmex-benchmark-analytical", None),
                             "cases_json_sha256": sha256(ROOT/"cases.json"),
                             "analytic_sha256": sha256(ROOT/"benchmarks/analytic.py")}}
    env = dict(os.environ)
    if args.code == "vmex":
        if args.vmex_path is not None:
            env["PYTHONPATH"] = str(args.vmex_path)
        cmd = [sys.executable, "-m", "vmex", deck_path.name]
        module = imported_source("vmex", env)
        if module is None:
            raise SystemExit("vmex is not importable with the given --vmex-path / environment")
        receipt["code_identity"] = source_metadata(module, "uwplasma/vmex", None)
    elif args.code == "vmec2000":
        if not args.vmec2000:
            raise SystemExit("xvmec2000 not found; pass --vmec2000")
        cmd = [args.vmec2000, deck_path.name]
        binary = Path(args.vmec2000).resolve()
        receipt["code_identity"] = {**source_metadata(binary, "PrincetonUniversity/STELLOPT (VMEC2000)", None),
                                    "binary_sha256": sha256(binary)}
    else:
        import importlib.metadata
        cmd = [sys.executable, "-c",
               "import vmecpp,sys; o=vmecpp.run(vmecpp.VmecInput.from_file(sys.argv[1]), verbose=True); "
               "o.wout.save(sys.argv[2])", deck_path.name, f"wout_{name}.nc"]
        receipt["code_identity"] = {"repository": "proximafusion/vmecpp",
                                    "version": importlib.metadata.version("vmecpp")}
    receipt["command"] = cmd
    write_receipt(out, receipt)  # a killed run still leaves its starting record
    t0 = time.perf_counter()
    with open(out/"stdout.log", "w") as log:
        proc = subprocess.run(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
    receipt["wall_seconds"] = time.perf_counter()-t0
    receipt["returncode"] = proc.returncode
    wout = out/f"wout_{name}.nc"
    receipt["wout"] = {"path": wout.name, "sha256": sha256(wout)} if wout.exists() else None
    receipt["status"] = "wout_written" if wout.exists() else "failed_no_wout"
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_receipt(out, receipt)
    print(run_id, receipt["status"], "%.1fs" % receipt["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
