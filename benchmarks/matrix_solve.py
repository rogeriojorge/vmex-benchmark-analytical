"""Stage 4 recovery-matrix solve driver: one (case, code, protocol, NS) run, one directory.

Protocols (kept separate, never mixed in one record):

* ``cold``   : the committed deck's own initial guess (boundary + VMEC axis guess, nothing of the
               interior supplied), a SINGLE radial stage at NS.
* ``ladder`` : "continued": the NS ladder 17 -> 33 -> 65 -> 129 up to NS, each stage restarted from
               the interpolated previous stage (intermediate stages stop at ftol 1e-12).
* ``restart``: parameter continuation: a single stage at NS restarted (``restart_from``) from the WOUT
               of an earlier solve (``--restart-wout``), e.g. the previous point of a parameter path.
* ``seed``   : "exact-seed": a single radial stage at NS started from the exact state projected onto the
               VMEC representation (exact flux surfaces, exact full-flux lambda for non-integer
               families; VMEX only: VMEC2000 accepts no interior initial state).

All three VMEC-family runs of a case read the same deck bytes except that the stage arrays differ with
the protocol.  The last iterate is written to a WOUT whether or not NITER was exhausted; the true
status (``fsq_final``, ``fsq_leq_ftol``) is in the receipt.  ``receipt.json`` is written first and
updated, so a killed run leaves its starting record.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

T_PROCESS = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LADDER = (17, 33, 65, 129, 257)
SCHEMA = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stages_for(protocol, ns):
    return [ns] if protocol in ("cold", "seed", "restart") else [n for n in LADDER if n < ns] + [ns]


def write_deck(case, protocol, ns, ftol, niter, tcon0, path, ftol_stages=1e-12):
    import vmex
    inp = vmex.VmecInput.from_file(ROOT/"inputs"/f"input.{case}_iota")
    stages = stages_for(protocol, ns)
    ftols = [ftol_stages]*(len(stages)-1) + [ftol]
    changes = dict(ns_array=np.asarray(stages), ftol_array=np.asarray(ftols), niter_array=np.asarray([niter]*len(stages)))
    if tcon0 is not None:
        changes["tcon0"] = tcon0
    inp = replace(inp, **changes)
    inp.to_indata(path)
    return inp, {"ns_array": stages, "ftol_array": ftols, "niter_per_stage": niter, "tcon0": tcon0}


def fortran_times(text):
    out = {}
    for key, pat in (("code_total_seconds", r"TOTAL COMPUTATIONAL TIME \(SEC\)\s+([\d.]+)"),
                     ("code_solver_seconds", r"TIME IN SOLVER\s+([\d.]+)")):
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def thin_history(history, every=25):
    """[iteration, max(fsqr, fsqz, fsql)] every ``every`` iterations plus the last one."""
    h = np.asarray(history, float)
    if h.ndim != 2 or len(h) == 0:
        return []
    m = np.max(h[:, :3], axis=1)
    idx = sorted(set(range(0, len(h), every)) | {len(h)-1})
    return [[int(i+1), float(m[i])] for i in idx]


def run_vmex(a, out, name, receipt):
    import jax
    jax.config.update("jax_enable_x64", True)
    import vmex
    from vmex.core.multigrid import solve_multigrid
    from evidence import source_metadata
    receipt["code_identity"] = source_metadata(vmex.__file__, "uwplasma/vmex", None)
    receipt["jax"] = jax.__version__
    receipt["vmex_version"] = vmex.__version__
    receipt["compilation_cache"] = os.environ.get("VMEX_COMPILATION_CACHE_DIR")
    inp = vmex.VmecInput.from_file(out/f"input.{name}")
    state = None
    if a.protocol == "seed":
        from analytic import cases
        from project_integer_vmex import project, with_resolution
        t = time.perf_counter()
        seed_inp = with_resolution(inp, a.ns)
        state, info = project(seed_inp, cases()[a.case], a.ns, lambda_mode=a.lambda_mode)
        receipt["seed"] = {"seconds": time.perf_counter()-t, "type": info["seed"], "lambda_certificates": info["lambda_certificates"]}
        (out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    restart = None
    if a.protocol == "restart":
        restart = Path(a.restart_wout)
        receipt["restart_from"] = {"path": str(restart.name), "sha256": sha256(restart)}
    t0 = time.perf_counter()
    result = solve_multigrid(inp, initial_state=state, restart_from=restart, mode="cli", verbose=True,
                             raise_on_max_iterations=False)
    receipt["wall_seconds"] = time.perf_counter()-t0
    w = vmex.wout_from_result(inp, result)
    path = out/f"wout_{name}.nc"
    vmex.write_wout(path, w)
    hist = np.asarray(result.fsq_history)
    receipt.update(wout={"path": path.name, "sha256": sha256(path)}, converged=bool(result.converged),
                   ier_flag=int(result.ier_flag), iterations_final_stage=int(result.iterations),
                   fsq_final=[float(result.fsqr), float(result.fsqz), float(result.fsql)],
                   jacobian_resets=int(result.jacobian_resets))
    if len(hist):
        receipt["fsq_first_iteration"] = [float(v) for v in hist[0, :3]]
    (out/"fsq_history.json").write_text(json.dumps(thin_history(hist), allow_nan=False)+"\n")


def run_vmec2000(a, out, name, receipt):
    from evidence import source_metadata
    binary = Path(a.xvmec).resolve()
    receipt["code_identity"] = {**source_metadata(binary, "PrincetonUniversity/STELLOPT (VMEC2000)", None),
                                "binary_sha256": sha256(binary)}
    cmd = [str(binary), f"input.{name}"]
    receipt["command"] = cmd
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    t0 = time.perf_counter()
    with open(out/"stdout.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT)
        _, status, ru = os.wait4(proc.pid, 0)
    receipt["wall_seconds"] = time.perf_counter()-t0
    receipt["returncode"] = os.waitstatus_to_exitcode(status)
    receipt["cpu_seconds"] = ru.ru_utime+ru.ru_stime
    receipt["peak_rss_bytes"] = int(ru.ru_maxrss*1024)
    receipt.update(fortran_times((out/"stdout.log").read_text(errors="replace")))
    wout = out/f"wout_{name}.nc"
    if wout.exists():
        import vmex
        w = vmex.read_wout(wout)
        receipt.update(wout={"path": wout.name, "sha256": sha256(wout)}, ier_flag=int(getattr(w, "ier_flag", -1)),
                       fsq_final=[float(w.fsqr), float(w.fsqz), float(w.fsql)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--code", choices=("vmex", "vmec2000"), required=True)
    p.add_argument("--protocol", choices=("cold", "ladder", "seed", "restart"), required=True)
    p.add_argument("--ns", type=int, required=True)
    p.add_argument("--ftol", type=float, default=1e-14)
    p.add_argument("--ftol-stages", type=float, default=1e-12, help="tolerance of the intermediate ladder stages")
    p.add_argument("--niter", type=int, default=20000, help="iteration cap per stage")
    p.add_argument("--restart-wout", help="WOUT to restart from (protocol restart)")
    p.add_argument("--tcon0", type=float, default=None)
    p.add_argument("--lambda-mode", default="auto", choices=("auto", "zero", "full_flux"))
    p.add_argument("--tag", default="", help="suffix of the run id (e.g. _tcon0, _warm, _ftol1e-8)")
    p.add_argument("--xvmec", default=shutil.which("xvmec"))
    p.add_argument("--out-parent", type=Path, default=ROOT/"results/matrix/runs")
    a = p.parse_args(argv)
    if a.code == "vmec2000" and a.protocol == "seed":
        raise SystemExit("VMEC2000 takes no interior initial state: the exact-seed protocol is VMEX only")
    run_id = f"{a.case}-{a.protocol}{a.tag}-ns{a.ns}-{a.code}"
    out = a.out_parent/run_id
    out.mkdir(parents=True, exist_ok=False)
    name = run_id
    deck = out/f"input.{name}"
    _, ladder = write_deck(a.case, a.protocol, a.ns, a.ftol, a.niter, a.tcon0, deck, a.ftol_stages)
    from evidence import source_metadata
    receipt = {"schema": SCHEMA, "run_id": run_id, "case": a.case, "code": a.code, "protocol": a.protocol, "ns": a.ns,
               "ladder": ladder, "ftol": a.ftol, "deck_sha256": sha256(deck), "started_utc": datetime.now(timezone.utc).isoformat(),
               "status": "running", "host": os.uname().nodename.split(".")[0], "argv": sys.argv,
               "runner_sha256": sha256(__file__), "omp_threads": os.environ.get("OMP_NUM_THREADS"),
               "load1_start": os.getloadavg()[0],
               "benchmark": {**source_metadata(__file__, "rogeriojorge/vmex-benchmark-analytical", None),
                             "cases_json_sha256": sha256(ROOT/"cases.json"), "analytic_sha256": sha256(ROOT/"benchmarks/analytic.py")}}
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    err = None
    try:
        (run_vmex if a.code == "vmex" else run_vmec2000)(a, out, name, receipt)
    except Exception as exc:  # recorded, not hidden
        err = f"{type(exc).__name__}: {exc}"
        receipt["error"] = err
    receipt["process_seconds"] = time.perf_counter()-T_PROCESS
    receipt["load1_end"] = os.getloadavg()[0]
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["status"] = "failed" if err else ("wout_written" if receipt.get("wout") else "failed_no_wout")
    if receipt.get("fsq_final"):
        receipt["fsq_leq_ftol"] = bool(max(receipt["fsq_final"]) <= a.ftol)
    (out/"receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n")
    print(run_id, receipt["status"], "%.0fs" % receipt.get("wall_seconds", float("nan")), receipt.get("fsq_final"), err or "", flush=True)


if __name__ == "__main__":
    main()
