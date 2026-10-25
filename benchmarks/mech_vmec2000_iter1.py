"""Stage 6 (negative result, kept for the record): VMEC2000's own residual at the exact seed (``xvmec input reset=<wout>``), against VMEX at the identical state.

VMEC2000 (``runvmec``/``initialize_radial``) can start from the interior of a WOUT: ``load_xc_from_wout`` reads rmnc, zmns, lmns
(+ rmns, zmnc, lmnc for LASYM), converts the m = 1 pairs to the constrained basis and lambda to its internal full-mesh form, and
``profil3d`` keeps it (``lreset = .false.``).  The exact projected state of ``mech_core.exact_seed`` is written as a WOUT with
VMEX's ``wout_from_state`` (a pure basis conversion, no solve), the same deck (NS single stage, ``NITER`` = 2, ``NSTEP`` = 1,
``TCON0`` as requested) is run by VMEC2000, and its iteration-1 FSQR/FSQZ/FSQL are read from the screen output and compared
with VMEX's residual of the same seed (``parts_report``).  A round trip check (VMEX reads its own WOUT back through
``state_from_wout``) records how well the interior survives the file.

RESULT: in the PARVMEC 9.0 build used here ``reset=`` is parsed but does not restore the file's interior (VMEC2000's own
converged WOUT comes back at iteration-1 fsq 2.8e-2/4.4e-4/1.5e-2, the same numbers for every file, see
``results/mechanism/vmec2000_reset_check.json``), so the two operators were compared the other way round (``mech_crosscode``).

Run on a box that has ``xvmec``: ``python mech_vmec2000_iter1.py --xvmec PATH --out DIR``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech_core as M  # noqa: E402

ROOT = M.ROOT


def parse_iter1(text):
    """First ITER row after a header: (iter, fsqr, fsqz, fsql) or None."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.search(r"ITER\s+FSQR\s+FSQZ\s+FSQL", line):
            for nxt in lines[i+1:i+6]:
                parts = nxt.split()
                if len(parts) >= 4 and re.fullmatch(r"\d+", parts[0]):
                    try:
                        return [int(parts[0])] + [float(p.replace("D", "E")) for p in parts[1:4]]
                    except ValueError:
                        continue
    return None


def one(case, ns, tcon0, xvmec, out, deck=None, case_obj=None, mpol=None, tag="", polar=False):
    import vmex
    from dataclasses import replace
    name = f"{getattr(case, 'name', case)}-ns{ns}-tcon{tcon0:g}{'-polar' if polar else ''}{tag}"
    run = out/name
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    inp, rt, seed, info = M.exact_seed(case, ns, tcon0, deck=deck, mpol=mpol, polar=polar)
    inp = replace(inp, niter_array=np.asarray([2]), nstep=1, ftol_array=np.asarray([1e-30]), tcon0=float(tcon0))
    w = vmex.wout_from_state(inp=inp, state=seed, fsqr=0.0, fsqz=0.0, fsql=0.0)
    wpath = run/f"wout_seed_{name}.nc"
    vmex.write_wout(wpath, w)
    inp.to_indata(run/f"input.{name}")
    proc = subprocess.run([str(xvmec), f"input.{name}", f"reset={wpath.name}"], cwd=run, capture_output=True, text=True, timeout=1800)
    (run/"stdout.log").write_text(proc.stdout+proc.stderr)
    it1 = parse_iter1(proc.stdout)
    rec = {"case": getattr(case, "name", case), "ns": ns, "tcon0": tcon0, "polar": polar, "returncode": proc.returncode, "vmec2000_iter1": it1}
    from vmex.core.restart import state_from_wout
    back = state_from_wout(vmex.read_wout(wpath), inp=inp)
    rec["roundtrip_max_abs"] = {k: float(np.abs(np.asarray(getattr(back, k))-np.asarray(getattr(seed, k))).max()) for k in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")}
    rt_b = M.with_baseline(rt, seed, "vmec")
    cache, res = M.fresh_cache(seed, rt_b)
    rec["vmex_iter1"] = [float(res.residuals.fsqr), float(res.residuals.fsqz), float(res.residuals.fsql)]
    if it1:
        rec["ratio_vmec2000_over_vmex"] = [it1[i+1]/max(rec["vmex_iter1"][i], 1e-300) for i in range(3)]
    print(name, "VMEC2000", it1, "VMEX", ["%.3e" % v for v in rec["vmex_iter1"]], flush=True)
    (run/"record.json").write_text(json.dumps(rec, indent=1, allow_nan=False)+"\n")
    for f in run.glob("*.nc"):
        if f.name != wpath.name:
            f.unlink()
    wpath.unlink()
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xvmec", type=Path, required=True)
    p.add_argument("--case", nargs="+", default=["solovev_symmetric"])
    p.add_argument("--ns", type=int, nargs="+", default=[17])
    p.add_argument("--tcon0", type=float, nargs="+", default=[1.0, 0.0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--polar", action="store_true")
    a = p.parse_args(argv)
    rows = []
    for case in a.case:
        for ns in a.ns:
            for t in a.tcon0:
                rows.append(one(case, ns, t, a.xvmec, a.out, polar=a.polar))
    (a.out/"summary.json").write_text(json.dumps(rows, indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
