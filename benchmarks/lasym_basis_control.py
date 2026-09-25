"""C5 LASYM basis control: one symmetric problem solved with LASYM=F and LASYM=T.

Both decks describe the same stellarator-symmetric physics, so in the full
basis the asymmetric partners (rmns, zmnc, lmnc) must vanish up to the solve
tolerance and the symmetric blocks must equal the LASYM=F solution.  This is a
solver-state control; the LASYM field score comes from the fitted-state route
and is reported under that label only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analytic import ROOT
from evidence import sha256_file, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symmetric-run", type=Path, required=True)
    parser.add_argument("--full-basis-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from vmex.core.wout import read_wout
    sym, full = (read_wout(run/"wout.nc") for run in (args.symmetric_run, args.full_basis_run))
    if sym.lasym or not full.lasym:
        raise SystemExit("expected LASYM=F then LASYM=T")
    rows = {}
    for name in ("rmnc", "zmns", "lmns"):
        a, b = np.asarray(getattr(sym, name)), np.asarray(getattr(full, name))
        if a.shape != b.shape:
            raise SystemExit(f"{name} shapes differ: {a.shape} vs {b.shape}")
        rows[name] = {"max_abs_difference": float(np.max(np.abs(a-b))),
                      "relative_l2": float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-300))}
    partners = {name: float(np.max(np.abs(np.asarray(getattr(full, name)))))
                for name in ("rmns", "zmnc", "lmnc") if getattr(full, name, None) is not None}
    reports = {tag: json.loads((run/"forward.json").read_text())
               for tag, run in (("lasym_false", args.symmetric_run), ("lasym_true", args.full_basis_run))}
    record = {
        "schema": 1, "evidence": "lasym_basis_control_solver_state",
        "symmetric_blocks": rows, "asymmetric_partner_max_abs": partners,
        "runs": {tag: {"run_id": r["run_id"], "iterations": r["iterations"], "converged": r["solver_converged"],
                       "fsq": [r["fsqr"], r["fsqz"], r["fsql"]], "field_path": r.get("field_path"),
                       "native_score": {k: r["native_score"].get(k) for k in
                                        ("field_relative_l2", "current_relative_l2", "force_pressure_scale")}}
                 for tag, r in reports.items()},
        "wout_sha256": {"lasym_false": sha256_file(args.symmetric_run/"wout.nc"),
                        "lasym_true": sha256_file(args.full_basis_run/"wout.nc")},
    }
    write_json(args.output, record, exclusive=True)
    print(json.dumps({k: record[k] for k in ("symmetric_blocks", "asymmetric_partner_max_abs")}, indent=1))


if __name__ == "__main__":
    main()
