"""C3: discrete VMEX residual of exact projections versus NS (no solve).

If the preconditioned/raw force residual of the exact projection falls with
NS at a consistent order, the gap is discretization of VMEX's equations on the
exact field; if it stays flat, the seed conversion (lambda scaling, m=1
constraint, edge row) is suspect.  Each seed is evaluated in its own operator
(frozen at the seed).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from analytic import ROOT
from evidence import capture_execution_source, reserve_run_directory, sha256_file, write_json
import refinement_observer as observer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--seed", nargs=2, action="append", metavar=("NS", "NPZ"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/audit/seed_residuals")
    args = parser.parse_args(argv)
    import vmex
    from vmex.core import implicit
    from vmex.core.solver import SpectralState
    run_id, out = reserve_run_directory(
        args.output_parent, f"{args.label}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    here = Path(__file__).resolve().parent
    capture_execution_source(out, [Path(__file__), here/"refinement_observer.py", here/"evidence.py"],
                             packages={"uwplasma/vmex": vmex.__file__})
    inp = vmex.VmecInput.from_file(args.deck)
    rows = []
    for ns_text, path in args.seed:
        ns, path = int(ns_text), Path(path)
        cfg = implicit.make_config(inp, ns=ns, ftol=1e-12, max_iterations=10, refine_tol=1e-11)
        implicit._template_runtime(cfg)
        implicit._boundary_pack_tables(cfg)
        params = implicit.params_from_input(inp)
        mask = implicit._fixed_boundary_dof_mask(cfg)
        data = np.load(path)
        state = SpectralState(**{n: np.asarray(data[n]) for n in observer.FIELDS})
        record, _ = observer.residual_record(implicit, cfg, params, state, mask, state)
        rows.append({"ns": ns, "seed": str(path), "seed_sha256": sha256_file(path), **record})
        print(ns, "preconditioned %.3e raw %.3e" % (record["preconditioned"]["norm"], record["raw"]["norm"]),
              {k: "%.2e" % v for k, v in record["preconditioned"]["blocks"].items() if v}, flush=True)
    write_json(out/"seed_residuals.json", {"schema": 1, "run_id": run_id, "deck_sha256": sha256_file(args.deck),
                                          "tcon0": float(inp.tcon0), "rows": rows}, exclusive=True)
    print(run_id)


if __name__ == "__main__":
    main()
