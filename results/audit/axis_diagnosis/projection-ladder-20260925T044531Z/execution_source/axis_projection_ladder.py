"""C2 axis diagnosis: exact integer-3D projection versus NS on near-axis points.

The NS129 zero-TCON root and its exact projection share a 3.5e-4 axis |B|
error, so the defect precedes the solver.  If it is radial representation it
should fall with NS; an axis-closure/evaluation rule would not.  No solve.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import ROOT, cases, field
from evidence import capture_execution_source, reserve_run_directory, sha256_file, write_json
import refinement_observer as observer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, nargs="+", default=(33, 65, 129, 257))
    parser.add_argument("--axis-s", type=float, nargs="+", default=(1e-8, 1e-4, 1e-2, 0.5))
    parser.add_argument("--angles", type=int, default=16)
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/audit/axis_diagnosis")
    args = parser.parse_args(argv)
    saved_argv, sys.argv = sys.argv, [sys.argv[0]]  # project_integer_vmex parses argv at import
    import project_integer_vmex as projection
    sys.argv = saved_argv
    import vmex
    from vmex.core import implicit
    from vmex.core.extender import _cartesian_derivative, _invert_coordinates
    from vmex.core.virtual_casing import _state_field_spectra
    from measurement import integer_surface_numpy
    run_id, out = reserve_run_directory(
        args.output_parent, f"projection-ladder-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    here = Path(__file__).resolve().parent
    capture_execution_source(out, [Path(__file__), here/"project_integer_vmex.py", here/"analytic.py",
                                   here/"measurement.py", here/"evidence.py"],
                             packages={"uwplasma/vmex": vmex.__file__})
    started = perf_counter()
    case = cases()["integer_3d"]
    deck = ROOT/"inputs/input.integer_3d_iota"
    base = vmex.VmecInput.from_file(deck)
    n = args.angles
    th, ph = np.meshgrid(2*np.pi*np.arange(n)/n, (2*np.pi/case.nfp)*np.arange(n)/n, indexing="ij")
    rows = []
    for ns in args.ns:
        inp = replace(base, ns_array=np.asarray([ns]))
        state = projection.project(inp, case, ns=ns)
        cfg = implicit.make_config(inp, ns=ns, ftol=1e-12, max_iterations=10, refine_tol=1e-11)
        implicit._template_runtime(cfg)
        params = implicit.params_from_input(inp)
        runtime = implicit.runtime_from_params(params, cfg)
        spectra = _state_field_spectra(inp, state, runtime)
        row = {"ns": ns, "state_sha256": observer.tree_hash(state), "by_s": {}}
        for s in args.axis_s:
            q = np.column_stack([np.full(th.size, s), th.ravel(), ph.ravel()])
            xyz = integer_surface_numpy(case, q)
            points = jnp.asarray(xyz)
            seed = vmex.VmecInteriorField.from_state(inp, state, runtime=runtime).flux_coordinates(points)
            coords, valid = _invert_coordinates(spectra, points, newton_iterations=12, initial_flux=seed)
            B = np.asarray(_cartesian_derivative(spectra, 0, coords, valid))
            E = np.stack([np.asarray(field(case, jnp.asarray(p))[0]) for p in xyz])
            b = E/np.linalg.norm(E, axis=1)[:, None]
            dB = B-E
            scale = np.sqrt(np.mean(np.sum(E*E, 1)))
            row["by_s"][f"{s:g}"] = {
                "all_valid": bool(np.all(valid)),
                "B_relative_l2": float(np.linalg.norm(dB)/np.linalg.norm(E)),
                "parallel_rms_over_B": float(np.sqrt(np.mean(np.sum(dB*b, 1)**2))/scale),
                "flux_label_error_max": float(np.max(np.abs(np.asarray(coords)[..., 0]**2-s))),
            }
        rows.append(row)
        print(ns, {k: "%.2e" % v["B_relative_l2"] for k, v in row["by_s"].items()}, flush=True)
    record = {"schema": 1, "run_id": run_id, "evidence": "exact_projection_no_solve",
              "deck_sha256": sha256_file(deck), "points": "exact chart at s, n x n angles (one period)",
              "rows": rows, "elapsed_seconds": perf_counter()-started}
    write_json(out/"ladder.json", record, exclusive=True)
    print(run_id)


if __name__ == "__main__":
    main()
