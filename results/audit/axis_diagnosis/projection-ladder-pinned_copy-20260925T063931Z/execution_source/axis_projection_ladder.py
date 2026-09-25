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
    parser.add_argument("--axis-row", choices=("pinned_copy", "linear_extrapolation"), default="pinned_copy",
                        help="diagnostic only: replace the pinned regular[0] = regular[1] axis rule")
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/audit/axis_diagnosis")
    args = parser.parse_args(argv)
    saved_argv, sys.argv = sys.argv, [sys.argv[0]]  # project_integer_vmex parses argv at import
    import project_integer_vmex as projection
    sys.argv = saved_argv
    import vmex
    from vmex.core import implicit
    from vmex.core.extender import _cartesian_derivative, _invert_coordinates
    from vmex.core.virtual_casing import _state_field_spectra
    from measurement import integer_exact_fields, integer_surface_numpy
    import vmex.core.extender as extender
    if args.axis_row == "linear_extrapolation":
        original = extender._radial_table

        def patched(coefficients, modes, spline=True):
            # Same regularization as the pin, but extrapolate the smooth
            # regular coefficient linearly in s to the axis instead of copying.
            if modes is None:
                return original(coefficients, modes, spline)
            coefficients = jnp.asarray(coefficients)
            ns = coefficients.shape[0]
            powers = jnp.abs(jnp.asarray(modes))/2.0
            s_mesh = jnp.arange(ns, dtype=coefficients.dtype)/(ns-1)
            scale = s_mesh[:, None]**powers[None, :]
            regular = coefficients/jnp.where(scale == 0.0, 1.0, scale)
            regular = regular.at[0].set(jnp.where(powers > 0, 2*regular[1]-regular[2], regular[0]))
            return regular, extender._spline_moments(regular) if spline else jnp.zeros_like(regular)

        extender._radial_table = patched
    run_id, out = reserve_run_directory(
        args.output_parent, f"projection-ladder-{args.axis_row}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
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
            dB = np.asarray(_cartesian_derivative(spectra, 1, coords, valid)).reshape(-1, 3, 3)
            E = np.stack([np.asarray(field(case, jnp.asarray(p))[0]) for p in xyz])
            _, curl_exact, _, _, _ = integer_exact_fields(case, xyz)
            b = E/np.linalg.norm(E, axis=1)[:, None]
            dB = B-E
            scale = np.sqrt(np.mean(np.sum(E*E, 1)))
            row["by_s"][f"{s:g}"] = {
                "all_valid": bool(np.all(valid)),
                "B_relative_l2": float(np.linalg.norm(dB)/np.linalg.norm(E)),
                "parallel_rms_over_B": float(np.sqrt(np.mean(np.sum(dB*b, 1)**2))/scale),
                "flux_label_error_max": float(np.max(np.abs(np.asarray(coords)[..., 0]**2-s))),
            }
            # Tensor index order is not assumed: the s=0.5 control selects it.
            for tag, T in (("ij", dB), ("ji", np.swapaxes(dB, -1, -2))):
                curl = np.stack([T[:, 2, 1]-T[:, 1, 2], T[:, 0, 2]-T[:, 2, 0], T[:, 1, 0]-T[:, 0, 1]], -1)
                row["by_s"][f"{s:g}"][f"J_relative_l2_{tag}"] = float(
                    np.linalg.norm(curl-curl_exact)/np.linalg.norm(curl_exact))
        rows.append(row)
        print(ns, {k: "B %.2e J %.2e/%.2e" % (v["B_relative_l2"], v["J_relative_l2_ij"], v["J_relative_l2_ji"])
                   for k, v in row["by_s"].items()}, flush=True)
    record = {"schema": 1, "run_id": run_id, "evidence": "exact_projection_no_solve",
              "axis_row": args.axis_row,
              "deck_sha256": sha256_file(deck), "points": "exact chart at s, n x n angles (one period)",
              "rows": rows, "elapsed_seconds": perf_counter()-started}
    write_json(out/"ladder.json", record, exclusive=True)
    print(run_id)


if __name__ == "__main__":
    main()
