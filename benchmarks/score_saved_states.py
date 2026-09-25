"""Score saved VMEX states on a fixed Cartesian point cloud against the exact field.

Point-cloud screening only (uniform weights): it compares states on identical
points and reports B relative error, native flux-label error, inversion
validity and pairwise physical B differences.  It is not a volume certificate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import ROOT, cases, field
from build_inputs import FIELD_T, LENGTH_M
from evidence import capture_execution_source, reserve_run_directory, sha256_file, write_json
import refinement_observer as observer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--ns", type=int, required=True)
    parser.add_argument("--states", type=Path, required=True, help="npz with <label>_<field> arrays")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--points", type=Path, required=True,
                        help="npz with points_xyz_m and reference_s")
    parser.add_argument("--case", default="integer_axisymmetric")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/audit/saved_state_scores")
    args = parser.parse_args(argv)
    import vmex
    from vmex.core import implicit
    from vmex.core.extender import _cartesian_derivative, _invert_coordinates
    from vmex.core.solver import SpectralState
    from vmex.core.virtual_casing import _state_field_spectra
    run_id, out = reserve_run_directory(
        args.output_parent, f"{args.label}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    here = Path(__file__).resolve().parent
    capture_execution_source(out, [Path(__file__), here/"refinement_observer.py", here/"evidence.py"],
                             packages={"uwplasma/vmex": vmex.__file__})
    started = perf_counter()
    inp = vmex.VmecInput.from_file(args.deck)
    cfg = implicit.make_config(inp, ns=args.ns, ftol=1e-12, max_iterations=10000,
                               refine_tol=1e-11, adjoint_tol=1e-10)
    params = implicit.params_from_input(inp)
    implicit._template_runtime(cfg)
    runtime = implicit.runtime_from_params(params, cfg)
    archive = np.load(args.points)
    points_np = np.asarray(archive["points_xyz_m"])
    reference_s = np.asarray(archive["reference_s"])
    points = jnp.asarray(points_np)
    case = cases()[args.case]
    exact = np.stack([np.asarray(FIELD_T*field(case, jnp.asarray(p)/LENGTH_M)[0]) for p in points_np])
    data = np.load(args.states)
    states = {k: SpectralState(**{n: jnp.asarray(data[f"{k}_{n}"]) for n in observer.FIELDS})
              for k in args.labels}
    seed = vmex.VmecInteriorField.from_state(inp, states[args.labels[0]], runtime=runtime)
    initial_flux = seed.flux_coordinates(points)

    @jax.jit
    def evaluate(state):
        spectra = _state_field_spectra(inp, state, runtime)
        coordinates, valid = _invert_coordinates(spectra, points, newton_iterations=12,
                                                 initial_flux=initial_flux)
        return _cartesian_derivative(spectra, 0, coordinates, valid), coordinates, valid

    rows, fields = {}, {}
    for label, state in states.items():
        B, coordinates, valid = (np.asarray(v) for v in evaluate(state))
        fields[label] = B
        rows[label] = {
            "state_sha256": observer.tree_hash(state),
            "all_valid": bool(np.all(valid)), "valid_count": int(np.sum(valid)),
            "B_relative_l2": float(np.linalg.norm(B-exact)/np.linalg.norm(exact)),
            # Native inversion returns rho; compare rho**2 with normalized flux s.
            "max_flux_label_error": float(np.max(np.abs(coordinates[..., 0]**2-reference_s))),
        }
    pairs = {f"{a}_minus_{b}_relative_l2": float(np.linalg.norm(fields[a]-fields[b])/np.linalg.norm(exact))
             for i, a in enumerate(args.labels) for b in args.labels[i+1:]}
    npz = out/"fields.npz"
    with npz.open("xb") as handle:
        np.savez_compressed(handle, points_xyz_m=points_np, B_exact=exact,
                            **{f"B_{k}": v for k, v in fields.items()})
    record = {"schema": 1, "run_id": run_id, "metric": "point_cloud_uniform_weights_screening",
              "deck_sha256": sha256_file(args.deck), "states_npz_sha256": sha256_file(args.states),
              "points_npz_sha256": sha256_file(args.points), "points": int(len(points_np)),
              "states": rows, "pairwise": pairs, "arrays": {"path": npz.name, "sha256": sha256_file(npz)},
              "elapsed_seconds": perf_counter()-started}
    write_json(out/"scores.json", record, exclusive=True)
    print(run_id, rows, pairs)


if __name__ == "__main__":
    main()
