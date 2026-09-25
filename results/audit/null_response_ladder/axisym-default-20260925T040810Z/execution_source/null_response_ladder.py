"""C4: axisymmetric c (nonzero) and delta (null) branch responses versus NS.

For each NS, solve the base and the centered +/-h decks with the single-pass
observer (a second observed pass only when the first does not certify), gate
every endpoint on its root certificate, and difference the Cartesian B at the
saved 96 fixed points.  The branch FD needs no input-tangent solve, so q is
matched by construction.  Metrics use plan eq.(3) with a_star = base value and
B_star = 1 T; this is a point-cloud screening, not a volume norm.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
from measurement import exact_field_parameter_tangents, scaled_response_error
import refinement_observer as observer

POINTS = ROOT/"results/audit/axisym_radial_relabel/axisym-relabel-field-20260925T010115.952375Z/radial_relabel_field_ns65.npz"
INDEX = {"c": 2, "delta": 3}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, nargs="+", default=(17, 33, 65, 129))
    parser.add_argument("--step", type=float, default=1e-4)
    parser.add_argument("--parameters", nargs="+", default=("c", "delta"))
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/audit/null_response_ladder")
    args = parser.parse_args(argv)
    import vmex
    from vmex.core import implicit
    from vmex.core.extender import _cartesian_derivative, _invert_coordinates
    from vmex.core.virtual_casing import _state_field_spectra
    from axisymmetric_root_response import _input_for_parameters
    run_id, out = reserve_run_directory(
        args.output_parent, f"axisym-default-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    here = Path(__file__).resolve().parent
    capture_execution_source(out, [Path(__file__), here/"refinement_observer.py", here/"evidence.py",
                                   here/"measurement.py", here/"axisymmetric_root_response.py",
                                   here/"build_inputs.py", here/"analytic.py"],
                             packages={"uwplasma/vmex": vmex.__file__})
    started = perf_counter()
    case = cases()["integer_axisymmetric"]
    base_parameters = np.asarray(case.parameters, dtype=float)
    points_np = np.asarray(np.load(POINTS)["points_xyz_m"])
    points = jnp.asarray(points_np)
    exact_B = np.stack([np.asarray(FIELD_T*field(case, jnp.asarray(p)/LENGTH_M)[0]) for p in points_np])
    exact_dB = exact_field_parameter_tangents(case, points_np, (2, 3), length_m=LENGTH_M, field_t=FIELD_T)
    parsed = vmex.VmecInput.from_file(ROOT/"inputs/input.integer_axisymmetric_current")
    record = {"schema": 1, "run_id": run_id, "question": "C4 c/delta branch response vs NS, default TCON",
              "points_sha256": sha256_file(POINTS), "step": args.step,
              "metric": "eq3_point_cloud_uniform_weights", "rungs": []}
    decks = {"base": base_parameters}
    for name in args.parameters:
        for sign in (1, -1):
            trial = base_parameters.copy()
            trial[INDEX[name]] += sign*args.step
            decks[f"{name}_{'plus' if sign > 0 else 'minus'}"] = trial

    for ns in args.ns:
        rung = {"ns": ns, "roots": {}, "responses": {}}
        fields = {}
        for label, parameters in decks.items():
            inp, degrees, fit = _input_for_parameters(parsed, parameters)
            inp = replace(inp, ns_array=np.asarray([ns]), ftol_array=np.asarray([1e-12]),
                          niter_array=np.asarray([10000]))
            deck = out/f"input_ns{ns}_{label}.indata"
            inp.to_indata(deck)
            cfg = implicit.make_config(inp, ns=ns, ftol=1e-12, max_iterations=10000,
                                       refine_tol=1e-11, adjoint_tol=1e-10)
            params = implicit.params_from_input(inp)
            t = perf_counter()
            raw, mask, host = observer.raw_host_state(implicit, cfg, params)
            state, first, _ = observer.observe_refinement(implicit, cfg, params, raw, mask)
            passes = [first]
            if not first["status"]["root_certified"] and first["state_changed"]:
                state, second, _ = observer.observe_refinement(implicit, cfg, params, state, mask)
                passes.append(second)
            final = passes[-1]
            runtime = implicit.runtime_from_params(params, cfg)
            spectra = _state_field_spectra(inp, state, runtime)
            seed = vmex.VmecInteriorField.from_state(inp, state, runtime=runtime).flux_coordinates(points)
            coordinates, valid = _invert_coordinates(spectra, points, newton_iterations=12, initial_flux=seed)
            B = np.asarray(_cartesian_derivative(spectra, 0, coordinates, valid))
            fields[label] = B
            rung["roots"][label] = {
                "deck_sha256": sha256_file(deck), "raw_host": host,
                "passes": [{"steps": p["attempted_steps"], "state_changed": p["state_changed"],
                            "before": p["residual_before_refinement_operator"]["preconditioned"]["norm"],
                            "after": p["residual_after_refinement_operator"]["preconditioned"]["norm"]}
                           for p in passes],
                "root_certified": final["status"]["root_certified"],
                "state_sha256": observer.tree_hash(state), "all_points_valid": bool(np.all(valid)),
                "B_relative_l2": float(np.linalg.norm(B-exact_B)/np.linalg.norm(exact_B)),
                "seconds": perf_counter()-t,
            }
            print(ns, label, rung["roots"][label]["passes"], rung["roots"][label]["B_relative_l2"], flush=True)
        for name in args.parameters:
            fd = (fields[f"{name}_plus"]-fields[f"{name}_minus"])/(2*args.step)
            expected = exact_dB[:, :, INDEX[name]-2]
            certified = all(rung["roots"][k]["root_certified"] for k in ("base", f"{name}_plus", f"{name}_minus"))
            metric = scaled_response_error(fd, expected, np.ones(len(fd)),
                                           parameter_scale=base_parameters[INDEX[name]], field_scale=FIELD_T)
            rung["responses"][name] = {
                **metric, "all_endpoint_roots_certified": certified,
                "relative_error": (float(np.linalg.norm(fd-expected)/np.linalg.norm(expected))
                                   if np.linalg.norm(expected) > 0 else None),
                "derivative_may_be_accepted": False,  # point cloud + single step: diagnostic
            }
            np.save(out/f"B_fd_ns{ns}_{name}.npy", fd)
        record["rungs"].append(rung)
        write_json(out/"ladder.partial.json", record)
    record["elapsed_seconds"] = perf_counter()-started
    write_json(out/"ladder.json", record, exclusive=True)
    (out/"ladder.partial.json").unlink()
    for rung in record["rungs"]:
        print(rung["ns"], {k: (v["scaled_rms_error"], v["relative_error"], v["all_endpoint_roots_certified"])
                           for k, v in rung["responses"].items()})


if __name__ == "__main__":
    main()
