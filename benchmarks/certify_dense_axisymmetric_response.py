"""Independently solve the smallest axisymmetric residual tangent as a dense system."""
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import vmex
from vmex.core import implicit

from analytic import ROOT, cases
from axisymmetric_root_response import (
    PARAMETER_INDEX, _input_for_parameters, _root_and_anchor, _sample_points,
    _tree_norm,
)
from evidence import reserve_run_directory, sha256_file, source_metadata, write_json

PIN = "b5f5267efc0795c4a49a224e321e9b370975c14c"
NS = 17
INPUT = ROOT/"inputs/input.integer_axisymmetric_current"
SOURCE_REPORT = ROOT/"results/vmex/response_runs/axisym-root-response-20260924T230821.158785Z/response.json"
SOURCE_ARRAYS = ROOT/"results/vmex/response_runs/axisym-root-response-20260924T230821.158785Z/response_ns17.npz"
OUTPUT_PARENT = ROOT/"results/audit/dense_axisymmetric_response"
TANGENT_STEP = 3e-4
CHUNK = 8
FIELDS = implicit._STATE_FIELDS


def _tree_diff(left, right):
    return jax.tree.map(lambda a, b: a-b, left, right)


def main():
    started = perf_counter()
    source = source_metadata(vmex.__file__, "uwplasma/vmex",
                             importlib.metadata.version("vmex"))
    if source.get("commit") != PIN:
        raise SystemExit("imported VMEX source differs from historical benchmark pin")
    run_id = datetime.now(timezone.utc).strftime("dense-axisym-response-%Y%m%dT%H%M%S.%fZ")
    run_id, out = reserve_run_directory(OUTPUT_PARENT, run_id)
    parsed = vmex.VmecInput.from_file(INPUT)
    case = cases()["integer_axisymmetric"]
    base_parameters = np.asarray(case.parameters, dtype=float)
    base_input, degrees, fits = _input_for_parameters(parsed, base_parameters)
    base_input = replace(base_input, ns_array=np.asarray([NS]),
                         ftol_array=np.asarray([1e-12]),
                         niter_array=np.asarray([10000]))
    cfg = implicit.make_config(base_input, ns=NS, ftol=1e-12,
                               max_iterations=10000, refine_tol=1e-11,
                               adjoint_tol=1e-10)
    params = implicit.params_from_input(base_input)
    state, mask, root = _root_and_anchor(cfg, params)
    frozen = jax.tree.map(jax.lax.stop_gradient, state)
    project = implicit._dof_projector(cfg, mask)
    z_star = project(state)
    edge = implicit._edge_mask(cfg)
    residual = implicit.residual_fn(cfg, frozen, mask, formulation="preconditioned")
    active_fields = implicit._active_state_fields(cfg)
    indices = {
        name: np.flatnonzero(np.asarray(getattr(mask, name)).reshape(-1)).astype(np.int32)
        for name in active_fields
    }
    size = sum(int(values.size) for values in indices.values())
    offsets = {}
    cursor = 0
    for name in active_fields:
        offsets[name] = slice(cursor, cursor+indices[name].size)
        cursor += indices[name].size

    def unpack(vector):
        values = {}
        for name in FIELDS:
            base = jnp.zeros_like(getattr(state, name))
            if name in indices:
                base = base.reshape(-1).at[indices[name]].set(vector[offsets[name]])
                base = base.reshape(getattr(state, name).shape)
            values[name] = base
        return implicit.SpectralState(**values)

    def pack_active(tree):
        return jnp.concatenate([
            getattr(tree, name).reshape(-1)[indices[name]] for name in active_fields
        ])

    zero = jnp.zeros((size,), dtype=state.R_cos.dtype)

    def reduced_residual(vector, prm):
        trial = jax.tree.map(jnp.add, z_star, unpack(vector))
        return pack_active(residual(trial, prm))

    @jax.jit
    def jacobian_columns(prm, basis):
        return jax.vmap(lambda direction: jax.jvp(
            lambda vector: reduced_residual(vector, prm), (zero,), (direction,)
        )[1])(basis)

    matrix = np.empty((size, size), dtype=np.float64)
    eye = np.eye(size, dtype=np.float64)
    for lo in range(0, size, CHUNK):
        hi = min(size, lo+CHUNK)
        rows = jax.block_until_ready(jacobian_columns(params, jnp.asarray(eye[lo:hi])))
        matrix[:, lo:hi] = np.asarray(rows).T
    if not np.all(np.isfinite(matrix)):
        raise FloatingPointError("dense active residual Jacobian contains nonfinite values")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    smallest = float(singular_values[-1])
    largest = float(singular_values[0])
    condition = largest/smallest if smallest > 0 else float("inf")

    report_source = json.loads(SOURCE_REPORT.read_text())
    source_rung = next(row for row in report_source["rungs"] if row["ns"] == NS)
    krylov_tangent = implicit.implicit_state_tangent_multi_rhs
    tangent_inputs = []
    input_tangent_records = {}
    for name in ("c", "delta"):
        plus_parameters = base_parameters.copy()
        minus_parameters = base_parameters.copy()
        plus_parameters[PARAMETER_INDEX[name]] += TANGENT_STEP
        minus_parameters[PARAMETER_INDEX[name]] -= TANGENT_STEP
        plus_input, plus_degrees, plus_fits = _input_for_parameters(parsed, plus_parameters)
        minus_input, minus_degrees, minus_fits = _input_for_parameters(parsed, minus_parameters)
        if plus_degrees != degrees or minus_degrees != degrees:
            raise ValueError("basis changed inside dense tangent finite difference")
        plus_input = replace(plus_input, ns_array=base_input.ns_array,
                             ftol_array=base_input.ftol_array,
                             niter_array=base_input.niter_array)
        minus_input = replace(minus_input, ns_array=base_input.ns_array,
                              ftol_array=base_input.ftol_array,
                              niter_array=base_input.niter_array)
        plus_params = implicit.params_from_input(plus_input)
        minus_params = implicit.params_from_input(minus_input)
        tangent = jax.tree.map(lambda plus, minus: (plus-minus)/(2*TANGENT_STEP),
                               plus_params, minus_params)
        tangent_inputs.append(tangent)
        input_tangent_records[name] = {
            "step": TANGENT_STEP,
            "fit_errors_plus": plus_fits,
            "fit_errors_minus": minus_fits,
        }
    tangent_batch = jax.tree.map(lambda *items: jnp.stack(items), *tangent_inputs)
    krylov_state_tangent, krylov_report = implicit.implicit_state_tangent_multi_rhs(
        params, cfg, state, mask, tangent_batch,
        probe_chunk_size=2, response_chunk_size=1,
    )
    krylov_state_tangent = jax.block_until_ready(krylov_state_tangent)

    direction_results = {}
    dense_full_tangents = {}
    dense_vectors = {}
    dense_rhs = {}
    for direction, name in enumerate(("c", "delta")):
        p_dot = tangent_inputs[direction]
        rhs = -jax.jvp(lambda prm: reduced_residual(zero, prm),
                       (params,), (p_dot,))[1]
        rhs = np.asarray(jax.block_until_ready(rhs), dtype=np.float64)
        solution = np.linalg.solve(matrix, rhs)
        linear_defect = matrix@solution-rhs
        relative_linear_defect = float(np.linalg.norm(linear_defect)/max(np.linalg.norm(rhs), 1e-30))
        z_dot = unpack(jnp.asarray(solution))
        full_dot = jax.jvp(
            lambda z, prm: implicit._assemble(
                z, implicit.runtime_from_params(prm, cfg), frozen, project, edge),
            (z_star, params), (z_dot, p_dot),
        )[1]
        full_dot = jax.block_until_ready(full_dot)
        krylov_dot = jax.tree.map(lambda value: value[direction], krylov_state_tangent)
        relative_state_difference = float(_tree_norm(_tree_diff(full_dot, krylov_dot)) /
                                          max(_tree_norm(krylov_dot), 1e-30))
        direction_results[name] = {
            "rhs_l2": float(np.linalg.norm(rhs)),
            "dense_relative_residual": relative_linear_defect,
            "dense_vs_krylov_full_state_relative_l2": relative_state_difference,
            "krylov_report": {
                "residual_norm": float(np.asarray(krylov_report.residual_norm[direction])),
                "tolerance": float(np.asarray(krylov_report.tolerance[direction])),
                "converged": bool(np.asarray(krylov_report.converged[direction])),
                "iterations": int(np.asarray(krylov_report.iterations[direction])),
            },
        }
        dense_vectors[name] = solution
        dense_rhs[name] = rhs
        dense_full_tangents[name] = full_dot

    array_path = out/"dense_system_ns17.npz"
    np.savez_compressed(array_path, jacobian=matrix, singular_values=singular_values,
                        c_rhs=dense_rhs["c"], delta_rhs=dense_rhs["delta"],
                        c_solution=dense_vectors["c"], delta_solution=dense_vectors["delta"])
    report = {
        "schema": 1,
        "evidence": "independent_dense_active_residual_solve",
        "command": "python benchmarks/certify_dense_axisymmetric_response.py",
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "numpy": np.__version__,
        "devices": [str(device) for device in jax.devices()],
        "status": "diagnostic_dense_linearization_check",
        "accepted_derivative": False,
        "source": source,
        "source_response_report": SOURCE_REPORT.relative_to(ROOT).as_posix(),
        "source_response_report_sha256": sha256_file(SOURCE_REPORT),
        "source_response_arrays_sha256": sha256_file(SOURCE_ARRAYS),
        "input": INPUT.relative_to(ROOT).as_posix(),
        "input_sha256": sha256_file(INPUT),
        "ns": NS, "ftol": 1e-12,
        "active_fields": active_fields,
        "active_dof_count": size,
        "dense_chunk_size": CHUNK,
        "jacobian_singular_value_max": largest,
        "jacobian_singular_value_min": smallest,
        "jacobian_condition_2": condition,
        "root_residual": root,
        "input_tangents": input_tangent_records,
        "directions": direction_results,
        "dense_system_arrays": {"path": array_path.name,
                                 "sha256": sha256_file(array_path)},
        "elapsed_seconds": perf_counter()-started,
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if sys.platform == "darwin" else 1024)),
    }
    report_path = out/"dense_response.json"
    write_json(report_path, report, exclusive=True)
    print(json.dumps({"run_id": run_id,
                      "report": report_path.relative_to(ROOT).as_posix(),
                      "report_sha256": sha256_file(report_path),
                      "active_dof_count": size,
                      "directions": direction_results,
                      "elapsed_seconds": report["elapsed_seconds"],
                      "host_peak_rss_mib": report["host_peak_rss_mib"]}, indent=2))


if __name__ == "__main__":
    main()
