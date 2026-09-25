"""Compare VMEX equilibrium tangents with exact and reconverged responses."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import vmex
from vmex.core import implicit
from vmex.core.extender import _cartesian_derivative, _invert_coordinates
from vmex.core.solver import SpectralState
from vmex.core.virtual_casing import _state_field_spectra

from analytic import ROOT, cases, field, flux, iota, label_at_s, surface
from build_inputs import FIELD_T, LENGTH_M, MU0, boundary_coefficients, fit_profiles
from measurement import exact_field_parameter_tangents, scaled_response_error
import refinement_observer
from evidence import reserve_run_directory, sha256_file, source_metadata, write_json


PINNED_VMEX = "b5f5267efc0795c4a49a224e321e9b370975c14c"
PARAMETERS = ("c", "delta")
PARAMETER_INDEX = {"c": 2, "delta": 3}
PARAMETER_SCALES = {"c": 1.0, "delta": 1.0}
# Plan eq. (3) uses a_star = the base parameter value (fractional change).
BLOCK_SCALES = {
    "boundary_m": LENGTH_M,
    "pressure_am_pa": FIELD_T**2/MU0,
    "iota_ai": 1.0,
    "current_shape_ac": 1.0,
    "phiedge_wb": FIELD_T*LENGTH_M**2,
    "curtor_a": FIELD_T*LENGTH_M/MU0,
    "pres_scale": 1.0,
}
STATE_FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, nargs="+", default=(17, 33))
    parser.add_argument("--fd-steps", type=float, nargs="+",
                        default=(1e-3, 3e-4, 1e-4))
    parser.add_argument("--frozen-steps", type=float, nargs="+",
                        default=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6))
    parser.add_argument("--expected-vmex", default=PINNED_VMEX)
    parser.add_argument("--output-parent", type=Path,
                        default=ROOT/"results/vmex/response_runs")
    return parser


def _input_for_parameters(base_input, parameters):
    base_case = cases()["integer_axisymmetric"]
    case = replace(base_case, parameters=tuple(map(float, parameters)))
    rows, ntor, boundary_error, omitted = boundary_coefficients(case)
    pressure_fit, iota_fit, current_fit, profile_errors = fit_profiles(case)
    mpol = int(np.max(rows[:, 0]))+1
    shape = (2*ntor+1, mpol)
    boundary = {name: np.zeros(shape) for name in ("rbc", "rbs", "zbc", "zbs")}
    for m, n, rc, rs, zc, zs in rows:
        ni, mi = int(n+ntor), int(m)
        boundary["rbc"][ni, mi] = rc*LENGTH_M
        boundary["rbs"][ni, mi] = rs*LENGTH_M
        boundary["zbc"][ni, mi] = zc*LENGTH_M
        boundary["zbs"][ni, mi] = zs*LENGTH_M

    def pad(values, size=21):
        return np.pad(np.asarray(values, dtype=float), (0, max(0, size-len(values))))[:size]

    updated = replace(
        base_input,
        **boundary,
        am=pad(pressure_fit.coef*FIELD_T**2/MU0),
        # NCURR=1: AI is not a prescribed input; retain its inert deck value.
        phiedge=FIELD_T*LENGTH_M**2*float(flux(case, case.edge)),
        ac=pad(current_fit.deriv().coef),
        curtor=float(current_fit(1.)*FIELD_T*LENGTH_M/MU0),
    )
    if max(profile_errors.values()) > 1e-8 or boundary_error*LENGTH_M > 1e-6:
        raise ValueError("perturbed exact-family input did not meet its fit smoke gate")
    return updated, {"pressure": pressure_fit.degree(), "iota": iota_fit.degree(),
                     "current": current_fit.degree(), "ntor": ntor}, {
                         "boundary_max_error_m": boundary_error*LENGTH_M,
                         "omitted_symmetric_coefficient_m": omitted*LENGTH_M,
                         **profile_errors,
                     }


def _tree_dot(left, right):
    return sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(left),
                                              jax.tree.leaves(right)))


def _tree_norm(tree):
    return float(np.sqrt(sum(np.vdot(np.asarray(x), np.asarray(x)).real
                             for x in jax.tree.leaves(tree))))


def _tree_difference(left, right):
    return jax.tree.map(lambda a, b: a-b, left, right)


def _tree_scale(tree, scalar):
    return jax.tree.map(lambda a: a*scalar, tree)


def _tree_arrays(prefix, tree):
    return {f"{prefix}_{name}": np.asarray(getattr(tree, name)) for name in STATE_FIELDS}


def _block_norms(tangent):
    groups = {
        "boundary_m": ("rbc", "rbs", "zbc", "zbs"),
        "pressure_am_pa": ("am",),
        "iota_ai": ("ai",),
        "current_shape_ac": ("ac", "ac_aux_f"),
        "phiedge_wb": ("phiedge",),
        "curtor_a": ("curtor",),
        "pres_scale": ("pres_scale",),
    }
    result = {}
    for group, fields in groups.items():
        values = [np.asarray(getattr(tangent, name), dtype=float).ravel() for name in fields]
        result[group] = {
            "l2": float(np.linalg.norm(np.concatenate(values))),
            "scaled_l2": float(np.linalg.norm(np.concatenate(values))/BLOCK_SCALES[group]),
        }
    return result


def _block_scaled_difference(left, right):
    groups = {
        "boundary_m": ("rbc", "rbs", "zbc", "zbs"),
        "pressure_am_pa": ("am",),
        "iota_ai": ("ai",),
        "current_shape_ac": ("ac", "ac_aux_f"),
        "phiedge_wb": ("phiedge",),
        "curtor_a": ("curtor",),
        "pres_scale": ("pres_scale",),
    }
    result = {}
    for group, fields in groups.items():
        coarse = np.concatenate([np.asarray(getattr(left, name), dtype=float).ravel()
                                 for name in fields])
        fine = np.concatenate([np.asarray(getattr(right, name), dtype=float).ravel()
                               for name in fields])
        change = float(np.linalg.norm(coarse-fine)/BLOCK_SCALES[group])
        result[group] = {
            "absolute_change_in_declared_scale": change,
            "relative_change": change/max(float(np.linalg.norm(fine)/BLOCK_SCALES[group]),
                                           1e-30),
        }
    return result


def _to_json_tangent(tangent):
    return {name: np.asarray(getattr(tangent, name)).tolist()
            for name in implicit.ImplicitParams.__dataclass_fields__}


def _physical_field_function(inp, cfg, points, initial_flux):
    def evaluate(state, params):
        runtime = implicit.runtime_from_params(params, cfg)
        spectra = _state_field_spectra(inp, state, runtime)
        coordinates, valid = _invert_coordinates(
            spectra, points, newton_iterations=12, initial_flux=initial_flux)
        return _cartesian_derivative(spectra, 0, coordinates, valid)
    return jax.jit(evaluate)


def _sample_points(case):
    coordinates = ((0.23, 0.7, 0.31), (0.23, 2.2, 1.41),
                   (0.61, 1.1, 0.83), (0.61, 2.7, 2.14))
    rows = []
    for s, theta, phi in coordinates:
        label = label_at_s(case, s)
        rows.append(LENGTH_M*np.asarray(surface(case, label, theta, phi), dtype=float))
    return np.asarray(rows)


def _exact_field_tangents(case, points):
    # (point, B component, c/delta direction)
    return exact_field_parameter_tangents(case, points, (2, 3),
                                          length_m=LENGTH_M, field_t=FIELD_T)


def _root_and_anchor(cfg, params):
    """Raw host state plus exactly one observed, uncached refinement.

    Replaces the historical observer, which measured after the public callback
    had already refined and then re-called the memoized refinement (a cache
    hit).  Historical keys are kept; their values now come from the single
    pass, measured in the refinement's own operator (frozen at the raw state).
    """
    previous_stats = dict(implicit._SOLVE_STATS.get(cfg, {}))
    raw, mask, host = refinement_observer.raw_host_state(implicit, cfg, params)
    anchored, observation, _ = refinement_observer.observe_refinement(
        implicit, cfg, params, raw, mask)
    anchored = jax.block_until_ready(anchored)
    before = observation["residual_before_refinement_operator"]["preconditioned"]["norm"]
    after = observation["residual_after_refinement_operator"]["preconditioned"]["norm"]
    return anchored, mask, {"observer": "single_pass_v1",
                            "residual_before_anchor": before,
                            "residual_after_anchor": after,
                            "state_anchor_shift_l2": observation["correction_norm"],
                            "solver_state_l2": _tree_norm(raw),
                            "raw_host": host,
                            "refinement": observation,
                            "status": observation["status"],
                            "derivative_gate": refinement_observer.derivative_gate(observation),
                            "solver_stats_delta": {
                                key: (value-previous_stats.get(key, 0)
                                      if isinstance(value, (int, float)) else value)
                                for key, value in implicit._SOLVE_STATS.get(cfg, {}).items()
                            }}


def _score_response(actual, expected, fixed_scale, parameter_scale=None):
    """Historical raw-norm fields plus the invariant plan-eq.(3) record.

    ``*_l2`` and ``*_over_fixed_scale`` keep their original (sample-count
    dependent, B_star/L_star) meaning for comparison with saved reports.
    """
    difference = np.asarray(actual)-np.asarray(expected)
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    return {
        "actual_signed": actual.tolist(),
        "expected_signed": expected.tolist(),
        "actual_l2": float(np.linalg.norm(actual)),
        "expected_l2": float(np.linalg.norm(expected)),
        "absolute_error_l2": float(np.linalg.norm(difference)),
        "absolute_error_over_fixed_scale": float(np.linalg.norm(difference)/fixed_scale),
        "relative_error": (float(np.linalg.norm(difference)/np.linalg.norm(expected))
                           if np.linalg.norm(expected) > 1e-14*fixed_scale else None),
        **({"point_cloud_eq3": scaled_response_error(
            actual, expected, np.ones(len(actual)),
            parameter_scale=parameter_scale, field_scale=FIELD_T)}
           if parameter_scale is not None else {}),
    }


def _run_rung(ns, steps, frozen_steps, output, source, base_parameters,
              base_input_artifact):
    from vmex.core.statephysics import volume_average_beta
    start = perf_counter()
    base_input_path = ROOT/"inputs/input.integer_axisymmetric_current"
    parsed = vmex.VmecInput.from_file(base_input_path)
    base_input, base_degrees, base_fit_errors = _input_for_parameters(parsed, base_parameters)
    base_input = replace(base_input, ns_array=np.asarray([ns]),
                         ftol_array=np.asarray([1e-12]),
                         niter_array=np.asarray([10000]))
    cfg = implicit.make_config(base_input, ns=ns, ftol=1e-12,
                               max_iterations=10000, refine_tol=1e-11,
                               adjoint_tol=1e-10)
    params = implicit.params_from_input(base_input)
    base_state, mask, root_record = _root_and_anchor(cfg, params)
    runtime = implicit.runtime_from_params(params, cfg)
    case = cases()["integer_axisymmetric"]
    points = jnp.asarray(_sample_points(case))
    field = vmex.VmecInteriorField.from_state(base_input, base_state, runtime=runtime)
    initial_flux = field.flux_coordinates(points)
    jax.block_until_ready(initial_flux)
    field_fn = _physical_field_function(base_input, cfg, points, initial_flux)
    base_B = field_fn(base_state, params)
    jax.block_until_ready(base_B)

    input_pairs = {}
    input_tangents = {}
    fit_records = {}
    for name in PARAMETERS:
        index = PARAMETER_INDEX[name]
        for step in sorted(set((*steps, 3e-5))):
            plus_parameters = np.asarray(base_parameters, dtype=float).copy()
            minus_parameters = plus_parameters.copy()
            plus_parameters[index] += step
            minus_parameters[index] -= step
            plus_input, plus_degrees, plus_fits = _input_for_parameters(parsed, plus_parameters)
            minus_input, minus_degrees, minus_fits = _input_for_parameters(parsed, minus_parameters)
            plus_input = replace(plus_input, ns_array=base_input.ns_array,
                                 ftol_array=base_input.ftol_array,
                                 niter_array=base_input.niter_array)
            minus_input = replace(minus_input, ns_array=base_input.ns_array,
                                  ftol_array=base_input.ftol_array,
                                  niter_array=base_input.niter_array)
            if plus_degrees != base_degrees or minus_degrees != base_degrees:
                raise ValueError("profile/boundary basis changed inside the derivative ladder")
            plus_params = implicit.params_from_input(plus_input)
            minus_params = implicit.params_from_input(minus_input)
            tangent = jax.tree.map(lambda plus, minus: (plus-minus)/(2*step),
                                   plus_params, minus_params)
            key = (name, step)
            input_pairs[key] = (plus_parameters, minus_parameters,
                                plus_params, minus_params, plus_input, minus_input)
            input_tangents[key] = tangent
            fit_records[f"{name}:{step:g}"] = {
                "plus": plus_fits, "minus": minus_fits,
                "profile_degree": plus_degrees,
            }

    tangent_step = min(steps, key=lambda value: abs(value-3e-4))
    tangents = [input_tangents[(name, tangent_step)] for name in PARAMETERS]
    tangent_batch = jax.tree.map(lambda *values: jnp.stack(values), *tangents)
    tangent_started = perf_counter()
    state_tangent, tangent_report = implicit.implicit_state_tangent_multi_rhs(
        params, cfg, base_state, mask, tangent_batch,
        probe_chunk_size=2, response_chunk_size=1,
    )
    state_tangent = jax.block_until_ready(state_tangent)
    tangent_seconds = perf_counter()-tangent_started

    exact_B = _exact_field_tangents(case, np.asarray(points))
    response_records = {}
    jvp_arrays = {}
    beta_exact = {
        "c": 4*base_parameters[2]*base_parameters[3]
             /(1+base_parameters[2]**2*base_parameters[3])**2,
        "delta": 2*base_parameters[2]**2
                  /(1+base_parameters[2]**2*base_parameters[3])**2,
    }
    block_tangents = {}
    tangent_convergence = {}
    finite_difference_arrays = {}
    for direction_index, name in enumerate(PARAMETERS):
        state_dot = jax.tree.map(lambda value: value[direction_index], state_tangent)
        parameter_dot = tangents[direction_index]
        _, dB = jax.jvp(field_fn, (base_state, params), (state_dot, parameter_dot))
        dB = np.asarray(jax.block_until_ready(dB))
        _, (dbeta, diota) = jax.jvp(
            lambda state, p: (
                volume_average_beta(state, implicit.runtime_from_params(p, cfg)),
                implicit.iota_profile(state, implicit.runtime_from_params(p, cfg)),
            ),
            (base_state, params), (state_dot, parameter_dot),
        )
        dbeta = float(dbeta)
        diota = np.asarray(diota)
        expected_iota = np.zeros_like(diota)
        expected = exact_B[:, :, direction_index]
        frozen_path_rows = []
        for step in frozen_steps:
            plus_state = jax.tree.map(lambda value, direction: value+step*direction,
                                      base_state, state_dot)
            minus_state = jax.tree.map(lambda value, direction: value-step*direction,
                                       base_state, state_dot)
            plus_params = jax.tree.map(lambda value, direction: value+step*direction,
                                       params, parameter_dot)
            minus_params = jax.tree.map(lambda value, direction: value-step*direction,
                                        params, parameter_dot)
            plus_B = np.asarray(jax.block_until_ready(field_fn(plus_state, plus_params)))
            minus_B = np.asarray(jax.block_until_ready(field_fn(minus_state, minus_params)))
            frozen_fd = (plus_B-minus_B)/(2*step)
            frozen_path_rows.append({
                "step": step,
                "B_frozen_linear_path_fd": _score_response(
                    frozen_fd, expected, FIELD_T/LENGTH_M,
                    base_parameters[PARAMETER_INDEX[name]]),
                "B_frozen_fd_minus_jvp_over_fixed_scale": float(
                    np.linalg.norm(frozen_fd-dB)/(FIELD_T/LENGTH_M)),
            })
            finite_difference_arrays[f"dB_frozen_fd_{name}_h{step:.0e}"] = frozen_fd
        response_records[name] = {
            "parameter_scale": PARAMETER_SCALES[name],
            "field_at_fixed_cartesian_points": _score_response(
                dB, expected, FIELD_T/LENGTH_M, base_parameters[PARAMETER_INDEX[name]]),
            "beta_volume": {
                "vmex_jvp": dbeta,
                "exact_derivative": beta_exact[name],
                "absolute_error": abs(dbeta-beta_exact[name]),
            },
            "prescribed_current_iota_profile": {
                "vmex_jvp_signed": diota.tolist(),
                "exact_derivative_l2": float(np.linalg.norm(expected_iota)),
                "absolute_l2_over_iota_scale": float(np.linalg.norm(diota)/2.0),
            },
            "frozen_path_fd": frozen_path_rows,
        }
        jvp_arrays[f"dB_{name}"] = dB
        jvp_arrays[f"diota_{name}"] = diota
        block_tangents[name] = _block_norms(parameter_dot)
        ordered_steps = sorted(set((*steps, 3e-5)), reverse=True)
        tangent_convergence[name] = [
            {"coarser_step": coarse, "finer_step": fine,
             "by_block": _block_scaled_difference(
                 input_tangents[(name, coarse)], input_tangents[(name, fine)])}
            for coarse, fine in zip(ordered_steps, ordered_steps[1:])
        ]

    # One deterministic bilinear check relates the forward tangent and the
    # transpose response of the same frozen operator.
    cotangents = []
    for phase in (0.3, 1.1):
        cotangents.append(jax.tree.map(
            lambda value, p=phase: jnp.sin(jnp.arange(value.size, dtype=value.dtype)
                                             .reshape(value.shape)+p), base_state))
    cotangent_batch = jax.tree.map(lambda *values: jnp.stack(values), *cotangents)
    pullback_started = perf_counter()
    parameter_pullback = implicit.implicit_state_pullback_multi_rhs(
        params, cfg, base_state, mask, cotangent_batch,
        solver="block", probe_chunk_size=2, response_chunk_size=1,
    )
    parameter_pullback = jax.block_until_ready(parameter_pullback)
    pullback_seconds = perf_counter()-pullback_started
    duality = []
    for index in range(2):
        state_direction = jax.tree.map(lambda value, i=index: value[i], state_tangent)
        cotangent = jax.tree.map(lambda value, i=index: value[i], cotangent_batch)
        parameter_gradient = jax.tree.map(lambda value, i=index: value[i], parameter_pullback)
        lhs = float(_tree_dot(state_direction, cotangent))
        rhs = float(_tree_dot(tangents[index], parameter_gradient))
        duality.append({"direction": PARAMETERS[index], "jvp_dot_cotangent": lhs,
                        "input_tangent_dot_vjp": rhs,
                        "absolute_difference": abs(lhs-rhs),
                        "relative_difference": abs(lhs-rhs)/max(abs(lhs), abs(rhs), 1e-30)})

    branch_rows = {name: [] for name in PARAMETERS}
    variant_input_hashes = {}
    for name in PARAMETERS:
        for step in sorted(set(steps), reverse=True):
            plus_parameters, minus_parameters, plus_params, minus_params, plus_input, minus_input = input_pairs[(name, step)]
            variants = []
            for sign, p_variant, inp_variant in (("plus", plus_params, plus_input),
                                                  ("minus", minus_params, minus_input)):
                deck_name = f"input_{name}_ns{ns}_h{step:.0e}_{sign}.indata"
                deck_path = output/deck_name
                inp_variant.to_indata(deck_path)
                variant_input_hashes[f"{name}:{step:g}:{sign}"] = sha256_file(deck_path)
                variant_state, variant_mask, variant_root = _root_and_anchor(cfg, p_variant)
                variant_runtime = implicit.runtime_from_params(p_variant, cfg)
                variant_B = np.asarray(jax.block_until_ready(
                    field_fn(variant_state, p_variant)))
                beta = float(volume_average_beta(variant_state, variant_runtime))
                iota_values = np.asarray(implicit.iota_profile(variant_state, variant_runtime))
                variants.append((sign, variant_state, variant_B, beta, iota_values,
                                 variant_root))
                finite_difference_arrays.update(_tree_arrays(
                    f"state_{name}_h{step:.0e}_{sign}", variant_state))
            plus, minus = variants
            fd_B = (plus[2]-minus[2])/(2*step)
            fd_beta = (plus[3]-minus[3])/(2*step)
            fd_iota = (plus[4]-minus[4])/(2*step)
            jvp_B = jvp_arrays[f"dB_{name}"]
            target = exact_B[:, :, PARAMETERS.index(name)]
            row = {
                "step": step,
                "B_branch_fd": _score_response(fd_B, target, FIELD_T/LENGTH_M,
                                              base_parameters[PARAMETER_INDEX[name]]),
                "B_fd_minus_jvp_over_fixed_scale": float(
                    np.linalg.norm(fd_B-jvp_B)/(FIELD_T/LENGTH_M)),
                "beta_branch_fd": fd_beta,
                "beta_fd_minus_jvp": abs(fd_beta-response_records[name]["beta_volume"]["vmex_jvp"]),
                "beta_fd_minus_exact": abs(fd_beta-beta_exact[name]),
                "iota_branch_fd_l2_over_iota_scale": float(np.linalg.norm(fd_iota)/2.0),
                "plus_root": plus[5], "minus_root": minus[5],
            }
            branch_rows[name].append(row)
            finite_difference_arrays[f"dB_fd_{name}_h{step:.0e}"] = fd_B
            finite_difference_arrays[f"diota_fd_{name}_h{step:.0e}"] = fd_iota

    for name, rows in branch_rows.items():
        for coarse, fine in zip(rows, rows[1:]):
            coarse["successive_B_fd_change_over_fixed_scale"] = float(
                np.linalg.norm(
                    finite_difference_arrays[f"dB_fd_{name}_h{coarse['step']:.0e}"]
                    -finite_difference_arrays[f"dB_fd_{name}_h{fine['step']:.0e}"]
                )/(FIELD_T/LENGTH_M))

    response_records.update({"_branch_fd": branch_rows})
    arrays = {
        "points_xyz_m": np.asarray(points),
        "initial_vmex_flux_coordinates": np.asarray(initial_flux),
        "B_vmex_base_T": np.asarray(base_B),
        "dB_exact": exact_B,
        "dB_c": jvp_arrays["dB_c"],
        "dB_delta": jvp_arrays["dB_delta"],
        "diota_c": jvp_arrays["diota_c"],
        "diota_delta": jvp_arrays["diota_delta"],
        **_tree_arrays("state_base", base_state),
        **_tree_arrays("state_tangent_c", jax.tree.map(lambda value: value[0], state_tangent)),
        **_tree_arrays("state_tangent_delta", jax.tree.map(lambda value: value[1], state_tangent)),
        **finite_difference_arrays,
    }
    npz_path = output/f"response_ns{ns}.npz"
    with npz_path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)

    tangent_report = {
        "residual_norm": np.asarray(tangent_report.residual_norm).tolist(),
        "tolerance": np.asarray(tangent_report.tolerance).tolist(),
        "iterations": np.asarray(tangent_report.iterations).tolist(),
        "converged": np.asarray(tangent_report.converged).tolist(),
    }
    return {
        "ns": ns,
        "closure": "NCURR=1 prescribed current",
        "tcon0": float(base_input.tcon0),
        "ftol": cfg.ftol,
        "degrees_of_freedom": "VMEX fixed-boundary mask; anchored preconditioned residual",
        "fit_degrees": base_degrees,
        "fit_errors": base_fit_errors,
        "base_input_sha256": sha256_file(base_input_path),
        "base_solver_input_artifact_sha256": sha256_file(base_input_artifact),
        "root_anchor": root_record,
        # Accepted derivatives require the base root certificate (plan section 3).
        "derivative_gate": root_record["derivative_gate"],
        "sample_points_xyz_m": np.asarray(points).tolist(),
        "sample_flux_coordinate_seed": np.asarray(initial_flux).tolist(),
        "input_tangents": {
            "finite_difference_step_for_root_tangent": tangent_step,
            "input_parameter_scales": PARAMETER_SCALES,
            "output_block_scales": BLOCK_SCALES,
            "central_difference_signed_vectors": {
                name: _to_json_tangent(input_tangents[(name, tangent_step)])
                for name in PARAMETERS
            },
            "block_scaled_l2": block_tangents,
            "successive_step_changes": tangent_convergence,
            "profile_degree_checks": fit_records,
        },
        "linear_response": tangent_report,
        "tangent_seconds": tangent_seconds,
        "transpose_duality": duality,
        "transpose_seconds": pullback_seconds,
        "physical_responses": response_records,
        "response_arrays": {"path": npz_path.name, "sha256": sha256_file(npz_path)},
        "elapsed_seconds": perf_counter()-start,
    }


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.ns != sorted(set(args.ns)) or any(ns < 9 for ns in args.ns):
        raise SystemExit("NS rungs must be unique, increasing, and at least 9")
    if len(args.fd_steps) < 2 or any(not np.isfinite(h) or h <= 0 for h in args.fd_steps):
        raise SystemExit("at least two positive finite FD steps are required")
    if args.fd_steps != sorted(args.fd_steps, reverse=True):
        raise SystemExit("FD steps must be listed from coarse to fine")
    if (len(args.frozen_steps) < 2 or
            any(not np.isfinite(h) or h <= 0 for h in args.frozen_steps) or
            args.frozen_steps != sorted(args.frozen_steps, reverse=True)):
        raise SystemExit("frozen-path steps must be positive, finite and listed coarse to fine")
    source = source_metadata(vmex.__file__, "uwplasma/vmex",
                             importlib.metadata.version("vmex"))
    if source.get("commit") != args.expected_vmex:
        raise SystemExit("imported VMEX source does not match the historical pin")
    run_id = datetime.now(timezone.utc).strftime("axisym-root-response-%Y%m%dT%H%M%S.%fZ")
    run_id, output = reserve_run_directory(args.output_parent, run_id)
    base_case = cases()["integer_axisymmetric"]
    base_parameters = np.asarray(base_case.parameters, dtype=float)
    parsed = vmex.VmecInput.from_file(ROOT/"inputs/input.integer_axisymmetric_current")
    base_input, _, _ = _input_for_parameters(parsed, base_parameters)
    write_json(output/"run_receipt.json", {
        "schema": 1, "run_id": run_id, "status": "running",
        "benchmark_branch": "t0-t1-d5484d1", "upstream_pr": None,
        "source": source, "input_sha256": sha256_file(ROOT/"inputs/input.integer_axisymmetric_current"),
        "ns_rungs": args.ns, "fd_steps": args.fd_steps,
        "frozen_path_steps": args.frozen_steps,
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/axisymmetric_root_response.py --ns "
                   +" ".join(map(str, args.ns))+" --fd-steps "
                   +" ".join(f"{x:g}" for x in args.fd_steps)+" --frozen-steps "
                   +" ".join(f"{x:g}" for x in args.frozen_steps),
    }, exclusive=True)
    rung_records = []
    try:
        for ns in args.ns:
            rung_input = replace(base_input, ns_array=np.asarray([ns]),
                                 ftol_array=np.asarray([1e-12]),
                                 niter_array=np.asarray([10000]))
            rung_input_path = output/f"base_input_ns{ns}.indata"
            rung_input.to_indata(rung_input_path)
            rung_records.append(_run_rung(ns, args.fd_steps, args.frozen_steps, output, source,
                                          base_parameters, rung_input_path))
    except Exception as error:
        write_json(output/"failure.json", {
            "schema": 1, "run_id": run_id, "status": "failed",
            "error_type": type(error).__name__, "error": str(error),
            "completed_rungs": [row["ns"] for row in rung_records],
        }, exclusive=True)
        raise
    record = {
        "schema": 1,
        "run_id": run_id,
        "status": "diagnostic_axisymmetric_equilibrium_tangents",
        "accepted_derivative": False,
        "source": source,
        "evidence_separation": {
            "input_map": "signed central finite differences of the generated fixed-basis inputs",
            "equilibrium_tangent": "VMEX residual-level forward linear response",
            "reconverged_branch_fd": "independent solves at perturbed exact-family inputs",
            "analytic_reference": "exact Cartesian B, beta and prescribed-current iota derivatives",
        },
        "parameterization": "integer_axisymmetric exact family; c and delta varied; NCURR=1",
        "command": "python benchmarks/axisymmetric_root_response.py --ns "
                   +" ".join(map(str, args.ns))+" --fd-steps "
                   +" ".join(f"{x:g}" for x in args.fd_steps)+" --frozen-steps "
                   +" ".join(f"{x:g}" for x in args.frozen_steps),
        "python": sys.version.split()[0], "jax": jax.__version__,
        "numpy": np.__version__, "platform": platform.platform(),
        "devices": [str(device) for device in jax.devices()],
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if sys.platform == "darwin" else 1024)),
        "rungs": rung_records,
        "script_sha256": sha256_file(Path(__file__)),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    report = output/"response.json"
    write_json(report, record, exclusive=True)
    print(json.dumps({"run_id": run_id, "report": str(report.relative_to(ROOT)),
                      "report_sha256": sha256_file(report),
                      "rungs": [{"ns": row["ns"], "tangent": row["linear_response"],
                                 "root": row["root_anchor"],
                                 "responses": row["physical_responses"]}
                                for row in rung_records]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
