"""Finite-difference the complete fixed-basis axisymmetric integer input map."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from analytic import ROOT, cases, field, integer_targets, surface
from build_inputs import (
    FIELD_T,
    LENGTH_M,
    MU0,
    boundary_coefficients,
    fit_profiles,
)
from evidence import reserve_run_directory, sha256_file, write_json
from measurement import integer_surface_jax


STEPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
HELDOUT_X, _ = leggauss(47)
HELDOUT_S = (HELDOUT_X+1)/2


def axisymmetric_integer_input_map(parameters):
    """Evaluate boundary, profiles, current convention, flux and scalars together."""
    base = cases()["integer_axisymmetric"]
    case = replace(base, parameters=tuple(map(float, parameters)))
    rows, ntor, boundary_error, omitted = boundary_coefficients(case)
    polp, poli, polI, profile_errors = fit_profiles(case)
    boundary_values = rows[:, 2:6]
    pressure_am = polp.coef*FIELD_T**2/MU0
    current_shape = polI.deriv().coef
    beta = float(integer_targets(*case.parameters)["beta"])
    values = {
        "boundary_coefficients": boundary_values.ravel(),
        "pressure_AM_coefficients": pressure_am,
        "iota_AI_coefficients": poli.coef,
        "current_AC_coefficients": current_shape,
        "pressure_at_heldout_s": np.asarray(polp(HELDOUT_S))*FIELD_T**2/MU0,
        "iota_at_heldout_s": np.asarray(poli(HELDOUT_S)),
        "current_shape_at_heldout_s": np.asarray(polI.deriv()(HELDOUT_S)),
        "PHIEDGE_Wb": np.asarray(FIELD_T*LENGTH_M**2*case.parameters[0]
                                  *case.parameters[1]*case.parameters[2]
                                  *np.pi*case.edge),
        "CURTOR_A": np.asarray(polI(1.)*FIELD_T*LENGTH_M/MU0),
        "volume_m3": np.asarray(2*np.pi**2*np.prod(case.parameters[:3])*case.edge
                                 *LENGTH_M**3),
        "beta_volume": np.asarray(beta),
    }
    degrees = {
        "pressure": polp.degree(),
        "iota": poli.degree(),
        "enclosed_current": polI.degree(),
        "ntor": ntor,
    }
    return values, degrees, {
        "boundary_max_error_m": boundary_error*LENGTH_M,
        "omitted_symmetric_coefficient_m": omitted*LENGTH_M,
        **profile_errors,
    }


def _flatten(values):
    names = tuple(sorted(values))
    return np.concatenate([np.asarray(values[name], dtype=float).ravel() for name in names]), names


def _exact_beta_derivative(parameters, index):
    _, _, c, delta = parameters
    denominator = (1+c*c*delta)**2
    if index == 2:
        return 4*c*delta/denominator
    if index == 3:
        return 2*c*c/denominator
    raise ValueError("only the axisymmetric c and delta directions are recorded")


def independent_ampere_profile_check(case, enclosed_current_fit, *,
                                     angular_counts=(512, 1024),
                                     steps=(2e-4, 1e-4, 5e-5)):
    """Check raw input AC(s) against independent Ampere loop-current derivatives."""
    if len(angular_counts) < 2:
        raise ValueError("at least two angular quadrature counts are required")
    sample_s = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    results = {}
    edge_values = {}
    integrators = {}
    for count in angular_counts:
        theta = jnp.arange(count)*2*jnp.pi/count

        def loop_current(s):
            q = jnp.stack((jnp.full_like(theta, s), theta, jnp.zeros_like(theta)), axis=-1)
            position = jax.vmap(lambda coordinate: integer_surface_jax(case, coordinate))(q)
            chart_jacobian = jax.vmap(jax.jacfwd(
                lambda coordinate: integer_surface_jax(case, coordinate)))(q)
            tangent = chart_jacobian[:, :, 1]
            magnetic_field = field(case, position)[0]
            return -2*jnp.pi*jnp.mean(jnp.sum(magnetic_field*tangent, axis=-1))

        loop_current = jax.jit(loop_current)
        integrators[str(count)] = loop_current
        edge_values[str(count)] = float(loop_current(jnp.asarray(1.0)))
        count_results = []
        for location in sample_s:
            for step in steps:
                derivative = (
                    -loop_current(location+2*step)+8*loop_current(location+step)
                    -8*loop_current(location-step)+loop_current(location-2*step)
                )/(12*step)
                fitted = enclosed_current_fit.deriv()(location)
                count_results.append({
                    "s": float(location),
                    "step": step,
                    "independent_dI_norm_ds": float(derivative),
                    "input_AC_at_s": float(fitted),
                    "absolute_difference": float(abs(derivative-fitted)),
                    "relative_difference": float(abs(derivative-fitted)
                                                 /max(abs(float(derivative)), 1e-14)),
                })
        results[str(count)] = count_results

    angular_convergence = {}
    low_count, high_count = map(str, (angular_counts[0], angular_counts[-1]))
    for label in (*sample_s, 1.0):
        current_low = (float(integrators[low_count](label))/edge_values[low_count])
        current_high = (float(integrators[high_count](label))/edge_values[high_count])
        angular_convergence[str(float(label))] = {
            f"normalized_current_{low_count}": current_low,
            f"normalized_current_{high_count}": current_high,
            "absolute_difference": abs(current_low-current_high),
        }
    input_curtor = float(enclosed_current_fit(1.))*FIELD_T*LENGTH_M/MU0
    independent_curtor = edge_values[high_count]*FIELD_T*LENGTH_M/MU0
    return {
        "angular_counts": list(angular_counts),
        "steps": list(steps),
        "sample_s": sample_s.tolist(),
        "current_profile_convention": (
            "AC(s) is dI/ds for the raw enclosed-current fit; CURTOR separately records "
            "the edge-current normalization."
        ),
        "edge_current_dimensionless": edge_values,
        "CURTOR_scaling_check": {
            "input_CURTOR_A": input_curtor,
            "independent_Ampere_CURTOR_A": independent_curtor,
            "relative_difference": abs(input_curtor-independent_curtor)
                                  /abs(independent_curtor),
        },
        "samples": results,
        "max_relative_AC_difference_by_angular_count": {
            count: max(row["relative_difference"] for row in rows)
            for count, rows in results.items()
        },
        "angular_quadrature_convergence": angular_convergence,
    }


def measure_axisymmetric_input_derivatives():
    base = cases()["integer_axisymmetric"]
    parameters = np.asarray(base.parameters, dtype=float)
    baseline, baseline_degrees, baseline_errors = axisymmetric_integer_input_map(parameters)
    baseline_vector, output_order = _flatten(baseline)
    measurements = {}
    for parameter_name, index in (("c", 2), ("delta", 3)):
        estimates = []
        degree_checks = []
        beta_errors = []
        for step in STEPS:
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step
            minus[index] -= step
            plus_values, plus_degrees, _ = axisymmetric_integer_input_map(plus)
            minus_values, minus_degrees, _ = axisymmetric_integer_input_map(minus)
            plus_vector, plus_order = _flatten(plus_values)
            minus_vector, minus_order = _flatten(minus_values)
            if plus_order != output_order or minus_order != output_order:
                raise RuntimeError("fixed input-map output basis changed across perturbations")
            if plus_degrees != baseline_degrees or minus_degrees != baseline_degrees:
                raise RuntimeError("profile fit degree changed across a derivative pair")
            estimates.append((plus_vector-minus_vector)/(2*step))
            degree_checks.append({"step": step, "plus": plus_degrees, "minus": minus_degrees})
            beta_error = (float((plus_values["beta_volume"]-
                                 minus_values["beta_volume"])/(2*step))
                          -_exact_beta_derivative(parameters, index))
            beta_errors.append({"step": step, "absolute_error": beta_error})

        convergence = []
        for index_step in range(len(STEPS)-1):
            coarse, fine = estimates[index_step:index_step+2]
            convergence.append({
                "coarser_step": STEPS[index_step],
                "finer_step": STEPS[index_step+1],
                "relative_vector_change": float(np.linalg.norm(coarse-fine)
                                                 /np.linalg.norm(fine)),
            })
        estimates_by_output = {}
        for name in output_order:
            indices = []
            start = 0
            for key in output_order:
                count = np.asarray(baseline[key]).size
                if key == name:
                    indices = list(range(start, start+count))
                    break
                start += count
            selected = estimates[-1]
            estimates_by_output[name] = float(np.linalg.norm(selected[indices]))
        measurements[parameter_name] = {
            "parameter_index": index,
            "steps": list(STEPS),
            "profile_degree_checks": degree_checks,
            "successive_relative_vector_changes": convergence,
            "beta_derivative_exact": _exact_beta_derivative(parameters, index),
            "beta_derivative_errors": beta_errors,
            "finest_step_output_derivative_l2_by_output": estimates_by_output,
        }

    point = jnp.asarray(surface(base, base.edge*0.37, 0.41, 0.73))

    def field_at(params):
        case = replace(base, parameters=(1., 1., params[0], params[1]))
        return field(case, point)[0]

    field_jacobian = np.asarray(jax.jacfwd(field_at)(jnp.asarray(parameters[2:])))
    field_c_fd = []
    for step in STEPS:
        plus = parameters.copy()
        minus = parameters.copy()
        plus[2] += step
        minus[2] -= step
        finite_difference = (
            np.asarray(field_at(jnp.asarray(plus[2:])))
            -np.asarray(field_at(jnp.asarray(minus[2:])))
        )/(2*step)
        field_c_fd.append({
            "step": step,
            "relative_error_to_jax": float(np.linalg.norm(finite_difference-field_jacobian[:, 0])
                                            /np.linalg.norm(field_jacobian[:, 0])),
        })

    if np.linalg.norm(baseline_vector) == 0:
        raise RuntimeError("unexpected zero complete input map")
    _, _, current_fit, _ = fit_profiles(base)
    current_conversion = independent_ampere_profile_check(base, current_fit)
    return {
        "schema": 1,
        "evidence": "axisymmetric_exact_family_complete_input_map_finite_differences",
        "solver_executed": False,
        "case": base.name,
        "parameters": parameters.tolist(),
        "profile_basis": baseline_degrees,
        "profile_and_boundary_fit_errors": baseline_errors,
        "heldout_normalized_toroidal_flux_nodes": HELDOUT_S.tolist(),
        "outputs_differentiated": list(output_order),
        "measurements": measurements,
        "current_profile_conversion": current_conversion,
        "fixed_cartesian_field_derivative": {
            "point_m": np.asarray(point).tolist(),
            "dB_dc_T_per_unit": field_jacobian[:, 0].tolist(),
            "dB_ddelta_T": field_jacobian[:, 1].tolist(),
            "dB_dc_finite_difference_errors": field_c_fd,
            "delta_derivative_is_exact_zero": bool(np.array_equal(field_jacobian[:, 1],
                                                                    np.zeros(3))),
        },
    }


def main():
    result = measure_axisymmetric_input_derivatives()
    run_id = datetime.now(timezone.utc).strftime("axisym-input-map-%Y%m%dT%H%M%S.%fZ")
    parent = ROOT/"results/reference/derivative_runs"
    run_id, output = reserve_run_directory(parent, run_id)
    result.update(
        run_id=run_id,
        script_sha256=sha256_file(Path(__file__)),
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )
    report = output/"derivatives.json"
    write_json(report, result, exclusive=True)
    print(json.dumps({
        "run_id": run_id,
        "report": report.relative_to(ROOT).as_posix(),
        "report_sha256": sha256_file(report),
        "measurements": result["measurements"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
