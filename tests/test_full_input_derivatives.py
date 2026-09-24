"""Regression checks for finite differences of the complete analytic input map."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

from analytic import cases
from build_inputs import fit_profiles
from full_input_derivatives import (
    _exact_beta_derivative,
    _flatten,
    axisymmetric_integer_input_map,
    independent_ampere_profile_check,
)


def test_axisymmetric_complete_input_map_has_stable_fixed_basis_tangents():
    base = cases()["integer_axisymmetric"]
    parameters = np.asarray(base.parameters, dtype=float)
    derivatives = {}
    for parameter_name, index in (("c", 2), ("delta", 3)):
        estimates = []
        for step in (1e-3, 3e-4):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += step
            minus[index] -= step
            plus_values, plus_degrees, _ = axisymmetric_integer_input_map(plus)
            minus_values, minus_degrees, _ = axisymmetric_integer_input_map(minus)
            assert plus_degrees == minus_degrees == {
                "pressure": 8, "iota": 8, "enclosed_current": 8, "ntor": 0,
            }
            plus_vector, names = _flatten(plus_values)
            minus_vector, minus_names = _flatten(minus_values)
            assert names == minus_names
            estimates.append((plus_vector-minus_vector)/(2*step))
            if parameter_name == "delta":
                beta_difference = (plus_values["beta_volume"]-
                                   minus_values["beta_volume"])/(2*step)
                np.testing.assert_allclose(
                    beta_difference, _exact_beta_derivative(parameters, index),
                    rtol=1e-6, atol=0,
                )
                np.testing.assert_allclose(
                    (plus_values["PHIEDGE_Wb"]-minus_values["PHIEDGE_Wb"])/(2*step),
                    np.pi,
                    rtol=1e-10,
                    atol=0,
                )
        derivatives[parameter_name] = estimates

    for estimates in derivatives.values():
        relative_change = np.linalg.norm(estimates[0]-estimates[1])/np.linalg.norm(estimates[1])
        assert relative_change < 1e-5
        assert np.isfinite(estimates[1]).all()


def test_input_current_derivative_matches_independent_ampere_loop_quadrature():
    case = cases()["integer_axisymmetric"]
    _, _, enclosed_current, _ = fit_profiles(case)
    report = independent_ampere_profile_check(
        case, enclosed_current, angular_counts=(256, 512), steps=(1e-4, 5e-5))
    assert max(report["angular_quadrature_convergence"][str(s)]["absolute_difference"]
               for s in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) < 1e-13
    assert max(report["max_relative_AC_difference_by_angular_count"].values()) < 1e-9
    assert report["CURTOR_scaling_check"]["relative_difference"] < 1e-10
