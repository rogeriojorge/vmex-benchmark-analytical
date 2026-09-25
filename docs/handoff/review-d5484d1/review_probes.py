"""Independent mathematical probes for the d5484d1 review; no VMEX imports.

Run: python review_probes.py
These experiments illustrate source-review findings, not solver performance.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import leggauss
import scipy
from scipy.integrate import solve_ivp


def composite_integral(function, breaks: np.ndarray, order: int) -> float:
    """Integrate on explicitly supplied intervals with positive Gauss weights."""
    breaks = np.asarray(breaks, dtype=float)
    if breaks.ndim != 1 or not np.all(np.diff(breaks) > 0) or order < 1:
        raise ValueError('Invalid interval partition or quadrature order')
    nodes, weights = leggauss(order)
    half = np.diff(breaks) / 2
    centres = (breaks[1:] + breaks[:-1]) / 2
    return float(np.sum(half[:, None] * weights * function(
        centres[:, None] + half[:, None] * nodes)))


def knot_partition_probe() -> dict:
    """An exactly integrable example of unequal reference/native flux labels."""
    alpha, native_knot = 0.12, 3/8
    # Native label g(s) is smooth, monotone, and fixes both endpoints.
    g = Polynomial([0, 1 + alpha, -alpha])
    crossing = 2*native_knot / (
        1 + alpha + np.sqrt((1 + alpha)**2 - 4*alpha*native_knot))
    primitive = ((g - native_knot)**2).integ()
    exact = float(primitive(1) - primitive(crossing))
    integrand = lambda s: np.maximum(g(s) - native_knot, 0)**2
    reference_breaks = np.linspace(0, 1, 9)
    corrected_breaks = np.sort(np.r_[reference_breaks, crossing])
    rows = []
    for order in (2, 3, 4, 6, 8):
        wrong = composite_integral(integrand, reference_breaks, order)
        correct = composite_integral(integrand, corrected_breaks, order)
        rows.append(dict(order=order,
                         reference_partition_absolute_error=abs(wrong-exact),
                         crossing_partition_absolute_error=abs(correct-exact)))
    assert abs(g(crossing)-native_knot) < 1e-15
    assert min(g.deriv()(np.linspace(0, 1, 21))) > 0
    assert rows[1]['crossing_partition_absolute_error'] < 2e-15
    assert rows[1]['reference_partition_absolute_error'] > 1e-10
    return dict(status='passed', meaning='synthetic quadrature example, not VMEX data',
                native_knot=native_knot, reference_crossing=crossing,
                reference_label_displacement=crossing-native_knot,
                exact_integral=exact, rows=rows)


def grid_count_probe() -> dict:
    """Reproduce the arithmetic contract in the reviewed composite driver."""
    rows = []
    for ns in (33, 65, 129):
        for order, old_count in ((2, 64), (4, 128)):
            correct_count = (ns-1)*order
            rows.append(dict(ns=ns, order=order, reviewed_hardcoded_count=old_count,
                             derived_count=correct_count,
                             reviewed_count_is_valid=old_count == correct_count))
    assert all(r['reviewed_count_is_valid'] == (r['ns'] == 33) for r in rows)
    return dict(status='passed', meaning='source-contract arithmetic reproduced without importing driver',
                rows=rows)


def integer_field(position: np.ndarray, stretches: np.ndarray) -> np.ndarray:
    """Landreman integer field with diagonal stretch, preserving complex dtype."""
    position = np.asarray(position)
    a, b, c = stretches
    x, y, z = np.moveaxis(position, -1, 0)
    q = (x/a)**2 + (y/b)**2
    f = np.sqrt(2*q-q*q-4*(z/c)**2)
    return np.stack(((2*z*x/c-a/b*f*y)/q,
                     (2*z*y/c+b/a*f*x)/q, c*(1-q)), axis=-1)


def seed_point(stretches: np.ndarray, delta: float) -> np.ndarray:
    a, b, c = stretches
    centre = -(a*a-b*b)/(4*c*c)
    if abs(centre)+np.sqrt(delta) >= 0.5:
        raise ValueError('Pressure torus leaves the smooth integer-field chart')
    u = centre + np.sqrt(0.43*delta)*np.cos(0.37)
    v = np.sqrt(0.43*delta)*np.sin(0.37)
    t = 0.61
    ell = np.sqrt((1+np.sqrt(1-4*(u*u+v*v)))/2)
    return np.array((a*(ell*np.cos(t)+(u*np.cos(t)+v*np.sin(t))/ell),
                     b*(ell*np.sin(t)+(v*np.cos(t)-u*np.sin(t))/ell),
                     c*(v*np.cos(2*t)-u*np.sin(2*t))))


def exact_flow(x0: np.ndarray, t, stretches: np.ndarray) -> np.ndarray:
    frequencies = np.array([1., 1., 2.])
    phase = np.asarray(t)[..., None]*frequencies
    return np.cos(phase)*x0 + np.sin(phase)/frequencies*integer_field(x0, stretches)


def complex_jacobian(function, point: np.ndarray) -> np.ndarray:
    directions = np.eye(3)*1e-30j
    return np.stack([np.imag(function(np.asarray(point, dtype=complex)+direction))/1e-30
                     for direction in directions], axis=-1)


def exact_flow_probe() -> dict:
    """Check exact trajectories, closure, tangent flow and volume preservation."""
    rows = []
    for stretches in (np.ones(3), np.array([np.sqrt(1.5), np.sqrt(.5), 1.]),
                      np.array([np.sqrt(1.5), np.sqrt(.5), 1.2])):
        x0 = seed_point(stretches, 1/64)
        times = np.linspace(0, 2*np.pi, 101)
        trajectories = exact_flow(x0, times, stretches)
        frequency = np.array([1., 1., 2.])
        velocities = (-frequency*np.sin(times[:, None]*frequency)*x0
                      + np.cos(times[:, None]*frequency)*integer_field(x0, stretches))
        equation_error = np.max(np.abs(velocities-integer_field(trajectories, stretches)))
        db = complex_jacobian(lambda x: integer_field(x, stretches), x0)
        tension_error = np.max(np.abs(db@integer_field(x0, stretches)
                                     + frequency**2*x0))
        t = .73
        tangent = np.diag(np.cos(frequency*t)) + np.diag(np.sin(frequency*t)/frequency)@db
        tangent_cs = complex_jacobian(lambda x: exact_flow(x, t, stretches), x0)
        composition = exact_flow(exact_flow(x0, .21, stretches), .52, stretches)
        numerical = solve_ivp(lambda _, x: integer_field(x, stretches),
                              (0, 2*np.pi), x0, method='DOP853',
                              rtol=2e-12, atol=2e-14, t_eval=times)
        if not numerical.success:
            raise RuntimeError(numerical.message)
        errors = dict(equation_max_abs=float(equation_error),
                      tension_max_abs=float(tension_error),
                      closure_max_abs=float(np.max(np.abs(trajectories[-1]-x0))),
                      group_property_max_abs=float(np.max(np.abs(composition-exact_flow(x0,t,stretches)))),
                      tangent_max_abs=float(np.max(np.abs(tangent-tangent_cs))),
                      volume_preservation_error=float(abs(np.linalg.det(tangent)-1)),
                      dop853_trajectory_max_abs=float(np.max(np.abs(numerical.y.T-trajectories))))
        assert max(errors[k] for k in errors if not k.startswith('dop853')) < 1e-11
        assert errors['dop853_trajectory_max_abs'] < 2e-9
        rows.append(dict(stretches=stretches.tolist(), **errors))
    return dict(status='passed', meaning='exact magnetic-field-line flow, not a particle orbit or VMEX trace',
                rows=rows)


def derivative_scaling_probe() -> dict:
    true = np.array([1e-3, 1e6])  # illustrative boundary and current derivatives
    perturbed = np.array([1.5e-3, 1e6])
    global_error = float(np.linalg.norm(perturbed-true)/np.linalg.norm(true))
    boundary_error = float(abs((perturbed[0]-true[0])/true[0]))
    scales = np.array([1e-3, 1e6])
    scaled_error = float(np.linalg.norm((perturbed-true)/scales)/np.linalg.norm(true/scales))
    assert global_error < 1e-8 and boundary_error == .5
    return dict(status='passed', unscaled_concatenated_relative_error=global_error,
                boundary_block_relative_error=boundary_error,
                dimensionless_block_scaled_relative_error=scaled_error,
                meaning='illustration of dimensionally incompatible aggregate norms')


def regularity_probe() -> dict:
    # Map x=s+eta*(s-k)_+^3. A solenoidal contravariant field has
    # B_y=1/(dx/ds). Geometry is C2, while B_y is only C1 in x.
    eta = .4
    eps = 1e-7
    u = eps
    denominator = 1+3*eta*u*u
    second_right = -6*eta/denominator**4 + 108*eta*eta*u*u/denominator**5
    # d/dx = (dx/ds)^(-1) d/ds.
    assert abs(second_right + 6*eta) < 1e-11
    return dict(status='passed', geometry_class='C2 piecewise cubic',
                field_class='C1, piecewise smooth',
                second_field_derivative_left=0.,
                second_field_derivative_right=float(second_right),
                limiting_jump=-6*eta,
                meaning='local Cartesian model illustrating derivative loss; not a toroidal equilibrium')


def streaming_probe() -> dict:
    rng = np.random.default_rng(92)
    weights = rng.uniform(.1, 2, 1031)
    reference = rng.normal(size=(1031,3))
    error = 1e-3*rng.normal(size=(1031,3))
    totals = np.zeros(3)
    maxima = 0.
    for start in range(0, len(weights), 127):
        w, r, e = weights[start:start+127], reference[start:start+127], error[start:start+127]
        totals += np.array([np.sum(w), np.sum(w*np.sum(r*r, axis=1)),
                            np.sum(w*np.sum(e*e, axis=1))])
        maxima = max(maxima, float(np.max(np.linalg.norm(e, axis=1))))
    streamed = np.sqrt(totals[2]/totals[1])
    direct = np.sqrt(np.sum(weights*np.sum(error*error, axis=1)) /
                     np.sum(weights*np.sum(reference*reference, axis=1)))
    assert abs(streamed-direct) < 1e-17
    return dict(status='passed', sample_count=len(weights), batch_size=127,
                final_batch_size=len(weights)%127, relative_norm=float(streamed),
                batch_vs_stream_difference=float(abs(streamed-direct)),
                error_max_norm=maxima,
                meaning='sufficient statistics check; production reductions should control summation error')


def main() -> None:
    result = dict(schema=1, solver_executed=False, benchmark_suite_executed=False,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, platform=platform.system()),
                  probes=dict(knot_partition=knot_partition_probe(),
                              grid_counts=grid_count_probe(), exact_flow=exact_flow_probe(),
                              derivative_scaling=derivative_scaling_probe(),
                              regularity=regularity_probe(), streaming=streaming_probe()))
    source = Path(__file__).resolve()
    result['script_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    output = source.parent/'results'/'review_probes.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
