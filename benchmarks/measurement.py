"""Integer-family exact fields and small response/stress helpers used by the tests and the axis study."""

from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from analytic import Case


def integer_surface_numpy(case: Case, coordinates: np.ndarray) -> np.ndarray:
    """NumPy chart used for independent complex-step quadrature weights."""
    if case.family != "integer" or case.phase != 0.0 or case.zshift != 0.0:
        raise ValueError("the R1 chart currently requires an unshifted integer-family case")
    a, b, c, edge = case.parameters
    coordinates = np.asarray(coordinates)
    s, theta, phi = coordinates.T
    centre = -(a*a-b*b)/(4*c*c)
    radius = np.sqrt(edge*s)
    u = centre + radius*np.cos(2*phi+theta)
    v = radius*np.sin(2*phi+theta)
    ell = np.sqrt((1+np.sqrt(1-4*(u*u+v*v)))/2)
    A, C = a*(ell+u/ell), a*v/ell
    D, E = b*v/ell, b*(ell-u/ell)
    ct = E*np.cos(phi)-C*np.sin(phi)
    st = A*np.sin(phi)-D*np.cos(phi)
    norm = np.sqrt(ct*ct+st*st)
    ct, st = ct/norm, st/norm
    return np.column_stack((A*ct+C*st, D*ct+E*st,
                           c*(v*(ct*ct-st*st)-2*u*st*ct)))



def integer_exact_fields(case: Case, xyz: np.ndarray, *, step: float = 1e-30):
    """Numpy/complex-step B, curl(B), grad(p), dB and flux label."""
    if case.family != "integer":
        raise ValueError("independent complex-step fields currently support integer cases")
    a, b, c, _ = case.parameters
    points = np.asarray(xyz, dtype=np.complex128)

    def field_and_label(position):
        x, y, z = position.T
        q = (x/a)**2+(y/b)**2
        f = np.sqrt(2*q-q*q-4*(z/c)**2)
        B = np.column_stack(((2*z*x/c-a/b*f*y)/q,
                             (2*z*y/c+b/a*f*x)/q,
                             c*(1-q)))
        Ha = (a*a+b*b)/2-(a*a-b*b)**2/(8*c*c)
        H = (x*x+y*y+4*z*z+np.sum(B*B, axis=1))/2
        psi = (H-Ha)/(2*c*c)
        return B, psi

    B, label = field_and_label(points)
    dB = np.empty((len(points), 3, 3), dtype=float)
    dpsi = np.empty((len(points), 3), dtype=float)
    for direction in range(3):
        shifted = points.copy()
        shifted[:, direction] += 1j*step
        B_shift, psi_shift = field_and_label(shifted)
        dB[:, :, direction] = B_shift.imag/step
        dpsi[:, direction] = psi_shift.imag/step
    curl = np.column_stack((dB[:, 2, 1]-dB[:, 1, 2],
                            dB[:, 0, 2]-dB[:, 2, 0],
                            dB[:, 1, 0]-dB[:, 0, 1]))
    gradp = case.pressure_slope*dpsi
    # The analytic label is an observation, not a complex-step derivative.
    # Taking its real part explicitly avoids a lossy complex-to-real cast.
    label = np.real(label)
    return tuple(np.asarray(np.real(value), dtype=float)
                 for value in (B, curl, gradp, dB, label))



def weighted_square_sum(values: np.ndarray, weights: np.ndarray) -> float:
    """Chunk-reducible weighted squared norm for streamed field statistics."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (values.ndim < 2 or weights.shape != (values.shape[0],) or
            not np.isfinite(values).all() or not np.isfinite(weights).all() or
            np.any(weights < 0)):
        raise ValueError("values and nonnegative finite sample weights have incompatible shapes")
    return float(np.sum(weights*np.sum(values*values, axis=tuple(range(1, values.ndim)))))



def exact_field_parameter_tangents(case: Case, points_m: np.ndarray, indices, *,
                                   length_m: float, field_t: float) -> np.ndarray:
    """d B / d parameters at fixed physical points, shape (point, component, index).

    The reference is dimensionless, so it is evaluated at x/L_star and scaled
    by B_star.  For a dimensionless parameter the result is in tesla.
    """
    from dataclasses import replace
    from analytic import field
    indices = tuple(indices)
    values = []
    for point in np.asarray(points_m, dtype=float):
        def evaluate(selected, point=point):
            parameters = list(case.parameters)
            for index, value in zip(indices, selected):
                parameters[index] = value
            changed = replace(case, parameters=tuple(parameters))
            return field_t*field(changed, jnp.asarray(point)/length_m)[0]
        base = jnp.asarray([case.parameters[i] for i in indices])
        values.append(np.asarray(jax.jacfwd(evaluate)(base)))
    return np.stack(values)



def scaled_response_error(actual: np.ndarray, expected: np.ndarray, weights: np.ndarray, *,
                          parameter_scale: float, field_scale: float) -> dict:
    """Plan equation (3): (a*/B*) times the weighted RMS of a field-response error.

    ``actual``/``expected`` have shape (point, component); for a dimensionless
    parameter they are in tesla.  The value is invariant under duplicating a
    point while splitting its weight, unlike a raw Euclidean vector norm.  It is
    a point-cloud metric unless the weights are a resolved volume quadrature.
    """
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if actual.shape != expected.shape or weights.shape != (actual.shape[0],):
        raise ValueError("response arrays and weights have incompatible shapes")
    scale = parameter_scale/field_scale
    total = weighted_square_sum(np.ones((len(weights), 1)), weights)
    error = np.sqrt(weighted_square_sum(actual-expected, weights)/total)
    size = np.sqrt(weighted_square_sum(expected, weights)/total)
    return {"scaled_rms_error": float(scale*error),
            "scaled_rms_expected": float(scale*size),
            "scaled_rms_actual": float(scale*np.sqrt(weighted_square_sum(actual, weights)/total)),
            "parameter_scale": float(parameter_scale), "field_scale": float(field_scale),
            "points": int(len(weights))}



def cube_bump_test_field(x: np.ndarray, center, half_width: float, component: int):
    """v = e_component * prod_i (1 - t_i^2)^2, t = (x - center)/h, zero outside the cube.

    v and its gradient vanish on the cube faces, so eq.(6) has no boundary term
    when the cube lies inside the plasma.  Returns (v, grad_v[..., i, j] = d v_i/d x_j).
    """
    t = (np.asarray(x, dtype=float)-np.asarray(center, dtype=float))/half_width
    inside = np.all(np.abs(t) < 1, axis=-1)
    f = np.where(np.abs(t) < 1, (1-t*t)**2, 0.0)
    df = np.where(np.abs(t) < 1, -4*t*(1-t*t)/half_width, 0.0)
    phi = np.prod(f, axis=-1)*inside
    grad_phi = np.stack([df[..., j]*np.prod(np.delete(f, j, axis=-1), axis=-1)
                         for j in range(3)], axis=-1)*inside[..., None]
    v = np.zeros(t.shape)
    v[..., component] = phi
    grad_v = np.zeros(t.shape+(3,))
    grad_v[..., component, :] = grad_phi
    return v, grad_v



def weak_stress_moment(B: np.ndarray, p: np.ndarray, grad_v: np.ndarray, weights: np.ndarray, *,
                       mu0: float, v: np.ndarray | None = None,
                       div_B: np.ndarray | None = None) -> dict:
    """Plan eq.(6): the weak force moment from B and p only, without a numerical curl.

    ``-sum w T:grad v - (1/mu0) sum w (v.B) div B`` with
    ``T = BB/mu0 - (p + B^2/(2 mu0)) I``; equals ``sum w v.(J x B - grad p)`` for
    a boundary-zero v.  The divergence term is included only when ``div_B`` is
    supplied; it is reported as unavailable, never assumed zero.
    """
    B = np.asarray(B, dtype=float)
    T = B[..., :, None]*B[..., None, :]/mu0 - (np.asarray(p)+0.5*np.sum(B*B, axis=-1)/mu0)[..., None, None]*np.eye(3)
    stress = -float(np.sum(weights*np.sum(T*grad_v, axis=(-2, -1))))
    divergence = None
    if div_B is not None:
        if v is None:
            raise ValueError("the divergence term needs v")
        divergence = -float(np.sum(weights*np.sum(v*B, axis=-1)*div_B))/mu0
    return {"stress_term": stress, "divergence_term": divergence,
            "moment": stress + (divergence or 0.0),
            "divergence_term_included": divergence is not None}

