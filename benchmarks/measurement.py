"""Independent quadrature and integer-family field routines for R1."""
from __future__ import annotations

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

from analytic import Case, volume


TARGETS = {
    "field_relative_l2": 1e-5,
    "current_relative_l2": 1e-3,
    "gradp_relative_l2": 1e-3,
    "force_pressure_scale": 1e-3,
}
DEFAULT_GRID_SHIFT = (0.173, 0.371)
MAX_SAVED_SAMPLES = 16_384


def deterministic_sample_indices(count: int, maximum: int = MAX_SAVED_SAMPLES) -> np.ndarray:
    """Select a bounded, endpoint-inclusive sample independent of chunking."""
    if count < 0 or maximum < 1:
        raise ValueError("count must be nonnegative and maximum must be positive")
    retained = min(count, maximum)
    if retained == count:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count-1, retained, dtype=np.int64)


def composite_spread(rows: list[dict], *, shift=DEFAULT_GRID_SHIFT) -> tuple[dict | None, bool]:
    """Certify radial score stability over native uniform-s cells."""
    required = {
        ("cell_gauss", 0.0, 0.0),
        ("cell_gauss", float(shift[0]), float(shift[1])),
        ("cell_midpoint", 0.0, 0.0),
    }
    levels = {}
    for row in rows:
        if (row.get("sample_measure") == "full_torus_physical_volume" and
                row.get("radial_rule") in {"cell_gauss", "cell_midpoint"}):
            levels.setdefault(row.get("radial_order"), []).append(row)
    fine_rows = {}
    for order in (2, 4):
        candidates = levels.get(order, [])
        signatures = {(row["radial_rule"], row["theta_shift_fraction"],
                       row["phi_shift_fraction"]) for row in candidates}
        if not required <= signatures:
            return None, False
        fine_rows[order] = [next(row for row in candidates if
                         (row["radial_rule"], row["theta_shift_fraction"],
                          row["phi_shift_fraction"]) == signature)
                            for signature in required]

    resolved = True
    by_order = {}
    for order, order_rows in fine_rows.items():
        order_spread = {}
        for metric, limit in TARGETS.items():
            values = [row["route_A"][metric] for row in order_rows]
            order_spread[metric] = max(values)-min(values)
            if order_spread[metric] > 0.1*limit:
                resolved = False
        order_spread["max_volume_relative_error"] = max(
            row["volume_relative_error"] for row in order_rows)
        if order_spread["max_volume_relative_error"] > 1e-10:
            resolved = False
        order_spread["max_route_A_B_difference"] = {
            metric: max(row["route_A_B_difference"][metric] for row in order_rows)
            for metric in TARGETS
        }
        if any(order_spread["max_route_A_B_difference"][metric] > 0.1*limit
               for metric, limit in TARGETS.items()):
            resolved = False
        by_order[str(order)] = order_spread

    coarse = next(row for row in fine_rows[2]
                  if row["radial_rule"] == "cell_gauss" and
                  row["theta_shift_fraction"] == 0.0 and
                  row["phi_shift_fraction"] == 0.0)
    fine = next(row for row in fine_rows[4]
                if row["radial_rule"] == "cell_gauss" and
                row["theta_shift_fraction"] == 0.0 and
                row["phi_shift_fraction"] == 0.0)
    level_change = {}
    for metric, limit in TARGETS.items():
        level_change[metric] = abs(fine["route_A"][metric]-coarse["route_A"][metric])
        if level_change[metric] > 0.1*limit:
            resolved = False
    return {
        "radial_partition": "uniform-s cells aligned with native spline knots",
        "native_cell_count": coarse.get("radial_cells"),
        "by_order": by_order,
        "order2_to_order4_gauss_change": level_change,
    }, resolved


def native_knot_crossings(s_reference: np.ndarray, s_native: np.ndarray,
                          native_knots: np.ndarray) -> np.ndarray:
    """Interpolate actual native-knot crossings along one sampled physical ray."""
    reference = np.asarray(s_reference, dtype=float)
    native = np.asarray(s_native, dtype=float)
    knots = np.asarray(native_knots, dtype=float)
    if (reference.ndim != 1 or reference.shape != native.shape or len(reference) < 2 or
            not np.isfinite(reference).all() or not np.isfinite(native).all() or
            not np.isfinite(knots).all()):
        raise ValueError("reference/native labels must be equal-length finite vectors")
    order = np.argsort(native, kind="stable")
    native, reference = native[order], reference[order]
    if np.any(np.diff(native) <= 0):
        raise ValueError("native labels along a ray must be strictly increasing")
    if np.any(np.diff(reference) <= 0):
        raise ValueError("reference labels along a ray must be strictly increasing")
    if np.any(knots < native[0]) or np.any(knots > native[-1]):
        raise ValueError("every requested knot must be bracketed by the sampled ray")
    return np.interp(knots, native, reference)


def native_radial_probes(ns: int) -> dict[str, list[float]]:
    """Return interior native-s probes near the axis, edge, cells, and knots."""
    if ns < 4:
        raise ValueError("at least four radial mesh points are required")
    h = 1/(ns-1)
    epsilon = h*1e-5
    knots = (1, ns//2, ns-2)
    interiors = (0, 1, ns//2, ns-2)
    around_knots = [k*h+direction*epsilon for k in knots for direction in (-1, 1)]
    cell_interiors = [(k+0.5)*h for k in interiors]
    return {
        "near_axis": [1e-8, 1e-6, 1e-4, 1e-2],
        "near_edge": [0.95, 0.99, 0.9999, 1-1e-8],
        "native_knots_and_cell_interiors": sorted(around_knots+cell_interiors),
    }


def integer_surface_jax(case: Case, q: jax.Array) -> jax.Array:
    """Independent exact position chart, parameterized by (s, theta, phi)."""
    if case.family != "integer" or case.phase != 0.0 or case.zshift != 0.0:
        raise ValueError("the R1 chart currently requires an unshifted integer-family case")
    a, b, c, edge = case.parameters
    s, theta, phi = q
    centre = -(a*a-b*b)/(4*c*c)
    radius = jnp.sqrt(edge*s)
    u = centre + radius*jnp.cos(2*phi+theta)
    v = radius*jnp.sin(2*phi+theta)
    ell = jnp.sqrt((1+jnp.sqrt(1-4*(u*u+v*v)))/2)
    A, C = a*(ell+u/ell), a*v/ell
    D, E = b*v/ell, b*(ell-u/ell)
    ct = E*jnp.cos(phi)-C*jnp.sin(phi)
    st = A*jnp.sin(phi)-D*jnp.cos(phi)
    norm = jnp.hypot(ct, st)
    ct, st = ct/norm, st/norm
    return jnp.stack((A*ct+C*st, D*ct+E*st,
                      c*(v*(ct*ct-st*st)-2*u*st*ct)))


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


def reference_grid(
    case: Case,
    *,
    nradial: int,
    ntheta: int,
    nphi: int,
    theta_shift: float = 0.0,
    phi_shift: float = 0.0,
    radial_rule: str = "gauss",
    radial_cells: int | None = None,
    radial_order: int = 1,
    length_m: float = 1.0,
) -> dict[str, np.ndarray | float | int | str]:
    """Build a physical-volume grid over one period and replicate its weights."""
    if min(nradial, ntheta, nphi) < 1 or not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("grid sizes and length scale must be positive")
    if radial_rule == "gauss":
        nodes, radial_weights = leggauss(nradial)
        radial, radial_weights = (nodes+1)/2, radial_weights/2
    elif radial_rule == "midpoint":
        radial = (np.arange(nradial)+0.5)/nradial
        radial_weights = np.full(nradial, 1/nradial)
    elif radial_rule in {"cell_gauss", "cell_midpoint"}:
        if radial_cells is None or radial_cells < 1 or radial_order < 1:
            raise ValueError("cell rules require positive radial_cells and radial_order")
        if nradial != radial_cells*radial_order:
            raise ValueError("nradial must equal radial_cells times radial_order")
        if radial_rule == "cell_gauss":
            nodes, local_weights = leggauss(radial_order)
            local_nodes, local_weights = (nodes+1)/2, local_weights/2
        else:
            local_nodes = (np.arange(radial_order)+0.5)/radial_order
            local_weights = np.full(radial_order, 1/radial_order)
        cells = np.arange(radial_cells)[:, None]
        radial = ((cells+local_nodes[None, :])/radial_cells).ravel()
        radial_weights = (np.broadcast_to(local_weights, (radial_cells, radial_order))
                          /radial_cells).ravel()
    else:
        raise ValueError("unknown radial quadrature rule")
    theta = 2*np.pi*(np.arange(ntheta)+theta_shift)/ntheta
    period = 2*np.pi/case.nfp
    phi = period*(np.arange(nphi)+phi_shift)/nphi
    s, th, ph = np.meshgrid(radial, theta, phi, indexing="ij")
    coordinates = np.column_stack((s.ravel(), th.ravel(), ph.ravel()))
    jacobian = np.empty((len(coordinates), 3, 3), dtype=float)
    for direction in range(3):
        shifted = coordinates.astype(np.complex128)
        shifted[:, direction] += 1e-30j
        jacobian[:, :, direction] = integer_surface_numpy(case, shifted).imag/1e-30
    signed_det = np.linalg.det(jacobian)
    density = np.abs(signed_det)
    if not np.isfinite(density).all() or np.any(density <= 0):
        raise ValueError("analytical coordinate chart has a singular/nonfinite Jacobian")
    tensor_weights = (radial_weights[:, None, None]
                      * (2*np.pi/ntheta)
                      * (period/nphi)
                      * case.nfp)
    weights = (density.reshape(s.shape)*tensor_weights).ravel()*length_m**3
    xyz = np.asarray(integer_surface_numpy(case, coordinates), dtype=float)*length_m
    return {
        "xyz": xyz,
        "weights": weights,
        "s_reference": s.ravel(),
        "coordinates": coordinates,
        "signed_jacobian": signed_det,
        "nfp_replication": case.nfp,
        "radial_rule": radial_rule,
        "radial_cells": radial_cells,
        "radial_order": radial_order if radial_rule.startswith("cell_") else None,
    }


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


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.sum(weights*np.sum(values*values, axis=-1))/np.sum(weights)))


def weighted_square_sum(values: np.ndarray, weights: np.ndarray) -> float:
    """Chunk-reducible weighted squared norm for streamed field statistics."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (values.ndim < 2 or weights.shape != (values.shape[0],) or
            not np.isfinite(values).all() or not np.isfinite(weights).all() or
            np.any(weights < 0)):
        raise ValueError("values and nonnegative finite sample weights have incompatible shapes")
    return float(np.sum(weights*np.sum(values*values, axis=tuple(range(1, values.ndim)))))


def decompose_physical_force_error(
    B: np.ndarray, J: np.ndarray, gradp: np.ndarray,
    B_reference: np.ndarray, J_reference: np.ndarray, gradp_reference: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split force error into exact bilinear field/current/profile terms."""
    arrays = [np.asarray(value, dtype=float) for value in
              (B, J, gradp, B_reference, J_reference, gradp_reference)]
    if (any(value.ndim != 2 or value.shape[1] != 3 for value in arrays) or
            any(value.shape != arrays[0].shape for value in arrays) or
            not all(np.isfinite(value).all() for value in arrays)):
        raise ValueError("physical field inputs must be finite arrays with shape (N, 3)")
    B, J, gradp, B_reference, J_reference, gradp_reference = arrays
    dB, dJ, dgradp = B-B_reference, J-J_reference, gradp-gradp_reference
    terms = {
        "dJ_cross_B_reference": np.cross(dJ, B_reference),
        "J_reference_cross_dB": np.cross(J_reference, dB),
        "dJ_cross_dB": np.cross(dJ, dB),
        "minus_dgradp": -dgradp,
    }
    total_error = (np.cross(J, B)-gradp)-(
        np.cross(J_reference, B_reference)-gradp_reference)
    return {
        "dB": dB, "dJ": dJ, "dgradp": dgradp,
        **terms,
        "total_force_error": total_error,
        "identity_defect": total_error-sum(terms.values()),
    }


def score_fields(
    numerical: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    field_t: float,
    length_m: float,
    mu0: float,
) -> dict:
    """Score physical fields at identical points with the full-torus measure."""
    B0 = float(field_t)
    L = float(length_m)
    weights = np.asarray(weights, dtype=float)
    B_ref = B0*exact["B"]
    J_ref = B0/(mu0*L)*exact["J"]
    gp_ref = B0**2/(mu0*L)*exact["gradp"]
    B = np.asarray(numerical["B"], dtype=float)
    J = np.asarray(numerical["J"], dtype=float)
    gp = np.asarray(numerical["gradp"], dtype=float)
    force = np.cross(J, B)-gp
    gradB = np.asarray(numerical["gradB"], dtype=float)
    def relative(error, reference):
        return _weighted_rms(error, weights)/_weighted_rms(reference, weights)
    return {
        "field_relative_l2": relative(B-B_ref, B_ref),
        "current_relative_l2": relative(J-J_ref, J_ref),
        "gradp_relative_l2": relative(gp-gp_ref, gp_ref),
        "force_pressure_scale": _weighted_rms(force, weights)/_weighted_rms(gp_ref, weights),
        "force_fixed_magnetic_scale": _weighted_rms(force, weights)/(B0*B0/(mu0*L)),
        "field_max_abs_T": float(np.max(np.linalg.norm(B-B_ref, axis=-1))),
        "current_max_abs_A_per_m2": float(np.max(np.linalg.norm(J-J_ref, axis=-1))),
        "force_max_abs_Pa_per_m": float(np.max(np.linalg.norm(force, axis=-1))),
        "divergence_max_abs_per_m": float(np.max(np.abs(np.trace(gradB, axis1=1, axis2=2)))),
    }


def compare_pressure_profiles(
    pressure_native: np.ndarray,
    pressure_fit_at_reference_label: np.ndarray,
    pressure_exact: np.ndarray,
    weights: np.ndarray,
    pressure_scale: float,
) -> dict[str, float]:
    """Separate profile-fit, native-label, and combined pressure errors."""
    pressure_native = np.asarray(pressure_native, dtype=float)
    pressure_fit_at_reference_label = np.asarray(
        pressure_fit_at_reference_label, dtype=float)
    pressure_exact = np.asarray(pressure_exact, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not (pressure_native.shape == pressure_fit_at_reference_label.shape == pressure_exact.shape):
        raise ValueError("pressure arrays must have identical shapes")
    if (weights.shape != pressure_native.shape or not np.isfinite(weights).all() or
            np.any(weights < 0) or np.sum(weights) <= 0 or
            not np.isfinite(pressure_scale) or pressure_scale <= 0 or
            not np.isfinite(pressure_native).all() or
            not np.isfinite(pressure_fit_at_reference_label).all() or
            not np.isfinite(pressure_exact).all()):
        raise ValueError("pressure samples, weights, and scale must be finite and valid")
    components = {
        "input_profile_fit": pressure_fit_at_reference_label-pressure_exact,
        "native_label_shift": pressure_native-pressure_fit_at_reference_label,
        "combined_at_physical_points": pressure_native-pressure_exact,
    }
    result = {}
    for name, error in components.items():
        rms = float(np.sqrt(np.sum(weights*error*error)/np.sum(weights)))
        result[f"{name}_rms_fixed_scale"] = rms/pressure_scale
        result[f"{name}_max_abs_fixed_scale"] = float(np.max(np.abs(error))/pressure_scale)
    return result


def exact_volume_m3(case: Case, length_m: float = 1.0) -> float:
    return float(volume(case)*length_m**3)
