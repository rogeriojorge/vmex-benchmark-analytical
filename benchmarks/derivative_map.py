"""Parameter -> input map, exact tangents, volume quadrature and gates for the integer family.

Everything here is independent of VMEX and of any solve.  The map is written in JAX so the input
tangent P_a is an exact automatic derivative rather than a finite difference of a fitted deck.
The only truncation is the Fourier fit of the boundary, which is linear in the sampled boundary, so
its derivative is the fit of the exact boundary derivative.  Profiles are exact for this family
(tau = 0): iota = -2 (counterclockwise R-Z angle), p(s) = 2 c^2 delta (1 - s) and
Psi_edge = pi a b c delta, with mu0 = 4e-7 pi, L* = 1 m, B* = 1 T as in ``build_inputs``.

Gates (proposed engineering targets, fixed before any result was seen, see STATUS.md):
nonzero response relative RMS error <= 1e-3; null response scaled RMS <= 1e-5.
"""
from dataclasses import replace

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from analytic import cases, label_at_s, surface
from measurement import exact_field_parameter_tangents

MU0 = 4e-7*np.pi  # same fixed convention as build_inputs.py
PARAMETERS = {"a": 0, "b": 1, "c": 2, "delta": 3}
GATES = {"nonzero_relative": 1e-3, "null_scaled": 1e-5}
ROOT_GATE = 1e-9  # largest accepted |F| (preconditioned and raw) at the returned state
CLOUD_S = (1e-4, 1e-3, 1e-2, .05, .15, .3, .5, .7, .85, .95)


def integer_surface(p, label, theta, phi):
    """JAX copy of analytic.surface for the integer family (tau = 0, no phase or shift)."""
    a, b, c = p[0], p[1], p[2]
    u0 = -(a*a-b*b)/(4*c*c)
    beta = 2*phi+theta
    u, v = u0+jnp.sqrt(label)*jnp.cos(beta), jnp.sqrt(label)*jnp.sin(beta)
    ell = jnp.sqrt((1+jnp.sqrt(1-4*(u*u+v*v)))/2)
    A, C, D, E = a*(ell+u/ell), a*v/ell, b*v/ell, b*(ell-u/ell)
    ct, st = E*jnp.cos(phi)-C*jnp.sin(phi), A*jnp.sin(phi)-D*jnp.cos(phi)
    norm = jnp.hypot(ct, st)
    ct, st = ct/norm, st/norm
    return jnp.stack((A*ct+C*st, D*ct+E*st, c*(v*(ct*ct-st*st)-2*u*st*ct)), -1)


def boundary_rz(p, theta, phi):
    """R, Z of the boundary at CCW angle theta and cylindrical angle phi."""
    xyz = integer_surface(p, p[3], theta, phi)
    return jnp.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]


def vmex_boundary(p, mpol, ntor, nfp, grid=128):
    """Dense INDATA rbc/zbs arrays [n+ntor, m]; the FFT of ``build_inputs.boundary_coefficients``."""
    theta = jnp.arange(grid)*2*jnp.pi/grid
    phi = jnp.arange(grid)*2*jnp.pi/(grid*nfp)
    R, Z = boundary_rz(p, theta[:, None], phi[None, :])
    fr, fz = jnp.fft.fft2(R)/grid**2, jnp.fft.fft2(Z)/grid**2
    rbc = jnp.zeros((2*ntor+1, mpol))
    zbs = jnp.zeros((2*ntor+1, mpol))
    for m in range(mpol):
        for n in range(0 if m == 0 else -ntor, ntor+1):
            factor = 1. if m == n == 0 else 2.
            rbc = rbc.at[n+ntor, m].set(factor*fr[m, -n].real)
            zbs = zbs.at[n+ntor, m].set(-factor*fz[m, -n].imag)
    return rbc, zbs


def profiles(p):
    """Exact profile coefficients: AM in Pa (powers of s), AI, PHIEDGE in Wb."""
    a, b, c, delta = p
    am = jnp.array([2*c*c*delta, -2*c*c*delta])/MU0
    return am, jnp.array([-2.]), jnp.pi*a*b*c*delta


def unit(name):
    return jnp.zeros(4).at[PARAMETERS[name]].set(1.)


def point_cloud(case, s_values=CLOUD_S, thetas=(.3, 1.9, 3.5, 5.1), phis=(.2, .9)):
    """Fixed physical points: the historical 80-point cloud on exact flux surfaces."""
    rows = [(s, t, f) for s in s_values for t in thetas for f in phis]
    s, t, f = np.asarray(rows).T
    xyz = np.asarray(surface(case, label_at_s(case, s), t, f), dtype=float)
    return xyz, np.asarray(rows)


PANEL_EDGES = (0.0, 0.01, 0.03, 0.1, 0.3)


def volume_rule(case, *, ns=24, ntheta=32, nphi=2, s_max=0.9, panels=False):
    """Cell-resolved volume quadrature over {s <= s_max} of the exact chart, one field period.

    Gauss-Legendre in the pressure label (s = label/edge for this family), uniform in theta and
    phi; ``panels=True`` makes it composite (ns nodes per panel) to resolve the near-axis shells.  The weight is the absolute determinant of the exact chart's automatic derivative times
    the rule weights, so no solver geometry enters the measure.  Returns points, weights, (s,theta,phi).
    """
    x, w = leggauss(ns)
    if panels:  # composite rule: ns Gauss nodes on each of the panels (0,.01,.03,.1,.3,s_max)
        edges = [e for e in PANEL_EDGES if e < s_max]+[s_max]
        label = np.concatenate([case.edge*(lo+(hi-lo)*(x+1)/2) for lo, hi in zip(edges[:-1], edges[1:])])
        wl = np.concatenate([case.edge*(hi-lo)*w/2 for lo, hi in zip(edges[:-1], edges[1:])])
        ns = len(label)
    else:
        label = case.edge*s_max*(x+1)/2
        wl = case.edge*s_max*w/2
    theta = 2*np.pi*np.arange(ntheta)/ntheta
    phi = 2*np.pi*(np.arange(nphi)+.5)/(nphi*case.nfp)
    L, T, F = (a.ravel() for a in np.meshgrid(label, theta, phi, indexing="ij"))
    p = jnp.asarray(case.parameters)

    def chart(q):
        return integer_surface(p, q[0], q[1], q[2])
    det = jax.jit(jax.vmap(lambda q: jnp.abs(jnp.linalg.det(jax.jacfwd(chart)(q)))))
    jac = np.asarray(det(jnp.stack((jnp.asarray(L), jnp.asarray(T), jnp.asarray(F)), -1)))
    shape = (ns, ntheta, nphi)
    weights = jac*np.broadcast_to(wl[:, None, None], shape).ravel()*(2*np.pi/ntheta)*(2*np.pi/case.nfp/nphi)
    xyz = np.asarray(jax.vmap(chart)(jnp.stack((jnp.asarray(L), jnp.asarray(T), jnp.asarray(F)), -1)))
    coords = np.stack((L/case.edge, T, F), -1)
    return xyz, weights, coords


def exact_tangents(case, xyz, names=("c", "delta")):
    """(point, component, parameter) exact dB/dp at fixed Cartesian points, tesla."""
    return exact_field_parameter_tangents(case, xyz, tuple(PARAMETERS[n] for n in names),
                                          length_m=1., field_t=1.)


def weighted_rms(x, weights=None):
    x = np.asarray(x, dtype=float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    finite = np.isfinite(x).all(axis=tuple(range(1, x.ndim)))
    return float(np.sqrt(np.sum(w[finite]*np.sum(x[finite]**2, axis=tuple(range(1, x.ndim))))/np.sum(w[finite])))


def scores(dB, exact, name, case, weights=None):
    """Nonzero parameters: relative weighted RMS error.  delta (null): (delta0/B*) * weighted RMS |dB|.

    ``weights=None`` is a point-cloud metric; with a volume rule it is the whole-volume norm.
    The null scaled metric uses the absolute scale a* = delta0, B* = 1 T, never a relative error
    divided by a zero exact response.
    """
    diff = np.asarray(dB)-np.asarray(exact)
    out = {"rms_error_T": weighted_rms(diff, weights), "rms_exact_T": weighted_rms(exact, weights),
           "finite_points": int(np.isfinite(diff).all(-1).sum()), "points": int(len(diff))}
    if name == "delta":
        out["null_scaled_rms"] = case.edge*out["rms_error_T"]
    else:
        out["relative_error"] = out["rms_error_T"]/out["rms_exact_T"]
    return out


def gate(name, score):
    """Pass/fail of the pre-registered gate for one score dictionary."""
    if name == "delta":
        return bool(score["null_scaled_rms"] <= GATES["null_scaled"])
    return bool(score["relative_error"] <= GATES["nonzero_relative"])


def check():
    """The JAX surface reproduces analytic.surface."""
    for name in ("integer_axisymmetric", "integer_3d"):
        case = cases()[name]
        th, ph = np.random.default_rng(1).uniform(0, 2*np.pi, (2, 50))
        ref = surface(case, case.edge*.7, th, ph)
        new = integer_surface(jnp.asarray(case.parameters), case.edge*.7, th, ph)
        assert np.max(abs(ref-np.asarray(new))) < 1e-13, name


if __name__ == "__main__":
    check()
    print("integer_surface matches analytic.surface")
