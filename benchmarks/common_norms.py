"""Whole-volume norms on the exact chart, for every family, and the DESC evaluator that feeds them.

``certify_projection`` scores a VMEC-family state with relative L2 norms of B, J = curl B / mu0 and
the force J x B - grad p (against the exact grad p) over exact-chart volume points whose weight is
the absolute chart Jacobian, so no solver geometry enters the measure.  That rule exists only for
the sheared chart there.  This module supplies the same rule for the integer and Solov'ev charts
(``surface`` of ``analytic``, ported to JAX so the Jacobian is an automatic derivative) and reuses
``certify_projection`` unchanged for the sheared family.  Weights, points, levels and region
bookkeeping are therefore identical to the VMEC-family scorer wherever both exist.

DESC is evaluated at the exact-chart Cartesian points themselves: ``map_coordinates`` inverts the
solved (rho, theta, zeta) -> (R, phi, Z) map, B and J are DESC's own, the pressure gradient is the
exact one (as for the VMEC family).  Points the inversion cannot reach lie outside the solved last
surface; they are excluded from the sums and counted, never silently dropped.

Conventions: exact fields are dimensionless with mu0 = 1 and B* = 1 T, L* = 1 m; DESC is SI, so
J_DESC * mu_0 is the exact curl and p_DESC * mu_0 the exact pressure.  The exact angle
theta_ccw is the negative of DESC's theta.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from analytic import differential_fields, flux, rotate, volume
import certify_projection as cp
from certify_projection import GATES, LEVELS, REGIONS, metrics, radial_nodes, uncertainty  # noqa: F401


def chart_xyz(case, q):
    """JAX port of ``analytic.surface`` for the integer and Solov'ev families: q = (label, theta, phi)."""
    label, theta, phi = q[..., 0], q[..., 1], q[..., 2]
    phi0 = phi-case.phase
    if case.family == "integer":
        a, b, c, _ = case.parameters
        w = jnp.sqrt(1-case.tau**2)
        u0 = -w*w*(a*a-b*b)/(4*c*c)
        beta = 2*phi0+theta
        u, v = u0+jnp.sqrt(label)*jnp.cos(beta), jnp.sqrt(label)*jnp.sin(beta)
        ell = jnp.sqrt((1+jnp.sqrt(1-4*(u*u+v*v)))/2)
        A, C, D, E = a*(ell+u/ell), a*v/ell, b*v/ell, b*(ell-u/ell)
        ct, st = E*jnp.cos(phi0)-C*jnp.sin(phi0), A*jnp.sin(phi0)-D*jnp.cos(phi0)
        norm = jnp.hypot(ct, st)
        ct, st = ct/norm, st/norm
        c2, s2 = ct*ct-st*st, 2*st*ct
        c2, s2 = w*c2-case.tau*s2, w*s2+case.tau*c2
        xyz = jnp.stack(((A*ct+C*st)/w, (D*ct+E*st)/w, c*(v*c2-u*s2-case.tau/2)/(w*w)), -1)
    elif case.family == "solovev":
        R0, b, g, chi, _, _ = case.parameters
        U = jnp.sqrt(label/b)/jnp.sqrt(1-chi*chi)*jnp.cos(theta)
        Z = jnp.sqrt(label/g)*(jnp.sin(theta)-chi/jnp.sqrt(1-chi*chi)*jnp.cos(theta))
        R = jnp.sqrt(R0*R0+U)
        xyz = jnp.stack((R*jnp.cos(phi0), R*jnp.sin(phi0), Z), -1)
    else:
        raise ValueError("chart_xyz covers the integer and Solov'ev families; sheared uses certify_projection")
    return rotate(xyz, case.phase)+jnp.array([0., 0., case.zshift])


def jacobian(case, label, theta, phi):
    """|det d xyz / d (label, theta, phi)| of the exact chart by forward-mode AD."""
    one = lambda q: jnp.abs(jnp.linalg.det(jax.jacfwd(lambda p: chart_xyz(case, p))(q)))  # noqa: E731
    return np.asarray(jax.jit(jax.vmap(one))(jnp.stack((label, theta, phi), -1)))


def rule_angles(case, nth, nt, shift):
    """Angular nodes of the non-sheared rules: theta over 2 pi, phi over one field period.

    The integer and Solov'ev fields repeat in phi with the field-period (``case.nfp``); a Solov'ev
    state is axisymmetric, so one plane is exact.  Weights carry the actual angular extent, so the
    returned rule integrates over exactly that wedge; ratios (the norms) do not depend on it.
    """
    th = 2*np.pi*(np.arange(nth)+shift)/nth
    if case.family == "solovev":
        return th, np.array([0.0]), 2*np.pi
    return th, 2*np.pi/case.nfp*(np.arange(nt)+shift)/nt, 2*np.pi/case.nfp


def volume_rule(case, name):
    """Points, weights, exact fields and radial bookkeeping for one level, any family.

    The sheared family is delegated to ``certify_projection.volume_rule``.  For the others the
    additional keys ``label``, ``theta``, ``phi`` (chart coordinates), ``wedge`` (phi extent of the
    rule) and ``volume_total`` (exact volume of that wedge) are provided.
    """
    if case.family == "sheared":
        rule = cp.volume_rule(case, name)
        rule["wedge"] = 2*np.pi
        rule["volume_total"] = float(volume(case))
        return rule
    nr, nth, nt, _, shift, panels = LEVELS[name]
    r, wr = radial_nodes(case.edge, nr, "gauss", panels)
    th, ph, wedge = rule_angles(case, nth, nt, shift)
    L, T, A = (a.ravel() for a in np.meshgrid(r, th, ph, indexing="ij"))
    shape = (len(r), len(th), len(ph))
    jac = jacobian(case, jnp.asarray(L), jnp.asarray(T), jnp.asarray(A))
    weights = jac*np.broadcast_to(wr[:, None, None], shape).ravel()*(2*np.pi/len(th))*(wedge/len(ph))
    xyz = np.asarray(chart_xyz(case, jnp.stack((jnp.asarray(L), jnp.asarray(T), jnp.asarray(A)), -1)))
    B, curl, gp, _, _ = (np.concatenate(p) for p in zip(*[
        differential_fields(case, xyz[i:i+16384]) for i in range(0, len(xyz), 16384)]))
    edge_flux = float(flux(case, case.edge))
    s = np.broadcast_to((np.asarray([float(flux(case, x)) for x in r])/edge_flux)[:, None, None], shape).ravel()
    return dict(name=name, xyz=xyz, w=weights, B=B, J=curl, gp=gp, s=s, label=L, theta=T, phi=A,
                wedge=wedge, volume_total=float(volume(case))*wedge/(2*np.pi))


def rule_volume_error(case, rule):
    """Relative error of the rule's total weight against the exact volume of its wedge."""
    return float(np.sum(rule["w"])/rule["volume_total"]-1)


def desc_guess(rule):
    """Computational-coordinate guess for the exact-chart points: rho = sqrt(s), theta = -theta_ccw, zeta = phi."""
    xyz = rule["xyz"]
    phi = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2*np.pi)
    theta = rule["theta"] if "theta" in rule else None
    if theta is None:
        raise KeyError("sheared rules need the chart angle: use desc_guess_sheared")
    return np.column_stack((np.sqrt(rule["s"]), np.mod(-theta, 2*np.pi), phi))


def desc_guess_sheared(case, rule):
    """Same guess for the sheared rule, whose points carry only xyz and s: recover theta from the chart.

    The chart angle theta is the angle of (X, Y) = -k (cos theta, sin theta) in the transverse plane,
    with Y = sin(-lam z) and X from the chart's own definition; here it is read back from the
    meshgrid ordering of ``certify_projection.volume_rule`` (label, theta, t with theta the middle axis).
    """
    nr, nth, nt, rule_name, shift, panels = LEVELS[rule["name"]]
    th = 2*np.pi*(np.arange(nth)+shift)/nth
    theta = np.broadcast_to(th[None, :, None], (nr*panels, nth, nt)).ravel()
    phi = np.mod(np.arctan2(rule["xyz"][:, 1], rule["xyz"][:, 0]), 2*np.pi)
    return np.column_stack((np.sqrt(rule["s"]), np.mod(-theta, 2*np.pi), phi))


def guess_for(case, rule):
    return desc_guess_sheared(case, rule) if case.family == "sheared" else desc_guess(rule)


def desc_BJ(eq, case, rule, map_chunk=8000, chunk=800, tol=1e-9, maxiter=30):
    """DESC's B and J*mu_0 at the exact-chart points of ``rule`` and a validity mask.

    ``valid`` is False where the coordinate inversion failed (point outside the solved last surface)
    or did not reproduce the target position to 1e-7 m.  The inversion is done in large chunks (its
    cost is dominated by a per-call overhead), the field evaluation in small ones (memory).
    """
    from desc.grid import Grid
    xyz = rule["xyz"]
    rpz = np.column_stack((np.hypot(xyz[:, 0], xyz[:, 1]), np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2*np.pi), xyz[:, 2]))
    guess = guess_for(case, rule)
    n = len(xyz)
    nodes = np.empty((n, 3))
    for i in range(0, n, map_chunk):
        sl = slice(i, min(i+map_chunk, n))
        nodes[sl] = np.asarray(eq.map_coordinates(rpz[sl], inbasis=("R", "phi", "Z"), outbasis=("rho", "theta", "zeta"),
                                                  guess=guess[sl], period=(np.inf, 2*np.pi, np.inf), tol=tol, maxiter=maxiter))
    B, J = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    ok = np.zeros(n, bool)
    good = np.flatnonzero(np.isfinite(nodes).all(1) & (nodes[:, 0] <= 1.0+1e-9) & (nodes[:, 0] >= 0))
    for i in range(0, len(good), chunk):
        idx = good[i:i+chunk]
        data = eq.compute(["R", "Z", "B", "J"], grid=Grid(nodes[idx], sort=False), override_grid=True)
        phi = nodes[idx, 2]
        R, Z = np.asarray(data["R"]), np.asarray(data["Z"])
        found = np.linalg.norm(np.column_stack((R*np.cos(phi), R*np.sin(phi), Z))-xyz[idx], axis=1) < 1e-7
        for key, out in (("B", B), ("J", J)):
            v = np.asarray(data[key])
            cart = np.column_stack((v[:, 0]*np.cos(phi)-v[:, 1]*np.sin(phi), v[:, 0]*np.sin(phi)+v[:, 1]*np.cos(phi), v[:, 2]))
            out[idx] = cart*(mu_0 if key == "J" else 1.0)
        ok[idx] = found & np.isfinite(B[idx]).all(1) & np.isfinite(J[idx]).all(1)
    return B, J, ok


def score_desc(eq, case, levels=("q2",), rules=None):
    """Whole-volume B/J/force norms of a DESC equilibrium on exact-chart rules: {level: metrics()}."""
    out, times, rules = {}, {}, rules if rules is not None else {}
    from time import perf_counter
    for name in levels:
        if name not in rules:
            rules[name] = volume_rule(case, name)
        t0 = perf_counter()
        B, J, valid = desc_BJ(eq, case, rules[name])
        out[name] = metrics(rules[name], np.nan_to_num(B), np.nan_to_num(J), valid)
        times[name] = perf_counter()-t0
    return out, times


def uncertainty_if_complete(scores):
    keys = (cp.FINEST, cp.PREVIOUS, cp.SHIFTED, cp.PANELS)
    return uncertainty(scores) if all(k in scores for k in keys) else None
