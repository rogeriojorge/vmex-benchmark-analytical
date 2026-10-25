"""Lopez configuration D in MRX: the exact-chart map, the vacuum (harmonic) field, scores against the exact field.

MRX (ToBlick/mrx) represents B as a Dirichlet 2-form on a spline map Phi(rho, theta, zeta) of a solid torus
and needs no nested surfaces. The vacuum field of a solid torus with B . n = 0 is one harmonic 2-form (one
linear solve), fixed up to the toroidal flux. For Lopez D the boundary is a flux surface of the exact vacuum
field, so the exact field is that harmonic form and a fixed-boundary MRX solve must converge to it; the
resolution ladder measures how fast.

Two charts of the same domain are provided (the discrete harmonic form and its lines are analysed in
the logical (rho, theta, zeta) chart of each):

``LabelMap``
    Phi = X(lambda, beta, 2 pi zeta) of the paper's forward map with (lambda, beta) = (a0 + A rho cos 2 pi
    t, b0 + B rho sin 2 pi t), t = theta + twist zeta (twist = 0: the lines are straight in the chart and
    the logical rotational transform is 0; twist = +-1 puts the rotation of the unperturbed lines at -+1
    in the logical angle, the transform a VMEC-style chart sees): logical surfaces are the exact flux surfaces (ellipse level sets of
    the labels, h = rho^2) and logical zeta is the source phase. The exact field is then B = X_phase /
    |X_phase|^2 with B^rho = B^theta = 0, so the harmonic form is smooth and aligned with the chart. Optional
    ``pert`` entries (eps, m, n, phase) scale the label radius by 1 + eps rho^m cos(2 pi (m t - n zeta +
    phase)) (n counts periods of the source phase): a boundary deformation that is not a union of field lines for n != 0.

``CylFourier``
    Phi = (R cos 2 pi zeta, R sin 2 pi zeta, Z) in the rigid frame ``OPT_FRAME`` with R, Z the VMEC Fourier
    series (m < mpol, |n| <= ntor) of the exact flux surfaces: every surface carries the harmonics of
    ``lopez_surfaces.boundary_rows`` and each coefficient is rho^m times a polynomial in rho^2 fitted to the
    exact surfaces. This is the chart (cylindrical angle, flux-uniform poloidal angle plus one turn of twist)
    and the deck truncation a VMEC-family code sees; the interior is the exact foliation, so the only domain
    error is the truncation of the boundary (``lopez_surfaces.boundary_rows``).

Conventions of the scores follow ``score_lopez``: ``curl`` is a_minor ||curl B|| / ||B|| (rms), the surface
error at s = 0.5 is area weighted, energy is compared with C Phi_edge, C = 2 pi. The weak discrete curl of a
harmonic 2-form vanishes identically (``weak_curl_over_B_over_a``, round-off), so the informative curl is the
pointwise curl of the interpolated field (``pointwise_curl_over_B_over_a``, automatic differentiation through
the map); div B of the Dirichlet 2-form is zero to round-off by construction. The chart ``twist = 0`` is the
best case (the harmonic form has b^rho = b^theta = 0 exactly, the return map is the identity to round-off),
so closed-line statistics are quoted for ``twist = -1`` (unit rotation) and the cylindrical chart.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MRX_DTYPE", "float64")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lopez  # noqa: E402
import lopez_surfaces as ls  # noqa: E402

TWO_PI = 2*np.pi
A0, B0, A_RAD, B_RAD = ls.A0, ls.B0, ls.A_RAD, ls.B_RAD


# ---------------------------------------------------------------- exact forward map in JAX
def forward_x(lam, beta, phase):
    """X(lambda, beta, phase), eq. (13), for JAX scalars (the NumPy twin is ``lopez.forward``)."""
    zeta = jnp.sqrt(lam)*jnp.exp(1j*phase)
    u, v = zeta.real, zeta.imag
    n_c = jnp.stack((1 - zeta**2, 1j*(1 + zeta**2), 2*zeta), -1)
    h = 2*zeta**2 + 2j*zeta**3
    w = 1 + lam
    nu = jnp.stack((2*u, 2*v, lam - 1), -1)/w
    kappa = 4*lam*u*(1 - v)/w**2
    return (h.real*n_c.real + h.imag*n_c.imag)/w**2 + (beta - kappa)*nu


class LabelMap(eqx.Module):
    """The exact chart, see the module docstring. ``pert`` is a tuple of (eps, m, n, phase)."""

    pert: tuple = eqx.field(static=True, default=())
    twist: int = eqx.field(static=True, default=0)

    def __call__(self, x):
        rho, theta, zeta = x[0], x[1], x[2]
        theta = theta + self.twist*zeta   # label angle = logical theta + twist zeta (integer twist: still periodic)
        scale = 1.0
        for eps, m, n, phase in self.pert:
            scale = scale + eps*rho**m*jnp.cos(TWO_PI*(m*theta - n*zeta + phase))
        alpha = TWO_PI*theta
        return forward_x(A0 + A_RAD*rho*scale*jnp.cos(alpha), B0 + B_RAD*rho*scale*jnp.sin(alpha), TWO_PI*zeta)


class CylFourier(eqx.Module):
    """Cylindrical chart (R, phi = 2 pi zeta, Z) of the exact flux surfaces truncated to VMEC harmonics.

    Every surface rho (h = rho^2, flux fraction s_of_h(h)) is the VMEC Fourier series of its exact (R, Z)
    section in the frame ``frame`` with modes m < mpol, |n| <= ntor, theta = sigma eta + c phi (the series of
    ``lopez_surfaces.boundary_rows``). The coefficient of mode (m, n) is rho^m times a polynomial in rho^2
    fitted to the exact surfaces (``coef`` (n_modes, 4, K): RBC, RBS, ZBC, ZBS), so the interior is the
    exact foliation truncated like a VMEC state, with the correct axis. The map returns the point in the
    exact (paper) frame."""

    modes: jnp.ndarray     # (n_modes, 2): m, n
    coef: jnp.ndarray
    rot: jnp.ndarray
    origin: jnp.ndarray
    flip: int = eqx.field(static=True, default=1)

    def __call__(self, x):
        rho, theta, zeta = x[0], x[1], x[2]
        phi = TWO_PI*zeta
        m, n = self.modes[:, 0], self.modes[:, 1]
        k = jnp.arange(self.coef.shape[2])
        amp = (rho**m)[:, None]*(self.coef@(rho**2)**k)          # (n_modes, 4)
        ang = TWO_PI*self.flip*theta*m - phi*n*1.0
        c, s = jnp.cos(ang), jnp.sin(ang)
        r = c@amp[:, 0] + s@amp[:, 1]
        z = c@amp[:, 2] + s@amp[:, 3]
        xc = jnp.stack((r*jnp.cos(phi), r*jnp.sin(phi), z))
        return self.rot.T@xc + self.origin


def cyl_fourier_map(mpol, ntor, frame=ls.OPT_FRAME, n_rad=33, degree=9, grid=None, cache=None):
    """CylFourier truncated at (mpol, ntor). Returns (map, boundary truncation error, fit error).

    ``grid`` is the (theta, phi) sampling of the exact surfaces (default 8x / 6x the highest harmonics, at
    least (64, 256)); radii are Chebyshev nodes in rho. The fit error is the max |dX| of the radial fit
    against the truncated series at the sampled surfaces' mid radii. Handedness is detected: a negative
    Jacobian flips theta."""
    cache = Path(cache) if cache else None
    if cache is not None and cache.exists():
        z = np.load(cache)
        modes, coef = z["modes"], z["coef"]
        err, fit_err = float(z["err"]), float(z["fit_err"])
    else:
        sigma, c = ls.gauge(frame)
        grid = grid or (max(64, 8*mpol), max(256, 6*ntor))
        th = TWO_PI*np.arange(grid[0])/grid[0]
        ph = TWO_PI*np.arange(grid[1])/grid[1]
        rho = 0.5*(1 - np.cos(np.pi*(np.arange(n_rad) + 0.5)/n_rad))      # Chebyshev nodes on (0, 1), incl. near-edge
        rho = np.concatenate([rho, [1.0]])
        rows_all = []
        for r_ in rho:
            s_ = float(ls.s_of_h(r_**2))
            R, Z = ls.boundary_rz(th[:, None], ph[None, :], frame, sigma, c, s=s_)
            rows_all.append(ls.fourier_rows(*ls.fourier_spectrum(R, Z), mpol, ntor))
        rows_all = np.array(rows_all)                     # (n_rho, n_modes, 6)
        modes = rows_all[0, :, :2].copy()
        vals = rows_all[:, :, 2:]                          # (n_rho, n_modes, 4)
        k = np.arange(degree + 1)
        coef = np.zeros((len(modes), 4, degree + 1))
        for i, (m_, n_) in enumerate(modes):
            design = (rho**m_)[:, None]*(rho**2)[:, None]**k[None, :]
            coef[i] = np.linalg.lstsq(design, vals[:, i, :], rcond=None)[0].T
        # fit error at mid radii and the boundary truncation error
        rmid = 0.5*(rho[1:-1] + rho[:-2])
        rows_mid = []
        for r_ in rmid[::4]:
            s_ = float(ls.s_of_h(r_**2))
            R, Z = ls.boundary_rz(th[:, None], ph[None, :], frame, sigma, c, s=s_)
            rows_mid.append(ls.fourier_rows(*ls.fourier_spectrum(R, Z), mpol, ntor))
        fit = []
        for r_, rows_ in zip(rmid[::4], rows_mid):
            pred = np.einsum("mck,k->mc", coef, (r_**2)**k)*(r_**modes[:, :1])
            fit.append(np.abs(pred - rows_[:, 2:]).max())
        fit_err = float(max(fit))
        _, err = ls.boundary_rows(frame, sigma, c, mpol, ntor)
        if cache is not None:
            np.savez(cache, modes=modes, coef=coef, err=err, fit_err=fit_err)
    phi_map = CylFourier(jnp.asarray(modes), jnp.asarray(coef), jnp.asarray(frame[0]), jnp.asarray(frame[1]), 1)
    pts = jnp.asarray(np.array([[0.5, 0.3, 0.2], [0.7, 0.6, 0.6], [0.9, 0.1, 0.9]]))
    dets = np.asarray(jax.vmap(lambda y: jnp.linalg.det(jax.jacfwd(phi_map)(y)))(pts))
    if dets.mean() < 0:
        phi_map = CylFourier(phi_map.modes, phi_map.coef, phi_map.rot, phi_map.origin, -1)
    return phi_map, float(err), float(fit_err)


# ---------------------------------------------------------------- sequence and vacuum field
def build_sequence(ns, p, phi_map, tol=None, verbose=False):
    """A full-torus (nfp = 1, no symmetry) MRX sequence with the map installed, preconditioners built and
    the harmonic forms computed."""
    from mrx.derham_sequence import DeRhamSequence
    from mrx.nullspace import compute_nullspaces
    seq = DeRhamSequence(tuple(ns), p, nfp=1, symmetry="none", tol=tol)
    seq.set_map(phi_map)
    seq.build_preconditioners()
    compute_nullspaces(seq, verbose=verbose)
    return seq


def logical_field_fn(seq):
    from mrx.differential_forms import DiscreteFunction
    return lambda dof: DiscreteFunction(dof, seq.basis_2, seq.E(2))


def cut_flux(seq, dof, zeta=0.37, n_rho=24, n_theta=48):
    """Flux of the 2-form through the logical cut zeta = const: int b^zeta drho dtheta (Gauss in rho)."""
    f = logical_field_fn(seq)(dof)
    xg, wg = np.polynomial.legendre.leggauss(n_rho)
    rho, w = (xg + 1)/2, wg/2
    th = (np.arange(n_theta) + .5)/n_theta
    pts = jnp.asarray(np.stack(np.broadcast_arrays(rho[:, None], th[None, :], zeta), -1).reshape(-1, 3))
    b = np.asarray(jax.vmap(f)(pts))[:, 2].reshape(n_rho, n_theta)
    return float(np.sum(w[:, None]*b)/n_theta)


def vacuum_field(seq, flux=None, zeta_probe=(0.11, 0.37, 0.73)):
    """The harmonic 2-form scaled to toroidal flux ``flux`` (default Phi_edge) through the cut zeta = const.

    Returns (dof, spread) with spread the relative spread of the flux over the probe cuts (a divergence /
    boundary-flux check)."""
    flux = float(ls.flux_edge()) if flux is None else flux
    h = jnp.asarray(seq.nullspace(2)[0])
    fl = np.array([cut_flux(seq, h, z) for z in zeta_probe])
    return h*(flux/fl.mean()), float((fl.max() - fl.min())/abs(fl.mean()))


def physical_field(seq, dof):
    """Callable x_logical -> B (physical, exact frame) of the 2-form: B = DPhi b / det DPhi."""
    f = logical_field_fn(seq)(dof)
    phi_map = seq.map
    jac = jax.jacfwd(phi_map)

    def b(x):
        d = jac(x)
        return d @ f(x)/jnp.linalg.det(d)
    return b


def curl_div_points(seq, dof, pts):
    """Pointwise curl B (physical vector) and div B of the interpolated field at logical points, by
    automatic differentiation of B = DPhi b / det DPhi through the map."""
    jac = jax.jacfwd(seq.map)
    bfun = physical_field(seq, dof)

    def one(x):
        dinv = jnp.linalg.inv(jac(x))
        db = jax.jacfwd(bfun)(x) @ dinv        # db[i, k] = d B_i / d X_k
        curl = jnp.stack((db[2, 1] - db[1, 2], db[0, 2] - db[2, 0], db[1, 0] - db[0, 1]))
        return curl, jnp.trace(db)
    return jax.jit(jax.vmap(one))(jnp.asarray(pts))


# ---------------------------------------------------------------- scores
def a_minor(frame=ls.OPT_FRAME, nth=48, nph=96):
    """RMS edge distance from the axis in the (R, Z) section at fixed geometric phi (score_lopez definition)."""
    sigma, c = ls.gauge(frame)
    th = TWO_PI*np.arange(nth)[:, None]/nth
    ph = TWO_PI*np.arange(nph)[None, :]/nph
    re, ze = ls.boundary_rz(th, ph, frame, sigma, c)
    ra, za = ls.boundary_rz(0., ph, frame, sigma, c, s=0.)
    return float(np.sqrt(np.mean((re - ra)**2 + (ze - za)**2)))


def _exact_b(x):
    return lopez.field(np.asarray(x, float))


def score(seq, dof, a_min=None, s_report=0.5, max_points=150_000, seed=3, surface=(24, 96), probes=24):
    """Compare the discrete field with the exact field, on the domain of ``seq.map``.

    Volume L2 uses the sequence's own Gauss rule, thinned by a regular stride to about ``max_points`` points
    (energy and volume come from the mass matrix and the full rule, not the thinned sum); the surface error at s_report is area weighted on the exact flux surface of the same label
    level (valid for the exact chart: logical surface rho = sqrt(h)); curl, div, enclosed current,
    circulation and energy are the vacuum identities of the discrete field. Only meaningful for fields scaled
    by ``vacuum_field``."""
    from mrx.relaxation.physics import compute_divergence_norm, compute_force
    a_min = a_minor() if a_min is None else a_min
    bphys = jax.jit(jax.vmap(physical_field(seq, dof)))
    phi_map = seq.map
    out = {}
    # ---- volume L2 on the quadrature grid, in zeta slabs
    x = np.asarray(seq.quad.x)
    w = np.asarray(seq.quad.w)
    jac_j = np.asarray(seq.jacobian_j)
    n = x.shape[0]
    step = max(1, n//max_points)
    sel = np.arange(0, n, step)
    num = den = s_hb = s_hh = 0.0
    for chunk in np.array_split(sel, max(1, len(sel)//20000)):
        xs = jnp.asarray(x[chunk])
        bh = np.asarray(bphys(xs))
        pos = np.asarray(jax.vmap(phi_map)(xs))
        be = _exact_b(pos)
        wt = (w*jac_j)[chunk]
        num += float(np.sum(wt*np.sum((bh - be)**2, -1)))
        den += float(np.sum(wt*np.sum(be**2, -1)))
        s_hb += float(np.sum(wt*np.sum(bh*be, -1)))
        s_hh += float(np.sum(wt*np.sum(bh**2, -1)))
    out["B_rel_volume_l2"] = float(np.sqrt(num/den))
    # shape error after the best scalar factor: separates the field from its flux normalisation (a deformed
    # domain encloses a different exact flux, so the unscaled error carries an O(eps^2) normalisation part)
    out["B_rel_volume_l2_scaled"] = float(np.sqrt(max(s_hh - s_hb**2/den, 0.0)/(s_hb**2/den)))
    out["flux_scale_fit"] = float(s_hb/den)
    en = float(seq.l2_norm_sq(dof, 2))
    vol = float(np.sum(w*jac_j))
    out["thinning"] = int(step)
    # ---- surface s_report, area weighted
    h = float(ls.h_of_s(s_report))
    nt, nz = surface
    th = (np.arange(nt) + .5)/nt
    ze = (np.arange(nz) + .5)/nz
    pts = np.stack(np.broadcast_arrays(np.sqrt(h), th[:, None], ze[None, :]), -1).reshape(-1, 3)
    pj = jnp.asarray(pts)
    bh = np.asarray(bphys(pj))
    pos_f = jax.vmap(phi_map)
    be = _exact_b(np.asarray(pos_f(pj)))
    dpt = jax.vmap(jax.jacfwd(phi_map))(pj)
    area = np.linalg.norm(np.cross(np.asarray(dpt[:, :, 1]), np.asarray(dpt[:, :, 2])), axis=-1)
    out["B_rel_surface_l2"] = float(np.sqrt(np.sum(area*np.sum((bh - be)**2, -1))/np.sum(area*np.sum(be**2, -1))))
    out["B_rel_surface_max"] = float(np.max(np.linalg.norm(bh - be, axis=-1)/np.linalg.norm(be, axis=-1)))
    # near-axis ring
    pa = np.stack(np.broadcast_arrays(0.05, th[:, None], ze[None, :]), -1).reshape(-1, 3)
    pa = jnp.asarray(pa)
    bha = np.asarray(bphys(pa))
    bea = _exact_b(np.asarray(pos_f(pa)))
    out["B_rel_near_axis_l2"] = float(np.sqrt(np.sum((bha - bea)**2)/np.sum(bea**2)))
    # ---- identities of the discrete field
    _, _, jcur, _ = compute_force(dof, seq)
    out["weak_curl_over_B_over_a"] = float(a_min*seq.l2_norm(jcur, 1)/seq.l2_norm(dof, 2))
    out["div_over_B_over_a"] = float(a_min*compute_divergence_norm(dof, seq)/seq.l2_norm(dof, 2))
    rng0 = np.random.default_rng(seed + 1)
    pc = np.stack((np.sqrt(rng0.uniform(0.0025, 0.9, 3000)), rng0.uniform(0, 1, 3000), rng0.uniform(0, 1, 3000)), -1)
    curl_p, div_p = curl_div_points(seq, dof, pc)
    b_p = np.asarray(bphys(jnp.asarray(pc)))
    out["pointwise_curl_over_B_over_a"] = float(a_min*np.sqrt(np.mean(np.sum(np.asarray(curl_p)**2, -1))/np.mean(np.sum(b_p**2, -1))))
    out["pointwise_div_over_B_over_a"] = float(a_min*np.sqrt(np.mean(np.asarray(div_p)**2)/np.mean(np.sum(b_p**2, -1))))
    out["energy"] = en
    out["energy_target"] = TWO_PI*float(ls.flux_edge())
    out["energy_rel_error"] = abs(en/out["energy_target"] - 1)
    out["volume"] = vol
    out["volume_rel_error"] = abs(vol/ls.volume()[0] - 1)
    # circulation int B . Phi_zeta dzeta at fixed (rho, theta) and enclosed current int B . Phi_theta dtheta
    rng = np.random.default_rng(seed)
    gz = (np.arange(256) + .5)/256
    circ, curr = [], []
    for rho0, th0 in zip(np.sqrt(rng.uniform(0.05, 0.95, probes)), rng.uniform(0, 1, probes)):
        px = jnp.asarray(np.stack(np.broadcast_arrays(rho0, th0, gz), -1))
        d = jax.vmap(jax.jacfwd(phi_map))(px)
        b = bphys(px)
        circ.append(float(np.mean(np.asarray(jnp.einsum("ni,ni->n", b, d[:, :, 2])))))
    out["circulation_rel_error_max"] = float(np.max(np.abs(np.abs(circ) - TWO_PI)/TWO_PI))
    gt = (np.arange(256) + .5)/256
    for rho0, ze0 in zip(np.sqrt(rng.uniform(0.05, 0.95, probes)), rng.uniform(0, 1, probes)):
        px = jnp.asarray(np.stack(np.broadcast_arrays(rho0, gt, ze0), -1))
        d = jax.vmap(jax.jacfwd(phi_map))(px)
        b = bphys(px)
        curr.append(float(np.mean(np.asarray(jnp.einsum("ni,ni->n", b, d[:, :, 1])))))
    b_rms = float(np.sqrt(en/vol))
    out["enclosed_current_over_aB_max"] = float(np.max(np.abs(curr))/(a_min*b_rms))
    return out


# ---------------------------------------------------------------- field lines, return map, closure
def _poincare():
    from mrx.diagnostics import poincare
    return poincare


def disk_seeds(n_r=8, n_theta=12, r_min=0.08, r_max=0.92, offset=0.37):
    """Polar seeds (r, theta) at zeta = 0 on a radial x angular grid (angles offset off the symmetry axes)."""
    r = np.linspace(r_min, r_max, n_r)
    th = (np.arange(n_theta) + offset)/n_theta
    return np.stack(np.broadcast_arrays(r[:, None], th[None, :]), -1).reshape(-1, 2)


def trace_lines(seq, dof, seeds, turns, steps=48):
    """Lines of the discrete field in the logical (u, v) = r (cos 2 pi theta, sin 2 pi theta) chart,
    sampled ``steps`` times per toroidal turn. Returns ys (n_seeds, turns*steps + 1, 2) and ok."""
    pc = _poincare()
    ys, ok = pc.trace(pc.logical_field(seq), jnp.asarray(dof), jnp.asarray(seeds), int(turns), int(steps))
    return np.asarray(ys), np.asarray(ok)


def return_defect(ys, steps):
    """One-turn displacement P(y) - y in the (u, v) chart (n_seeds, 2) and the radius-normalised size."""
    d = ys[:, steps] - ys[:, 0]
    return d, np.linalg.norm(d, axis=-1)


def axis_track(ys0, steps):
    """The magnetic axis in the (u, v) chart at every saved step: the mean over the traced turns of the innermost
    line ``ys0`` (n_saves, 2) at the same phase within the turn (the MRX tracer's definition)."""
    inner = ys0[:-1].reshape(-1, steps, 2)
    centre = inner.mean(0)
    reps = -(-ys0.shape[0]//steps)
    return np.tile(centre, (reps, 1))[:ys0.shape[0]]


def rotation_per_turn(ys, steps, center=None):
    """Signed poloidal turns per toroidal turn of every line about ``center`` (n_saves, 2) or a fixed point
    (default: the logical axis r = 0), from the unwrapped angle at every saved step (a rotation of exactly one
    turn per toroidal turn is invisible when sampling only at integer turns)."""
    n = ys.shape[1]
    c = np.zeros((n, 2)) if center is None else np.broadcast_to(np.asarray(center), (n, 2))
    ang = np.unwrap(np.arctan2(ys[..., 1] - c[:, 1], ys[..., 0] - c[:, 0]), axis=1)
    k = (np.arange(n) - (n - 1)/2)/steps
    return ((ang - ang.mean(1, keepdims=True))@k)/(k@k)/TWO_PI


def closure_summary(seq, dof, n_r=8, n_theta=12, turns=20, steps=48, expected_rotation=0.0):
    """Closed-line statistics of a field: the one-turn displacement of lines seeded over the disk
    (max/rms in the logical (u, v) chart, whose radius is 1) and the deviation of the rotation per turn
    from ``expected_rotation`` (the rotation of the chart for a field whose lines all close)."""
    seeds = np.concatenate([[[0.01, 0.0]], disk_seeds(n_r, n_theta)])
    ys, ok = trace_lines(seq, dof, seeds, turns, steps)
    ys, ys_axis = ys[1:], ys[0]
    d, mag = return_defect(ys, steps)
    rot = rotation_per_turn(ys, steps, axis_track(ys_axis, steps))
    # max excursion in logical radius over all turns
    rad = np.linalg.norm(ys, axis=-1)
    exc = rad.max(1) - rad.min(1)
    return dict(ok=bool(ok.all()), turns=int(turns), steps=int(steps), n_lines=int(len(seeds)),
                return_defect_max=float(mag.max()), return_defect_rms=float(np.sqrt(np.mean(mag**2))),
                rotation_dev_max=float(np.minimum(np.abs(rot - expected_rotation), np.abs(rot + expected_rotation)).max()),
                radial_excursion_max=float(exc.max()))


# ---------------------------------------------------------------- effective Hamiltonian of the one-turn map
def one_turn_grid(seq, dof, n=61, r_max=0.97, steps=48):
    """P(y) - y of the one-turn map on the zeta = 0 cut at the nodes of an n x n Cartesian (u, v) grid inside
    r <= r_max. Returns (u, v, D) with D (n, n, 2) and NaN outside the disk."""
    ax = np.linspace(-r_max, r_max, n)
    U, V = np.meshgrid(ax, ax, indexing="ij")
    inside = np.hypot(U, V) <= r_max
    r = np.hypot(U, V)[inside]
    th = (np.arctan2(V, U)[inside]/TWO_PI) % 1.0
    ys, ok = trace_lines(seq, dof, np.stack([np.maximum(r, 1e-3), th], -1), 1, steps)
    D = np.full((n, n, 2), np.nan)
    D[inside] = ys[:, steps] - ys[:, 0]
    return ax, ax, D


def effective_hamiltonian(u, v, D):
    """H on the grid with D = J grad H, J = [[0, 1], [-1, 0]] (D_u = H_v, D_v = -H_u), least squares, H = 0 on
    average; the generator of a near-identity area-preserving map. Returns H (n, n), NaN outside."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    n = len(u)
    h = u[1] - u[0]
    inside = np.isfinite(D[..., 0])
    idx = -np.ones((n, n), int)
    idx[inside] = np.arange(inside.sum())
    rows, cols, vals, rhs = [], [], [], []
    k = 0
    for a in range(n):
        for b in range(n):
            if not inside[a, b]:
                continue
            if a + 1 < n and inside[a + 1, b]:       # H_u = -D_v at the midpoint
                rows += [k, k]; cols += [idx[a + 1, b], idx[a, b]]; vals += [1/h, -1/h]
                rhs.append(-0.5*(D[a, b, 1] + D[a + 1, b, 1])); k += 1
            if b + 1 < n and inside[a, b + 1]:       # H_v = D_u
                rows += [k, k]; cols += [idx[a, b + 1], idx[a, b]]; vals += [1/h, -1/h]
                rhs.append(0.5*(D[a, b, 0] + D[a, b + 1, 0])); k += 1
    A = coo_matrix((vals, (rows, cols)), shape=(k, inside.sum())).tocsr()
    sol = lsqr(A, np.array(rhs), atol=1e-14, btol=1e-14, iter_lim=20000)[0]
    H = np.full((n, n), np.nan)
    H[inside] = sol - sol.mean()
    return H


def critical_points(u, v, D, n_start=1):
    """Zeros of D (fixed points of the one-turn map) by Newton on a cubic interpolant, classified by det dD:
    > 0 O-point, < 0 X-point (D = J grad H, so det dD = det Hess H). Returns a list of (u, v, kind)."""
    from scipy.interpolate import RegularGridInterpolator
    fill = np.nan_to_num(D)
    fu = RegularGridInterpolator((u, v), fill[..., 0], method="cubic")
    fv = RegularGridInterpolator((u, v), fill[..., 1], method="cubic")
    inside = np.isfinite(D[..., 0])
    mag = np.linalg.norm(fill, axis=-1)
    pts = []
    h = u[1] - u[0]
    # starts: interior grid nodes that are local minima of |D| (8-neighbourhood)
    for a in range(1, len(u) - 1):
        for b in range(1, len(v) - 1):
            if not inside[a-1:a+2, b-1:b+2].all():
                continue
            blk = mag[a-1:a+2, b-1:b+2]
            if mag[a, b] > blk.min() + 1e-300:
                continue
            x = np.array([u[a], v[b]])
            for _ in range(30):
                f = np.array([fu(x)[0], fv(x)[0]])
                e = 0.3*h
                J = np.array([[fu(x + [e, 0])[0] - fu(x - [e, 0])[0], fu(x + [0, e])[0] - fu(x - [0, e])[0]],
                              [fv(x + [e, 0])[0] - fv(x - [e, 0])[0], fv(x + [0, e])[0] - fv(x - [0, e])[0]]])/(2*e)
                try:
                    dx = np.linalg.solve(J, -f)
                except np.linalg.LinAlgError:
                    break
                if np.hypot(*dx) > 2*h:
                    dx *= 2*h/np.hypot(*dx)
                x = x + dx
                if np.hypot(*dx) < 1e-10*h or not (u[0] < x[0] < u[-1] and v[0] < x[1] < v[-1]):
                    break
            if not (u[1] < x[0] < u[-2] and v[1] < x[1] < v[-2]) or not np.isfinite(x).all():
                continue
            f = np.array([fu(x)[0], fv(x)[0]])
            if np.hypot(*f) > 1e-6*max(mag[inside].max(), 1e-300) + 1e-12 or np.hypot(*x) > u[-1]:
                continue
            e = 0.3*h
            J = np.array([[fu(x + [e, 0])[0] - fu(x - [e, 0])[0], fu(x + [0, e])[0] - fu(x - [0, e])[0]],
                          [fv(x + [e, 0])[0] - fv(x - [e, 0])[0], fv(x + [0, e])[0] - fv(x - [0, e])[0]]])/(2*e)
            if all(np.hypot(*(x - p[:2])) > 0.5*h for p in pts):
                pts.append((float(x[0]), float(x[1]), "O" if np.linalg.det(J) > 0 else "X"))
    return pts


def island_regions(u, v, H, pts):
    """For each O-point the connected region {H on the O side of the innermost X level} around it: area and the
    radial extent max r - min r (logical units); X levels are H at the X-points. Returns a list of dicts."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import label
    fill = np.nan_to_num(H)
    hi = RegularGridInterpolator((u, v), fill, method="cubic")
    xs = [p for p in pts if p[2] == "X"]
    out = []
    if not xs:
        return out
    U, V = np.meshgrid(u, v, indexing="ij")
    R = np.hypot(U, V)
    inside = np.isfinite(H)
    dA = (u[1] - u[0])**2
    for p in pts:
        if p[2] != "O":
            continue
        h_o = float(hi([p[:2]])[0])
        levels = [float(hi([q[:2]])[0]) for q in xs]
        best = None
        for hx in sorted(levels, key=lambda h_: abs(h_ - h_o)):   # nearest barrier first
            mask = inside & ((H - hx)*(h_o - hx) > 0)
            lab, _ = label(mask)
            i, j = np.unravel_index(np.argmin((U - p[0])**2 + (V - p[1])**2 + 1e9*(~mask)), U.shape)
            if lab[i, j] == 0:
                continue
            comp = lab == lab[i, j]
            if (comp & (R > R[inside].max() - 1.5*(u[1] - u[0]))).any():     # region reaches the edge: not an island
                continue
            best = dict(O=p[:2], H_O=h_o, H_sep=hx, area=float(comp.sum()*dA),
                        r_min=float(R[comp].min()), r_max=float(R[comp].max()), radial_extent=float(R[comp].max() - R[comp].min()),
                        diameter=float(max(np.ptp(U[comp]), np.ptp(V[comp]))))
            break
        if best:
            out.append(best)
    return out


def hamiltonian_record(seq, dof, n=61, steps=48):
    """Effective-Hamiltonian analysis of the one-turn map as one JSON-able dict plus the grid arrays:
    O/X fixed points (zeros of P - id), the largest displacement and the island regions that close inside
    the disk. Returns (record, (u, D, H))."""
    u, v, D = one_turn_grid(seq, dof, n=n, steps=steps)
    H = effective_hamiltonian(u, v, D)
    pts = critical_points(u, v, D)
    rec = dict(grid=n, n_O=sum(p[2] == "O" for p in pts), n_X=sum(p[2] == "X" for p in pts),
               points=[dict(u=p[0], v=p[1], kind=p[2]) for p in pts], regions=island_regions(u, v, H, pts),
               D_max=float(np.nanmax(np.abs(np.nan_to_num(D)))))
    return rec, (u, D, H)


def exact_line_closure(n_lines=8, seed=4, rtol=1e-12):
    """Trace lines of the exact field (``lopez.field``, label-selected root) in physical space for one source
    period and measure how well they close and conserve the labels.

    The line through X(lambda, beta, 0) is integrated with dX/dt = B(X); the period is T = int |X_phase|^2
    dphase (B = X_phase / |X_phase|^2). Returns the max return distance |X(T) - X(0)|, the max label drift and
    the max |B x X_phase| residual of the start points (tangency), all from NumPy/SciPy only."""
    from scipy.integrate import solve_ivp
    rng = np.random.default_rng(seed)
    rho = np.sqrt(rng.uniform(0.05, 0.95, n_lines))
    al = rng.uniform(0, TWO_PI, n_lines)
    lam, beta = A0 + A_RAD*rho*np.cos(al), B0 + B_RAD*rho*np.sin(al)
    ph = np.linspace(0, TWO_PI, 2001)
    ret, drift = 0.0, 0.0
    for l_, b_ in zip(lam, beta):
        x, xp = lopez.forward(l_, b_, ph)
        period = np.trapezoid(np.sum(xp*xp, -1), ph)
        sol = solve_ivp(lambda t, y: lopez.field(y), (0, period), x[0], rtol=rtol, atol=1e-14, method="DOP853")
        ret = max(ret, float(np.linalg.norm(sol.y[:, -1] - x[0])))
        lam1, beta1 = lopez.labels(sol.y[:, -1])
        drift = max(drift, abs(float(lam1) - l_), abs(float(beta1) - b_))
    return dict(max_return_distance=ret, max_label_drift=drift, minor_radius_scale=float(A_RAD))
