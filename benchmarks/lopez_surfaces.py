"""Lopez (arXiv:2610.29921v1) configuration D as a fixed-boundary VMEC-family benchmark.

Exact flux map, flux-uniform angle, geometric-angle inversion, boundary Fourier
representation with measured truncation error, exact transform, and the vacuum
INDATA generator. NumPy float64 only; no equilibrium solve is performed.

Points are generated from known field-line labels (lambda, beta) and source phase,
so the selected cubic root is known (zeta = sqrt(lambda) exp(i phase)); this module
does not depend on a Cartesian root selector. Notation: lambda, beta are field-line
labels (not VMEC lambda, not plasma beta); phase is the source phase of the paper;
phi is the geometric cylindrical angle of the chosen proper rigid frame.

Frame: X_c = Rot (X - origin), Rot a proper rotation (paper: X_c = x + 7/20,
Y_c = -z - 17/1150, Z_c = y);
R = hypot(X_c, Y_c), phi = atan2(Y_c, X_c). The paper's frame puts the chart origin
about 4e-3 from the D boundary (dphi/dphase spans 8e3:1); ``optimize_frame`` finds a
better rigid frame (see ``chart_check`` and ``truncation_ladder``).

Poloidal angle: theta = sigma*eta + c*phi (eta = flux-uniform label angle,
sigma = +-1 chosen so (R, Z) is counterclockwise, c the integer transform
= field-line rotation about the axis per toroidal turn). Total flux (not /2pi).
"""
import json
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]

# Configuration D: ellipse in (lambda, beta) label space, level h in [0, 1].
A0, B0, A_RAD, B_RAD = 3/20, 1/50, 1/300, 1/100
NFP = 1
# A frame is (Rot, origin): X_c = Rot @ (X - origin), Rot a proper rotation.
PAPER_FRAME = (np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]), np.array([-7/20, 0., -17/1150]))
TWO_PI = 2*np.pi


def make_frame(polar, azimuth, origin):
    """Proper rotation whose third row is the cylindrical axis n(polar, azimuth); X_c = Rot (X - origin)."""
    n = np.array([np.sin(polar)*np.cos(azimuth), np.sin(polar)*np.sin(azimuth), np.cos(polar)])
    e1 = np.cross([1., 0, 0] if abs(n[0]) < .9 else [0, 1., 0], n)
    e1 /= np.linalg.norm(e1)
    return np.array([e1, np.cross(n, e1), n]), np.asarray(origin, float)


# Tilted frame from ``optimize_frame(10, 56)`` (guards: one turn, R > 0.05, spread < 60).
OPT_FRAME = make_frame(1.0456200151, 1.6147828511, (-0.3036348093, -0.0041797475, -0.0256278372))


# ---------------------------------------------------------------- exact flux map
def _delta():
    return A_RAD**2/(A0 + np.sqrt(A0**2 - A_RAD**2))


def flux_of_h(h):
    """Total toroidal flux through the level-h ellipse, cancellation-safe."""
    h = np.asarray(h, float)
    return np.pi*B_RAD/A_RAD*A_RAD**2*h/(A0 + np.sqrt(A0**2 - A_RAD**2*h))


def flux_edge():
    return np.pi*B_RAD/A_RAD*_delta()


def h_of_s(s):
    d = _delta()
    s = np.asarray(s, float)
    return (2*A0*d*s - d*d*s*s)/A_RAD**2


def s_of_h(h):
    return flux_of_h(h)/flux_edge()


def radius_of_rho(rho):
    """r = A sqrt(h) with rho = sqrt(s); smooth through the axis. Returns (r, dr/drho)."""
    d, rho = _delta(), np.asarray(rho, float)
    root = np.sqrt(2*A0*d - d*d*rho*rho)
    return rho*root, root - d*d*rho*rho/root


# ---------------------------------------------------------------- forward map
def forward(lam, beta, phase, tangents=False):
    """X(lambda, beta, phase) (paper eq. 13); optionally (X, X_lam, X_beta, X_phase)."""
    lam, beta, phase = np.broadcast_arrays(*(np.asarray(v, float) for v in (lam, beta, phase)))
    zeta = np.sqrt(lam)*np.exp(1j*phase)
    u, v = zeta.real, zeta.imag
    one = np.ones_like(u)
    N = np.stack((1 - zeta**2, 1j*(1 + zeta**2), 2*zeta), -1)
    Np = np.stack((-2*zeta, 2j*zeta, 2*one), -1)
    H, Hp = 2*zeta**2 + 2j*zeta**3, 4*zeta + 6j*zeta**2
    w = 1 + lam
    n = np.stack((2*u, 2*v, lam - 1), -1)/w[..., None]
    kappa = 4*lam*u*(1 - v)/w**2
    X = ((H.real[..., None]*N.real + H.imag[..., None]*N.imag)/w[..., None]**2
         + (beta - kappa)[..., None]*n)
    if not tangents:
        return X

    def directional(dzeta, dlam, dbeta):
        du, dv = dzeta.real, dzeta.imag
        dH, dN = Hp*dzeta, Np*dzeta[..., None]
        dn = np.stack((2*du, 2*dv, dlam*one), -1)/w[..., None] - n*(dlam/w)[..., None]
        dkappa = (4*(dlam*u*(1 - v) + lam*du*(1 - v) - lam*u*dv)/w**2 - 2*kappa*dlam/w)
        dX = ((dH.real[..., None]*N.real + H.real[..., None]*dN.real
               + dH.imag[..., None]*N.imag + H.imag[..., None]*dN.imag)/w[..., None]**2
              - 2*dlam[..., None]*(H.real[..., None]*N.real + H.imag[..., None]*N.imag)/w[..., None]**3
              + (dbeta - dkappa)[..., None]*n + (beta - kappa)[..., None]*dn)
        return dX
    zero = np.zeros_like(u)
    return (X, directional(zeta/(2*lam), one, zero), directional(zero + 0j, zero, one),
            directional(1j*zeta, zero, zero))


def field_at_root(lam, beta, phase, X=None):
    """B = Im(N/(zeta D)) at X(lam, beta, phase) using the Cartesian formula and the known root."""
    zeta = np.sqrt(np.asarray(lam, float))*np.exp(1j*np.asarray(phase, float))
    X = forward(lam, beta, phase) if X is None else X
    w = X[..., 0] + 1j*X[..., 1]
    D = 6j*zeta**2 + 2*(w.conj() + 2)*zeta - 2*X[..., 2]
    N = np.stack((1 - zeta**2, 1j*(1 + zeta**2), 2*zeta), -1)
    return (N/(zeta*D)[..., None]).imag


# ---------------------------------------------------------------- flux-uniform angle
def eta_of_alpha(alpha, h):
    """eta = atan2(k sin(alpha), a cos(alpha) + r), continuous (winding preserved)."""
    r = A_RAD*np.sqrt(h)
    return np.arctan2(np.sqrt(A0**2 - r*r)*np.sin(alpha), A0*np.cos(alpha) + r)


def labels(r, angle, kind="eta"):
    """(lambda, beta, d/dr, d/dangle) for ellipse radius r = A sqrt(h), at label angle.

    kind 'eta': flux-uniform angle (closed forms lambda = k^2/(a - r cos eta));
    kind 'alpha': ordinary ellipse angle.
    """
    ratio = B_RAD/A_RAD
    c, s = np.cos(angle), np.sin(angle)
    if kind == "alpha":
        return (A0 + r*c, B0 + ratio*r*s, c, ratio*s, -r*s, ratio*r*c)
    k = np.sqrt(A0**2 - r*r)
    D = A0 - r*c
    lam, bet = k*k/D, B0 + ratio*r*k*s/D
    dlam_r, dlam_a = (-2*r*D + k*k*c)/D**2, -k*k*r*s/D**2
    dbet_r = ratio*s*((k*k - r*r)/k*D + r*k*c)/D**2
    dbet_a = ratio*r*k*(c*D - r*s*s)/D**2
    return lam, bet, dlam_r, dbet_r, dlam_a, dbet_a


# ---------------------------------------------------------------- chart
def to_frame(X, frame):
    Xc = (np.asarray(X) - frame[1]) @ frame[0].T
    return Xc[..., 0], Xc[..., 1], Xc[..., 2]


def cylindrical(X, frame):
    Xc, Yc, Zc = to_frame(X, frame)
    return np.hypot(Xc, Yc), np.arctan2(Yc, Xc), Zc


def phase_of_phi(lam, beta, phi, frame, iterations=64):
    """Source phase in [0, 2 pi) with geometric angle phi, by bisection.

    Needs one toroidal turn per source period and monotone phi(phase) (``chart_check``):
    F = mod(phi(phase) - phi(0), 2 pi) is then nondecreasing, so no unwrapping is used.
    """
    lam, beta, phi = np.broadcast_arrays(*(np.asarray(v, float) for v in (lam, beta, phi)))
    phi0 = cylindrical(forward(lam, beta, 0.*lam), frame)[1]
    target = np.mod(phi - phi0, TWO_PI)
    lo, hi = np.zeros_like(lam), np.full_like(lam, TWO_PI)
    for _ in range(iterations):
        mid = (lo + hi)/2
        below = np.mod(cylindrical(forward(lam, beta, mid), frame)[1] - phi0, TWO_PI) < target
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi)/2


def section_point(r, angle, phi, frame, kind="eta"):
    """(R, Z, lam, beta, phase) at ellipse radius r, label angle, geometric angle phi."""
    lam, beta = labels(r, angle, kind)[:2]
    phase = phase_of_phi(lam, beta, phi, frame)
    R, _, Z = cylindrical(forward(lam, beta, phase), frame)
    return R, Z, lam, beta, phase


def section_jacobian(r, dr, angle, phi, frame, kind="eta"):
    """Analytic (R, Z, dR/dr, dZ/dr, dR/dangle, dZ/dangle, lam, beta, phase) at fixed geometric phi.

    Label derivatives are propagated through X, with the phase moving so phi stays
    fixed (implicit function theorem); ``dr`` is dr/drho or 1 for a derivative in r.
    """
    lam, beta, dlr, dbr, dla, dba = labels(r, angle, kind)
    phase = phase_of_phi(lam, beta, phi, frame)
    X, Xl, Xb, Xp = forward(lam, beta, phase, tangents=True)
    Xc, Yc, _ = to_frame(X, frame)
    R2 = Xc*Xc + Yc*Yc

    def geo(T):  # (dR, dZ, dphi) of a Cartesian tangent in the frame
        Tc = T @ frame[0].T
        dXc, dYc, dZc = Tc[..., 0], Tc[..., 1], Tc[..., 2]
        return ((Xc*dXc + Yc*dYc)/np.sqrt(R2), dZc, (Xc*dYc - Yc*dXc)/R2)
    gl, gb, gp = geo(Xl), geo(Xb), geo(Xp)
    out = [np.sqrt(R2), to_frame(X, frame)[2]]
    for dlam, dbet in ((dlr*dr, dbr*dr), (dla, dba)):
        dphase = -(gl[2]*dlam + gb[2]*dbet)/gp[2]
        out += [gl[i]*dlam + gb[i]*dbet + gp[i]*dphase for i in (0, 1)]
    return (*out, lam, beta, phase)


def chart_check(frame, nr=9, nalpha=256, nphase=4096):
    """Sampled chart conditions on the whole disk (not an interval certificate).

    Reports min R, extreme dphi/dphase, and the number of phase samples where the
    wrapped phi increment fails to be positive (a fold or missed winding).
    """
    phase = np.linspace(0, TWO_PI, nphase, endpoint=False)
    alpha = np.linspace(0, TWO_PI, nalpha, endpoint=False)[:, None]
    stats = dict(rmin=np.inf, dphi_min=np.inf, dphi_max=-np.inf, bad_increments=0)
    for r in A_RAD*np.sqrt(np.linspace(0, 1, nr)):
        lam, beta = labels(r, alpha)[:2]
        X, _, _, Xp = forward(lam, beta, phase[None, :], tangents=True)
        Xc, Yc, _ = to_frame(X, frame)
        R2 = Xc*Xc + Yc*Yc
        Tc = Xp @ frame[0].T
        dphi = (Xc*Tc[..., 1] - Yc*Tc[..., 0])/R2
        ang = np.arctan2(Yc, Xc)
        step = np.mod(np.roll(ang, -1, axis=1) - ang, TWO_PI)
        stats["bad_increments"] += int(np.sum(step > np.pi))
        stats.update(rmin=min(stats["rmin"], float(np.sqrt(R2.min()))),
                     dphi_min=min(stats["dphi_min"], float(dphi.min())),
                     dphi_max=max(stats["dphi_max"], float(dphi.max())))
    stats["dphi_ratio"] = stats["dphi_max"]/stats["dphi_min"] if stats["dphi_min"] > 0 else np.inf
    return stats


def orientation_and_rotation(frame, n=2048):
    """sigma = orientation of eta in (R, Z) about the axis; c = field-line rotation per turn.

    Both are measured with the boundary section and a dense unwrapped polar angle about
    the exact axis, not assumed. Returns (sigma, c_by_s) with c an integer per surface.
    """
    phi = np.linspace(0, TWO_PI, n, endpoint=False)
    Rax, Zax = section_point(0., 0., phi, frame)[:2]
    eta = np.linspace(0, TWO_PI, n, endpoint=False)
    R, Z = section_point(A_RAD, eta, phi[0], frame)[:2]
    psi = np.unwrap(np.arctan2(Z - Zax[0], R - Rax[0]))
    sigma = int(np.sign(psi[-1] - psi[0]))
    rotation = {}
    for s in (0.1, 0.5, 1.0):
        r = radius_of_rho(np.sqrt(s))[0]
        lam, beta = labels(r, 0.7)[:2]            # one field line: eta fixed
        ph = np.linspace(0, TWO_PI, 4*n, endpoint=False)  # source-phase samples
        Xl = forward(np.full_like(ph, lam), np.full_like(ph, beta), ph)
        Rl, pl, Zl = cylindrical(Xl, frame)
        Ra, Za = section_point(0., 0., pl, frame)[:2]
        ang = np.unwrap(np.append(np.arctan2(Zl - Za, Rl - Ra), np.arctan2(Zl - Za, Rl - Ra)[0]))
        rotation[s] = (ang[-1] - ang[0])/TWO_PI
    return sigma, rotation


# ---------------------------------------------------------------- boundary Fourier series
def boundary_rz(theta, phi, frame, sigma, c, s=1.0, kind="eta"):
    """Exact (R, Z) of the flux surface s at VMEC angles theta, phi (theta = sigma*eta + c*phi)."""
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    return section_point(radius_of_rho(np.sqrt(s))[0], sigma*(theta - c*phi), phi, frame, kind)[:2]


def fourier_spectrum(R, Z):
    """Complex FFT coefficients of R, Z on a uniform (theta, phi) grid (NFP = 1)."""
    n = R.size
    return np.fft.fft2(R)/n, np.fft.fft2(Z)/n


def fourier_rows(fr, fz, mpol, ntor):
    """VMEC rows [m, n, RBC, RBS, ZBC, ZBS] of R = sum RBC cos(m th - n phi) + RBS sin(...)."""
    rows = []
    for m in range(mpol):
        for n in range(0 if m == 0 else -ntor, ntor + 1):
            f = 1 if m == n == 0 else 2
            r, z = fr[m, -n], fz[m, -n]
            rows.append([m, n, f*r.real, -f*r.imag, f*z.real, -f*z.imag])
    return np.array(rows)


def evaluate_rows(rows, theta, phi):
    ang = theta[:, None]*rows[:, 0] - phi[:, None]*rows[:, 1]
    cs, sn = np.cos(ang), np.sin(ang)
    return cs@rows[:, 2] + sn@rows[:, 3], cs@rows[:, 4] + sn@rows[:, 5]


def truncation_ladder(frame, sigma, c, resolutions, grid=(96, 256), npoints=1500, kind="eta", seed=11):
    """Held-out max |(dR, dZ)| of the truncated boundary series versus the exact surface.

    One high-resolution FFT supplies every truncation; ``tail`` is the largest coefficient
    at the edge of the grid (how far the reference spectrum itself is resolved).
    """
    th = TWO_PI*np.arange(grid[0])/grid[0]
    ph = TWO_PI*np.arange(grid[1])/grid[1]
    R, Z = boundary_rz(th[:, None], ph[None, :], frame, sigma, c, kind=kind)
    fr, fz = fourier_spectrum(R, Z)
    edge_m, edge_n = slice(grid[0]//2 - 2, grid[0]//2 + 3), slice(grid[1]//2 - 4, grid[1]//2 + 5)
    tail = float(max(*(abs(f[edge_m]).max() for f in (fr, fz)), *(abs(f[:, edge_n]).max() for f in (fr, fz))))
    rng = np.random.default_rng(seed)
    tt, pp = rng.uniform(0, TWO_PI, (2, npoints))
    Re, Ze = boundary_rz(tt, pp, frame, sigma, c, kind=kind)
    out = []
    for mpol, ntor in resolutions:
        rows = fourier_rows(fr, fz, mpol, ntor)
        Rf, Zf = evaluate_rows(rows, tt, pp)
        out.append(dict(mpol=mpol, ntor=ntor, modes=len(rows),
                        max_error=float(np.max(np.hypot(Rf - Re, Zf - Ze))), tail=tail))
    return out


def optimize_frame(mpol, ntor, start=(1.0239, 1.5631, -.2776, -.0714, -.0672), grid=(24, 160),
                   rmin=.05, ratio=60.):
    """Nelder-Mead over (polar, azimuth, origin) of the frame; minimizes the on-grid truncation
    error at (mpol, ntor). Rejects frames that fail the one-turn / monotone chart test, have
    R < rmin, or dphi/dphase spread above ``ratio`` (conditioning guard)."""
    from scipy.optimize import minimize
    th = TWO_PI*np.arange(grid[0])/grid[0]
    ph = TWO_PI*np.arange(grid[1])/grid[1]
    pc = np.linspace(0, TWO_PI, 1024, endpoint=False)
    lb, bb = labels(A_RAD, np.linspace(0, TWO_PI, 8, endpoint=False)[:, None])[:2]
    X, _, _, Xp = forward(np.concatenate([[[A0]], lb]), np.concatenate([[[B0]], bb]), pc[None, :],
                          tangents=True)
    keep = ((np.abs(np.fft.fftfreq(grid[0], 1/grid[0]))[:, None] < mpol) &
            (np.abs(np.fft.fftfreq(grid[1], 1/grid[1]))[None, :] <= ntor))

    def error(p):
        frame = make_frame(p[0], p[1], p[2:])
        Xc, Yc, _ = to_frame(X, frame)
        step = np.mod(np.roll(np.arctan2(Yc, Xc), -1, 1) - np.arctan2(Yc, Xc), TWO_PI)
        Tc = Xp @ frame[0].T
        dphi = (Xc*Tc[..., 1] - Yc*Tc[..., 0])/(Xc*Xc + Yc*Yc)
        if not ((step < np.pi).all() and (step > 0).all() and np.abs(step.sum(1) - TWO_PI).max() < 1e-9
                and np.hypot(Xc, Yc).min() > rmin and dphi.min() > 0 and dphi.max() < ratio*dphi.min()):
            return 1.
        sigma = -1   # the orientation/rotation of accepted frames is re-measured afterwards
        d = [np.fft.ifft2(np.where(keep, np.fft.fft2(q), 0)).real - q
             for q in boundary_rz(th[:, None], ph[None, :], frame, sigma, 1)]
        return float(np.hypot(*d).max())
    result = minimize(lambda p: np.log(error(p) + 1e-12), start, method="Nelder-Mead",
                      options=dict(xatol=1e-6, fatol=1e-6, maxiter=1000))
    return result.x, float(np.exp(result.fun))


# ---------------------------------------------------------------- flux from the exact field
def disk_flux(h, n_r, n_alpha, phase=0.3):
    """Flux of the exact field through the level-h ellipse at fixed source phase.

    Integrates B . (X_lam x X_beta) over (lam, beta) = (a + r cos al, b + (B/A) r sin al); B comes from
    the Cartesian formula at the known root, X_lam, X_beta from the analytic forward-map tangents.
    Gauss-Legendre in r, trapezoid in alpha (both exponentially convergent: the integrand is analytic).
    """
    x, w = leggauss(n_r)
    r = A_RAD*np.sqrt(h)*(x + 1)/2
    al = TWO_PI*np.arange(n_alpha)/n_alpha
    lam, beta = A0 + r[:, None]*np.cos(al), B0 + B_RAD/A_RAD*r[:, None]*np.sin(al)
    X, Xl, Xb, _ = forward(lam, beta, phase, tangents=True)
    dflux = np.einsum("...i,...i", field_at_root(lam, beta, phase, X), np.cross(Xl, Xb))
    return float(np.einsum("i,ij->", w*r*A_RAD*np.sqrt(h)/2, dflux)*B_RAD/A_RAD*TWO_PI/n_alpha)


def flux_ladder(h=1.0, sizes=((2, 4), (3, 6), (4, 8), (6, 12), (8, 16), (12, 24), (16, 32))):
    """Resolution ladder of ``disk_flux`` against the closed form, with the successive-difference
    error estimate (what a user without the closed form would see)."""
    exact, out, previous = float(flux_of_h(h)), [], None
    for n_r, n_a in sizes:
        value = disk_flux(h, n_r, n_a)
        out.append(dict(n_r=n_r, n_alpha=n_a, flux=value, rel_error=abs(value/exact - 1),
                        successive_diff=None if previous is None else abs(value - previous)/exact))
        previous = value
    return out


def volume(n_r=8, n_alpha=16, n_phase=256):
    """Exact volume inside the edge surface: int det(X_lam, X_beta, X_phase) dlam dbeta dphase
    (Gauss-Legendre in r, trapezoid in alpha and phase). Also int B^2 dV = 2 pi Phi_edge is not assumed."""
    x, w = leggauss(n_r)
    r = A_RAD*(x + 1)/2
    al = TWO_PI*np.arange(n_alpha)/n_alpha
    ph = TWO_PI*np.arange(n_phase)/n_phase
    lam = (A0 + r[:, None, None]*np.cos(al)[None, :, None])*np.ones(n_phase)
    beta = (B0 + B_RAD/A_RAD*r[:, None, None]*np.sin(al)[None, :, None])*np.ones(n_phase)
    _, Xl, Xb, Xp = forward(lam, beta, ph, tangents=True)
    jac = np.einsum("...i,...i", np.cross(Xl, Xb), Xp)
    X = forward(lam, beta, ph)
    b2 = np.sum(field_at_root(lam, beta, ph, X)**2, -1)
    cell = B_RAD/A_RAD*TWO_PI/n_alpha*TWO_PI/n_phase*(w*r*A_RAD/2)[:, None, None]
    return float(np.sum(cell*jac)), float(np.sum(cell*jac*b2))


def volume_ladder(sizes=((4, 8, 64), (6, 12, 128), (8, 16, 256), (12, 24, 512))):
    """(V, int B^2 dV) ladder; the second entry is checked against 2 pi Phi_edge."""
    out, previous = [], None
    for n in sizes:
        v, e = volume(*n)
        out.append(dict(n_r=n[0], n_alpha=n[1], n_phase=n[2], volume=v, b2_integral=e,
                        b2_rel_error=abs(e/(TWO_PI*float(flux_edge())) - 1),
                        successive_diff=None if previous is None else abs(v/previous - 1)))
        previous = v
    return out


def section_flux(rho_max, n_rho, n_eta, frame, phi=0.9, kind="eta"):
    """Flux of the exact field through the geometric-phi section inside s = rho_max^2, plus the
    pointwise deviation of the flux density from the uniform value (Phi_edge/pi) rho drho deta.

    B_phi |dR dZ| comes from the analytic section Jacobian at Gauss-Legendre nodes in rho and
    uniform nodes in eta; there are no finite differences and h(s) enters only through r(rho).
    ``orientation`` is the sign of dR dZ / (drho deta) (sigma of the module docstring).
    """
    x, w = leggauss(n_rho)
    rho = rho_max*(x + 1)/2
    eta = TWO_PI*(np.arange(n_eta) + .5)/n_eta
    r, drdrho = radius_of_rho(rho)
    *_, lam, beta, phase = jac = section_jacobian(r[:, None], drdrho[:, None], eta[None, :], phi, frame, kind)
    Rr, Zr, Ra, Za = jac[2:6]
    g = Rr*Za - Ra*Zr
    Bc = field_at_root(lam, beta, phase) @ frame[0].T
    bphi = -Bc[..., 0]*np.sin(phi) + Bc[..., 1]*np.cos(phi)
    dflux = bphi*np.abs(g)
    flux = float(np.einsum("i,ij->", w*rho_max/2, dflux)*TWO_PI/n_eta)
    return dict(flux=flux, orientation=int(np.sign(g[0, 0])) if (np.sign(g) == np.sign(g[0, 0])).all() else 0,
                min_bphi=float(bphi.min()),
                density_max_dev=float(np.abs(dflux/(flux_edge()/np.pi*rho[:, None]) - 1).max()))


def section_flux_ladder(frame, sizes=((4, 8), (6, 16), (8, 32), (12, 64), (16, 128)), s=1.0):
    exact = float(flux_edge()*s)
    out = []
    for n_rho, n_eta in sizes:
        res = section_flux(np.sqrt(s), n_rho, n_eta, frame)
        res.update(n_rho=n_rho, n_eta=n_eta, rel_error=abs(res["flux"]/exact - 1))
        out.append(res)
    return out


def inversion_check(frame, n=4000, seed=5):
    """Forward/inverse agreement of phase_of_phi on random surfaces, and seam continuity of R, Z."""
    rng = np.random.default_rng(seed)
    r = A_RAD*np.sqrt(rng.random(n))
    lam, beta = labels(r, rng.uniform(0, TWO_PI, n))[:2]
    phi = rng.uniform(-np.pi, np.pi, n)
    phase = phase_of_phi(lam, beta, phi, frame)
    back = cylindrical(forward(lam, beta, phase), frame)[1]
    err = float(np.abs(np.angle(np.exp(1j*(back - phi)))).max())
    e = 1e-9
    side = [section_point(A_RAD, 1.1, np.array([-e, e]) + k*TWO_PI, frame)[:2] for k in (0, 1)]
    seam = float(max(abs(side[0][0][0] - side[0][0][1]), abs(side[0][1][0] - side[0][1][1]),
                     abs(side[1][0][0] - side[0][0][0]), abs(side[1][1][0] - side[0][1][0])))
    return dict(max_phi_error=err, seam_jump=seam, samples=n)


# ---------------------------------------------------------------- transform, axis, VMEC input decks
LENGTH_M, FIELD_T = 1.0, 1.0   # reference units are used as SI (same convention as build_inputs.py)


def gauge(frame):
    """(sigma, c): orientation of eta in (R, Z) and the integer field-line rotation per toroidal turn,
    both measured (``orientation_and_rotation``); the rotation must be an integer, equal on all surfaces."""
    sigma, rotation = orientation_and_rotation(frame)
    values = np.array(list(rotation.values()))
    c = int(np.round(values[0]))
    if np.abs(values - c).max() > 1e-6:
        raise ValueError(f"non-integer or surface-dependent field-line rotation: {rotation}")
    return sigma, c


def axis_harmonics(frame, sigma, c, nmax, n=1024):
    """Exact magnetic axis R(phi) = sum cr_n cos(n phi) + sr_n sin(n phi), Z likewise (n = 0..nmax)."""
    phi = TWO_PI*np.arange(n)/n
    R, Z = boundary_rz(0., phi, frame, sigma, c, s=0.)
    out = {}
    for name, q in (("R", R), ("Z", Z)):
        f = np.fft.rfft(q)[:nmax + 1]/n*2
        f[0] /= 2
        out[name + "_cos"], out[name + "_sin"] = f.real.tolist(), (-f.imag).tolist()
    return out


def boundary_rows(frame, sigma, c, mpol, ntor, grid=None):
    """VMEC boundary rows from one FFT, plus the held-out truncation error (max |(dR, dZ)|)."""
    grid = grid or (max(64, 8*mpol), max(256, 6*ntor))
    th, ph = TWO_PI*np.arange(grid[0])/grid[0], TWO_PI*np.arange(grid[1])/grid[1]
    R, Z = boundary_rz(th[:, None], ph[None, :], frame, sigma, c)
    rows = fourier_rows(*fourier_spectrum(R, Z), mpol, ntor)
    tt, pp = np.random.default_rng(97).uniform(0, TWO_PI, (2, 2000))
    Re, Ze = boundary_rz(tt, pp, frame, sigma, c)
    Rf, Zf = evaluate_rows(rows, tt, pp)
    return rows, float(np.max(np.hypot(Rf - Re, Zf - Ze)))


def deck_text(rows, closure, sigma, c):
    """INDATA for the zero-pressure vacuum case; closure 1 = zero toroidal current (transform
    predicted), 0 = prescribed constant iota = c (control only)."""
    mpol, ntor = int(rows[:, 0].max()) + 1, int(rows[:, 1].max())
    lines = ["&INDATA", "  LFREEB = F", "  LASYM = T", f"  NFP = {NFP}", f"  MPOL = {mpol}",
             f"  NTOR = {ntor}",
             "  NS_ARRAY = 17 33 65", "  FTOL_ARRAY = 1e-10 1e-12 1e-14", "  NITER_ARRAY = 10000 20000 30000",
             "  NSTEP = 200", "  DELT = 0.5", "  GAMMA = 0", "  PRES_SCALE = 1",
             "  PMASS_TYPE = 'power_series'",
             f"  PHIEDGE = {FIELD_T*LENGTH_M**2*float(flux_edge()):.17e}", "  AM = 0.0",
             f"  NCURR = {closure}"]
    if closure == 0:
        lines += ["  PIOTA_TYPE = 'power_series'", f"  AI = {float(c):.17e}"]
    else:
        lines += ["  PCURR_TYPE = 'power_series'", "  CURTOR = 0.0", "  AC = 0.0"]
    for m, n, rc, rs, zc, zs in rows:
        for key, value in (("RBC", rc), ("ZBS", zs), ("RBS", rs), ("ZBC", zc)):
            if abs(value) > 2e-15:
                lines.append(f"  {key}({int(n)},{int(m)}) = {LENGTH_M*value:.17e}")
    return "\n".join(lines + ["/"]) + "\n"


def write_decks(mpol=10, ntor=72, frame=OPT_FRAME, out=ROOT/"inputs"):
    """Write input.lopez_D_current / input.lopez_D_iota and update inputs/manifest.json (this case only)."""
    sigma, c = gauge(frame)
    rows, error = boundary_rows(frame, sigma, c, mpol, ntor)
    manifest = json.loads((out/"manifest.json").read_text())
    manifest["records"] = [r for r in manifest["records"] if r["case"] != "lopez_D"]
    lopez = dict(source="arXiv:2610.29921v1 configuration D", frame_rotation=frame[0].tolist(),
                 frame_origin=frame[1].tolist(), sigma=sigma, iota=c, theta="sigma*eta + c*phi (Lambda = 0)",
                 phiedge=FIELD_T*LENGTH_M**2*float(flux_edge()), volume=volume()[0],
                 axis=axis_harmonics(frame, sigma, c, ntor))
    for closure, suffix in ((1, "current"), (0, "iota")):
        path = out/f"input.lopez_D_{suffix}"
        path.write_text(deck_text(rows, closure, sigma, c))
        manifest["records"].append(dict(case="lopez_D", file=path.name, ncurr=closure, axisymmetric=False,
                                        lasym=True, boundary_max_error_m=error*LENGTH_M,
                                        symmetric_omitted_coeff_m=0.0, profile_errors={}, vmex_run=False,
                                        status="candidate_input_not_equilibrium",
                                        resolution=dict(mpol=mpol, ntor=ntor, ns=[17, 33, 65]), lopez=lopez))
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return error


def report():
    """Every ladder of this module as one JSON-serializable dict."""
    sigma, c = gauge(OPT_FRAME)
    s = np.linspace(0, 1, 10001)
    out = dict(flux_edge=float(flux_edge()), max_h_minus_s=float(np.abs(h_of_s(s) - s).max()),
               disk_flux_ladder=flux_ladder(1.0),
               disk_flux_ladder_h03=flux_ladder(0.3, ((2, 4), (4, 8), (8, 16))),
               volume_ladder=volume_ladder(), sigma=sigma, iota=c, frames={})
    out["section_flux_ladder"] = {str(v): section_flux_ladder(OPT_FRAME, s=v) for v in (0.1, 0.5, 1.0)}
    out["section_flux_alpha_gauge_control"] = section_flux(1.0, 8, 64, OPT_FRAME, kind="alpha")
    ladders = {"paper": (PAPER_FRAME, [(m, n) for m in (12, 24) for n in (24, 56, 96, 192, 320, 480)],
                         (96, 1024)),
               "optimized": (OPT_FRAME, [(m, n) for m in (4, 8, 10, 12) for n in (16, 24, 40, 56, 72, 96)],
                             (64, 256))}
    for name, (frame, resolutions, grid) in ladders.items():
        sg, cc = gauge(frame)
        out["frames"][name] = dict(chart=chart_check(frame), sigma=sg, iota=cc,
                                   inversion=inversion_check(frame),
                                   truncation=truncation_ladder(frame, sg, cc, resolutions, grid=grid))
    return out


if __name__ == "__main__":
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))   # e.g. --mpol 10 --ntor 72 [--report out.json]
    if "--report" in args:
        Path(args["--report"]).write_text(json.dumps(report(), indent=1, default=float) + "\n")
    print("boundary max error:", write_decks(int(args.get("--mpol", 10)), int(args.get("--ntor", 72))))
