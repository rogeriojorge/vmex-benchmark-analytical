"""Alternative nested foliations of the same Lopez D vacuum field.

The exact field of configuration D has closed field lines, so any family of nested contours in the
(lambda, beta) label disk with the prescribed outer contour is a valid set of magnetic surfaces
(López, Section on foliation gauge). This module builds one such family beyond the paper's concentric
ellipses: homothetic contractions of the boundary ellipse towards an interior point displaced from the
ellipse centre by ``d = (dl, db)`` in units of the semi-axes. The contour centre moves like sigma^2, not
like sigma: a homothety about the interior point would give contours whose centre is displaced from the
limit point by sigma D, a conical (non-smooth) coordinate singularity at the axis that no VMEC radial
representation can hold (measured: a projection error proportional to |d| that does not converge). Here

    p(sigma, alpha) = c0 + (1 - sigma^2) D + sigma (A cos alpha, B sin alpha),   D = (dl A, db B),

is smooth at the axis (the contours are concentric ellipses about it to first order) and is nested iff
2 |d| < 1 (the Jacobian carries the factor 1 - 2 sigma (dl cos(alpha) + db sin(alpha))).

Flux inside the contour (closed form, the same integral as for the concentric family with the
centre moved to lambda_c = a + (1 - sigma^2) dl A):

    Phi(sigma) = pi (B/A) (lambda_c - sqrt(lambda_c^2 - sigma^2 A^2)).

The poloidal label ``eta`` is flux-uniform: with (sigma, alpha) as base coordinates the flux two-form is
w(alpha) d(sigma) d(alpha), w = AB (1 - 2 sigma (dl cos alpha + db sin alpha)) / (lambda_c + sigma A cos alpha)
up to a factor depending on sigma only, so eta = 2 pi int_0^alpha w / int_0^{2 pi} w gives (Phi_edge / 2 pi) ds d(eta)
exactly. ``w`` is analytic and periodic, so its integral is done spectrally. The foliation is a different
VMEC state with the same B(x); every identity of the scorer applies to it unchanged. d = 0 reproduces the
concentric family of ``lopez_surfaces`` with eta = the paper's flux-uniform angle (up to the origin of eta).
"""
from __future__ import annotations

import numpy as np

import lopez_surfaces as ls

A0, B0, A_RAD, B_RAD = ls.A0, ls.B0, ls.A_RAD, ls.B_RAD
TWO_PI = 2*np.pi
_NFOUR = 64


def parse_kind(kind):
    """'shift:dl:db' -> (dl, db); anything else -> None."""
    if isinstance(kind, str) and kind.startswith("shift:"):
        _, dl, db = kind.split(":")
        return float(dl), float(db)
    return None


def _check(d):
    if 2*np.hypot(*d) >= 1.0:
        raise ValueError(f"|d| = {np.hypot(*d):.3f} >= 1/2: contours are not nested")


def flux_of_sigma(sigma, d):
    """Total toroidal flux inside the contour with contraction factor sigma (cancellation-safe)."""
    sigma = np.asarray(sigma, float)
    lc = A0 + (1 - sigma**2)*d[0]*A_RAD
    return np.pi*B_RAD/A_RAD*(sigma*A_RAD)**2/(lc + np.sqrt(lc**2 - (sigma*A_RAD)**2))


def sigma_of_s(s, d):
    """Contraction factor of the contour enclosing the flux fraction s (monotone, by bisection)."""
    _check(d)
    target = float(s)*float(ls.flux_edge())
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5*(lo + hi)
        if flux_of_sigma(mid, d) < target:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo + hi)


def _density_series(sigma, d):
    """Fourier coefficients (w0, a_k, b_k) of w(alpha) / (A B), w = a_0 + sum a_k cos + b_k sin."""
    alpha = TWO_PI*np.arange(_NFOUR)/_NFOUR
    lc = A0 + (1 - sigma**2)*d[0]*A_RAD
    w = (1 - 2*sigma*(d[0]*np.cos(alpha) + d[1]*np.sin(alpha)))/(lc + sigma*A_RAD*np.cos(alpha))
    f = np.fft.rfft(w)/_NFOUR
    return f[0].real, 2*f[1:].real, -2*f[1:].imag


def alpha_of_eta(eta, sigma, d):
    """Newton inversion of eta(alpha) = alpha + sum_k [a_k sin(k alpha) + b_k (1 - cos(k alpha))]/(k w0)."""
    w0, a, b = _density_series(sigma, d)
    k = np.arange(1, len(a) + 1)
    eta = np.asarray(eta, float)
    alpha = eta.copy()
    for _ in range(30):
        kk = k*alpha[..., None]
        value = alpha + np.sum((a*np.sin(kk) + b*(1 - np.cos(kk)))/k, -1)/w0
        slope = (w0 + np.sum(a*np.cos(kk) + b*np.sin(kk), -1))/w0
        step = (value - eta)/slope
        alpha = alpha - step
        if np.abs(step).max() < 1e-15:
            break
    return alpha


def labels_of(s, eta, d):
    """(lambda, beta) of the contour with flux fraction s at flux-uniform label eta."""
    if s <= 0.0:
        return np.full_like(np.asarray(eta, float), A0 + d[0]*A_RAD), np.full_like(np.asarray(eta, float), B0 + d[1]*B_RAD)
    sigma = sigma_of_s(s, d)
    alpha = alpha_of_eta(eta, sigma, d)
    lam = A0 + (1 - sigma**2)*d[0]*A_RAD + sigma*A_RAD*np.cos(alpha)
    beta = B0 + (1 - sigma**2)*d[1]*B_RAD + sigma*B_RAD*np.sin(alpha)
    return lam, beta


def boundary_rz(theta, phi, frame, sigma_orient, c, s, d):
    """(R, Z) of the contour s of the shifted foliation at VMEC angles (theta = sigma_orient*eta + c*phi)."""
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    lam, beta = labels_of(s, sigma_orient*(theta - c*phi), d)
    phase = ls.phase_of_phi(lam, beta, phi, frame)
    R, _, Z = ls.cylindrical(ls.forward(lam, beta, phase), frame)
    return R, Z
