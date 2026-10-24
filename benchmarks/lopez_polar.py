"""The Lopez D exact state in VMEC's m = 1 polar gauge (lambda != 0).

VMEC (and VMEX) evolve the m = 1 R/Z harmonics in a constrained basis, rss = zcs and rsc = zcc
for m = 1 (``residue.f90``, ``readin.f``). In terms of the cross-section at fixed (s, phi): with
R_1(theta) = A cos theta + B sin theta and Z_1(theta) = C cos theta + D sin theta the constraint is
B = C at every phi. The flux-uniform angle eta of the exact construction does not satisfy it, so a
state built in the eta gauge (lambda = 0) is *not* representable by the solver: its boundary is rotated
by VMEC's own theta shift and the interior is then inconsistent with the edge (``hot_restart_state``
shears the edge against the interior, the Jacobian changes sign, and the axis retry rebuilds a cold
interior guess; every earlier "exact seed" run was silently a cold start).

The same surfaces in the polar gauge: theta' = theta - omega(s, phi) with

    omega = - arg z+,   z+ = [(A + D) + i (C - B)]/2  (the e^{i theta} coefficient of R + i Z),

so that z+ becomes real and positive (B' = C'); evaluated on the exact surface s. The phase of z+ stays
within a band as phi runs around (winding 0), so the VMEC transform in this gauge is still c. The field line of transform c then obeys
d(theta')/d(phi) = c - omega_phi, i.e. lambda(s, theta', phi) = omega(s, phi) (m = 0 harmonics only)
and B^theta' = (c - omega_phi) B^phi, B^phi = Phi_edge / (2 pi sqrt(g)) unchanged (theta' = theta - omega
has unit Jacobian at fixed s and phi, so the flux density stays uniform). omega(s, phi) is smooth, so it is
held as a Fourier series in phi and evaluated at arbitrary phi.

Kinds: 'polar:<base>' with base 'eta', 'alpha' or 'shift:dl:db'.
"""
from __future__ import annotations

import numpy as np

import lopez_foliation as lf
import lopez_surfaces as ls

TWO_PI = 2*np.pi
_NTH, _NPHI = 64, 512
_S_AXIS = 1e-4          # omega at the axis is the limit s -> 0 (the m = 1 content vanishes there)


def is_polar(kind):
    return isinstance(kind, str) and kind.startswith("polar:")


def base_kind(kind):
    return kind.split(":", 1)[1] if is_polar(kind) else kind


def base_rz(kind, theta, phi, frame, sigma, c, s):
    """(R, Z) of the unshifted surface family (no polar gauge)."""
    shift = lf.parse_kind(kind)
    if shift is not None:
        return lf.boundary_rz(theta, phi, frame, sigma, c, s, shift)
    return ls.boundary_rz(theta, phi, frame, sigma, c, s=s, kind=kind)


def m1_harmonics(R, Z):
    """(A, B, C, D) per phi column of R, Z sampled uniformly in theta (axis 0)."""
    n = R.shape[0]
    fr, fz = np.fft.rfft(R, axis=0)[1]/n, np.fft.rfft(Z, axis=0)[1]/n
    return 2*fr.real, -2*fr.imag, 2*fz.real, -2*fz.imag


def omega_grid(kind, s, frame, sigma, c, nphi=_NPHI):
    """omega(phi_k) on a uniform phi grid for the surface s of the base family."""
    th = TWO_PI*np.arange(_NTH)/_NTH
    ph = TWO_PI*np.arange(nphi)/nphi
    R, Z = base_rz(kind, th[:, None], ph[None, :], frame, sigma, c, max(s, _S_AXIS))
    a, b, cc, d = m1_harmonics(R, Z)
    phase = np.unwrap(np.angle((a + d) + 1j*(cc - b)))      # arg of z+ = [(A + D) + i (C - B)]/2
    winding = (phase[-1] - phase[0])*len(phase)/(len(phase) - 1)/TWO_PI
    if abs(winding) > 0.05:
        raise ValueError(f"the m = 1 phase winds {winding:.3f} times in phi: the polar gauge shifts the transform")
    return -phase


_CACHE = {}


def omega_series(kind, s, frame, sigma, c):
    """Fourier coefficients of omega(s, phi) = a0 + sum_{n>=1} 2 Re(a_n exp(i n phi)) (cached)."""
    key = (kind, float(s), int(sigma), int(c), np.asarray(frame[0]).tobytes(), np.asarray(frame[1]).tobytes())
    if key not in _CACHE:
        w = omega_grid(kind, s, frame, sigma, c)
        _CACHE[key] = np.fft.rfft(w)/len(w)
    return _CACHE[key]


def eval_omega(series, phi, derivative=False):
    phi = np.asarray(phi, float)
    unique, inverse = np.unique(phi, return_inverse=True)     # omega does not depend on theta
    n = np.arange(len(series))
    e = np.exp(1j*np.multiply.outer(unique, n))
    if derivative:
        value = 2*np.real((e*(1j*n*series)).sum(-1))
    else:
        value = series[0].real + 2*np.real((e[..., 1:]*series[1:]).sum(-1))
    return value[inverse].reshape(phi.shape)


def polar_rz(kind, theta, phi, frame, sigma, c, s):
    """(R, Z) of the surface s of ``polar:<base>`` at polar-gauge angles (theta', phi)."""
    base = base_kind(kind)
    theta, phi = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
    w = eval_omega(omega_series(base, s, frame, sigma, c), phi)
    return base_rz(base, theta + w, phi, frame, sigma, c, s)


def lambda_and_derivative(kind, ns, nphi, frame, sigma, c):
    """omega and d omega / d phi of every full-mesh surface on the uniform phi grid: (ns, nphi) each."""
    base = base_kind(kind)
    ph = TWO_PI*np.arange(nphi)/nphi
    out_w, out_d = np.empty((ns, nphi)), np.empty((ns, nphi))
    for j in range(ns):
        ser = omega_series(base, j/(ns - 1), frame, sigma, c)
        out_w[j], out_d[j] = eval_omega(ser, ph), eval_omega(ser, ph, derivative=True)
    return out_w, out_d
