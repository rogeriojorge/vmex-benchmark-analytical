"""Periodic lambda reconstruction from both signed contravariant flux densities.

Independent NumPy reference, not a VMEX solver or a Boozer transform.
Angles are theta in [0, 2*pi) and physical phi in [0, 2*pi/nfp).
"""
from __future__ import annotations

import numpy as np


def derivative(values: np.ndarray, axis: int, period: float) -> np.ndarray:
    """Differentiate resolved real Fourier data on an odd periodic grid."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or axis not in (0, 1):
        raise ValueError("Expected a 2-D array and axis 0 or 1")
    if values.shape[axis] % 2 != 1 or period <= 0:
        raise ValueError("Use an odd grid and a positive period")
    wave = 2 * np.pi * np.fft.fftfreq(values.shape[axis], d=period / values.shape[axis])
    shape = [1, 1]
    shape[axis] = len(wave)
    return np.fft.ifft(np.fft.fft(values, axis=axis) * (1j * wave.reshape(shape)), axis=axis).real


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def recover_lambda(U: np.ndarray, V: np.ndarray, *, poloidal: float,
                   toroidal: float, nfp: int, tolerance: float = 1e-9
                   ) -> tuple[np.ndarray, dict[str, float]]:
    """Recover zero-mean lambda from U=J B^theta and V=J B^phi.

    U = poloidal - toroidal*lambda_phi;
    V = toroidal*(1 + lambda_theta).
    `poloidal` and `toroidal` are SIGNED coordinate flux derivatives, not
    unsigned WOUT fluxes. Bad compatibility or periods are rejected rather
    than silently projected out. The tolerance is dimensionless and scaled
    to max(1, RMS of the supplied lambda gradient).
    """
    U, V = np.asarray(U, dtype=float), np.asarray(V, dtype=float)
    if U.ndim != 2 or U.shape != V.shape or min(U.shape) < 3:
        raise ValueError("Expected two equal 2-D periodic arrays")
    if any(n % 2 != 1 for n in U.shape):
        raise ValueError("Use odd grids to exclude the real Nyquist ambiguity")
    if (not np.isfinite(U).all() or not np.isfinite(V).all()
            or not np.isfinite([poloidal, toroidal, tolerance]).all()
            or toroidal == 0 or tolerance <= 0 or int(nfp) != nfp or nfp < 1):
        raise ValueError("Invalid flux densities, signed fluxes, NFP, or tolerance")
    gtheta, gphi = V / toroidal - 1, (poloidal - U) / toroidal
    scale = max(1.0, np.hypot(rms(gtheta), rms(gphi)))
    curl = derivative(gtheta, 1, 2 * np.pi / nfp) - derivative(gphi, 0, 2 * np.pi)
    periods = max(float(np.max(np.abs(np.mean(gtheta, axis=0)))),
                  float(np.max(np.abs(np.mean(gphi, axis=1)))))
    m = np.fft.fftfreq(U.shape[0], d=1 / U.shape[0])[:, None]
    kphi = nfp * np.fft.fftfreq(U.shape[1], d=1 / U.shape[1])[None, :]
    k2 = m*m + kphi*kphi
    numerator = -1j * (m*np.fft.fft2(gtheta) + kphi*np.fft.fft2(gphi))
    coeff = np.zeros_like(numerator)
    np.divide(numerator, k2, out=coeff, where=k2 > 0)
    lam = np.fft.ifft2(coeff).real
    dt = derivative(lam, 0, 2*np.pi)
    dp = derivative(lam, 1, 2*np.pi/nfp)
    error = np.hypot(rms(dt - gtheta), rms(dp - gphi)) / scale
    certificate = dict(gradient_reconstruction_relative=float(error),
                       closedness_scaled_rms=rms(curl)/scale,
                       period_mean_scaled_max=periods/scale,
                       lambda_mean=float(np.mean(lam)),
                       minimum_straight_angle_derivative=float(np.min(1+dt)))
    if max(error, periods/scale) > tolerance:
        raise ValueError(f"Incompatible periodic flux data: {certificate}")
    # Curl is reported separately: its units include an angular derivative,
    # so a grid-independent gradient tolerance is not also a curl tolerance.
    return lam, certificate
