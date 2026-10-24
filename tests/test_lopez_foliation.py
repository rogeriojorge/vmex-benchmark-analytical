"""Shifted foliation of the Lopez D field: same outer contour, flux-uniform eta, nested."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import lopez_foliation as F  # noqa: E402
import lopez_surfaces as ls  # noqa: E402

D = (0.3, -0.2)


def test_zero_shift_is_the_concentric_flux_uniform_family():
    eta = np.linspace(0, 2*np.pi, 17)
    for s in (0.1, 0.5, 1.0):
        r = ls.radius_of_rho(np.sqrt(s))[0]
        lam, beta = ls.labels(r, eta, "eta")[:2]
        lam2, beta2 = F.labels_of(s, eta, (0.0, 0.0))
        assert np.abs(lam - lam2).max() < 1e-13 and np.abs(beta - beta2).max() < 1e-13


def test_flux_closed_form_matches_concentric_and_is_monotone():
    h = ls.h_of_s(np.array([0.2, 0.7, 1.0]))
    assert np.allclose(F.flux_of_sigma(np.sqrt(h), (0.0, 0.0)), ls.flux_of_h(h), rtol=1e-14)
    sigma = np.linspace(1e-3, 1, 400)
    assert (np.diff(F.flux_of_sigma(sigma, D)) > 0).all()
    assert F.flux_of_sigma(1.0, D) == pytest.approx(float(ls.flux_edge()), rel=1e-14)   # same boundary, same flux
    for s in (0.05, 0.5, 0.95):
        assert F.flux_of_sigma(F.sigma_of_s(s, D), D) == pytest.approx(s*float(ls.flux_edge()), rel=1e-13)


def test_outer_contour_is_the_boundary_ellipse_and_axis_is_the_shifted_centre():
    eta = np.linspace(0, 2*np.pi, 33)
    lam, beta = F.labels_of(1.0, eta, D)
    assert np.abs(((lam - F.A0)/F.A_RAD)**2 + ((beta - F.B0)/F.B_RAD)**2 - 1).max() < 1e-12
    lam0, beta0 = F.labels_of(1e-12, eta, D)
    assert np.allclose(lam0, F.A0 + D[0]*F.A_RAD, atol=1e-6*F.A_RAD) and np.allclose(beta0, F.B0 + D[1]*F.B_RAD, atol=1e-6*F.B_RAD)


def test_flux_density_is_uniform_in_s_and_eta():
    rng = np.random.default_rng(2)
    s0, eta0 = rng.uniform(0.1, 0.9, 6), rng.uniform(0, 2*np.pi, 6)
    ds, de = 1e-6, 1e-5
    for s, e in zip(s0, eta0):
        l_s = (np.array(F.labels_of(s + ds, np.array([e]), D)) - np.array(F.labels_of(s - ds, np.array([e]), D)))/(2*ds)
        l_e = (np.array(F.labels_of(s, np.array([e + de]), D)) - np.array(F.labels_of(s, np.array([e - de]), D)))/(2*de)
        lam = F.labels_of(s, np.array([e]), D)[0]
        density = np.abs(l_s[0]*l_e[1] - l_s[1]*l_e[0])/(2*lam)       # flux per ds deta
        assert density[0] == pytest.approx(float(ls.flux_edge())/(2*np.pi), rel=1e-7)


def test_eta_inversion_is_exact_and_rejects_crossing_contours():
    sigma = F.sigma_of_s(0.6, D)
    alpha = np.linspace(-1, 7, 20)
    w0, a, b = F._density_series(sigma, D)
    k = np.arange(1, len(a) + 1)
    eta = alpha + np.sum((a*np.sin(k*alpha[:, None]) + b*(1 - np.cos(k*alpha[:, None])))/k, -1)/w0
    assert np.abs(F.alpha_of_eta(eta, sigma, D) - alpha).max() < 1e-13
    assert (np.diff(eta) > 0).all()                                      # strictly increasing
    with pytest.raises(ValueError):
        F.sigma_of_s(0.5, (0.6, 0.0))


def test_kind_parser():
    assert F.parse_kind("shift:0.3:-0.2") == (0.3, -0.2) and F.parse_kind("eta") is None
