"""Algebraic, quadrature, geometry and sensitivity checks independent of VMEX."""
import sys
from dataclasses import replace
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from analytic import (cases, differential_fields, field, flux, flux_rate, integer_targets,
                      iota, label_at_s, pressure, rotate, shear_flux_rates, surface,
                      validate, volume)
from build_inputs import boundary_coefficients, enclosed_current

CASES = cases()


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_force_surface_and_angle(case):
    validate(case)
    rng = np.random.default_rng(73)
    label = case.edge*rng.uniform(.01, .95, 24)
    th, ph = rng.uniform(0, 2*np.pi, (2, 24))
    x = surface(case, label, th, ph)
    B, J, gp, db, psi = differential_fields(case, x)
    np.testing.assert_allclose(psi, label, atol=case.edge*2e-12, rtol=0)
    np.testing.assert_allclose(np.cross(J, B), gp, atol=2e-12)
    np.testing.assert_allclose(np.trace(db, axis1=1, axis2=2), 0., atol=2e-12)
    np.testing.assert_allclose(np.sin(np.arctan2(x[:, 1], x[:, 0])-ph), 0., atol=2e-13)


def test_asymmetric_grad_shafranov_identity_symbolically():
    R, Z, R0, b, g, chi = sp.symbols("R Z R0 b g chi", positive=True)
    psi = b*(R**2-R0**2)**2+g*Z**2+2*chi*sp.sqrt(b*g)*(R**2-R0**2)*Z
    assert sp.simplify(sp.diff(psi, R, 2)-sp.diff(psi, R)/R+sp.diff(psi, Z, 2)-8*b*R**2-2*g) == 0


@pytest.mark.parametrize("name", ["integer_3d_stretched", "sheared_A", "solovev_asymmetric"])
def test_volume_from_boundary_integral(name):
    case = CASES[name]
    n = 256
    th = np.arange(n)*2*np.pi/n
    ph = np.arange(n)*2*np.pi/n
    x = surface(case, case.edge, th[:, None], ph[None, :])
    r2, z = x[..., 0]**2+x[..., 1]**2, x[..., 2]
    dz = np.fft.ifft(1j*np.fft.fftfreq(n, 1/n)[:, None]*np.fft.fft(z, axis=0), axis=0).real
    actual = (2*np.pi)**2*np.mean(r2*dz/2)
    np.testing.assert_allclose(actual, volume(case), rtol=2e-10)


@pytest.mark.parametrize("name", ["integer_3d", "sheared_A", "solovev_asymmetric"])
def test_flux_from_independent_cross_section_quadrature(name):
    case = CASES[name]
    n, w = leggauss(32)
    psi = case.edge*(n+1)/2
    th = np.arange(256)*2*np.pi/256
    x = surface(case, psi[:, None], th[None, :], 0.)
    h = 1e-4*psi[:, None]
    # Fourth-order radial differentiation of the explicit surface map.
    deriv = (-surface(case, psi[:, None]+2*h, th[None, :], 0.)
             +8*surface(case, psi[:, None]+h, th[None, :], 0.)
             -8*surface(case, psi[:, None]-h, th[None, :], 0.)
             +surface(case, psi[:, None]-2*h, th[None, :], 0.))/(12*h[..., None])
    dt = np.fft.ifft(1j*np.fft.fftfreq(256, 1/256)[None, :, None]*np.fft.fft(x, axis=1), axis=1).real
    jac = deriv[..., 0]*dt[..., 2]-dt[..., 0]*deriv[..., 2]
    B = np.asarray(field(case, x)[0])
    actual = case.edge/2*np.sum(w*np.mean(B[..., 1]*jac, axis=1))*2*np.pi
    np.testing.assert_allclose(actual, flux(case, case.edge), rtol=2e-8)


def test_two_independent_axisymmetric_field_formulas():
    case = CASES["integer_axisymmetric"]
    x = surface(case, .01, np.linspace(0, 2*np.pi, 31), .41)
    np.testing.assert_allclose(field(case, x)[0], field(CASES["solovev_symmetric"], x)[0], atol=1e-14)


def test_genuine_up_down_asymmetry():
    c = CASES["solovev_asymmetric"]
    x = np.array([[1.04, 0., .025]])
    mirrored = x*np.array([1., 1., -1.])
    assert abs(float(pressure(c, x)[0]-pressure(c, mirrored)[0])) > 1e-4


def test_frame_covariance_and_nonzero_asymmetric_coefficients():
    base, framed = CASES["integer_3d"], CASES["integer_3d_reframed"]
    x = surface(base, .01, np.linspace(.1, 6, 24), .2)
    xr = np.asarray(rotate(x, framed.phase))+[0.,0.,framed.zshift]
    np.testing.assert_allclose(field(framed, xr)[0], rotate(field(base, x)[0], framed.phase), atol=1e-13)
    rows, _, _, _ = boundary_coefficients(framed)
    assert np.max(abs(rows[:, 3:5])) > .01


def test_profile_inversion_and_ampere_current():
    case = CASES["sheared_A"]
    s = np.array([0., .15, .65, 1.])
    psi = label_at_s(case, s)
    np.testing.assert_allclose(flux(case, psi)/flux(case, case.edge), s, atol=2e-12)
    coarse, fine = enclosed_current(case, psi, 256), enclosed_current(case, psi, 1024)
    np.testing.assert_allclose(coarse, fine, atol=1e-12)
    assert fine[0] < 1e-12 and fine[-1] > 0


def test_scalar_derivatives_and_eulerian_null():
    a = jnp.array([np.sqrt(1.5), np.sqrt(.5), 1., 1/64])
    fun = lambda x: integer_targets(*x, xp=jnp)["beta"]
    ad = np.asarray(jax.grad(fun)(a))
    fd = np.array([(float(fun(a+1e-5*e))-float(fun(a-1e-5*e)))/2e-5 for e in np.eye(4)])
    np.testing.assert_allclose(ad, fd, rtol=3e-7, atol=1e-9)
    base = CASES["integer_3d"]
    x = jnp.asarray(surface(base, .005, .3, .4))
    null = jax.jacfwd(lambda d: field(replace(base, parameters=(*base.parameters[:-1], d)), x)[0])(jnp.array(base.edge))
    np.testing.assert_array_equal(null, np.zeros(3))


def test_lambda_transform_null_and_flux_scaling():
    k = jnp.array([0., .3, .7])
    fn = lambda lam: shear_flux_rates(k, 1.08, 3., lam, xp=jnp)
    ratio = lambda lam: fn(lam)[1]/fn(lam)[0]
    np.testing.assert_allclose(jax.jacfwd(ratio)(jnp.array(3.5)), 0., atol=1e-14)
    q = lambda lam: fn(lam)[0]
    np.testing.assert_allclose(jax.jacfwd(q)(jnp.array(3.5)), -q(3.5)/3.5, atol=1e-14)


def test_invalid_domains_rejected():
    for base, params in [(CASES["integer_3d"], (2., .3, .5, .1)),
                         (CASES["sheared_A"], (1., .1, 3., .245)),
                         (CASES["solovev_asymmetric"], (1., .25, 1., 1.1, 1., .01))]:
        with pytest.raises(ValueError):
            validate(replace(base, parameters=params))


def test_physical_scorer_scaling_and_sign_detection():
    from score_samples import score
    case = CASES["integer_3d"]
    s = np.linspace(.1, .8, 8)
    x = surface(case, case.edge*s, np.linspace(.1, 5., 8), .3)
    B, J, gp, _, _ = differential_fields(case, x)
    L, B0, mu0 = 2., 3., 4e-7*np.pi
    data = dict(case_name=case.name, xyz=L*x, B=B0*B, J=B0/(mu0*L)*J,
                gradp=B0*B0/(mu0*L)*gp, weights=np.ones(8),
                s=s, length_m=L, field_t=B0, mu0=mu0)
    good = score(data)
    assert good["field_relative_l2"] < 1e-14
    assert good["current_relative_l2"] < 1e-14
    assert good["surface_label_max_over_edge"] < 1e-12
    bad = score({**data, "B": -data["B"]})
    np.testing.assert_allclose(bad["field_relative_l2"], 2.)
    with pytest.raises(ValueError):
        score({**data, "weights": -data["weights"]})
    with pytest.raises(ValueError):
        score({**data, "field_t": np.inf})
    with pytest.raises(ValueError):
        score({**data, "xyz": L*surface(case, 1.2*case.edge, np.linspace(.1, 5., 8), .3)})
