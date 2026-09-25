"""Independent mathematical/contract probes; these do not execute VMEX."""
import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest

from flux_potential import derivative, recover_lambda, rms


@pytest.mark.parametrize("nfp", [1, 2, 5])
def test_two_flux_components_fix_periodic_lambda(nfp):
    th = 2*np.pi*np.arange(33)[:, None]/33
    ph = 2*np.pi*np.arange(35)[None, :]/(35*nfp)
    exact = 0.04*np.sin(3*th-2*nfp*ph)+0.02*np.cos(th+nfp*ph)
    dt, dp = derivative(exact, 0, 2*np.pi), derivative(exact, 1, 2*np.pi/nfp)
    P, T = -0.13, 0.31
    lam, cert = recover_lambda(T-P*dp, P*(1+dt), poloidal=T, toroidal=P, nfp=nfp)
    np.testing.assert_allclose(lam, exact, atol=2e-15)
    assert cert["gradient_reconstruction_relative"] < 2e-14
    assert cert["minimum_straight_angle_derivative"] > 0


def test_straightness_alone_does_not_fix_resonant_flux_density():
    th = 2*np.pi*np.arange(33)[:, None]/33
    ph = 2*np.pi*np.arange(35)[None, :]/70
    exact = 0.08*np.sin(2*th-2*ph)
    dt, dp = derivative(exact, 0, 2*np.pi), derivative(exact, 1, np.pi)
    U, V = 1-dp, 1+dt
    np.testing.assert_allclose(U/V, 1, atol=2e-14)
    assert rms(dp+dt) < 2e-14  # both exact lambda and zero solve transport
    lam, _ = recover_lambda(U, V, poloidal=1, toroidal=1, nfp=2)
    np.testing.assert_allclose(lam, exact, atol=2e-15)
    assert rms(V-1) > 0.1  # zero lambda would give the wrong field amplitude


def test_incompatible_one_form_is_rejected():
    th = 2*np.pi*np.arange(33)[:, None]/33
    ph = 2*np.pi*np.arange(35)[None, :]/35
    gtheta = np.sin(ph)+np.zeros_like(th)
    with pytest.raises(ValueError, match="Incompatible"):
        recover_lambda(np.zeros_like(gtheta), 1+gtheta, poloidal=0, toroidal=1, nfp=1)


def test_nonzero_period_is_not_silently_removed():
    with pytest.raises(ValueError, match="Incompatible"):
        recover_lambda(np.zeros((33,35)), np.full((33,35), 1.1), poloidal=0, toroidal=1, nfp=1)


def test_weak_stress_identity_and_pressure_defect():
    q, w = leggauss(8)
    x = np.stack(np.meshgrid(q,q,q,indexing="ij"), axis=-1)
    weights = w[:,None,None]*w[None,:,None]*w[None,None,:]
    X,Y,Z = np.moveaxis(x,-1,0)
    B = np.stack((-Y,X,np.zeros_like(X)),axis=-1)
    p = 3-X*X-Y*Y
    shape = (1-X*X)*(1-Y*Y)*(1-Z*Z)
    # v=(X*shape,0,0), zero on all faces; gradv stores component, derivative.
    gradv = np.zeros(x.shape[:-1]+(3,3))
    gradv[...,0,0]=(1-3*X*X)*(1-Y*Y)*(1-Z*Z)
    gradv[...,0,1]=-2*X*Y*(1-X*X)*(1-Z*Z)
    gradv[...,0,2]=-2*X*Z*(1-X*X)*(1-Y*Y)
    def weak(pressure):
        T=B[..., :,None]*B[..., None,:]-(0.5*np.sum(B*B,axis=-1)+pressure)[...,None,None]*np.eye(3)
        return -np.sum(weights*np.sum(T*gradv,axis=(-2,-1)))
    assert abs(weak(p)) < 2e-14
    epsilon=0.2
    expected=np.sum(weights*(-2*epsilon*X)*(X*shape))
    np.testing.assert_allclose(weak(p+epsilon*X*X), expected, atol=2e-14)
    assert abs(expected) > 0.05


def test_weighted_response_norm_is_not_sample_count_dependent():
    B=np.array([[1.,2.,3.],[2.,0.,1.]])
    weights=np.array([2.,1.])
    norm=lambda a,w: np.sqrt(np.sum(w*np.sum(a*a,axis=1))/np.sum(w))
    np.testing.assert_allclose(norm(B,weights),norm(np.repeat(B,7,axis=0),np.repeat(weights/7,7)))
    # dB/da has field units for dimensionless a. Scale with a_star/B_star.
    a_star,B_star=0.02,3.
    np.testing.assert_allclose(norm(a_star*B/B_star,weights),a_star/B_star*norm(B,weights))


def test_own_context_roots_do_not_verify_one_frozen_branch():
    # F(z,p;g)=z-p-g, own contexts g(p)=2p: own roots z=3p.
    h=1e-4
    zp,zm=3*h,-3*h
    np.testing.assert_allclose(zp-h-2*h, 0., atol=1e-18)
    measured=(zp-zm)/(2*h)
    frozen_tangent=1.
    assert abs(measured-frozen_tangent) > 1.9
    np.testing.assert_allclose(measured-1.,2.)  # omitted F_g*g_p term


def test_inexact_newton_true_residual_controls_descent():
    A=np.array([[2.,0.3],[0.1,1.]])
    r=np.array([0.4,-0.8])
    d=np.linalg.solve(A,-r)
    e=A@d+r
    np.testing.assert_allclose(r@(A@d),-r@r+r@e,atol=1e-15)
    assert r@(A@d) < 0
