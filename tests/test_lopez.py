"""Lopez D reference field: root selection, chart (sampled), vacuum identities, seed control."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import lopez


def domain_points(n, seed, scale=0.1):
    rng = np.random.default_rng(seed)
    rho = np.sqrt(rng.random(n))
    alpha, phase = rng.uniform(0, 2*np.pi, (2, n))
    lam = lopez.A0 + scale*rho*np.cos(alpha)/30
    beta = lopez.B0 + scale*rho*np.sin(alpha)/10
    return lam, beta, phase, lopez.forward(lam, beta, phase)[0]


def test_label_selection_recovers_forward_map_and_rivals_are_outside():
    for scale in (0.1, 1.0):  # D, then the principal domain it sits inside
        lam, beta, phase, x = domain_points(3000, 5, scale)
        root = lopez.evaluate(x)
        np.testing.assert_allclose(root.zeta, np.sqrt(lam)*np.exp(1j*phase), atol=1e-13, rtol=0)
        np.testing.assert_allclose(root.lam, lam, atol=1e-13, rtol=0)
        np.testing.assert_allclose(root.beta, beta, atol=1e-13, rtol=0)
        assert root.residual.max() < 1e-14
        assert root.q.max() <= scale**2*(1 + 1e-9) and root.q_rival.min() >= 2  # certificate bounds
        assert (np.abs(root.d)**2).min() >= 1/100
    xp = lopez.forward(lam, beta, phase)[1]  # last pass: the principal domain
    np.testing.assert_allclose(lopez.field(x), xp/np.sum(xp**2, -1)[:, None], rtol=1e-11, atol=1e-12)


def test_points_outside_d_are_flagged_not_clipped():
    lam, beta, _, x = domain_points(500, 6, scale=1.0)  # principal domain, mostly outside D
    root = lopez.evaluate(x)
    assert np.mean(lopez.D_SCALE*root.q > 1) > 0.9
    np.testing.assert_allclose(root.q, lopez.q_label(lam, beta), atol=1e-12)  # labels not relabelled
    # a point far from the image of the domain is reported with q >> 1, not mapped into D
    far = lopez.evaluate(np.array([[0.3, -0.2, 0.5]]))
    assert far.q[0] > 1


def test_vacuum_identities_and_exact_jacobian():
    res = lopez.verify(npoints=1000)
    assert res["div_over_grad"] < 1e-14 and res["curl_over_grad"] < 1e-14
    assert res["max_abs_div"] < 1e-12 and res["max_norm_curl"] < 1e-12
    for key in ("b_dot_grad_lambda", "b_dot_grad_beta", "b_dot_grad_q", "clebsch_error",
                "tangent_error", "label_error"):
        assert res[key] < 1e-13, key
    fd = res["jacobian_fd_error_by_step"]  # label-reselecting central differences, O(h^2)
    assert fd["1e-05"] < 1e-6
    assert 50 < fd["0.001"]/fd["0.0001"] < 200 and 50 < fd["0.0001"]/fd["1e-05"] < 200


def test_curl_of_finite_differences_is_far_from_roundoff():
    """The exact identity is not a property of the differencing: FD curl carries O(h^2) error."""
    _, _, _, x = domain_points(200, 7)
    _, db = lopez.field_jacobian(x)
    h = 1e-3
    jac = np.stack([(lopez.field(x + h*e) - lopez.field(x - h*e))/(2*h) for e in np.eye(3)], -1)
    assert np.abs(np.trace(jac, axis1=1, axis2=2)).max() > 1e-6*np.abs(db).max()
    assert np.abs(np.trace(db, axis1=1, axis2=2)).max() < 1e-13*np.abs(db).max()


def test_chart_sampled():
    assert abs(np.linalg.det(lopez.ROT) - 1) < 1e-15
    res = lopez.chart_check(nrho=4, nalpha=32, nphase=512)
    assert res["sampled_not_certified"]
    assert res["r_min"] > 3e-3 and res["dphi_dphase_min"] > 1e-2 and res["orientation_numerator_min"] > 1e-3
    assert res["wrapped_step_min"] > 0 and res["wrapped_step_max"] < np.pi  # no fold, no aliasing
    assert abs(res["winding_turns_min"] - 1) < 1e-12 and abs(res["winding_turns_max"] - 1) < 1e-12
    assert res["root_error_max"] < 1e-13 and res["phase_error_max"] < 1e-12
    assert res["q_d_max"] <= 1 + 1e-9 and res["q_d_rival_min"] > 200 and res["d_squared_min"] > 1/100
    main = lopez.chart_check(nrho=2, nalpha=128, nphase=2048, scale=1.0)  # the main torus folds in this frame
    assert main["dphi_dphase_min"] < 0 and main["orientation_numerator_min"] < 0
    assert main["winding_turns_min"] < 0.5  # some main-torus lines do not wind once: no single-valued chart


def test_smallest_modulus_root_is_wrong_for_most_points():
    res = lopez.modulus_failure(4000)
    assert res["label_wrong_fraction"] == 0 and res["wrong_fraction"] > 0.7
    assert res["median_field_error_when_wrong"] > 0.1
    bins = np.array(res["wrong_by_phase_bin"])
    assert (bins == 1).sum() >= 8 and (bins == 0).sum() >= 1  # right only on a window of source phases


def seed_error(uniform, scale=0.1, n=256):
    """Max relative field error of the zero-lambda seed C X_phase/J, C = flux/(2 pi), from labels
    built from (s, alpha) or from the flux-uniform angle eta; J by finite differences."""
    a, aa, bb = lopez.A0, scale/30, scale/10
    delta = aa*aa/(a + np.sqrt(a*a - aa*aa))
    flux = np.pi*bb/aa*delta

    def lab(s, angle):
        h = (2*a*delta*s - delta**2*s*s)/aa**2
        r, ca, sa = aa*np.sqrt(h), np.cos(angle), np.sin(angle)
        if uniform:
            ca, sa = (a*ca - r)/(a - r*ca), np.sqrt(a*a - r*r)*sa/(a - r*ca)
        return a + r*ca, lopez.B0 + bb*np.sqrt(h)*sa

    rng = np.random.default_rng(182)
    s = rng.uniform(.05, .95, n)
    angle, phase = rng.uniform(0, 2*np.pi, (2, n))
    x, xp = lopez.forward(*lab(s, angle), phase)
    exact = lopez.field(x)
    ds, da = 1e-5, 1e-4
    xs = (lopez.forward(*lab(s + ds, angle), phase)[0] - lopez.forward(*lab(s - ds, angle), phase)[0])/(2*ds)
    xa = (lopez.forward(*lab(s, angle + da), phase)[0] - lopez.forward(*lab(s, angle - da), phase)[0])/(2*da)
    seed = flux/(2*np.pi)*xp/np.einsum("ij,ij->i", xs, np.cross(xa, xp))[:, None]
    return (np.linalg.norm(seed - exact, axis=1)/np.linalg.norm(exact, axis=1)).max()


def test_direction_only_seed_fails_and_flux_uniform_angle_passes():
    assert seed_error(False) > 1e-2  # theta = alpha - phi, Lambda = 0: right direction, wrong amplitude
    assert seed_error(True) < 1e-7  # eta = theta + phi: finite-difference limited
