"""MRX glue for Lopez D: exact-chart map, chart handedness, vacuum solve, closure statistics, WOUT line tools.

The MRX solves need the MRX clone on PYTHONPATH (``importorskip``); the line tools are NumPy only.
"""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

os.environ.setdefault("MRX_DTYPE", "float64")   # before the first import of mrx (importorskip below imports it)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import lopez_surfaces as ls  # noqa: E402
import lopez_wout_lines as wl  # noqa: E402


def test_spectral_eval_reproduces_a_trigonometric_polynomial():
    nth, nz = 16, 40
    th = np.arange(nth)[:, None]/nth
    ze = np.arange(nz)[None, :]/nz
    f = lambda t, z: 1.3 + np.cos(2*np.pi*(2*t - 3*z)) - 0.4*np.sin(2*np.pi*(t + 5*z))  # noqa: E731
    spec = wl._spectrum(f(th, ze) + 0*ze)
    rng = np.random.default_rng(1)
    tt, zz = rng.uniform(0, 1, (2, 50))
    assert np.abs(wl._eval(spec, tt, zz) - f(tt, zz)).max() < 1e-12


@pytest.fixture(scope="module")
def mrx_modules():
    pytest.importorskip("equinox")
    pytest.importorskip("diffrax")
    pytest.importorskip("mrx")
    import lopez_mrx as lm
    return lm


def test_label_map_is_the_exact_forward_map_and_right_handed(mrx_modules):
    import jax
    import jax.numpy as jnp
    lm = mrx_modules
    rng = np.random.default_rng(2)
    x = rng.uniform(0.1, 0.9, (30, 3))
    for twist in (0, -1):
        phi = lm.LabelMap(twist=twist)
        got = np.asarray(jax.vmap(phi)(jnp.asarray(x)))
        t = x[:, 1] + twist*x[:, 2]
        want = ls.forward(ls.A0 + ls.A_RAD*x[:, 0]*np.cos(2*np.pi*t), ls.B0 + ls.B_RAD*x[:, 0]*np.sin(2*np.pi*t),
                          2*np.pi*x[:, 2])
        assert np.abs(got - want).max() < 1e-13
        det = np.asarray(jax.vmap(lambda y, phi=phi: jnp.linalg.det(jax.jacfwd(phi)(y)))(jnp.asarray(x)))
        assert (det > 0).all()


def test_boundary_deformation_n0_is_a_union_of_field_lines(mrx_modules):
    """n = 0 deformation: boundary labels depend on the angle only, so the exact field stays the vacuum field."""
    import jax.numpy as jnp
    import lopez
    lm = mrx_modules
    phi = lm.LabelMap(pert=((0.2, 2, 0, 0.1),), twist=0)
    x = jnp.array([1.0, 0.3, 0.1])
    x2 = jnp.array([1.0, 0.3, 0.8])
    lam1, beta1 = lopez.labels(np.asarray(phi(x)))
    lam2, beta2 = lopez.labels(np.asarray(phi(x2)))
    assert abs(float(lam1 - lam2)) < 1e-9 and abs(float(beta1 - beta2)) < 1e-9


def test_cyl_fourier_boundary_is_the_truncated_exact_surface(mrx_modules, tmp_path):
    import jax
    import jax.numpy as jnp
    lm = mrx_modules
    phi_map, err, fit = lm.cyl_fourier_map(6, 24, cache=tmp_path/"c.npz", n_rad=17, degree=9, grid=(48, 192))
    assert 1e-5 < err < 5e-3 and fit < 1e-9
    sigma, c = ls.gauge(ls.OPT_FRAME)
    th, ph = np.array([0.31, 1.7, 4.4]), np.array([0.2, 2.9, 5.5])
    r, z = ls.boundary_rz(th, ph, ls.OPT_FRAME, sigma, c)
    # the map's logical theta is +-theta/2 pi depending on the detected handedness
    pts = jnp.asarray(np.stack([np.ones(3), phi_map.flip*th/(2*np.pi), ph/(2*np.pi)], -1))
    got = np.asarray(jax.vmap(phi_map)(pts)) @ ls.OPT_FRAME[0].T - ls.OPT_FRAME[1] @ ls.OPT_FRAME[0].T
    assert np.abs(np.hypot(got[:, 0], got[:, 1]) - r).max() < 2*err
    assert np.abs(got[:, 2] - z).max() < 2*err
    det = np.asarray(jax.vmap(lambda y: jnp.linalg.det(jax.jacfwd(phi_map)(y)))(
        jnp.asarray(np.random.default_rng(0).uniform(0.05, 1, (200, 3)))))
    assert (det > 0).all()


def test_vacuum_solve_recovers_exact_field_and_closes_lines(mrx_modules):
    lm = mrx_modules
    seq = lm.build_sequence((6, 8, 12), 3, lm.LabelMap(twist=0))
    dof, spread = lm.vacuum_field(seq)
    assert spread < 1e-6        # flux through different cuts: divergence-free, quadrature-limited
    s = lm.score(seq, dof, probes=4)
    assert s["B_rel_volume_l2"] < 1e-4 and s["div_over_B_over_a"] < 1e-10
    assert s["energy_rel_error"] < 1e-7 and s["enclosed_current_over_aB_max"] < 1e-8
    assert abs(s["flux_scale_fit"] - 1) < 1e-4
    c = lm.closure_summary(seq, dof, 3, 4, 2, 24, expected_rotation=0.0)
    assert c["return_defect_max"] < 1e-10      # B^rho = B^theta = 0 to round-off at (6, 8, 12), p = 3 (measured)
    seq1 = lm.build_sequence((6, 8, 12), 3, lm.LabelMap(twist=-1))
    dof1, _ = lm.vacuum_field(seq1)
    c1 = lm.closure_summary(seq1, dof1, 3, 4, 3, 24, expected_rotation=1.0)
    assert c1["rotation_dev_max"] < 1e-3        # twisted chart: unit rotation of the exact field (measured 2e-5)
    assert c1["return_defect_max"] < 1e-3


def test_exact_field_lines_close_and_conserve_labels(mrx_modules):
    out = mrx_modules.exact_line_closure(n_lines=3)
    assert out["max_return_distance"] < 1e-8*out["minor_radius_scale"]*100 and out["max_label_drift"] < 1e-9
