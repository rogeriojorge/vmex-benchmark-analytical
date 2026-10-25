"""Fast checks of the Lopez D scorer on synthetic states (no solver, no JAX)."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import lopez  # noqa: E402
import lopez_recovery as lr  # noqa: E402
import lopez_surfaces as ls  # noqa: E402
import score_lopez as sl  # noqa: E402


# ---------------------------------------------------------------- harmonic transforms
@pytest.mark.parametrize("nfp", [1, 3])
def test_synth_analyze_roundtrip_and_derivatives(nfp):
    rng = np.random.default_rng(3)
    xm, xn = sl.mode_table(4, 3)
    xn = xn*nfp
    c, s = rng.normal(size=(2, len(xm)))
    s[(xm == 0) & (xn == 0)] = 0.
    nth, nz = 24, 32
    f = sl.synth(c, s, xm, xn, nfp, nth, nz)
    c2, s2 = sl.analyze(f, xm, xn, nfp)
    assert np.allclose(c, c2, atol=1e-13) and np.allclose(s, s2, atol=1e-13)
    th = 2*np.pi*np.arange(nth)[:, None]/nth
    ze = 2*np.pi*np.arange(nz)[None, :]/(nfp*nz)
    ang = th[..., None]*xm - ze[..., None]*xn
    assert np.allclose(f, (np.cos(ang)*c + np.sin(ang)*s).sum(-1), atol=1e-12)
    d_th = (-np.sin(ang)*c*xm + np.cos(ang)*s*xm).sum(-1)
    d_ze = (np.sin(ang)*c*xn - np.cos(ang)*s*xn).sum(-1)
    assert np.allclose(sl.synth(c, s, xm, xn, nfp, nth, nz, dtheta=True), d_th, atol=1e-11)
    assert np.allclose(sl.synth(c, s, xm, xn, nfp, nth, nz, dzeta=True), d_ze, atol=1e-11)


def test_grid_too_small_is_an_error():
    xm, xn = sl.mode_table(8, 8)
    with pytest.raises(ValueError):
        sl.synth(np.ones(len(xm)), None, xm, xn, 1, 8, 8)


def test_frame_map_inverts_the_chart_frame():
    frame, _, _ = lr.gauge_record()
    x = np.random.default_rng(0).normal(size=(50, 3))
    xc = np.stack(ls.to_frame(x, frame), -1)
    case = sl.lopez_case()
    back, _ = sl.to_exact_frame(case, xc)
    assert np.allclose(back, x, atol=1e-13)


# ---------------------------------------------------------------- toy vacuum torus: B = R0 B0 / R toroidal
R0, A_MINOR = 1.0, 0.3


def torus_case():
    flux = R0*2*np.pi*(R0 - np.sqrt(R0**2 - A_MINOR**2))

    def exact(xyz):
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        r2 = x*x + y*y
        b = R0*np.stack((-y/r2, x/r2, 0*x), -1)
        s = ((np.sqrt(r2) - R0)**2 + z*z)/A_MINOR**2
        return b, s, 0*s, 10 + 0*s

    def axis(phi):
        return R0 + 0*phi, 0*phi
    return sl.Case("torus", (np.eye(3), np.zeros(3)), exact, 0.0, flux, 2*np.pi**2*R0*A_MINOR**2, axis,
                   2*np.pi*R0)


def torus_wout(ns, mpol=3, ntor=2):
    xm, xn = sl.mode_table(mpol, ntor)
    rc, zs = np.zeros((ns, len(xm))), np.zeros((ns, len(xm)))
    s = np.linspace(0, 1, ns)
    rho = A_MINOR*np.sqrt(s)
    rc[:, (xm == 0) & (xn == 0)] = R0
    rc[:, (xm == 1) & (xn == 0)] = rho[:, None]
    zs[:, (xm == 1) & (xn == 0)] = rho[:, None]
    xmn, xnn = sl.nyquist_table(2*mpol, 2*ntor)
    zero = np.zeros((ns, len(xmn)))
    w = sl.as_wout(SimpleNamespace(
        nfp=1, ns=ns, mpol=mpol, ntor=ntor, lasym=False, xm=xm, xn=xn, xm_nyq=xmn, xn_nyq=xnn, rmnc=rc, zmns=zs,
        bsupumnc=zero, bsupvmnc=zero, iotaf=np.zeros(ns), phi=np.linspace(0, 1, ns)*torus_case().flux_edge,
        fsqr=0., fsqz=0., fsql=0., ier_flag=0, signgs=-1))
    grid = sl.grid_for(w)
    half = sl.evaluate_surfaces(w, "half", grid)
    bv = R0/half.R**2   # B^zeta = B_phi/R, B_phi = R0/R; theta component zero
    cos, _ = sl.analyze(bv, xmn, xnn, 1)
    w.bsupvmnc = np.vstack([np.zeros((1, len(xmn))), cos])
    return w


def test_torus_vacuum_identities_converge_second_order():
    case = torus_case()
    rows = {ns: sl.score_wout(torus_wout(ns), case, keep_profile=False) for ns in (9, 17, 33)}
    v = {ns: r["vacuum"] for ns, r in rows.items()}
    assert v[33]["curl_over_B_over_a_rms_outer_s_ge_0.25"] < 1e-3   # floor: Nyquist truncation of 1/R^2
    for key in ("toroidal_flux_rel_error", "volume_rel_error", "circulation_rel_error_max", "energy_rel_error"):
        assert v[33][key] < 1e-3, key   # floors are the Nyquist/geometry truncation of this tiny basis
    assert (rows[33]["B_relative_surface"]["value"] < rows[9]["B_relative_surface"]["value"] / 3)
    assert v[33]["volume_rel_error"] < v[9]["volume_rel_error"] / 3
    assert rows[33]["B_relative_surface"]["value"] < 2e-3
    assert rows[33]["vacuum"]["div_over_B_over_a_rms"] < 1e-9
    assert rows[33]["vacuum"]["enclosed_current_over_aB_max"] < 1e-9
    assert rows[33]["axis_offset"] < 1e-12
    assert rows[33]["iota_max_abs_error"] == 0.0
    assert rows[33]["flux_label_error_max"] < 1e-6


def test_torus_scaling_and_transform_errors_are_seen():
    case, w = torus_case(), torus_wout(17)
    base = sl.score_wout(w, case, keep_profile=False)
    scaled = deepcopy(w)
    scaled.bsupvmnc = w.bsupvmnc*1.01
    r = sl.score_wout(scaled, case, keep_profile=False)
    assert r["B_relative_surface"]["value"] == pytest.approx(0.01, rel=0.05)
    assert r["vacuum"]["toroidal_flux_rel_error"] == pytest.approx(0.01, rel=0.01)
    assert r["vacuum"]["energy_rel_error"] == pytest.approx(0.0201, rel=0.05)
    assert r["vacuum"]["circulation_rel_error_max"] > 0.009   # B_zeta scales with the field
    shifted = deepcopy(w)
    shifted.iotaf = w.iotaf + 0.25
    assert sl.score_wout(shifted, case, keep_profile=False)["iota_max_abs_error"] == pytest.approx(0.25)
    # a toroidal current in the field is a curl: add a poloidal field B^theta = 0.05 B^zeta
    twisted = deepcopy(w)
    twisted.bsupumnc = 0.05*w.bsupvmnc
    t = sl.score_wout(twisted, case, keep_profile=False)
    assert t["vacuum"]["enclosed_current_over_aB_max"] > 100*base["vacuum"]["enclosed_current_over_aB_max"] + 1e-3
    assert t["vacuum"]["curl_over_B_over_a_rms_at_report"] > 3*base["vacuum"]["curl_over_B_over_a_rms_at_report"]


# ---------------------------------------------------------------- Lopez D: exact state projected and the negative controls
def test_lopez_projection_resolves_and_direction_only_seed_fails():
    case = sl.lopez_case()
    eta = {ns: sl.score_wout(lr.oracle_wout(ns, 8, 40), case, keep_profile=False) for ns in (9, 17)}
    alpha = sl.score_wout(lr.oracle_wout(17, 8, 40, kind="alpha"), case, keep_profile=False)
    # radial refinement converges at fixed angular resolution; the boundary truncation is the floor
    assert eta[17]["B_relative_surface"]["value"] < eta[9]["B_relative_surface"]["value"]
    assert eta[17]["B_relative_surface"]["value"] < 1e-2
    assert eta[17]["vacuum"]["toroidal_flux_rel_error"] < 1e-12  # B^zeta sqrt(g) uniform by construction
    # the ordinary ellipse angle with Lambda = 0 has the right direction but not the right amplitude
    assert alpha["B_relative_surface"]["value"] > 1e-2
    assert alpha["B_relative_surface"]["value"] > 3*eta[17]["B_relative_surface"]["value"]


def test_modulus_root_selection_is_flagged_as_wrong():
    case = sl.lopez_case()
    w = lr.oracle_wout(9, 8, 40)
    good = sl.score_wout(w, case, keep_profile=False)
    bad = sl.score_wout(w, sl.lopez_case(select="modulus"), keep_profile=False)
    assert bad["B_relative_surface"]["value"] > 10*good["B_relative_surface"]["value"]


def test_exact_reference_labels_agree_with_the_flux_map():
    rng = np.random.default_rng(1)
    r = ls.A_RAD*np.sqrt(rng.random(200))
    lam, beta = ls.labels(r, rng.uniform(0, 2*np.pi, 200))[:2]
    phase = rng.uniform(0, 2*np.pi, 200)
    x = ls.forward(lam, beta, phase)
    b, s, q, margin = sl.lopez_case().exact(x)
    assert np.allclose(s, ls.s_of_h((r/ls.A_RAD)**2), atol=1e-9)
    assert np.allclose(b, lopez.field(x), atol=1e-12)
    assert q.max() <= 1 + 1e-9 and margin.min() >= 2


def test_fsq_history_uses_the_last_radial_stage(tmp_path):
    import lopez_tables as lt
    log = tmp_path/"log"
    log.write_text("  NS =   17 NO. FOURIER MODES = 3\n    1  1.0E-02  2.0E-02  3.0E-02  x\n"
                   "  NS =   33 NO. FOURIER MODES = 3\n    1  5.0E-01  2.0E-01  1.0E-01  x\n"
                   "  200  4.0E-06  2.0E-06  1.0E-06  x\n")
    assert lt.fsq_history(log) == [(1, 0.5, 0.2, 0.1), (200, 4e-6, 2e-6, 1e-6)]
