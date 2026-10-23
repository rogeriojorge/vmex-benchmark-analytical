"""Lopez configuration D: exact flux map, flux-uniform angle, chart, boundary series, vacuum decks."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import lopez_surfaces as L

OPT = L.OPT_FRAME


def _labels(n=6, seed=1):
    rng = np.random.default_rng(seed)
    r = L.A_RAD*np.sqrt(rng.random(n))
    lam, beta = L.labels(r, rng.uniform(0, 2*np.pi, n))[:2]
    return lam, beta, rng.uniform(0, 2*np.pi, n)


def test_forward_tangents_field_and_flux_identity():
    lam, beta, phase = _labels()
    X, Xl, Xb, Xp = L.forward(lam, beta, phase, tangents=True)
    e = 1e-6
    for T, d in ((Xl, (e, 0, 0)), (Xb, (0, e, 0)), (Xp, (0, 0, e))):
        fd = (L.forward(lam + d[0], beta + d[1], phase + d[2]) - L.forward(lam - d[0], beta - d[1], phase - d[2]))/(2*e)
        assert np.abs(fd - T).max() < 1e-9*np.abs(T).max()
    B = L.field_at_root(lam, beta, phase, X)                       # Cartesian formula at the known root
    assert np.abs(B - Xp/np.sum(Xp**2, -1)[:, None]).max() < 1e-13*np.abs(B).max()   # tangent identity
    assert np.allclose(np.einsum("ij,ij->i", B, np.cross(Xl, Xb))*2*lam, 1, rtol=0, atol=1e-13)  # B.(Xl x Xb)=1/(2 lam)


def test_flux_map_closed_form_and_quadrature_ladder():
    s = np.linspace(0, 1, 10001)
    assert np.abs(L.h_of_s(s) - s).max() == pytest.approx(3.0871821e-5, rel=1e-7)   # handoff value
    assert L.flux_edge() == pytest.approx(3.491089555922127e-4, rel=1e-14)
    assert np.abs(L.s_of_h(L.h_of_s(s)) - s).max() < 1e-15
    ladder = L.flux_ladder(1.0)
    errors = [r["rel_error"] for r in ladder]
    assert errors[0] > 1e-9 > errors[1] > errors[2] and errors[-1] < 1e-15       # converges to rounding
    for r in ladder[1:]:                                                          # estimator bounds the error
        assert r["rel_error"] < 3*r["successive_diff"] + 1e-15


def test_volume_and_energy_identity():
    v, b2 = L.volume(6, 12, 128)
    assert b2 == pytest.approx(2*np.pi*L.flux_edge(), rel=1e-13)                  # int B^2 dV = 2 pi Phi
    assert v == pytest.approx(L.volume(8, 16, 256)[0], rel=1e-13)
    assert v == pytest.approx(5.040841624776e-4, rel=1e-11)


def test_chart_conditions_and_paper_frame_numbers():
    paper = L.chart_check(L.PAPER_FRAME, nr=2, nalpha=128, nphase=2048)
    assert paper["rmin"] == pytest.approx(3.78e-3, rel=1e-2)                       # handoff: 3.7798e-3
    assert paper["dphi_min"] == pytest.approx(0.0153927, rel=2e-3)
    assert paper["dphi_max"] == pytest.approx(126.42, rel=5e-3) and paper["dphi_ratio"] > 8e3
    opt = L.chart_check(OPT, nr=3, nalpha=64, nphase=1024)
    assert opt["bad_increments"] == 0 and opt["rmin"] > .049 and opt["dphi_ratio"] < 60
    assert np.linalg.det(OPT[0]) == pytest.approx(1.0, abs=1e-14)
    assert np.allclose(OPT[0] @ OPT[0].T, np.eye(3), atol=1e-14)


def test_inversion_and_seam():
    for frame in (OPT, L.PAPER_FRAME):
        check = L.inversion_check(frame, n=500)
        assert check["max_phi_error"] < 1e-12 and check["seam_jump"] < 1e-7


def test_gauge_is_measured_and_matches_jacobian_sign():
    sigma, c = L.gauge(OPT)
    assert (sigma, c) == (-1, 1)          # eta clockwise in (R, Z); field lines rotate +1 about the axis per turn
    flux = L.section_flux(1.0, 4, 8, OPT)
    assert flux["orientation"] == sigma   # independent: sign of the analytic section Jacobian
    assert flux["min_bphi"] > 0           # B . e_phi > 0 so PHIEDGE > 0 with a right-handed (R, phi, Z)


def test_flux_uniform_angle_and_geometric_section_flux():
    ladder = L.section_flux_ladder(OPT, sizes=((4, 8), (8, 32)), s=1.0)
    assert all(r["rel_error"] < 1e-13 and r["density_max_dev"] < 1e-12 for r in ladder)
    inner = L.section_flux_ladder(OPT, sizes=((6, 16),), s=0.3)[0]                 # exercises h(s) at s != 1
    assert inner["rel_error"] < 1e-13
    control = L.section_flux(1.0, 8, 32, OPT, kind="alpha")                       # direction-only angle fails
    assert control["density_max_dev"] > 1e-2 and control["flux"] == pytest.approx(L.flux_edge(), rel=1e-13)


def test_truncation_decreases_and_frame_matters():
    sigma, c = L.gauge(OPT)
    ladder = L.truncation_ladder(OPT, sigma, c, [(8, 24), (8, 40), (8, 56)], grid=(48, 192), npoints=400)
    errors = [r["max_error"] for r in ladder]
    assert errors[0] > 5*errors[1] > 25*errors[2] and errors[2] < 2e-5 and ladder[0]["tail"] < 1e-8
    paper = L.truncation_ladder(L.PAPER_FRAME, -1, c, [(8, 56)], grid=(48, 192), npoints=400)[0]
    assert paper["max_error"] > 1e3*errors[2]                                     # paper frame is unusable


def test_decks_manifest_and_axis(tmp_path):
    out = tmp_path/"inputs"
    out.mkdir()
    shutil.copy(ROOT/"inputs/manifest.json", out/"manifest.json")
    error = L.write_decks(mpol=4, ntor=8, out=out)
    assert 1e-4 < error < 5e-2                                                    # coarse deck, measured not assumed
    cur, iota = ((out/f"input.lopez_D_{k}").read_text() for k in ("current", "iota"))
    assert "NCURR = 1" in cur and "CURTOR = 0.0" in cur and "AC = 0.0" in cur and "AI" not in cur
    assert "NCURR = 0" in iota and "AI = 1.00000000000000000e+00" in iota
    for text in (cur, iota):
        assert "LASYM = T" in text and "NFP = 1" in text and "AM = 0.0" in text and "MPOL = 4" in text
        assert f"PHIEDGE = {float(L.flux_edge()):.17e}" in text and "LFREEB = F" in text
        assert "RBS(" in text and "ZBC(" in text                                  # asymmetric: all four arrays
    records = [r for r in json.loads((out/"manifest.json").read_text())["records"] if r["case"] == "lopez_D"]
    assert [r["ncurr"] for r in records] == [1, 0] and records[0]["status"] == "candidate_input_not_equilibrium"
    assert records[0]["lopez"]["iota"] == 1 and records[0]["lopez"]["volume"] == pytest.approx(5.040841624776e-4)
    original = json.loads((ROOT/"inputs/manifest.json").read_text())["records"]
    assert len(json.loads((out/"manifest.json").read_text())["records"]) == len(original)  # others untouched
    assert len(records[0]["lopez"]["axis"]["R_cos"]) == 9                       # axis series stored to ntor


def test_axis_harmonics_reproduce_the_axis():
    ax = L.axis_harmonics(OPT, -1, 1, 96)
    phi = np.array([0.1, 1.3, 2.9, 4.4, 6.0])
    k = np.arange(97)[:, None]
    R = np.array(ax["R_cos"])@np.cos(k*phi) + np.array(ax["R_sin"])@np.sin(k*phi)
    Z = np.array(ax["Z_cos"])@np.cos(k*phi) + np.array(ax["Z_sin"])@np.sin(k*phi)
    Rx, Zx = L.boundary_rz(0., phi, OPT, -1, 1, s=0.)
    assert np.abs(R - Rx).max() < 1e-6 and np.abs(Z - Zx).max() < 1e-6


def test_committed_deck_matches_generator():
    """The committed deck is the write_decks(10, 72) output: header, flux, and the low modes."""
    text = (ROOT/"inputs/input.lopez_D_current").read_text()
    assert "MPOL = 10" in text and "NTOR = 72" in text and f"PHIEDGE = {float(L.flux_edge()):.17e}" in text
    rows = L.boundary_rows(OPT, -1, 1, 6, 24, grid=(48, 128))[0]
    committed = {line.split("=")[0].strip(): float(line.split("=")[1]) for line in text.splitlines() if "(" in line}
    for m, n in ((0, 0), (1, 0), (1, 1), (2, -1), (0, 2)):
        row = rows[(rows[:, 0] == m) & (rows[:, 1] == n)][0]
        for key, value in zip(("RBC", "RBS", "ZBC", "ZBS"), row[2:]):
            assert abs(committed.get(f"{key}({n},{m})", 0.0) - value) < 1e-6
