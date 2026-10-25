"""Lopez D in VMEC's m = 1 polar gauge: constraint satisfied, omega periodic, boundary rows consistent."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import lopez_polar as lp  # noqa: E402
import lopez_recovery as lr  # noqa: E402
import score_lopez as sl  # noqa: E402

TH = 2*np.pi*np.arange(48)[:, None]/48
PH = 2*np.pi*np.arange(96)[None, :]/96


@pytest.mark.parametrize("kind", ["polar:eta", "polar:shift:0.1:0.0"])
def test_polar_surfaces_satisfy_the_m1_constraint(kind):
    frame, sigma, c = lr.gauge_with()
    for s in (0.3, 1.0):
        R, Z = lr.surface_rz(kind, TH, PH, frame, sigma, c, s)
        a, b, cc, d = lp.m1_harmonics(R, Z)
        assert np.abs(b - cc).max() < 1e-13*max(1.0, np.abs(a).max())     # R_1^s = Z_1^c at every phi
        assert ((a + d) > 0).mean() > 0.3                                   # z+ real and positive: A' + D' = 2 |z+|
        zp = (a + d)/2
        assert zp.min() > 0
    # the unshifted eta gauge violates it (that is why the old exact seed was not a VMEC state)
    R, Z = lr.surface_rz("eta", TH, PH, frame, sigma, c, 1.0)
    a, b, cc, d = lp.m1_harmonics(R, Z)
    assert np.abs(b - cc).max() > 1e-3


def test_omega_is_periodic_without_winding_and_series_matches_grid():
    frame, sigma, c = lr.gauge_with()
    w = lp.omega_grid("eta", 0.5, frame, sigma, c)
    assert abs(w[-1] + (w[1] - w[0]) - w[0] - 0) < 0.2                        # closes up to one step
    ser = lp.omega_series("eta", 0.5, frame, sigma, c)
    ph = 2*np.pi*np.arange(len(w))/len(w)
    assert np.abs(lp.eval_omega(ser, ph) - w).max() < 1e-12
    h = 1e-6
    fd = (lp.eval_omega(ser, ph + h) - lp.eval_omega(ser, ph - h))/(2*h)
    assert np.abs(lp.eval_omega(ser, ph, derivative=True) - fd).max() < 1e-6


def _m1_block_mismatch(kind):
    """VMEC's m = 1 constraint in block form: rsc = zcc (cos(n phi) sin(theta) of R vs cos(theta) cos(n phi) of Z)
    and rss = zcs, from the signed (m, n) rows of the surface s = 1."""
    xm, xn, rc, rs, zc, zs = lr.surface_harmonics(2, 6, 24, kind, (48, 128))
    out = 0.0
    for n in range(0, 25):
        i, j = np.flatnonzero((xm == 1) & (xn == n)), np.flatnonzero((xm == 1) & (xn == -n))
        rs_n, zc_n = rs[-1][i[0]] + (rs[-1][j[0]] if n else 0), zc[-1][i[0]] + (zc[-1][j[0]] if n else 0)
        out = max(out, abs(rs_n - zc_n))
        if n:
            out = max(out, abs((rc[-1][i[0]] - rc[-1][j[0]]) - (zs[-1][j[0]] - zs[-1][i[0]])))
    return out


def test_boundary_rows_of_the_polar_gauge_obey_the_m1_constraint():
    assert _m1_block_mismatch("polar:eta") < 1e-12
    assert _m1_block_mismatch("eta") > 1e-4


def test_polar_oracle_state_has_small_field_error():
    w = lr.oracle_wout(9, 8, 48, "polar:eta")
    rec = sl.score_wout(w, sl.lopez_case(), keep_profile=False)
    assert rec["B_relative_surface"]["value"] < 5e-3
    assert rec["vacuum"]["energy_rel_error"] < 1e-3
