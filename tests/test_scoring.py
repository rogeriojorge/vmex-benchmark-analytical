"""Scorer geometry checks on synthetic WOUT-like objects (no solver needed)."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

from score_wout_exact import minor_radius  # noqa: E402


def _torus(R0, a, nfp=1, shift=0.0):
    """Two-surface circular torus: axis row (R0 + shift cos(nfp phi)) and an edge of radius a."""
    xm = np.array([0, 1, 0])
    xn = np.array([0, 0, nfp])
    rmnc = np.array([[R0, 0.0, shift], [R0, a, shift]])
    zmns = np.array([[0.0, 0.0, 0.0], [0.0, a, 0.0]])
    return SimpleNamespace(xm=xm, xn=xn, rmnc=rmnc, zmns=zmns, lasym=False, nfp=nfp, ns=2)


@pytest.mark.parametrize("ntheta, nzeta", [(32, 16), (17, 9), (64, 5)])
def test_minor_radius_of_circular_torus(ntheta, nzeta):
    """The old scorer paired each boundary point with the wrong axis point: 1.418 for a = 0.1."""
    theta = 2*np.pi*np.arange(ntheta)/ntheta
    zeta = 2*np.pi*np.arange(nzeta)/nzeta
    assert minor_radius(_torus(1.0, 0.1), theta, zeta) == pytest.approx(0.1, rel=1e-12)


def test_minor_radius_with_toroidally_varying_axis():
    theta = 2*np.pi*np.arange(24)/24
    zeta = 2*np.pi*np.arange(11)/(11*3)
    assert minor_radius(_torus(1.0, 0.05, nfp=3, shift=0.02), theta, zeta) == pytest.approx(0.05, rel=1e-12)
