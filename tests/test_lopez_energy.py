"""Energy bookkeeping for the Lopez D recovery study (signed, flux-corrected) and the gauge override of the scorer."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import lopez_energy as le  # noqa: E402
import score_lopez as sl  # noqa: E402


def _score(w_ratio, flux_ratio, e_b):
    target = 2*np.pi*3.49e-4
    return dict(run_id="x", ns=17, B_relative_volume_l2=e_b, fsq=[1e-6]*3,
                vacuum=dict(energy_target_circulation_phi=target, energy_b2_integral=target*w_ratio,
                            toroidal_flux=3.49e-4*flux_ratio, toroidal_flux_exact=3.49e-4, volume=1.0, volume_exact=1.0))


def test_signed_energy_and_flux_correction_follow_the_dirichlet_identity():
    # B = lam B_vac + b with b of zero flux: W/W_vac = lam^2 + |b|^2/W_vac, e_B^2 = (lam - 1)^2 + |b|^2/W_vac
    lam, b2 = 1.0007, 3e-5
    row = le.energy_row(_score(lam**2 + b2, lam, np.sqrt((lam - 1)**2 + b2)))
    assert row["signed_dW"] == pytest.approx(lam**2 + b2 - 1)
    assert row["flux_rel_error"] == pytest.approx(lam - 1)
    assert row["signed_dW_flux_corrected"] == pytest.approx(b2/lam**2, rel=1e-9)
    assert row["ratio_corrected_to_eB2"] == pytest.approx(1.0, rel=2e-2)
    below = le.energy_row(_score(0.999, 1.0, 1e-3))     # an energy below the exact vacuum would be flagged by its sign
    assert below["signed_dW"] < 0


def test_iota_override_changes_only_the_scored_transform():
    base, shifted = sl.lopez_case(), sl.lopez_case(iota=0)
    assert base.iota == 1.0 and shifted.iota == 0.0
    assert base.flux_edge == shifted.flux_edge and base.volume == shifted.volume
