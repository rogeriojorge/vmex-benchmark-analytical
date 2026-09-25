"""C7 metric contract for the direct integer-family optimization (no solver)."""
from dataclasses import replace
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

from analytic import cases  # noqa: E402
from optimize_integer_family import make_metrics, quadrature  # noqa: E402

THETA = jnp.asarray([np.sqrt(1.5), np.sqrt(0.5), 1/64])


def _metrics(base=None):
    base = base or cases()["integer_3d"]
    return make_metrics(base, *quadrature(8, 16, 16, base.nfp))


def test_quadrature_matches_closed_forms():
    m = _metrics()(THETA)
    assert float(m["volume"]) == pytest.approx(float(m["volume_closed"]), rel=1e-12)
    assert float(m["beta_quadrature"]) == pytest.approx(float(m["beta_closed"]), rel=1e-9)
    assert float(m["beta_closed"]) == pytest.approx(0.0350877192982, rel=1e-11)


def test_objectives_are_invariant_under_family_self_similarity():
    from analytic import integer_field
    x = jnp.asarray([1.1, 0.2, 0.05])
    B1 = integer_field(x, 1.2, 0.7, 1.0)[0]
    B2 = integer_field(3*x, 3*1.2, 3*0.7, 3.0)[0]
    np.testing.assert_allclose(np.asarray(B2), 3*np.asarray(B1), rtol=1e-13)


def test_objective_gradients_match_central_differences():
    metrics = _metrics()
    grad = jax.jacfwd(metrics)(THETA)
    for key in ("C_J", "C_B", "margin"):
        fd = np.asarray([(float(metrics(THETA+1e-5*e)[key])-float(metrics(THETA-1e-5*e)[key]))/2e-5
                         for e in np.eye(3)])
        np.testing.assert_allclose(np.asarray(grad[key]), fd, rtol=1e-6, atol=1e-9)
