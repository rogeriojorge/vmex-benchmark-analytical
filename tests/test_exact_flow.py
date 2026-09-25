"""C5 exact-flow control (plan eq. 11): field-line flow of the integer family.

For dx/dt = B(x) the integer family (and its commuting diagonal stretch) has
x(t) = cos(Omega t) x0 + Omega^-1 sin(Omega t) B(x0), Omega = diag(1, 1, 2).
The ODE and variational equations are integrated independently.  Reference
only; field-line time is neither toroidal angle nor particle time.
"""
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

from analytic import cases, field, label_at_s, surface  # noqa: E402

OMEGA = np.array([1.0, 1.0, 2.0])
CASES = ("integer_axisymmetric", "integer_3d")


def _B(case, x):
    return field(case, jnp.asarray(x))[0]


def _flow(case, x0, t):
    return np.cos(OMEGA*t)*x0 + np.sin(OMEGA*t)/OMEGA*np.asarray(_B(case, x0))


def _tangent(case, x0, t):
    grad = np.asarray(jax.jacfwd(lambda x: _B(case, x))(jnp.asarray(x0)))
    return np.diag(np.cos(OMEGA*t)) + (np.sin(OMEGA*t)/OMEGA)[:, None]*grad


def _start(case, s=0.4, theta=0.7, phi=0.3):
    return np.asarray(surface(case, float(label_at_s(case, s)), theta, phi), dtype=float)


@pytest.mark.parametrize("name", CASES)
def test_flow_tangent_volume_and_surface_closure(name):
    case = cases()[name]
    x0, t = _start(case), 1.3
    jac = jax.jit(jax.jacfwd(lambda x: _B(case, x)))

    def rhs(_, y):
        x, M = y[:3], y[3:].reshape(3, 3)
        return np.concatenate([np.asarray(_B(case, x)), (np.asarray(jac(x)) @ M).ravel()])

    sol = solve_ivp(rhs, (0, t), np.concatenate([x0, np.eye(3).ravel()]),
                    method="DOP853", rtol=1e-13, atol=1e-14)
    x_ode, M_ode = sol.y[:3, -1], sol.y[3:, -1].reshape(3, 3)
    np.testing.assert_allclose(_flow(case, x0, t), x_ode, atol=1e-11)
    M = _tangent(case, x0, t)
    np.testing.assert_allclose(M, M_ode, atol=1e-9)
    assert np.linalg.det(M) == pytest.approx(1.0, abs=1e-12)  # div B = 0
    psi = [float(field(case, jnp.asarray(x))[1]) for x in (x0, _flow(case, x0, t))]
    assert psi[1] == pytest.approx(psi[0], abs=1e-13)
    # Composition: phi_{t1+t2} = phi_{t2} o phi_{t1}.
    np.testing.assert_allclose(_flow(case, _flow(case, x0, 0.4), 0.9), _flow(case, x0, 1.3), atol=1e-13)


def test_physical_scaling_of_field_line_time():
    case = cases()["integer_3d"]
    length, field_t = 2.0, 3.0
    x0 = length*_start(case)
    t = 0.8
    sol = solve_ivp(lambda _, x: field_t*np.asarray(_B(case, x/length)), (0, t), x0,
                    method="DOP853", rtol=1e-13, atol=1e-14)
    # x = L xbar(B* t / L).
    np.testing.assert_allclose(sol.y[:, -1], length*_flow(case, x0/length, field_t*t/length), atol=1e-10)
