"""Contract tests for the single-pass refinement observer and response units.

The fake ``implicit`` module reproduces only the seams and memo contract of the
pinned VMEX refinement; it is a contract test, not a physical root.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import NamedTuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import refinement_observer as observer  # noqa: E402
from analytic import cases  # noqa: E402
from measurement import exact_field_parameter_tangents, scaled_response_error  # noqa: E402


class State(NamedTuple):
    R_cos: jax.Array
    R_sin: jax.Array
    Z_cos: jax.Array
    Z_sin: jax.Array
    L_cos: jax.Array
    L_sin: jax.Array


def _state(value):
    return State(*(jnp.full((3,), float(value)) for _ in range(6)))


def _fake_implicit(step_quality):
    """Residual F(z) = (z - 1) + 0.1 (z - 1)^2; frozen state is ignored.

    ``step_quality`` = "newton" takes exact Newton steps, "bad" steps away.
    """
    module = SimpleNamespace(_LAST_REFINED={}, _SOLVE_STATS={})

    def residual_fn(cfg, frozen, mask, formulation="preconditioned"):
        return lambda z, params: jax.tree.map(lambda v: (v-1.0)+0.1*(v-1.0)**2, z)

    module.residual_fn = residual_fn
    module._dof_projector = lambda cfg, mask: (lambda tree: tree)

    def factors(cfg, params, frozen, mask, z):
        return None

    def block_step(cfg, params, frozen, mask, z, factors):
        if step_quality == "newton":
            z_new = jax.tree.map(lambda v: v-((v-1.0)+0.1*(v-1.0)**2)/(1.0+0.2*(v-1.0)), z)
        else:
            z_new = jax.tree.map(lambda v: v+0.5, z)
        fz = residual_fn(cfg, frozen, mask)(z_new, params)
        return z_new, fz, observer.tree_norm(fz), 1e-12

    module._refine_block_factors = factors
    module._refine_block_step = block_step
    module._refine_step = None

    def refined_state(cfg, params, state, mask, *, initial_correction=None):
        F = residual_fn(cfg, state, mask)
        base = observer.tree_norm(F(state, params))
        if base <= cfg.refine_tol:
            return state
        factors_ = module._refine_block_factors(cfg, params, state, mask, state)
        z, best_z, best = state, state, base
        for _ in range(4):
            z, _, norm, _ = module._refine_block_step(cfg, params, state, mask, z, factors_)
            if float(norm) < best:
                best_z, best = z, float(norm)
        return state if best >= base else best_z

    module._refined_state = refined_state

    def refine_fixed_point(cfg, params, state, mask):
        hit = module._LAST_REFINED.get(cfg)
        if hit is None:
            stats = module._SOLVE_STATS.setdefault(cfg, {"refinements": 0})
            stats["refinements"] += 1
            hit = module._LAST_REFINED[cfg] = module._refined_state(cfg, params, state, mask)
        return hit

    module._refine_fixed_point = refine_fixed_point
    return module


class Config:
    """Identity-hashed like the real eq=False ImplicitConfig."""
    refine_tol = 1e-11


CFG = Config()


def test_observer_sees_a_genuinely_changed_certified_state():
    implicit = _fake_implicit("newton")
    start = _state(1.2)
    refined, record, _ = observer.observe_refinement(implicit, CFG, None, start, _state(1))
    assert record["state_changed"] and record["status"]["refinement_attempted"]
    assert record["status"]["root_certified"] and not record["status"]["response_certified"]
    assert record["memo_untouched"] and CFG not in implicit._LAST_REFINED
    steps = [row for row in record["trace"] if row["kind"] == "block_newton_step"]
    assert steps[0]["defect"]["true_linear_defect_over_residual"] < 1e-12
    assert steps[0]["defect"]["merit_directional_derivative_over_residual_sq"] == pytest.approx(-1)
    assert record["returned_trial_indices"]
    assert observer.derivative_gate(record)["derivative_may_be_accepted"]


def test_observer_distinguishes_unchanged_failed_refinement():
    implicit = _fake_implicit("bad")
    start = _state(1.2)
    refined, record, _ = observer.observe_refinement(implicit, CFG, None, start, _state(1))
    assert not record["state_changed"] and record["correction_norm"] == 0.0
    assert record["status"]["refinement_attempted"]
    assert not record["status"]["root_certified"]
    assert not any(row.get("accepted_as_returned_state") for row in record["trace"])
    before = record["residual_before_refinement_operator"]["preconditioned"]["norm"]
    after = record["residual_after_refinement_operator"]["preconditioned"]["norm"]
    assert before == after > 1e-3
    gate = observer.derivative_gate(record)
    assert not gate["derivative_may_be_accepted"] and gate["reason"]


def test_second_public_memo_call_is_a_hit_not_a_refinement():
    implicit = _fake_implicit("newton")
    cfg = Config()
    probe = observer.memo_probe(implicit, cfg, None, _state(1.2), _state(1))
    assert probe["first_call_refined"] and probe["second_call_memo_hit"] and probe["same_object"]
    # A different starting state at the same cfg/params still returns the memo:
    # the memo is not a general-purpose refiner of an arbitrary state.
    other = implicit._refine_fixed_point(cfg, None, _state(5.0), _state(1))
    assert observer.tree_hash(other) == probe["output_sha256"]


def test_tree_hash_is_sensitive_to_values_and_order():
    assert observer.tree_hash(_state(1.0)) != observer.tree_hash(_state(1.0 + 1e-15))


def test_scaled_response_is_invariant_under_duplicated_points():
    rng = np.random.default_rng(3)
    actual, expected = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    base = scaled_response_error(actual, expected, np.ones(4), parameter_scale=0.5, field_scale=2.0)
    doubled = scaled_response_error(np.vstack([actual, actual]), np.vstack([expected, expected]),
                                    np.full(8, 0.5), parameter_scale=0.5, field_scale=2.0)
    assert doubled["scaled_rms_error"] == pytest.approx(base["scaled_rms_error"], rel=1e-14)
    raw = np.linalg.norm(np.vstack([actual, actual])-np.vstack([expected, expected]))
    assert raw == pytest.approx(np.sqrt(2)*np.linalg.norm(actual-expected))
    # Dimensionless parameter response in tesla scales with a_star/B_star.
    assert base["scaled_rms_error"] == pytest.approx(
        0.25*np.sqrt(np.sum((actual-expected)**2)/4))


def test_exact_tangents_use_length_and_field_scales():
    case = cases()["integer_axisymmetric"]
    point = np.asarray([[1.05, 0.02, 0.03]])
    unit = exact_field_parameter_tangents(case, point, (2, 3), length_m=1.0, field_t=1.0)
    scaled = exact_field_parameter_tangents(case, 2.0*point, (2, 3), length_m=2.0, field_t=3.0)
    assert np.all(np.isfinite(unit)) and np.linalg.norm(unit) > 0
    np.testing.assert_allclose(scaled, 3.0*unit, rtol=1e-12, atol=1e-14)


def test_weak_stress_moment_matches_strong_force_and_detects_pressure_defect():
    from numpy.polynomial.legendre import leggauss
    from analytic import label_at_s, surface
    from measurement import cube_bump_test_field, integer_exact_fields, weak_stress_moment
    case = cases()["integer_3d"]
    center = np.asarray(surface(case, float(label_at_s(case, 0.3)), 0.4, 0.2), dtype=float)
    h = 0.02
    q, w = leggauss(12)
    grid = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    weights = (w[:, None, None]*w[None, :, None]*w[None, None, :]).ravel()*h**3
    x = center + h*grid
    B, curl, gradp, _, label = integer_exact_fields(case, x)
    assert np.all(label < 0.9*case.edge)  # test cube lies inside the plasma
    # Dimensionless reference: curl B = J (mu0 = 1); p from the same label.
    p = case.pressure_slope*(label - case.edge)
    J = curl
    force = np.cross(J, B) - gradp
    for component in range(3):
        v, grad_v = cube_bump_test_field(x, center, h, component)
        weak = weak_stress_moment(B, p, grad_v, weights, mu0=1.0)
        strong = float(np.sum(weights*np.sum(v*force, axis=-1)))
        scale = float(np.sum(weights*np.abs(v[:, component])*np.linalg.norm(gradp, axis=-1)))
        assert abs(weak["moment"]) < 1e-8*scale and abs(strong) < 1e-8*scale
        # A pressure defect epsilon*x_k is detected with the expected moment.
        epsilon = 1e-3*float(np.max(np.linalg.norm(gradp, axis=-1)))
        defect = weak_stress_moment(B, p + epsilon*x[:, component], grad_v, weights, mu0=1.0)
        expected = -epsilon*float(np.sum(weights*v[:, component]))
        assert defect["moment"] == pytest.approx(expected, rel=1e-8)
