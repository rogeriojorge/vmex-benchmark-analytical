"""Solved-state scoring with the projection certificate's norm set, without a VMEC solve.

A stand-in field built from the exact equilibrium (and perturbed copies of it) stands in for a solver
output, so the metrics, masks and quadrature rules are checked against known answers.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import certify_projection as cp  # noqa: E402
import score_solved as ss  # noqa: E402
from analytic import cases, field, surface, volume  # noqa: E402

FAMILIES = ["solovev_symmetric", "solovev_asymmetric", "integer_3d", "integer_3d_tau", "integer_3d_tau_mirror"]


class ExactField:
    """Quacks like VmecInteriorField: B(p) and gradB(p)[i, j] = dB_i/dx_j, optionally scaled / cut."""

    def __init__(self, case, scale=1.0, nan_beyond=None):
        self.case, self.scale, self.nan_beyond = case, scale, nan_beyond

    def _B(self, x):
        return field(self.case, x)[0]

    def B(self, p):
        b = self.scale*jax.vmap(self._B)(p)
        return self._cut(b, p)

    def gradB(self, p):
        g = self.scale*jax.vmap(jax.jacfwd(self._B))(p)
        return self._cut(g, p)

    def _cut(self, value, p):
        if self.nan_beyond is None:
            return value
        label = jax.vmap(lambda x: field(self.case, x)[1])(p)
        bad = (label > self.nan_beyond).reshape((-1,) + (1,)*(value.ndim-1))
        return jnp.where(bad, jnp.nan, value)


@pytest.mark.parametrize("name", ["solovev_asymmetric", "integer_3d_tau"])
def test_surface_sampler_is_the_same_function_in_numpy_and_jax(name):
    case = cases()[name]
    label, theta, phi = np.array([0.2, 0.9])*case.edge, np.array([0.3, 4.0]), np.array([1.1, 5.0])
    assert np.allclose(surface(case, label, theta, phi), np.asarray(surface(case, label, theta, phi, xp=jnp)), atol=1e-14)


def test_sheared_surface_refuses_a_traced_sampler():
    with pytest.raises(ValueError, match="NumPy only"):
        surface(cases()["sheared_A"], 0.1, 0.0, 0.0, xp=jnp)


@pytest.mark.parametrize("name", FAMILIES)
def test_exact_chart_volume_rule_reproduces_closed_forms(name):
    case = cases()[name]
    rules = {k: cp.volume_rule(case, k) for k in ("q1", "q2")}
    for row in cp.reference_certificate(case, rules).values():
        assert abs(row["volume_relative_error"]) < 1e-12
        assert row["exact_force_over_gradp"] < 1e-10
        if cp.closed_form_mean_B2(case) is not None:
            assert abs(row["mean_B2_relative_error"]) < 1e-12
    r = rules["q2"]
    assert abs(np.sum(r["w"])/float(volume(case))-1) < 1e-12
    assert np.all(r["s"] >= 0) and np.all(r["s"] <= 1)


@pytest.mark.parametrize("name", ["solovev_asymmetric", "integer_3d_tau"])
def test_the_exact_field_scores_zero_and_a_scaled_field_scores_its_scale(name):
    case = cases()[name]
    rules = {k: cp.volume_rule(case, k) for k in ("q1", "q2")}
    exact = ss.score_state(ExactField(case), rules)["scores"]["q2"]["all"]
    assert exact["B"] < 1e-13 and exact["J"] < 1e-11 and exact["force"] < 1e-9
    assert exact["outside_boundary_volume_fraction"] == 0.0
    scaled = ss.score_state(ExactField(case, scale=1.001), rules)["scores"]["q2"]
    # B and J scale linearly; the force residual of a scaled field is first order in the scale too
    assert scaled["all"]["B"] == pytest.approx(1e-3, rel=1e-9)
    assert scaled["all"]["J"] == pytest.approx(1e-3, rel=1e-9)
    assert 1e-4 < scaled["all"]["force"] < 1e-1
    for region in ("axis", "bulk", "edge"):
        assert scaled[region]["B"] == pytest.approx(1e-3, rel=1e-9)


def test_rejected_points_are_counted_with_their_volume_fraction_not_dropped_silently():
    case = cases()["solovev_symmetric"]
    rule = cp.volume_rule(case, "q2")
    cut = ss.score_state(ExactField(case, nan_beyond=0.5*case.edge), {"q2": rule})["scores"]["q2"]
    rejected = rule["w"][rule["s"] > 0.5]  # the cut is at label = edge/2, s is about 0.5 there
    assert cut["all"]["outside_boundary_points"] > 0
    assert cut["all"]["outside_boundary_volume_fraction"] == pytest.approx(np.sum(rejected)/np.sum(rule["w"]), abs=0.1)
    assert 0.3 < cut["all"]["outside_boundary_volume_fraction"] < 0.95
    assert cut["edge"]["outside_boundary_volume_fraction"] == pytest.approx(1.0)
    assert cut["edge"]["B"] is None and cut["edge"]["force"] is None  # nothing evaluable: no value, not NaN
    assert cut["axis"]["outside_boundary_volume_fraction"] == 0.0
    assert cut["axis"]["B"] < 1e-13  # the field is exact where it is defined


def test_native_evaluation_pads_the_last_chunk_and_flags_nan_points():
    xyz = np.array([[1.0, 0.0, 0.0]]*5 + [[1.0, 0.0, 0.01]]*2)

    def evaluate(p):
        bad = p[:, 2] > 0.005
        B = jnp.where(bad[:, None], jnp.nan, jnp.ones((len(p), 3)))
        g = jnp.where(bad[:, None, None], jnp.nan, jnp.zeros((len(p), 3, 3)))
        return B, g
    B, J, valid = ss.native_BJ_safe(evaluate, xyz, chunk=4)
    assert B.shape == (7, 3) and J.shape == (7, 3)
    assert valid.tolist() == [True]*5 + [False]*2


def test_headline_uses_the_finest_rule_and_reports_the_quadrature_uncertainty():
    case = cases()["integer_3d"]
    rules = {k: cp.volume_rule(case, k) for k in ("q1", "q2", "q3", "q4_shift", "q4_panels", "q4")}
    record = ss.score_state(ExactField(case, scale=1.0005), rules)
    head = ss.headline(record)
    assert head["level"] == "q4"
    assert head["B"] == pytest.approx(5e-4, rel=1e-9)
    assert head["uncertainty"]["B"] < 1e-12  # a perfectly smooth difference integrates identically on every rule
    assert record["uncertainty"]["all"]["B"]["uncertainty_below_tenth_gate"]


@pytest.mark.parametrize("name", ["solovev_asymmetric", "integer_3d_tau", "sheared_A"])
def test_mesh_rule_integrates_each_radial_cell_and_reproduces_the_volume(name):
    case = cases()[name]
    rule = cp.volume_rule(case, "m3", ns=9)
    assert abs(np.sum(rule["w"])/float(volume(case))-1) < 1e-11
    # 8 cells x 3 Gauss nodes, every s node inside its own cell of the 9-surface mesh
    s_levels = np.unique(np.round(rule["s"], 12))
    assert len(s_levels) == 8*3
    assert np.all(np.floor(s_levels*8-1e-9) >= 0) and len(set(np.floor(s_levels*8).astype(int))) == 8
    with pytest.raises(ValueError, match="need the state's NS"):
        cp.volume_rule(case, "m3")


def test_mesh_ladder_uncertainty_uses_step_and_shift_only():
    case = cases()["solovev_symmetric"]
    rules = {k: cp.volume_rule(case, k, ns=9) for k in cp.MESH_LEVELS}
    record = ss.score_state(ExactField(case, scale=1.0002), rules)
    assert cp.ladder_of(record["scores"]) == ("m4", "m3", "m4_shift", None)
    parts = record["uncertainty"]["all"]["B"]
    assert set(parts) >= {"ladder_step", "angular_shift"} and "radial_panels" not in parts
    assert ss.headline(record)["level"] == "m4"
