"""Exact-chart volume rules for every family, run-log helpers, and the DESC glue (skipped without DESC)."""
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import common_norms as cn  # noqa: E402
from analytic import cases, label_at_s, surface  # noqa: E402
import run_log  # noqa: E402

CASES = cases()


@pytest.mark.parametrize("name", ["solovev_symmetric", "solovev_asymmetric", "integer_3d", "integer_3d_tau",
                                  "integer_3d_tau_mirror", "integer_3d_reframed"])
def test_jax_chart_matches_numpy_surface(name):
    case = CASES[name]
    rng = np.random.default_rng(1)
    q = np.column_stack((rng.uniform(1e-3, case.edge, 40), rng.uniform(0, 2*np.pi, 40), rng.uniform(0, 2*np.pi, 40)))
    got = np.asarray(cn.chart_xyz(case, jnp.asarray(q)))
    assert np.abs(got-surface(case, q[:, 0], q[:, 1], q[:, 2])).max() < 1e-12


@pytest.mark.parametrize("name", ["solovev_symmetric", "solovev_asymmetric", "integer_3d", "integer_3d_tau_mirror"])
def test_rule_weights_integrate_to_exact_volume(name):
    case = CASES[name]
    rule = cn.volume_rule(case, "q1")
    assert abs(cn.rule_volume_error(case, rule)) < 1e-12
    assert np.all(rule["w"] > 0)
    assert rule["s"].min() > 0 and rule["s"].max() < 1


def test_sheared_rule_delegates_and_integrates_to_exact_volume():
    case = CASES["sheared_A"]
    rule = cn.volume_rule(case, "q1")
    assert abs(cn.rule_volume_error(case, rule)) < 1e-12


@pytest.mark.parametrize("name", ["integer_3d_tau", "sheared_A", "sheared_issan_tau_mirror"])
def test_desc_guess_reproduces_the_chart_points(name):
    """(rho, theta_DESC, zeta) = (sqrt s, -theta, phi) must land on the rule's own Cartesian points."""
    case = CASES[name]
    rule = cn.volume_rule(case, "q1")
    guess = cn.guess_for(case, rule)
    assert np.all(guess[:, 0] > 0) and np.all(guess[:, 0] < 1)
    pick = np.random.default_rng(2).choice(len(guess), 60, replace=False)
    label = np.asarray([float(label_at_s(case, g**2)) for g in guess[pick, 0]])
    xyz = surface(case, label, -guess[pick, 1], guess[pick, 2])
    assert np.abs(xyz-rule["xyz"][pick]).max() < 1e-9


def test_parse_table_reads_lsq_exact_and_trf_rows():
    text = ["   Iteration     Total nfev        Cost      Cost reduction    Step norm     Optimality   ",
            "       0              1         2.9571e-07                                    1.10e-03    ",
            "       1              2         1.9829e-08      2.76e-07       1.84e-03       1.08e-05    ",
            "`gtol` termination condition is satisfied."]
    rows = run_log.parse_table([(float(i), t) for i, t in enumerate(text)])
    assert [r["iteration"] for r in rows] == [0, 1]
    assert rows[0]["cost"] == pytest.approx(2.9571e-7) and rows[0]["step_norm"] is None
    assert rows[1]["optimality"] == pytest.approx(1.08e-5) and rows[1]["step_norm"] == pytest.approx(1.84e-3)


def test_json_safe_replaces_nonfinite():
    assert run_log.json_safe({"a": [np.nan, np.float64(1.5), np.int64(2)]}) == {"a": [None, 1.5, 2]}
    assert run_log.json_safe({"x": jnp.asarray(2.5), "y": jnp.asarray([1.0, jnp.inf])}) == {"x": 2.5, "y": [1.0, None]}


def test_exact_chart_force_residual_vanishes():
    """The exact fields on the rule satisfy J x B = grad p to quadrature/AD precision (rule self-check)."""
    case = CASES["integer_3d_tau"]
    rule = cn.volume_rule(case, "q1")
    force = np.cross(rule["J"], rule["B"])-rule["gp"]
    assert np.sqrt(np.sum(rule["w"]*np.sum(force**2, 1))/np.sum(rule["w"]*np.sum(rule["gp"]**2, 1))) < 1e-10


class TestWithDesc:
    @pytest.fixture(scope="class")
    def desc_modules(self):
        pytest.importorskip("desc")
        import desc
        desc.set_device("cpu")
        import desc_general_case as dgc
        import desc_matched as dm
        return dgc, dm

    def test_exact_seed_is_consistent_and_scoring_agrees_with_native(self, desc_modules):
        """On a built (unsolved) Solov'ev state the chart-point score equals DESC's native score to a few percent
        and the exact-field lambda gradient data are self-consistent."""
        dgc, dm = desc_modules
        case = CASES["solovev_symmetric"]
        eq, _ = dgc.build(case, 4, 0, 6, True)
        info = dm.exact_lambda(eq, case, 6, 4, 0)
        assert info["gradient_inconsistency_relative"] < 1e-2
        scores, _ = cn.score_desc(eq, case, levels=("q1",))
        chart = scores["q1"]["all"]
        native = dgc.score_everywhere(eq, case)["volume"]["gauss"]
        assert chart["outside_boundary_points"] == 0
        assert chart["B"] < 1e-1
        assert chart["B"] == pytest.approx(native["B_relative_l2"], rel=0.5)
