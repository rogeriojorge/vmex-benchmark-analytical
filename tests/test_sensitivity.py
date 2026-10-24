"""Stage 5 checks that need no solver: input map, exact tangents, volume rule, metrics, receipts."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import derivative_map as dm
from analytic import cases, field, volume
from measurement import exact_field_parameter_tangents

CASES = cases()


@pytest.mark.parametrize("name", ["integer_axisymmetric", "integer_3d"])
def test_ad_map_reproduces_committed_deck(name):
    vmex = pytest.importorskip("vmex")
    case = CASES[name]
    inp = vmex.VmecInput.from_file(ROOT/f"inputs/input.{name}_iota")
    p = jnp.asarray(case.parameters)
    rbc, zbs = dm.vmex_boundary(p, int(inp.mpol), int(inp.ntor), int(inp.nfp))
    am, ai, phiedge = dm.profiles(p)
    np.testing.assert_allclose(rbc, inp.rbc, atol=2e-15, rtol=0)
    np.testing.assert_allclose(zbs, inp.zbs, atol=2e-15, rtol=0)
    np.testing.assert_allclose(am, np.asarray(inp.am)[:2], rtol=1e-13)
    np.testing.assert_allclose(float(ai[0]), float(inp.ai[0]), atol=1e-14)
    np.testing.assert_allclose(float(phiedge), float(inp.phiedge), rtol=1e-15)


@pytest.mark.parametrize("name,mpol,ntor,params", [("integer_axisymmetric", 13, 0, "bcd"),
                                                   ("integer_3d", 13, 12, "abcd")])
def test_input_tangent_is_converged_against_map_differences(name, mpol, ntor, params):
    """P_a from automatic differentiation equals the centred difference of the map as h -> 0.

    Exact, because the Fourier fit is linear in the sampled boundary; this is the independent
    convergence certificate the plan requires for the input tangent.
    """
    case = CASES[name]
    p0 = jnp.asarray(case.parameters)

    def pack(q):
        rbc, zbs = dm.vmex_boundary(q, mpol, ntor, case.nfp)
        am, ai, phi = dm.profiles(q)
        return jnp.concatenate([rbc.ravel(), zbs.ravel(), am, ai, jnp.atleast_1d(phi)])
    for letter in params:
        u = dm.unit({"a": "a", "b": "b", "c": "c", "d": "delta"}[letter])
        tangent = np.asarray(jax.jvp(pack, (p0,), (u,))[1])
        errors = []
        for h in (1e-2, 1e-3, 1e-4):
            hh = h*float(jnp.dot(p0, u))
            fd = np.asarray(pack(p0+hh*u)-pack(p0-hh*u))/(2*hh)
            errors.append(np.linalg.norm(fd-tangent)/np.linalg.norm(tangent))
        floor = 1e-11  # map is already linear to roundoff in some directions
        assert errors[1] <= max(errors[0]/50, floor) and errors[2] <= max(errors[1]/50, floor)  # h^2
        assert errors[2] < 1e-6


@pytest.mark.parametrize("name", ["integer_axisymmetric", "integer_3d"])
def test_exact_delta_response_is_zero_and_c_matches_finite_difference(name):
    case = CASES[name]
    rng = np.random.default_rng(5)
    from analytic import surface
    xyz = np.asarray(surface(case, case.edge*rng.uniform(.05, .9, 12), *rng.uniform(0, 2*np.pi, (2, 12))))
    tan = dm.exact_tangents(case, xyz, ("c", "delta"))
    assert np.max(abs(tan[:, :, 1])) < 1e-13
    h = 1e-6
    f = lambda c: np.asarray(jax.vmap(lambda x: field(replace(case, parameters=(*case.parameters[:2], c, case.edge)), x)[0])(jnp.asarray(xyz)))
    fd = (f(case.parameters[2]+h)-f(case.parameters[2]-h))/(2*h)
    np.testing.assert_allclose(tan[:, :, 0], fd, rtol=1e-8, atol=1e-9)


def test_exact_tangent_scaling_with_nonunit_units():
    """x = L* xbar, B = B* Bbar: the reference is evaluated at x/L* and scaled by B*."""
    case = CASES["integer_axisymmetric"]
    from analytic import surface
    xbar = np.asarray(surface(case, case.edge*.4, np.array([.7, 2.0]), np.array([.2, 1.1])))
    base = exact_field_parameter_tangents(case, xbar, (2,), length_m=1., field_t=1.)
    scaled = exact_field_parameter_tangents(case, 3.*xbar, (2,), length_m=3., field_t=2.5)
    np.testing.assert_allclose(scaled, 2.5*base, rtol=1e-13)


@pytest.mark.parametrize("name,rule", [("integer_axisymmetric", (16, 24, 2)), ("integer_3d", (24, 32, 8))])
def test_volume_rule_reproduces_closed_form_volume_and_energy(name, rule):
    case = CASES[name]
    xyz, w, coords = dm.volume_rule(case, ns=rule[0], ntheta=rule[1], nphi=rule[2], s_max=1.0)
    assert abs(w.sum()*case.nfp/volume(case)-1) < 1e-7
    # the volume mean of |B|^2 converges under refinement of the rule
    B = np.asarray(jax.vmap(lambda x: field(case, x)[0])(jnp.asarray(xyz)))
    fine = dm.volume_rule(case, ns=rule[0]+8, ntheta=rule[1]+8, nphi=rule[2]+2, s_max=1.0)  # finer in every direction
    Bf = np.asarray(jax.vmap(lambda x: field(case, x)[0])(jnp.asarray(fine[0])))
    e, ef = np.sum(w*np.sum(B*B, -1))/w.sum(), np.sum(fine[1]*np.sum(Bf*Bf, -1))/fine[1].sum()
    assert abs(e/ef-1) < 1e-6


def test_volume_rule_points_lie_on_exact_flux_surfaces_inside_smax():
    case = CASES["integer_3d"]
    xyz, w, coords = dm.volume_rule(case, ns=8, ntheta=8, nphi=4, s_max=.9)
    label = np.asarray(jax.vmap(lambda x: field(case, x)[1])(jnp.asarray(xyz)))
    np.testing.assert_allclose(label/case.edge, coords[:, 0], atol=1e-12)
    assert coords[:, 0].max() < .9 and np.all(w > 0)


def test_scores_split_weights_and_vacuum_denominator_policy():
    case = CASES["integer_axisymmetric"]
    rng = np.random.default_rng(3)
    exact = rng.standard_normal((10, 3))
    dB = exact+1e-4*rng.standard_normal((10, 3))
    w = rng.uniform(.5, 1.5, 10)
    s0 = dm.scores(dB, exact, "c", case, w)
    # duplicate a point and split its weight: invariant
    dB2, ex2, w2 = np.vstack([dB, dB[:1]]), np.vstack([exact, exact[:1]]), np.append(w, w[0]/2)
    w2[0] /= 2
    s1 = dm.scores(dB2, ex2, "c", case, w2)
    np.testing.assert_allclose(s1["relative_error"], s0["relative_error"], rtol=1e-13)
    null = dm.scores(dB-exact, np.zeros_like(exact), "delta", case, w)
    assert "relative_error" not in null and null["rms_exact_T"] == 0.0
    np.testing.assert_allclose(null["null_scaled_rms"], case.edge*dm.weighted_rms(dB-exact, w), rtol=1e-14)


def test_gates_are_the_preregistered_values():
    assert dm.GATES == {"nonzero_relative": 1e-3, "null_scaled": 1e-5}
    assert dm.gate("c", {"relative_error": 9.9e-4}) and not dm.gate("c", {"relative_error": 1.1e-3})
    assert dm.gate("delta", {"null_scaled_rms": 9.9e-6}) and not dm.gate("delta", {"null_scaled_rms": 1.1e-5})


def test_nonfinite_points_are_reported_not_dropped_silently():
    case = CASES["integer_axisymmetric"]
    exact = np.ones((4, 3))
    dB = exact.copy()
    dB[2] = np.nan
    s = dm.scores(dB, exact, "c", case)
    assert s["finite_points"] == 3 and s["points"] == 4


def test_driver_helpers_without_a_solve():
    pytest.importorskip("vmex")
    import sensitivity_vmex as sv
    assert sv.chunks(1030, 512) == [(0, 512), (512, 1024), (1024, 1030)]
    tree = {"a": jnp.array([3., 0.]), "b": jnp.array([4.])}
    assert sv.tree_norm(tree) == 5.0 and sv.tree_dot(tree, tree) == 25.0


RECORDS = sorted((ROOT/"results/sensitivity").glob("*_ns*.json")) if (ROOT/"results/sensitivity").exists() else []


@pytest.mark.skipif(not RECORDS, reason="no committed sensitivity records")
@pytest.mark.parametrize("path", RECORDS, ids=[p.name for p in RECORDS])
def test_committed_records_are_internally_consistent(path):
    rec = json.loads(path.read_text())
    assert rec["vmex_commit"].startswith("caf09ee7")
    assert set(rec["root"]) == {"preconditioned", "raw"} and rec["vmex_version"] == "0.12.0"
    for name, entry in rec["params"].items():
        kind = "delta" if name == "delta" else "c"
        for obs in entry["jvp"]:
            score = entry["jvp"][obs]
            assert score["finite_points"] == score["points"]
            assert score["gate_pass"] == dm.gate(kind, score)
            for fd in entry["fd"]:
                if obs in fd["scores"]:
                    assert fd["scores"][obs]["gate_pass"] == dm.gate(kind, fd["scores"][obs])


def test_scalar_mean_derivative_matches_finite_difference_of_the_mean():
    import sensitivity_scalar as ss
    rng = np.random.default_rng(11)
    B = rng.standard_normal((50, 3))+3.
    dB = rng.standard_normal((50, 3))
    w = rng.uniform(.5, 2., 50)
    h = 1e-6
    mean = lambda x: np.sum(w*np.linalg.norm(x, axis=-1))/w.sum()
    fd = (mean(B+h*dB)-mean(B-h*dB))/(2*h)
    np.testing.assert_allclose(ss.mean_derivative(B, dB, w), fd, rtol=1e-8)


def test_region_metric_splits_the_set_without_dropping_points():
    import sensitivity_regions as sr
    rng = np.random.default_rng(2)
    exact = rng.standard_normal((20, 3))
    approx = exact+1e-3*rng.standard_normal((20, 3))
    w = rng.uniform(1, 2, 20)
    whole = sr.metric(approx-exact, exact, "c", w, 1.)
    left, right = np.arange(20) < 7, np.arange(20) >= 7
    parts = [sr.metric((approx-exact)[m], exact[m], "c", w[m], 1.) for m in (left, right)]
    assert min(parts) <= whole <= max(parts)
