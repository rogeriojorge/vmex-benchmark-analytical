"""Full-flux lambda seed, projection record and quadrature certificate for sheared cases."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import certify_projection as cp  # noqa: E402
from analytic import cases, shear_averages, volume  # noqa: E402
from flux_lambda import fourier_coefficients, recover_lambda, seed_type  # noqa: E402


def test_seed_type_names_each_family():
    c = cases()
    assert seed_type(c["integer_3d"]) == "exact_field_lambda_zero"
    assert seed_type(c["sheared_A"]) == "full_flux_lambda"
    assert seed_type(c["sheared_A"], "zero") == "geometry_only_lambda_zero"
    assert seed_type(c["integer_3d"], "full_flux") == "full_flux_lambda"
    with pytest.raises(ValueError):
        seed_type(c["sheared_A"], "transport")


@pytest.mark.parametrize("grid", [64, 65])
def test_fourier_coefficients_round_trip(grid):
    modes = SimpleNamespace(m=np.array([0, 0, 1, 1, 2, 3]), n=np.array([0, 2, -3, 0, 4, -1]))
    theta = 2*np.pi*np.arange(grid)[:, None]/grid
    zeta = 2*np.pi*np.arange(grid)[None, :]/grid
    truth_c = np.array([0.5, 0.25, -0.75, 0.125, 0.3, -0.2])
    truth_s = np.array([0.0, -0.4, 0.6, 0.35, -0.15, 0.05])
    values = sum(c*np.cos(m*theta-n*zeta)+s*np.sin(m*theta-n*zeta)
                 for c, s, m, n in zip(truth_c, truth_s, modes.m, modes.n))
    cos, sin = fourier_coefficients(values, modes)
    assert np.allclose(cos, truth_c, atol=1e-13)
    sin_expected = truth_s.copy()
    sin_expected[0] = 0.0  # sin(0) carries no (0, 0) amplitude
    assert np.allclose(sin, sin_expected, atol=1e-13)
    with pytest.raises(ValueError):
        fourier_coefficients(values[:8, :8], modes)


def test_flux_lambda_separates_lambda_from_the_straightness_null_space():
    """With iota = 1 and NFP = 2, lambda = 0.08 sin(2 theta - 2 phi) and lambda = 0 both satisfy the
    field-line equation, but only the true lambda reproduces both signed flux densities."""
    nfp, n = 2, 33
    theta = 2*np.pi*np.arange(n)[:, None]/n
    phi = 2*np.pi*np.arange(n)[None, :]/(n*nfp)
    P, T = -0.3, -0.3                                  # iota = T/P = 1
    lam = 0.08*np.sin(2*theta-2*phi)
    lam_theta, lam_phi = 0.16*np.cos(2*theta-2*phi), -0.16*np.cos(2*theta-2*phi)
    U, V = T-P*lam_phi, P*(1+lam_theta)
    recovered, certificate = recover_lambda(U, V, poloidal=T, toroidal=P, nfp=nfp)
    assert np.max(np.abs(recovered-lam)) < 1e-13
    assert certificate["gradient_reconstruction_relative"] < 1e-12
    # lambda = 0 reproduces neither flux density
    assert np.max(np.abs(V-P)) > 1e-2 and np.max(np.abs(U-T)) > 1e-2


def test_flux_lambda_rejects_incompatible_flux_data():
    n = 33
    rng = np.random.default_rng(1)
    U, V = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    with pytest.raises(ValueError, match="Incompatible periodic flux data"):
        recover_lambda(U, V, poloidal=0.2, toroidal=0.3, nfp=2)


def test_lambda_rebuilds_the_exact_field_on_a_heldout_shifted_grid():
    case = cases()["sheared_A"]
    record = cp.lambda_reconstruction(case, 0.5, n=49, n_check=57)
    assert record["B_rebuilt_relative_l2"] < 1e-7
    assert record["min_1_plus_lambda_theta"] > 0.5
    assert abs(record["iota_check"]) < 1e-12


@pytest.fixture(scope="module")
def tiny_rule():
    case = cases()["sheared_A"]
    cp.LEVELS["tiny"] = (10, 21, 21, "gauss", 0.0, 1)
    try:
        yield case, cp.volume_rule(case, "tiny")
    finally:
        del cp.LEVELS["tiny"]


def test_chart_volume_rule_matches_closed_forms(tiny_rule):
    """Exact Jacobian weights reproduce volume, field energy and the force balance of the reference."""
    case, rule = tiny_rule
    V = float(np.sum(rule["w"]))
    assert V == pytest.approx(float(volume(case)), rel=1e-11)
    mean_B2 = float(np.sum(rule["w"]*np.sum(rule["B"]**2, -1))/V)
    assert mean_B2 == pytest.approx(shear_averages(*case.parameters)["mean_B2"], rel=1e-10)
    force = np.cross(rule["J"], rule["B"])-rule["gp"]
    assert np.sqrt(np.sum(rule["w"]*np.sum(force**2, -1))/np.sum(rule["w"]*np.sum(rule["gp"]**2, -1))) < 1e-12


def test_metrics_scale_with_a_known_perturbation(tiny_rule):
    """A uniform 1e-3 relative field error scores 1e-3 in B; a missed point is not hidden by NaN."""
    _, rule = tiny_rule
    exact = cp.metrics(rule, rule["B"], rule["J"])
    assert exact["all"]["B"] == 0 and exact["all"]["J"] == 0 and exact["all"]["force"] < 1e-12
    scaled = cp.metrics(rule, (1+1e-3)*rule["B"], rule["J"])
    assert scaled["all"]["B"] == pytest.approx(1e-3, rel=1e-12)
    assert set(scaled) == {"all", "axis", "bulk", "edge"}
    assert scaled["axis"]["B"] == pytest.approx(1e-3, rel=1e-9)


def test_uncertainty_flags_quadrature_spread_against_a_tenth_of_the_gate():
    def block(b, j, f):
        return {region: {"B": b, "J": j, "force": f} for region in ("all", *cp.REGIONS)}
    stable = {cp.FINEST: block(2e-6, 1e-4, 1e-4), cp.PREVIOUS: block(2.01e-6, 1.01e-4, 1.01e-4),
              cp.SHIFTED: block(2e-6, 1e-4, 1e-4), cp.PANELS: block(2e-6, 1e-4, 1e-4)}
    out = cp.uncertainty(stable)["all"]
    assert all(out[k]["uncertainty_below_tenth_gate"] for k in out)
    assert out["B"]["uncertainty"] == pytest.approx(1e-8)
    noisy = dict(stable)
    noisy[cp.SHIFTED] = block(4e-6, 1e-4, 1e-4)  # 2e-6 spread in B exceeds gate/10 = 1e-6
    out = cp.uncertainty(noisy)["all"]
    assert not out["B"]["uncertainty_below_tenth_gate"] and out["J"]["uncertainty_below_tenth_gate"]
    assert out["B"]["uncertainty"] == pytest.approx(2e-6)


def test_sheared_projection_uses_the_full_flux_lambda_and_beats_the_zero_control():
    """Failed before: the projector zeroed lambda for every family. The seed type is recorded."""
    pytest.importorskip("vmex")
    import vmex
    from project_integer_vmex import project, surface_rows, with_resolution
    case = cases()["sheared_A"]
    inp = with_resolution(vmex.VmecInput.from_file(ROOT/"inputs/input.sheared_A_iota"), 9, 8, 6)
    state, info = project(inp, case, 9)
    control, control_info = project(inp, case, 9, lambda_mode="zero")
    assert info["seed"] == "full_flux_lambda" and control_info["seed"] == "geometry_only_lambda_zero"
    assert np.max(np.abs(np.asarray(control.L_sin))) == 0 and np.max(np.abs(np.asarray(state.L_sin))) > 1e-3
    assert np.max(np.abs(np.asarray(state.L_cos))) == 0  # stellarator-symmetric case
    assert info["lambda_certificates"]["minimum_straight_angle_derivative"] > 0.5
    error = max(r["B_relative_surface_l2"] for r in surface_rows(inp, case, state, 9)[1])
    zero = max(r["B_relative_surface_l2"] for r in surface_rows(inp, case, control, 9)[1])
    assert zero > 0.05 and error < zero/20
