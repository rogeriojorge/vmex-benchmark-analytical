"""Small mathematical checks motivating the review, not VMEX solver tests.

Run: python audit_probes.py
Only NumPy is required. Results are written beside this script.
"""
from pathlib import Path
import json
import numpy as np


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def main():
    coarse = 2 * np.pi * np.arange(8) / 8
    fine = 2 * np.pi * (np.arange(257) + 0.173) / 257
    alias = {
        "function": "sin(4*theta)",
        "eight_node_rms": rms(np.sin(4 * coarse)),
        "shifted_257_node_rms": rms(np.sin(4 * fine)),
        "exact_rms": float(1 / np.sqrt(2)),
        "interpretation": "A zero coarse-grid error does not certify a volume norm.",
    }
    assert alias["eight_node_rms"] < 1e-13
    assert abs(alias["shifted_257_node_rms"] - alias["exact_rms"]) < 1e-14

    # Circular torus, s = rho**2. Integrate one physical toroidal period.
    R0, minor, nfp = 3.0, 0.4, 3
    nodes, radial_weights = np.polynomial.legendre.leggauss(16)
    s = (nodes + 1) / 2
    theta = 2 * np.pi * np.arange(64) / 64
    jac = minor**2 / 2 * (R0 + minor * np.sqrt(s[:, None]) * np.cos(theta))
    one_period = float(np.sum(jac * radial_weights[:, None] / 2)
                       * (2 * np.pi / len(theta)) * (2 * np.pi / nfp))
    full_exact = float(2 * np.pi**2 * R0 * minor**2)
    weights = {
        "nfp": nfp, "one_period_volume": one_period,
        "full_torus_volume": nfp * one_period,
        "exact_full_torus_volume": full_exact,
        "interpretation": "Constant period factors cancel in relative norms, not volumes.",
    }
    assert abs(nfp * one_period / full_exact - 1) < 1e-14

    # First variation of x = rho*cos(theta-u) at u=0 is rho*u*sin(theta).
    # With u ~ rho**2*sin(3*theta), its m=4 coefficient scales as rho**3.
    theta = 2 * np.pi * np.arange(256) / 256
    rho = np.geomspace(1e-5, 1e-2, 10)
    amp = 0.1
    bad_u = amp * rho[:, None]**2 * (1-rho[:, None]**2) * np.sin(3*theta)
    good_u = amp * rho[:, None]**3 * (1-rho[:, None]**2)**2 * np.sin(3*theta)
    coeff = lambda u: 2*np.mean(rho[:, None]*u*np.sin(theta)*np.cos(4*theta), axis=1)
    bad, good = coeff(bad_u), coeff(good_u)
    bad_order = float(np.polyfit(np.log(rho), np.log(np.abs(bad)), 1)[0])
    good_order = float(np.polyfit(np.log(rho), np.log(np.abs(good)), 1)[0])
    assert abs(bad_order-3) < 1e-3 and abs(good_order-4) < 1e-3
    regularity = {
        "gauge_poloidal_mode": 3,
        "old_envelope": "rho**2*(1-rho**2)",
        "regular_envelope": "rho**3*(1-rho**2)**2",
        "induced_position_mode": 4,
        "old_coefficient_radial_order": bad_order,
        "regular_coefficient_radial_order": good_order,
        "required_analytic_leading_order_at_least": 4,
        "interpretation": "The old m=3 gauge is not an analytic polar chart at the axis.",
    }
    report = {
        "schema": 1, "evidence": "standalone_mathematical_checks",
        "vmex_executed": False, "repository_tests_executed": False,
        "numpy_version": np.__version__, "checks_passed": 3,
        "sampling_alias": alias, "period_normalization": weights,
        "gauge_axis_regularity": regularity,
    }
    path = Path(__file__).with_name("audit_probe_results.json")
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
