"""Project the exact symmetric Solov'ev case into VMEX's continuous basis.

The axis-regular series and straight-field-line gauge follow the derivation
in pinned VMEX ``tests/test_strong_force_solovev.py``. Coefficients here are
computed independently with binomial identities, then scored against this
repository's Cartesian analytical field without a nonlinear solve.
"""
import json
from math import comb
from pathlib import Path
import subprocess
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.special import binom
from vmex.core.radial_basis import BSplineBasis
from vmex.core.strong_force import (
    HighOrderEquilibriumState, evaluate_high_order_fields, certify_strong_force)

from analytic import ROOT, cases, differential_fields
from build_inputs import MU0

DEGREE = 5
SPANS = (2, 4, 8)
MMAX = 12


def project(case, spans):
    if case.name != "solovev_symmetric":
        raise ValueError("Only the exact symmetric Solov'ev chart is derived here")
    R0, b, g, chi, F0, psi_edge = case.parameters
    assert chi == 0 and case.nfp == 1
    a = np.sqrt(psi_edge/b)/R0**2
    c = np.sqrt(psi_edge/g)
    jac = -R0**2*a*c/4
    basis = BSplineBasis.clamped(np.linspace(0, 1, spans+1),
                                degree=DEGREE, quadrature_order=DEGREE+3)
    s = np.asarray(basis.collocation_nodes)
    zeros = np.zeros((MMAX+1, basis.size))
    R_cos, Z_sin, L_sin = zeros.copy(), zeros.copy(), zeros.copy()

    def q_radius(m, values):
        total = np.zeros_like(np.asarray(values), dtype=float)
        for k in range(m, 65, 2):
            cm = comb(k, (k-m)//2) / (2**k if m == 0 else 2**(k-1))
            total += binom(0.5, k)*cm*a**k*np.asarray(values)**((k-m)//2)
        return R0*total

    for m in range(MMAX+1):
        R_cos[m] = np.asarray(basis.fit(jnp.asarray(q_radius(m, s))))
        if m:
            x = a*np.sqrt(s)
            t = np.sqrt((1-x)/(1+x))
            tau_over_x = 2/((1+x)*(1+t)**2)
            q_lambda = 2*(-a*tau_over_x)**m/m
            L_sin[m] = np.asarray(basis.fit(jnp.asarray(q_lambda)))
    Z_sin[1] = np.asarray(basis.fit(jnp.full(s.shape, c)))
    F = np.sqrt(F0**2-4*g*psi_edge*s)
    phipf = F*jac/(R0**2*np.sqrt(1-a*a*s))
    pressure = 8*b*psi_edge*(1-s)/MU0
    boundary_R = np.array([q_radius(m, 1.0).item() for m in range(MMAX+1)])
    boundary_Z = np.zeros(MMAX+1)
    boundary_Z[1] = c
    return HighOrderEquilibriumState(
        radial_basis=basis, m=np.arange(MMAX+1), n=np.zeros(MMAX+1, dtype=int), nfp=1,
        R_cos=jnp.asarray(R_cos), R_sin=jnp.asarray(zeros),
        Z_cos=jnp.asarray(zeros), Z_sin=jnp.asarray(Z_sin),
        L_cos=jnp.asarray(zeros), L_sin=jnp.asarray(L_sin),
        phipf=basis.fit(jnp.asarray(phipf)),
        chipf=basis.fit(jnp.full(s.shape, psi_edge)),
        pressure=basis.fit(jnp.asarray(pressure)), jacobian_sign=-1,
        source="independent exact Solovev projection",
        boundary_R_cos=jnp.asarray(boundary_R), boundary_R_sin=jnp.zeros(MMAX+1),
        boundary_Z_cos=jnp.zeros(MMAX+1), boundary_Z_sin=jnp.asarray(boundary_Z))


def main():
    case = cases()["solovev_symmetric"]
    s, theta, phi = np.meshgrid([0.12, 0.5, 0.88],
        2*np.pi*np.arange(8)/8, 2*np.pi*np.arange(4)/4, indexing="ij")
    rows = []
    for spans in SPANS:
        state = project(case, spans)
        start = perf_counter()
        sample = evaluate_high_order_fields(state, jnp.sqrt(s), theta, phi)
        xyz, B = np.asarray(sample.position).reshape(-1, 3), np.asarray(sample.B).reshape(-1, 3)
        exact_B = differential_fields(case, xyz)[0]
        error = float(np.linalg.norm(B-exact_B)/np.linalg.norm(exact_B))
        report = certify_strong_force(state, angular_multiplier=1)
        rows.append(dict(spans=spans, spline_degree=DEGREE, mmax=MMAX,
                         field_relative_l2=error,
                         force_absolute_rms_N_m3=float(report.absolute_l2),
                         force_normalized_l2=float(report.normalized_l2),
                         radial_quadrature_difference=float(report.radial_refinement_difference),
                         evaluation_seconds=perf_counter()-start))
        print(rows[-1])
    root = Path(__import__("vmex").__file__).resolve().parents[1]
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    out = ROOT / "results/projection"
    out.mkdir(exist_ok=True)
    (out / "solovev_symmetric.json").write_text(json.dumps(dict(
        schema=1, evidence="analytic_projection", status="passed", vmex_commit=sha,
        case=case.name, no_nonlinear_solve=True, field_samples=96,
        method="exact regular series fitted to VMEX continuous spline basis",
        scope="Cartesian B projection; force certificate has a measured floor and finite quadrature difference",
        rows=rows), indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
