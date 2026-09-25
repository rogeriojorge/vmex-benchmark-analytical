"""C7: direct optimization of the exact integer family (no equilibrium solve).

Objectives (plan section 11), both dimensionless and invariant under the
family's self-similarity (a, b, c) -> lambda (a, b, c), so c = 1 fixes that
freedom and no L_star/B_star rescaling path needs differentiating:

    C_J = l_* <J^2>^(1/2) / B_rms,   l_* = V^(1/3)
    C_B = <(|B|/B_rms - 1)^2>

Volume averages use an explicit (s, theta, phi) tensor quadrature of the exact
chart with a JAX Jacobian; the reference is dimensionless (mu0 = 1, J = curl B).
Constraints: |u_a| + sqrt(delta) <= 1/2 - margin, beta_V in a band,
1 <= a/b <= max elongation, and (a^2 - b^2) >= asymmetry floor (keeps a 3-D
design).  An epsilon-constraint sweep on C_B gives a small Pareto set.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize

from analytic import ROOT, cases, integer_field
from measurement import integer_surface_jax


def quadrature(nr, nt, nphi, nfp):
    x, w = leggauss(nr)
    s, ws = (x+1)/2, w/2
    th = 2*np.pi*np.arange(nt)/nt
    ph = (2*np.pi/nfp)*np.arange(nphi)/nphi
    S, T, P = np.meshgrid(s, th, ph, indexing="ij")
    q = np.stack([S.ravel(), T.ravel(), P.ravel()], -1)
    w = (ws[:, None, None]*(2*np.pi/nt)*(2*np.pi/nphi)*np.ones_like(S)).ravel()
    return jnp.asarray(q), jnp.asarray(w)


def make_metrics(base_case, q, w):
    """Return metrics(theta) for theta = (a, b, delta) with c = 1."""
    def metrics(theta):
        a, b, delta = theta[0], theta[1], theta[2]
        c = 1.0
        case = replace(base_case, parameters=(a, b, c, delta))
        chart = lambda qq: integer_surface_jax(case, qq)  # noqa: E731
        x = jax.vmap(chart)(q)
        det = jnp.abs(jnp.linalg.det(jax.vmap(jax.jacfwd(chart))(q)))
        weights = w*det
        field_fn = lambda xx: integer_field(xx, a, b, c)[0]  # noqa: E731
        B = jax.vmap(field_fn)(x)
        grad = jax.vmap(jax.jacfwd(field_fn))(x)
        J = jnp.stack([grad[:, 2, 1]-grad[:, 1, 2], grad[:, 0, 2]-grad[:, 2, 0],
                       grad[:, 1, 0]-grad[:, 0, 1]], -1)
        V = jnp.sum(weights)
        B2 = jnp.sum(B*B, -1)
        B_rms = jnp.sqrt(jnp.sum(weights*B2)/V)
        J_rms = jnp.sqrt(jnp.sum(weights*jnp.sum(J*J, -1))/V)
        C_J = V**(1/3)*J_rms/B_rms
        C_B = jnp.sum(weights*(jnp.sqrt(B2)/B_rms-1)**2)/V
        Ha = (a*a+b*b)/2-(a*a-b*b)**2/(8*c*c)
        beta_closed = 2*c*c*delta/(Ha+c*c*delta)
        # <p> with p = 2 c^2 (delta - psi), psi = s delta on this chart.
        p_avg = jnp.sum(weights*2*c*c*delta*(1-q[:, 0]))/V
        beta_quad = 2*p_avg/(B_rms**2)
        u_a = -(a*a-b*b)/(4*c*c)
        return {"C_J": C_J, "C_B": C_B, "beta_closed": beta_closed, "beta_quadrature": beta_quad,
                "volume": V, "volume_closed": 2*jnp.pi**2*a*b*c*delta,
                "margin": 0.5-(jnp.abs(u_a)+jnp.sqrt(delta)), "elongation": a/b,
                "asymmetry": a*a-b*b}
    return metrics


def optimize(args):
    base = cases()["integer_3d"]
    q, w = quadrature(args.nr, args.nt, args.nphi, base.nfp)
    metrics = make_metrics(base, q, w)
    value = jax.jit(metrics)
    grads = jax.jit(jax.jacfwd(metrics))
    starts = [np.asarray([np.sqrt(1.5), np.sqrt(0.5), 1/64]),
              np.asarray([1.1, 0.9, 0.02]), np.asarray([1.4, 0.6, 0.01])]
    bounds = [(0.3, 3.0), (0.3, 3.0), (1e-4, 0.2)]
    cons = [
        {"type": "ineq", "fun": lambda t: float(value(jnp.asarray(t))["margin"]) - args.margin,
         "jac": lambda t: np.asarray(grads(jnp.asarray(t))["margin"])},
        {"type": "ineq", "fun": lambda t: float(value(jnp.asarray(t))["beta_closed"]) - args.beta_min,
         "jac": lambda t: np.asarray(grads(jnp.asarray(t))["beta_closed"])},
        {"type": "ineq", "fun": lambda t: args.beta_max - float(value(jnp.asarray(t))["beta_closed"]),
         "jac": lambda t: -np.asarray(grads(jnp.asarray(t))["beta_closed"])},
        {"type": "ineq", "fun": lambda t: args.max_elongation - t[0]/t[1],
         "jac": lambda t: np.asarray([-1/t[1], t[0]/t[1]**2, 0.0])},
        {"type": "ineq", "fun": lambda t: float(value(jnp.asarray(t))["asymmetry"]) - args.asymmetry,
         "jac": lambda t: np.asarray(grads(jnp.asarray(t))["asymmetry"])},
    ]
    results = []
    for kappa in args.cb_caps:
        for start_index, start in enumerate(starts):
            history = []

            def objective(t):
                m = value(jnp.asarray(t))
                history.append({"theta": list(map(float, t)), "C_J": float(m["C_J"]),
                                "C_B": float(m["C_B"]), "beta": float(m["beta_closed"]),
                                "margin": float(m["margin"])})
                return float(m["C_J"])

            cap = [{"type": "ineq", "fun": lambda t, k=kappa: k - float(value(jnp.asarray(t))["C_B"]),
                    "jac": lambda t: -np.asarray(grads(jnp.asarray(t))["C_B"])}]
            res = minimize(objective, start, jac=lambda t: np.asarray(grads(jnp.asarray(t))["C_J"]),
                           method="SLSQP", bounds=bounds, constraints=cons+cap,
                           options={"maxiter": 200, "ftol": 1e-12})
            results.append({"cb_cap": kappa, "start": start_index, "success": bool(res.success),
                            "message": str(res.message), "iterations": int(res.nit),
                            "theta": list(map(float, res.x)), "history": history})
    return base, metrics, results


def gradient_check(metrics, theta, steps=(1e-4, 1e-5, 1e-6)):
    grad = jax.jacfwd(metrics)(jnp.asarray(theta))
    rows = []
    for key in ("C_J", "C_B"):
        g = np.asarray(grad[key])
        for h in steps:
            fd = np.asarray([(float(metrics(jnp.asarray(theta)+h*e)[key])
                              - float(metrics(jnp.asarray(theta)-h*e)[key]))/(2*h) for e in np.eye(3)])
            rows.append({"metric": key, "step": h, "relative_error": float(np.linalg.norm(fd-g)/np.linalg.norm(g))})
    return rows


def main(argv=None):
    from evidence import capture_execution_source, reserve_run_directory, write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nr", type=int, default=12)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--nphi", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--beta-min", type=float, default=0.02)
    parser.add_argument("--beta-max", type=float, default=0.05)
    parser.add_argument("--max-elongation", type=float, default=3.0)
    parser.add_argument("--asymmetry", type=float, default=0.2)
    parser.add_argument("--cb-caps", type=float, nargs="+", default=(0.02, 0.04, 0.08, 0.16))
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/optimization/integer_family")
    args = parser.parse_args(argv)
    run_id, out = reserve_run_directory(
        args.output_parent, f"direct-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    here = Path(__file__).resolve().parent
    capture_execution_source(out, [Path(__file__), here/"analytic.py", here/"measurement.py", here/"evidence.py"])
    started = perf_counter()
    base, metrics, results = optimize(args)
    q2, w2 = quadrature(2*args.nr, 2*args.nt, 2*args.nphi, base.nfp)
    held_out = make_metrics(base, q2, w2)
    for row in results:
        theta = jnp.asarray(row["theta"])
        m = {k: float(v) for k, v in metrics(theta).items()}
        h = {k: float(v) for k, v in held_out(theta).items()}
        row["final"] = m
        row["held_out_refined_quadrature"] = h
        row["held_out_relative_change"] = {k: abs(h[k]-m[k])/max(abs(h[k]), 1e-300) for k in ("C_J", "C_B", "volume")}
        row["gradient_check"] = gradient_check(metrics, row["theta"])
    record = {"schema": 1, "run_id": run_id, "evidence": "direct_exact_family_optimization_no_solver",
              "family": "integer (c = 1 self-similarity gauge)", "settings": vars(args) | {"output_parent": None},
              "results": results, "elapsed_seconds": perf_counter()-started}
    write_json(out/"optimization.json", record, exclusive=True)
    for row in results:
        print(row["cb_cap"], row["start"], row["success"], ["%.4f" % v for v in row["theta"]],
              "CJ %.5f CB %.5f beta %.4f" % (row["final"]["C_J"], row["final"]["C_B"], row["final"]["beta_closed"]),
              "held %.1e" % max(row["held_out_relative_change"].values()),
              "grad %.1e" % min(r["relative_error"] for r in row["gradient_check"] if r["metric"] == "C_J"))
    print(run_id)


if __name__ == "__main__":
    main()
