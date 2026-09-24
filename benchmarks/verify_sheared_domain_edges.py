"""Probe admissibility margins and physical-angle branches of the sheared chart."""
from dataclasses import replace
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import (ROOT, cases, shear_chart, surface, validate,
                      validate_sheared_surface_chart_sampled)
from sheared_map import angle_cross, physical_angle_root

base = cases()["sheared_C"]
eps, _, lam, edge = base.parameters
floor = float(np.arcsin(np.sqrt(2*edge)))
bad = []
for label, parameters in (
    ("negative_epsilon", (-0.01, floor+0.1, lam, edge)),
    ("zero_lambda", (eps, floor+0.1, 0.0, edge)),
    ("singular_edge", (eps, 2.0, lam, 0.5)),
    ("domain_boundary", (eps, floor, lam, edge)),
    ("below_domain_boundary", (eps, floor-1e-4, lam, edge)),
):
    try:
        validate(replace(base, parameters=parameters))
        rejected = False
    except ValueError:
        rejected = True
    bad.append(dict(case=label, rejected=rejected))

theta = 2*np.pi*np.arange(64)/64
t = np.linspace(0.0, 2*np.pi, 1025)
theta_grid, t_grid = np.broadcast_arrays(theta[:, None], t[None, :])
rows = []
for margin in (0.001, 0.01, 0.1, 0.2, 0.5, 1.0, float(cases()["sheared_C"].parameters[1]-floor)):
    S = floor+margin
    trial = replace(base, parameters=(eps, S, lam, edge))
    validate(trial)
    xyz = shear_chart(np.sqrt(2*edge), theta_grid, t_grid, eps, S, lam)
    phi = np.unwrap(np.arctan2(xyz[..., 1], xyz[..., 0]), axis=-1)
    slope = np.diff(phi, axis=-1)/(2*np.pi/(len(t)-1))
    radii = np.hypot(xyz[..., 0], xyz[..., 1])
    imin = np.unravel_index(np.argmin(slope), slope.shape)
    target = float((phi[imin[0], imin[1]]+phi[imin[0], imin[1]+1])/2)
    crossings = int(np.count_nonzero(np.diff(np.signbit(phi[imin[0]]-target))))
    try:
        validate_sheared_surface_chart_sampled(trial)
        guard_rejected = False
    except ValueError:
        guard_rejected = True
    rows.append(dict(margin=margin, min_sampled_dphi_dt=float(slope.min()),
                     min_sampled_radius=float(radii.min()),
                     finite=bool(np.isfinite(xyz).all()),
                     winding_max_abs=float(np.max(abs(phi[:, -1]-phi[:, 0]-2*np.pi))),
                     fold_theta=float(theta[imin[0]]), fold_t=float(t[imin[1]]),
                     crossings_at_fold_target=crossings, graph_valid_sampled=bool(slope.min() > 0),
                     sampled_guard_rejected=guard_rejected))

branch = []
for center in (0.0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi):
    for offset in (-1e-6, 0.0, 1e-6):
        phi = center+offset
        k = 0.8*np.sqrt(2*edge)
        th = 0.71
        root = physical_angle_root(k, th, phi, eps, floor+0.01, lam)
        xyz = shear_chart(k, th, root, eps, floor+0.01, lam, xp=jnp)
        denominator = jax.grad(lambda tt: angle_cross(tt, k, th, phi, eps, floor+0.01, lam))(root)
        independent = surface(replace(base, parameters=(eps, floor+0.01, lam, edge)),
                              k*k/2, th, phi)
        branch.append(dict(phi=float(phi), position_error=float(np.linalg.norm(np.asarray(xyz)-independent)),
                           angular_derivative=float(denominator)))

passed = (all(item["rejected"] for item in bad) and
          all(item["finite"] and item["min_sampled_radius"] > 0 and
              item["winding_max_abs"] < 1e-10 for item in rows) and
          all(not item["graph_valid_sampled"] and item["sampled_guard_rejected"] and
              item["crossings_at_fold_target"] >= 3
              for item in rows[:3]) and
          all(item["graph_valid_sampled"] and not item["sampled_guard_rejected"]
              for item in rows[3:]) and
          all(item["position_error"] < 1e-10 and abs(item["angular_derivative"]) > 1e-3
              for item in branch))
out = ROOT / "results/reference/sheared_domain_edges.json"
out.write_text(json.dumps(dict(schema=1, evidence="analytic_reference_sampled",
    status="passed" if passed else "failed", edge=edge, domain_floor_S=floor,
    finding="smooth-domain validation alone does not ensure a single-valued physical-angle chart",
    rejected_cases=bad, near_domain_rows=rows, branch_rows=branch),
    indent=2, allow_nan=False)+"\n")
print({"status": "passed" if passed else "failed", "near_domain": rows,
       "max_branch_position_error": max(x["position_error"] for x in branch)})
if not passed:
    raise SystemExit("Sheared edge-domain check failed")
