"""Compare implicit angle-root JVP with independent physical-surface differences."""
from dataclasses import replace
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import ROOT, cases, shear_chart, surface
from sheared_map import angle_cross, physical_angle_root

DIRECTION = np.array([0.01, 0.02, 0.03, 0.02, 0.03, 0.01])
STEPS = (1e-3, 1e-4, 1e-5, 1e-6)
rows = []
for name in ("sheared_A", "sheared_B", "sheared_C"):
    case = cases()[name]
    eps, S, lam, edge = case.parameters
    for fraction, theta, phi in ((0.6, 0.4, 0.6), (0.9, 2.0, 1.2)):
        k = fraction*np.sqrt(2*edge)
        parameters = jnp.array([k, theta, phi, eps, S, lam])

        def mapped(a):
            kk, th, ph, ee, ss, ll = a
            t = physical_angle_root(kk, th, ph, ee, ss, ll)
            return shear_chart(kk, th, t, ee, ss, ll, xp=jnp)

        point, tangent = jax.jvp(mapped, (parameters,), (jnp.asarray(DIRECTION),))
        t = physical_angle_root(*parameters)
        denominator = jax.grad(lambda tt: angle_cross(tt, *parameters))(t)

        def independent(offset):
            a = np.asarray(parameters)+offset*DIRECTION
            trial = replace(case, parameters=(a[3], a[4], a[5], edge))
            return surface(trial, a[0]**2/2, a[1], a[2])

        position_error = float(np.linalg.norm(np.asarray(point)-independent(0.0)))
        errors = [float(np.linalg.norm((independent(h)-independent(-h))/(2*h)-tangent))
                  for h in STEPS]
        status = "passed" if (abs(float(denominator)) > 1e-3 and
                              position_error < 1e-10 and min(errors) < 1e-7) else "failed"
        rows.append(dict(case=name, fraction=fraction, status=status,
                         angular_derivative=float(denominator),
                         position_error=position_error, finite_difference_steps=STEPS,
                         directional_position_errors=errors))
        print(rows[-1])
out = ROOT / "results/reference/sheared_derivative.json"
out.write_text(json.dumps(dict(schema=1, evidence="analytic_reference_sampled",
    status="passed" if all(r["status"] == "passed" for r in rows) else "failed",
    method="custom JVP of implicit scalar angle root vs independent bisection surface",
    direction=DIRECTION.tolist(), rows=rows), indent=2, allow_nan=False)+"\n")
if any(r["status"] != "passed" for r in rows):
    raise SystemExit("Sheared implicit derivative check failed")
