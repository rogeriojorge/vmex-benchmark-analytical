"""Measured analytical sensitivities and Taylor curves; no VMEX solve."""
from dataclasses import replace
import json

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import ROOT, cases, field, integer_targets, shear_flux_rates, surface

p = jnp.array([np.sqrt(1.5), np.sqrt(.5), 1., 1/64])
f = lambda q: integer_targets(*q, xp=jnp)["beta"]
direction = jnp.array([.12, -.07, .15, .003])
grad = jax.grad(f)(p)
directional = float(jnp.dot(grad, direction))
rows = []
for h in np.logspace(-1, -7, 13):
    fp, fm, f0 = float(f(p+h*direction)), float(f(p-h*direction)), float(f(p))
    fd = (fp-fm)/(2*h)
    rows.append(dict(step=float(h), taylor_remainder=abs(fp-f0-h*directional),
                     central_fd_error=abs(fd-directional)))
case = cases()["integer_3d"]
point = jnp.asarray(surface(case, .005, .3, .4))
null = jax.jacfwd(lambda d: field(replace(case, parameters=(*case.parameters[:-1], d)), point)[0])(jnp.array(case.edge))
ratio = lambda lam: (lambda pair: pair[1]/pair[0])(
    shear_flux_rates(jnp.array([0., .3, .7]), 1.08, 3., lam, xp=jnp))
lam_null = jax.jacfwd(ratio)(jnp.array(3.5))
result = dict(schema=1, evidence="analytic_derivatives_only", vmex_executed=False,
              parameters=list(map(float, p)), beta_gradient=list(map(float, grad)),
              direction=list(map(float, direction)), directional_beta=directional,
              delta_eulerian_field_derivative=list(map(float, null)),
              lambda_iota_derivative=list(map(float, lam_null)), steps=rows)
(ROOT / "results/reference/derivatives.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
print(json.dumps({k:v for k,v in result.items() if k != "steps"}, indent=2))
