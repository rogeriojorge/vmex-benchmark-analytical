"""Scalar implicit derivative and reconverged finite differences, fixed boundary.

Usage: python benchmarks/run_gradient_smoke.py inputs/input.integer_3d_iota
This varies PHIEDGE alone: it is NOT an exact-family continuum derivative.
The local agent must add the complete family input map before P4 acceptance.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex
from vmex.core import implicit

from analytic import ROOT

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
inp = vmex.VmecInput.from_file(Path(sys.argv[1]))
if inp.lfreeb:
    raise SystemExit("Free-boundary differentiation needs the coupled API, not this fixed-boundary wrapper.")
p0 = implicit.params_from_input(inp)
FTOL = 1e-14
STEPS = np.logspace(-2, -6, 5)


def value(relative_flux_change):
    p = replace(p0, phiedge=p0.phiedge*(1+relative_flux_change))
    return implicit.run(inp, p, ftol=FTOL, max_iterations=30000).wb


f0, ad = jax.value_and_grad(value)(0.)
rows = []
for h in STEPS:
    fp, fm = float(value(h)), float(value(-h))
    fd = (fp-fm)/(2*h)
    rows.append(dict(step=float(h), central_fd=fd,
                     absolute_error=abs(fd-float(ad)),
                     relative_error=abs(fd-float(ad))/max(abs(float(ad)), 1e-30),
                     taylor_remainder=abs(fp-float(f0)-h*float(ad))))
out = ROOT / "results/vmex" / Path(sys.argv[1]).name.removeprefix("input.")
out.mkdir(parents=True, exist_ok=True)
(out / "gradient_smoke.json").write_text(json.dumps(dict(schema=1,
    evidence="discrete_derivative_consistency_only", full_benchmark_pass=False,
    objective="wb", parameter="relative_phiedge_at_fixed_boundary_and_profiles",
    value=float(f0), adjoint=float(ad), ftol=FTOL, root_refinement="default pinned implicit.run; local residual certificate required",
    steps=rows), indent=2, allow_nan=False)+"\n")
print(f"d(wb)/d(relative phiedge) = {float(ad):.12e}; inspect the step-size interval, not one minimum.")
