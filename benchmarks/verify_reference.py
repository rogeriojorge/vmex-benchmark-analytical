"""Reproduce sampled identities and quadrature checks; does not import VMEX."""
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from analytic import (ROOT, cases, differential_fields, field, flux, integer_targets,
                      iota, shear_averages, surface, validate, volume)

NPOINTS = 192
SEED = 260923
OUT = ROOT / "results/reference/identities.json"
rng = np.random.default_rng(SEED)
result = {"schema": 1, "evidence": "analytic_reference_sampled", "vmex_executed": False,
          "seed": SEED, "python": sys.version, "platform": platform.platform(),
          "versions": {k: importlib.metadata.version(k) for k in ("numpy", "scipy", "jax", "jaxlib")},
          "cases": []}
rms = lambda x: float(np.sqrt(np.mean(np.sum(x*x, axis=-1))))
for name, case in cases().items():
    validate(case)
    label = case.edge*rng.uniform(0.005, 0.99, NPOINTS)
    theta, phi = rng.uniform(0., 2*np.pi, (2, NPOINTS))
    xyz = surface(case, label, theta, phi)
    B, J, gp, db, ps = differential_fields(case, xyz)
    force = np.cross(J, B)-gp
    fscale = max(rms(gp), 1e-14)
    # Independent real central differences also work on the non-holomorphic
    # sheared evaluator. This is not another invocation of the JAX derivative.
    step = 2e-5
    fd = np.stack([(np.asarray(field(case, xyz+step*e)[0])-
                    np.asarray(field(case, xyz-step*e)[0]))/(2*step)
                   for e in np.eye(3)], -1)
    row = {"name": name, "npoints": NPOINTS, "volume": float(volume(case)),
           "toroidal_flux": float(flux(case, case.edge)),
           "iota_axis_ccw": float(iota(case, 0.)),
           "label_error_over_edge": float(np.max(abs(ps-label))/case.edge),
           "max_abs_divB": float(np.max(abs(np.trace(db, axis1=1, axis2=2)))),
           "force_rms_over_gradp_rms": rms(force)/fscale,
           "tangency_scaled": float(np.max(abs(np.sum(B*gp, -1))))/(rms(B)*fscale),
           "real_fd_dB_relative": float(np.linalg.norm(fd-db)/np.linalg.norm(db)),
           "physical_phi_error": float(np.max(abs(np.sin(np.arctan2(xyz[:, 1], xyz[:, 0])-phi))))}
    if case.family == "integer":
        row.update({k: float(v) for k, v in integer_targets(*case.parameters).items()})
    if case.family == "sheared":
        coarse, fine = shear_averages(*case.parameters), shear_averages(*case.parameters, nr=120, nt=512)
        row.update(fine)
        row["beta_quadrature_relative_change"] = abs(coarse["beta"]/fine["beta"]-1)
    row["passed"] = bool(row["label_error_over_edge"] < 2e-11 and
                         row["force_rms_over_gradp_rms"] < 2e-11 and
                         row["max_abs_divB"] < 2e-10 and
                         row["real_fd_dB_relative"] < 2e-6 and
                         row["tangency_scaled"] < 2e-11 and row["physical_phi_error"] < 2e-12 and
                         row.get("beta_quadrature_relative_change", 0) < 2e-9)
    result["cases"].append(row)
    print(f"{name:34s} force={row['force_rms_over_gradp_rms']:.2e} FD={row['real_fd_dB_relative']:.2e} pass={row['passed']}")
result["passed"] = all(r["passed"] for r in result["cases"])
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
if not result["passed"]:
    raise SystemExit("An analytical reference check failed; inspect the JSON.")
