"""Score native physical samples against an exact field, independent of WOUT.

Usage: python benchmarks/score_samples.py samples.npz scores.json
NPZ contract: case_name (scalar text), xyz/B/J/gradp (N,3), weights (N,),
length_m, field_t, mu0 (positive scalars). Optional s (N,) tests the flux labels.
weights are physical dV quadrature weights. No hidden interpolation is done.
"""
import json
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from analytic import cases, differential_fields, label_at_s


def score(data):
    case = cases()[str(np.asarray(data["case_name"]).item())]
    L, B0, mu0 = (float(data[k]) for k in ("length_m", "field_t", "mu0"))
    if not np.isfinite([L, B0, mu0]).all() or min(L, B0, mu0) <= 0:
        raise ValueError("Scales must be positive.")
    xyz, B, J, gp = [np.asarray(data[k], dtype=float) for k in ("xyz", "B", "J", "gradp")]
    weights = np.asarray(data["weights"], dtype=float)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or any(x.shape != xyz.shape for x in (B,J,gp))
            or weights.shape != (len(xyz),) or np.any(weights <= 0)):
        raise ValueError("Invalid sample shapes or nonpositive volume weights.")
    if not all(np.isfinite(x).all() for x in (xyz, B, J, gp, weights)):
        raise ValueError("Nonfinite physical samples are a failure, not missing data.")
    Be, Je, gpe, _, label = differential_fields(case, xyz/L)
    if not all(np.isfinite(x).all() for x in (Be, Je, gpe, label)):
        raise ValueError("Samples leave the smooth analytical field domain.")
    if np.any(label < -case.edge*1e-10) or np.any(label > case.edge*(1+1e-10)):
        raise ValueError("Score the common interior domain; measure boundary mismatch separately.")
    Be, Je, gpe = B0*Be, B0/(mu0*L)*Je, B0**2/(mu0*L)*gpe
    rms = lambda x: float(np.sqrt(np.sum(weights*np.sum(x*x, axis=-1))/np.sum(weights)))
    rel = lambda err, ref: rms(err)/rms(ref) if rms(ref) > 0 else None
    F = np.cross(J, B)-gp
    result = dict(schema=1, evidence="physical_samples_vs_analytic", case=case.name,
                  sample_count=len(xyz), quadrature_volume_m3=float(weights.sum()),
                  field_relative_l2=rel(B-Be, Be), current_relative_l2=rel(J-Je, Je),
                  gradp_relative_l2=rel(gp-gpe, gpe), force_rms_N_m3=rms(F),
                  force_fixed_magnetic_scale=rms(F)/(B0*B0/(mu0*L)),
                  force_pressure_scale=rel(F, gpe),
                  field_max_abs_T=float(np.max(np.linalg.norm(B-Be, axis=-1))),
                  accepted=None)
    if "s" in data:
        if np.shape(data["s"]) != (len(xyz),):
            raise ValueError("s must contain one normalized flux label per point.")
        target = label_at_s(case, np.asarray(data["s"]))
        result["surface_label_max_over_edge"] = float(np.max(abs(label-target))/case.edge)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    with np.load(sys.argv[1], allow_pickle=False) as data:
        result = score(data)
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
