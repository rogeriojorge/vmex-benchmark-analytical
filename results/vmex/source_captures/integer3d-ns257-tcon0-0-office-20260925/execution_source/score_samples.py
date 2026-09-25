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
from evidence import validate_point_cloud


POINT_CLOUD_KEYS = (
    "case_name", "xyz", "weights", "s_reference", "length_m", "field_t", "mu0",
)
OBSERVATION_KEYS = ("B", "J", "gradp", "s_native")


def point_cloud(data):
    """Return a solver-neutral point cloud from new or historical sample files."""
    cloud = {key: data[key] for key in POINT_CLOUD_KEYS if key in data}
    if "s_reference" not in cloud and "s" in data:
        cloud["s_reference"] = data["s"]
    validate_point_cloud(cloud)
    npoints = len(np.asarray(cloud["xyz"]))
    if np.shape(cloud["weights"]) != (npoints,) or np.shape(cloud["s_reference"]) != (npoints,):
        raise ValueError("point-cloud weights and s_reference must match xyz")
    return cloud


def join_samples(cloud, observations):
    """Join a neutral reference cloud with one solver's observations in memory."""
    cloud = dict(cloud)
    validate_point_cloud(cloud)
    if np.shape(observations.get("B")) != np.shape(cloud["xyz"]):
        raise ValueError("solver B observations must match the reference point cloud")
    joined = {**cloud, **observations}
    return joined


def score(data):
    data = dict(data)
    if "s_reference" not in data and "s" in data:
        data["s_reference"] = data["s"]
    if "s_native" not in data and "vmex_s" in data:
        data["s_native"] = data["vmex_s"]
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
    def rms(values):
        return float(np.sqrt(np.sum(weights*np.sum(values*values, axis=-1))/np.sum(weights)))

    def relative_rms(error, reference):
        reference_rms = rms(reference)
        return rms(error) / reference_rms if reference_rms > 0 else None

    F = np.cross(J, B)-gp
    result = dict(schema=2, evidence="physical_samples_vs_analytic", case=case.name,
                  sample_count=len(xyz), quadrature_volume_m3=float(weights.sum()),
                  field_relative_l2=relative_rms(B-Be, Be),
                  current_relative_l2=relative_rms(J-Je, Je),
                  gradp_relative_l2=relative_rms(gp-gpe, gpe), force_rms_N_m3=rms(F),
                  force_fixed_magnetic_scale=rms(F)/(B0*B0/(mu0*L)),
                  force_pressure_scale=relative_rms(F, gpe),
                  field_max_abs_T=float(np.max(np.linalg.norm(B-Be, axis=-1))))
    if "s_reference" in data:
        s_reference = np.asarray(data["s_reference"], dtype=float)
        if s_reference.shape != (len(xyz),) or not np.isfinite(s_reference).all():
            raise ValueError("s_reference must contain one finite normalized flux label per point")
        target = label_at_s(case, s_reference)
        result["surface_label_max_over_edge"] = float(np.max(abs(label-target))/case.edge)
    if "s_native" in data:
        s_native = np.asarray(data["s_native"], dtype=float)
        if s_native.shape != (len(xyz),) or not np.isfinite(s_native).all():
            raise ValueError("s_native must contain one finite solver flux label per point")
        if "s_reference" in data:
            delta_s = s_native-s_reference
            result["native_flux_label_minus_reference_max_abs"] = float(np.max(abs(delta_s)))
            result["native_flux_label_minus_reference_rms"] = float(np.sqrt(np.mean(delta_s**2)))
            # Historical plot data uses this label. Keep it only as an alias in
            # scores of legacy VMEX bundles; new records use the neutral name.
            if "vmex_s" in data:
                result["vmex_s_minus_reference_max_abs"] = result["native_flux_label_minus_reference_max_abs"]
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    with np.load(sys.argv[1], allow_pickle=False) as data:
        result = score(data)
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
