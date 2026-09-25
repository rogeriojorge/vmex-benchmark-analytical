"""Persist an exact physical force-error decomposition for a scored VMEX run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np

from analytic import cases, differential_fields
from evidence import sha256_file, write_json
from measurement import decompose_physical_force_error


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser


def _rms(values, weights):
    return float(np.sqrt(np.sum(weights*np.sum(values*values, axis=-1))/np.sum(weights)))


def _weighted_product(left, right, weights):
    return float(np.sum(weights*np.sum(left*right, axis=-1)))


def main(argv=None):
    args = _parser().parse_args(argv)
    run_dir = args.run_dir
    report_path = run_dir/"forward.json"
    cloud_path = run_dir/"point_cloud.npz"
    observations_path = run_dir/"vmex_observations.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != 2:
        raise SystemExit("physical decomposition requires a schema-2 forward report")
    if report.get("status") not in {"scored_diagnostic", "solver_converged", "solver_capped"}:
        raise SystemExit("forward report does not contain a completed solver state")
    for name, path in (("point_cloud", cloud_path), ("solver_observations", observations_path)):
        if report.get("artifacts", {}).get(name, {}).get("sha256") != sha256_file(path):
            raise SystemExit(f"{name} artifact hash does not match the forward report")
    with np.load(cloud_path, allow_pickle=False) as data:
        cloud = {key: np.asarray(data[key]) for key in data.files}
    with np.load(observations_path, allow_pickle=False) as data:
        observations = {key: np.asarray(data[key]) for key in data.files}
    case = cases()[str(cloud["case_name"].item())]
    L, B0, mu0 = (float(cloud[key]) for key in ("length_m", "field_t", "mu0"))
    exact_B, exact_J, exact_gradp, _, _ = differential_fields(case, cloud["xyz"]/L)
    B_reference = B0*exact_B
    J_reference = B0/(mu0*L)*exact_J
    gradp_reference = B0**2/(mu0*L)*exact_gradp
    decomposition = decompose_physical_force_error(
        observations["B"], observations["J"], observations["gradp"],
        B_reference, J_reference, gradp_reference,
    )
    weights = np.asarray(cloud["weights"], dtype=float)
    total = decomposition["total_force_error"]
    force_scale = _rms(gradp_reference, weights)
    terms = ("dJ_cross_B_reference", "J_reference_cross_dB", "dJ_cross_dB", "minus_dgradp")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    vectors_path = output_dir/"force_decomposition.npz"
    with vectors_path.open("xb") as output:
        np.savez_compressed(output, xyz=cloud["xyz"], weights=weights,
                            B_vmex=observations["B"], J_vmex=observations["J"],
                            gradp_vmex=observations["gradp"], B_reference=B_reference,
                            J_reference=J_reference, gradp_reference=gradp_reference,
                            **decomposition)
    vectors_sha = sha256_file(vectors_path)
    source_file = Path(__file__)
    measurement_file = Path(__file__).with_name("measurement.py")
    analytic_file = Path(__file__).with_name("analytic.py")
    code_hashes = {
        "benchmarks/decompose_physical_force.py": sha256_file(source_file),
        "benchmarks/measurement.py": sha256_file(measurement_file),
        "benchmarks/analytic.py": sha256_file(analytic_file),
    }
    record = {
        "schema": 1,
        "status": "diagnostic_physical_force_decomposition",
        "case": case.name,
        "run_id": report["run_id"],
        "solver_status": report["status"],
        "solver_converged": report.get("solver_converged"),
        "vmex_source": report["source"],
        "forward_report_sha256": sha256_file(report_path),
        "point_cloud_sha256": sha256_file(cloud_path),
        "solver_observations_sha256": sha256_file(observations_path),
        "input_sha256": report.get("input_sha256"),
        "seed_sha256": report.get("seed_sha256"),
        "sample_count": len(weights),
        "sample_weight_sum_m3": float(np.sum(weights)),
        "evidence_domain": "96-point native field sample with saved physical quadrature weights",
        "reference_field": "independent analytical Cartesian B/J/gradp at identical positions",
        "force_terms": {
            name: {"weighted_rms_N_per_m3": _rms(decomposition[name], weights),
                  "weighted_rms_over_reference_gradp": _rms(decomposition[name], weights)/force_scale}
            for name in terms
        },
        "total_force_error": {
            "weighted_rms_N_per_m3": _rms(total, weights),
            "weighted_rms_over_reference_gradp": _rms(total, weights)/force_scale,
        },
        "weighted_mean_term_inner_products_N2_per_m6": {
            f"{left}__dot__{right}": _weighted_product(
                decomposition[left], decomposition[right], weights)/np.sum(weights)
            for i, left in enumerate(terms) for right in terms[i+1:]
        },
        "identity_defect_max_abs_N_per_m3": float(
            np.max(np.abs(decomposition["identity_defect"]))),
        "benchmark_code_sha256": code_hashes,
        "artifacts": {"vectors": {"path": vectors_path.name, "sha256": vectors_sha}},
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir/"decomposition.json", record, exclusive=True)
    record_sha = sha256_file(output_dir/"decomposition.json")
    print(json.dumps({"status": record["status"], "record_sha256": record_sha,
                      "vectors_sha256": vectors_sha,
                      "total_force_error": record["total_force_error"],
                      "identity_defect_max_abs_N_per_m3":
                          record["identity_defect_max_abs_N_per_m3"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
