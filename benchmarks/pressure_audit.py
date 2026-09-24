"""Add pressure-profile and flux-surface error budgets to an R1 measurement."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import vmex
from vmex.core import profiles

from analytic import ROOT, cases
from build_inputs import FIELD_T, MU0
from evidence import sha256_file, source_metadata, write_json
from measurement import compare_pressure_profiles


def _profile(inp, s):
    return np.asarray(profiles.pressure(
        inp.pmass_type, inp.am, inp.am_aux_s, inp.am_aux_f, jnp.asarray(s),
        pres_scale=inp.pres_scale, bloat=inp.bloat, spres_ped=inp.spres_ped,
    ), dtype=float)


def _artifact(run_dir: Path, record: dict, key: str) -> Path:
    metadata = record["artifacts"].get(key)
    if metadata is None:
        raise ValueError(f"measurement has no {key} artifact")
    path = (run_dir/metadata["path"]).resolve()
    if path.parent != run_dir.resolve() or not path.is_file():
        raise ValueError(f"measurement {key} artifact is missing or outside its run directory")
    if sha256_file(path) != metadata["sha256"]:
        raise ValueError(f"measurement {key} artifact hash does not match its record")
    return path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_dir", type=Path)
    parser.add_argument("--expected-vmex", default="b5f5267efc0795c4a49a224e321e9b370975c14c")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    run_dir = args.measurement_dir.resolve()
    measurement_path = run_dir/"measurement.json"
    record = json.loads(measurement_path.read_text(encoding="utf-8"))
    source = source_metadata(vmex.__file__, "uwplasma/vmex", importlib.metadata.version("vmex"))
    if source.get("commit") != args.expected_vmex:
        raise SystemExit(f"wrong VMEX source pin: expected {args.expected_vmex}, loaded {source.get('commit')}")
    if record.get("vmex_source", {}).get("commit") != source.get("commit"):
        raise SystemExit("pressure audit source does not match the measured VMEX source")

    input_path = _artifact(run_dir, record, "input_copy")
    samples_path = _artifact(run_dir, record, "finest_samples")
    manifest_path = ROOT/"inputs/manifest.json"
    expected_manifest_hash = record.get("benchmark_code_sha256", {}).get("inputs/manifest.json")
    if expected_manifest_hash is not None and sha256_file(manifest_path) != expected_manifest_hash:
        raise SystemExit("current input manifest differs from the measurement's recorded manifest")
    input_deck = vmex.VmecInput.from_file(input_path)
    case = cases()[record["case"]]
    pressure_scale = FIELD_T**2/MU0*abs(case.pressure_slope*case.edge)
    with np.load(samples_path, allow_pickle=False) as samples:
        s_native = np.asarray(samples["s_native"])
        s_reference = np.asarray(samples["s_reference"])
        weights = np.asarray(samples["weights"])
    pressure_native = _profile(input_deck, s_native)
    pressure_fit_at_reference = _profile(input_deck, s_reference)
    pressure_exact = (FIELD_T**2/MU0*case.pressure_slope*case.edge*(s_reference-1))
    volume_result = compare_pressure_profiles(
        pressure_native, pressure_fit_at_reference, pressure_exact, weights, pressure_scale)

    target_path = _artifact(run_dir, record, "targeted_samples")
    target_rows = [row for row in record["rows"]
                   if row["grid_id"].startswith("target_") and row.get("status") == "measured"]
    target_results = {}
    with np.load(target_path, allow_pickle=False) as samples:
        for row in target_rows:
            group_id = row["grid_id"].removeprefix("target_")
            keys = {name: f"{group_id}__{name}" for name in
                    ("s_native", "s_reference", "weights")}
            native = np.asarray(samples[keys["s_native"]])
            reference = np.asarray(samples[keys["s_reference"]])
            target_weights = np.asarray(samples[keys["weights"]])
            p_native = _profile(input_deck, native)
            p_fit = _profile(input_deck, reference)
            p_exact = FIELD_T**2/MU0*case.pressure_slope*case.edge*(reference-1)
            metrics = compare_pressure_profiles(
                p_native, p_fit, p_exact, target_weights, pressure_scale)
            nradial = len(row["native_s_targets"])
            angular_count = row["n_theta"]*row["n_phi_one_period"]
            if len(reference) != nradial*angular_count:
                raise ValueError(f"targeted pressure sample shape does not match {group_id}")
            per_surface = np.ptp(p_exact.reshape(nradial, angular_count), axis=1)/pressure_scale
            metrics["exact_pressure_surface_spread_max_fixed_scale"] = float(np.max(per_surface))
            metrics["exact_pressure_surface_spread_by_native_s"] = [
                {"native_s": s_value, "spread_fixed_scale": float(spread)}
                for s_value, spread in zip(row["native_s_targets"], per_surface)
            ]
            target_results[group_id] = metrics

    manifest = json.loads((ROOT/"inputs/manifest.json").read_text(encoding="utf-8"))
    manifest_row = next(row for row in manifest["records"] if row["case"] == case.name)
    output = {
        "schema": 1,
        "run_id": record["run_id"],
        "parent_measurement_sha256": sha256_file(measurement_path),
        "vmex_source": source,
        "pressure_scale_pa": pressure_scale,
        "input_pressure_fit_errors": manifest_row["profile_errors"],
        "full_torus_finest_grid": volume_result,
        "targeted_native_surfaces": target_results,
        "artifacts": {
            "input_copy_sha256": sha256_file(input_path),
            "finest_samples_sha256": sha256_file(samples_path),
            "targeted_samples_sha256": sha256_file(target_path),
        },
        "pressure_audit_code_sha256": sha256_file(Path(__file__)),
    }
    write_json(run_dir/"pressure_audit.json", output, exclusive=True)
    print(json.dumps({"run_id": record["run_id"],
                      "pressure_audit_sha256": sha256_file(run_dir/"pressure_audit.json"),
                      "output": "pressure_audit.json"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
