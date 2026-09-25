"""Run a fixed-boundary VMEX case into a unique, checksummed run directory.

Usage: python benchmarks/run_vmex.py INPUT [NS [NITER [SEED_NPZ]]]
Optional NS/NITER select a single bounded stage. BENCH_FTOL overrides all
effective stage tolerances; BENCH_TCON0 overrides TCON0, with `default` using
the deck value. BENCH_RUN_ID and BENCH_PARENT_RUN_ID provide explicit lineage.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import scipy
import vmex
from vmex.core.solver import SpectralState

from analytic import ROOT, cases
from evidence import (
    acceptance_state, reserve_run_directory, sha256_file, source_metadata,
    write_json,
)
from native_samples import sample_lifted_lasym, sample_native, split_samples
from score_samples import join_samples, score


def _rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value / (1024**2 if platform.system() == "Darwin" else 1024))


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "external-input"


def _finite_float(value, label: str, *, positive: bool = False) -> float:
    number = float(value)
    if not np.isfinite(number) or (number <= 0 if positive else number < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise SystemExit(f"{label} must be finite and {qualifier}")
    return number


def _stage_controls(inp, ns_arg, niter_arg, ftol_env):
    input_ns = np.atleast_1d(np.asarray(inp.ns_array, dtype=np.int64)).ravel()
    input_ftol = np.atleast_1d(np.asarray(inp.ftol_array, dtype=np.float64)).ravel()
    input_niter = np.atleast_1d(np.asarray(inp.niter_array, dtype=np.int64)).ravel()
    if not input_ns.size or not input_ftol.size or not input_niter.size:
        raise ValueError("VMEX input contains an empty NS/FTOL/NITER stage array")
    if ns_arg is not None:
        ns_values = np.asarray([ns_arg], dtype=np.int64)
        ftol_base = input_ftol[-1]
        niter_base = input_niter[-1]
    else:
        ns_values = input_ns
        ftol_base = input_ftol
        niter_base = input_niter
    stage_count = len(ns_values)
    ftol_values = np.resize(np.atleast_1d(ftol_base), stage_count)
    niter_values = np.resize(np.atleast_1d(niter_base), stage_count)
    if ftol_env is not None:
        ftol_values[:] = _finite_float(ftol_env, "BENCH_FTOL", positive=True)
    if niter_arg is not None:
        niter_values[:] = int(niter_arg)
    if np.any(ns_values < 3) or np.any(niter_values < 1) or np.any(ftol_values <= 0):
        raise ValueError("effective NS/FTOL/NITER arrays contain invalid values")
    return ns_values, ftol_values, niter_values


def _load_manifest_case(input_path: Path):
    manifest = input_path.parent / "manifest.json"
    if not manifest.exists():
        raise ValueError("A manifest case is required for native physical sampling")
    rows = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    row = next((item for item in rows if item["file"] == input_path.name), None)
    if row is None:
        raise ValueError("Input file is not listed in its manifest")
    if row.get("boundary_max_error_m", float("inf")) > 1e-6 or row.get(
        "symmetric_omitted_coeff_m", float("inf")
    ) > 1e-10:
        raise ValueError("Input representation unresolved; refine its manifest first")
    return row


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("ns", nargs="?", type=int)
    parser.add_argument("niter", nargs="?", type=int)
    parser.add_argument("seed_npz", nargs="?", type=Path)
    parser.add_argument("--parent-run-id", default=os.environ.get("BENCH_PARENT_RUN_ID"))
    parser.add_argument("--run-id", default=os.environ.get("BENCH_RUN_ID"))
    parser.add_argument("--keep-terminal", action="store_true",
                        help="return and score a final state that reaches NITER")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input does not exist: {args.input}")
    if args.ns is not None and args.ns < 3:
        raise SystemExit("NS override must be at least 3")
    if args.niter is not None and args.niter < 1:
        raise SystemExit("NITER override must be positive")

    manifest_row = _load_manifest_case(input_path)
    inp = vmex.VmecInput.from_file(input_path)
    if inp.lfreeb:
        raise SystemExit("This runner is fixed-boundary only; use the R6 protocol for free boundary")
    ns_values, ftol_values, niter_values = _stage_controls(
        inp, args.ns, args.niter, os.environ.get("BENCH_FTOL"))
    tcon_text = os.environ.get("BENCH_TCON0")
    tcon_requested = None if tcon_text is None else tcon_text.strip()
    if tcon_requested is None or tcon_requested.lower() == "default":
        tcon_override = None
    else:
        tcon_override = _finite_float(tcon_requested, "BENCH_TCON0")
    inp = replace(
        inp,
        ns_array=ns_values,
        ftol_array=ftol_values,
        niter_array=niter_values,
    )
    effective_tcon0 = float(inp.tcon0) if tcon_override is None else tcon_override

    seed = None
    seed_hash = None
    if args.seed_npz is not None:
        seed_path = args.seed_npz.resolve()
        if not seed_path.is_file():
            raise SystemExit("Seed NPZ does not exist")
        with np.load(seed_path, allow_pickle=False) as arrays:
            seed = SpectralState(**{name: arrays[name] for name in
                ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
        seed_hash = sha256_file(seed_path)

    run_id, out = reserve_run_directory(ROOT / "results/vmex/runs", args.run_id)
    input_hash = sha256_file(input_path)
    source = source_metadata(
        vmex.__file__, "uwplasma/vmex", importlib.metadata.version("vmex"))
    requested = {
        "ns": args.ns,
        "niter": args.niter,
        "ftol": os.environ.get("BENCH_FTOL"),
        "tcon0": tcon_requested,
        "keep_terminal": bool(args.keep_terminal or tcon_text is not None),
    }
    effective = {
        "ns_array": [int(value) for value in ns_values],
        "ftol_array": [float(value) for value in ftol_values],
        "niter_array": [int(value) for value in niter_values],
        "tcon0": effective_tcon0,
    }
    controls = {"requested": requested, "effective": effective}
    input_copy = out / "input.indata"
    input_copy.write_bytes(input_path.read_bytes())
    write_json(out / "effective_input.json", {
        "schema": 1,
        "original_input_sha256": input_hash,
        "input_copy": input_copy.name,
        "controls": controls,
    }, exclusive=True)

    start_utc = datetime.now(timezone.utc).isoformat()
    record = {
        "schema": 2,
        "run_id": run_id,
        "parent_run_id": args.parent_run_id,
        "status": "running",
        "evidence": "vmex_fixed_boundary_diagnostic",
        "input_file": _repo_relative(input_path),
        "input_sha256": input_hash,
        "seed_sha256": seed_hash,
        "source": source,
        "controls": controls,
        "start_time_utc": start_utc,
        "environment": {
            "python": sys.version.split()[0],
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "jax": jax.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "devices": [str(device) for device in jax.devices()],
            "host_peak_rss_mib": None,
            "device_memory_peak_mib": None,
        },
        "artifacts": {
            "input_copy": {"path": input_copy.name, "sha256": sha256_file(input_copy)},
        },
        "representation_resolved": True,
        "measurement_resolved": False,
        "root_certified": False,
        "accepted": False,
    }
    write_json(out / "forward.json", record, exclusive=True)

    solve_start = perf_counter()
    try:
        result = vmex.solve_multigrid(
            inp,
            ns_array=ns_values,
            ftol_array=ftol_values,
            niter_array=niter_values,
            initial_state=seed,
            verbose=True,
            tcon0=tcon_override,
            raise_on_max_iterations=not requested["keep_terminal"],
        )
        jax.block_until_ready(result.state)
    except Exception as exc:
        record.update(
            status="solver_failed",
            solve_seconds=perf_counter() - solve_start,
            end_time_utc=datetime.now(timezone.utc).isoformat(),
            environment={**record["environment"], "host_peak_rss_mib": _rss_mib()},
            error_type=type(exc).__name__,
            error=str(exc),
            **acceptance_state(
                solver_converged=False,
                pointwise_thresholds_met=False,
                measurement_resolved=False,
                representation_resolved=record["representation_resolved"],
                root_certified=False,
            ),
        )
        write_json(out / "forward.json", record)
        raise SystemExit(f"VMEX run {run_id} failed; inspect its forward.json") from exc

    solve_seconds = perf_counter() - solve_start
    wout_path = out / "wout.nc"
    wout = vmex.wout_from_result(inp, result)
    vmex.write_wout(wout_path, wout)
    record.update(
        status="solver_converged" if bool(result.converged) else "solver_capped",
        solve_seconds=solve_seconds,
        solver_converged=bool(result.converged),
        iterations=int(result.iterations),
        ier_flag=int(result.ier_flag),
        fsqr=float(np.asarray(result.fsqr)),
        fsqz=float(np.asarray(result.fsqz)),
        fsql=float(np.asarray(result.fsql)),
        scalars={key: float(np.asarray(getattr(wout, key)))
                 for key in ("wb", "volume_p", "betatotal")},
        end_time_utc=datetime.now(timezone.utc).isoformat(),
        artifacts={**record["artifacts"], "wout": {
            "path": wout_path.name, "sha256": sha256_file(wout_path)}},
    )
    write_json(out / "forward.json", record)

    sample_start = perf_counter()
    try:
        case_name = manifest_row["case"]
        if inp.lasym:
            samples = sample_lifted_lasym(inp, result.state, cases()[case_name])
            field_path = "continuous_fitted_state_lasym"
        else:
            samples = sample_native(inp, result.state, cases()[case_name], chunk_size=32)
            field_path = "native_clebsch_cartesian"
        cloud, observations = split_samples(samples)
        point_cloud_path = out / "point_cloud.npz"
        observations_path = out / "vmex_observations.npz"
        np.savez_compressed(point_cloud_path, **cloud)
        np.savez_compressed(observations_path, **observations)
        merged = join_samples(cloud, observations)
        scores = score(merged)
    except Exception as exc:
        record.update(
            status="postprocess_failed",
            error_type=type(exc).__name__, error=str(exc),
            environment={**record["environment"], "host_peak_rss_mib": _rss_mib()},
            end_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        write_json(out / "forward.json", record)
        raise SystemExit(f"VMEX solve completed but sampling failed for run {run_id}") from exc

    thresholds = {
        "field_relative_l2": 1e-5,
        "current_relative_l2": 1e-3,
        "force_pressure_scale": 1e-3,
    }
    pointwise = all(scores[key] <= limit for key, limit in thresholds.items())
    gate_state = acceptance_state(
        solver_converged=bool(result.converged),
        pointwise_thresholds_met=pointwise,
        measurement_resolved=False,
        representation_resolved=True,
        root_certified=False,
    )
    score_path = out / "native_scores.json"
    write_json(score_path, {
        **scores,
        "thresholds": thresholds,
        "pointwise_thresholds_met": pointwise,
        "measurement_resolved": False,
        "root_certified": False,
    }, exclusive=True)
    record.update(
        status="scored_diagnostic",
        field_path=field_path,
        native_sample_and_score_seconds=perf_counter() - sample_start,
        environment={**record["environment"], "host_peak_rss_mib": _rss_mib()},
        pointwise_thresholds_met=pointwise,
        measurement_resolved=False,
        root_certified=False,
        accepted=gate_state["accepted"],
        native_score=scores,
        artifacts={**record["artifacts"],
            "point_cloud": {"path": point_cloud_path.name, "sha256": sha256_file(point_cloud_path)},
            "solver_observations": {"path": observations_path.name, "sha256": sha256_file(observations_path)},
            "native_scores": {"path": score_path.name, "sha256": sha256_file(score_path)}},
    )
    write_json(out / "forward.json", record)
    print(json.dumps({"run_id": run_id, "status": record["status"],
                      "accepted": record["accepted"], "scores": scores}, indent=2), flush=True)


if __name__ == "__main__":
    main()
