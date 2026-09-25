"""Small immutable-run and source-provenance helpers for benchmark records."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib
import uuid

import numpy as np


SOLVER_SAMPLE_FIELDS = frozenset({
    "B", "J", "gradp", "s_native", "vmex_s", "native_score",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_solver_report(report: dict) -> dict:
    """Normalize supported legacy/schema-2 solver reports without guessing identity."""
    if not isinstance(report, dict):
        raise ValueError("solver report must be a JSON object")
    schema = report.get("schema", 1)
    if isinstance(schema, bool) or schema not in {1, 2}:
        raise ValueError(f"unsupported solver report schema: {schema!r}")
    if schema == 1:
        source_commit = report.get("vmex_commit")
        version = report.get("vmex_version")
        converged = report.get("converged")
        if converged is not None and not isinstance(converged, bool):
            raise ValueError("schema-1 converged must be a boolean or null")
        old_status = report.get("status")
        if old_status is not None and not isinstance(old_status, str):
            raise ValueError("schema-1 status must be a string or null")
        if old_status in {"failed", "solver_failed"} and converged is True:
            raise ValueError("schema-1 failed status contradicts converged")
        if old_status in {"failed", "solver_failed"}:
            status = "solver_failed"
        elif converged is True:
            status = "solver_converged"
        elif converged is False:
            iterations, limit = report.get("iterations"), report.get("niter_limit")
            status = ("solver_capped" if isinstance(iterations, (int, float)) and
                      isinstance(limit, (int, float)) and iterations >= limit
                      else "solver_not_converged")
        else:
            status = old_status if isinstance(old_status, str) else "unknown"
        requested = {
            key: report.get(key) for key in
            ("ns_override", "niter_limit", "ftol", "tcon0_requested", "initialization")
            if key in report
        }
        effective = {
            key: report.get(key) for key in
            ("ns_override", "niter_limit", "ftol", "tcon0_effective")
            if key in report
        }
        artifacts = {
            key: report[key] for key in
            ("seed_sha256", "native_sample_sha256")
            if isinstance(report.get(key), str)
        }
        iterations = report.get("iterations")
        tracked_tree_clean = None
        tracked_diff_sha256 = None
        timing_seconds = report.get("solve_seconds_including_first_compile")
        host_peak_rss_mib = report.get("peak_rss_mib")
        device_memory_peak_mib = None
    elif schema == 2:
        source = report.get("source")
        controls = report.get("controls")
        if not isinstance(source, dict) or not isinstance(controls, dict):
            raise ValueError("schema-2 report requires source and controls objects")
        source_commit = source.get("commit")
        version = source.get("version")
        tracked_tree_clean = source.get("tracked_tree_clean")
        tracked_diff_sha256 = source.get("tracked_diff_sha256")
        if tracked_tree_clean is not None and not isinstance(tracked_tree_clean, bool):
            raise ValueError("schema-2 tracked_tree_clean must be boolean or null")
        if tracked_diff_sha256 is not None and not isinstance(tracked_diff_sha256, str):
            raise ValueError("schema-2 tracked_diff_sha256 must be a string or null")
        requested = controls.get("requested", {})
        effective = controls.get("effective", {})
        if not isinstance(requested, dict) or not isinstance(effective, dict):
            raise ValueError("schema-2 requested/effective controls must be objects")
        status = report.get("status", "unknown")
        if not isinstance(status, str):
            raise ValueError("schema-2 status must be a string")
        converged = report.get("solver_converged")
        if converged is not None and not isinstance(converged, bool):
            raise ValueError("schema-2 solver_converged must be a boolean or null")
        if status == "solver_converged" and converged is not True:
            raise ValueError("schema-2 converged status contradicts solver_converged")
        if status in {"solver_capped", "solver_failed"} and converged is not False:
            raise ValueError("schema-2 failed/capped status contradicts solver_converged")
        raw_artifacts = report.get("artifacts", {})
        if not isinstance(raw_artifacts, dict):
            raise ValueError("schema-2 artifacts must be an object")
        artifacts = {}
        for name, item in raw_artifacts.items():
            if isinstance(item, dict) and isinstance(item.get("sha256"), str):
                artifacts[name] = item["sha256"]
        iterations = report.get("iterations")
        timing_seconds = report.get("solve_seconds")
        environment = report.get("environment", {})
        if not isinstance(environment, dict):
            raise ValueError("schema-2 environment must be an object")
        host_peak_rss_mib = environment.get("host_peak_rss_mib")
        device_memory_peak_mib = environment.get("device_memory_peak_mib")
    else:
        raise ValueError(f"unsupported solver report schema: {schema!r}")
    if source_commit is not None and not isinstance(source_commit, str):
        raise ValueError("solver source commit must be a string or null")
    if source_commit == "":
        raise ValueError("solver source commit must not be empty")
    return {
        "schema": schema,
        "source_commit": source_commit,
        "version": version,
        "tracked_tree_clean": tracked_tree_clean,
        "tracked_diff_sha256": tracked_diff_sha256,
        "status": status,
        "solver_converged": converged,
        "controls": {"requested": requested, "effective": effective},
        "artifacts_sha256": artifacts,
        "iterations": iterations,
        "solve_seconds": timing_seconds,
        "host_peak_rss_mib": host_peak_rss_mib,
        "device_memory_peak_mib": device_memory_peak_mib,
    }


def parent_comparison_rows(output_parent: Path, parent_report: dict,
                           expected: tuple[str, str, str, str]) -> list[dict]:
    """Load and validate immutable refinement ancestry for a child profile."""
    chain = []
    seen = set()
    report = parent_report
    while True:
        run_id = report.get("run_id")
        identity = (
            report.get("vmex_source", {}).get("commit"),
            report.get("case"),
            report.get("state_source_sha256"),
            report.get("input_sha256"),
        )
        if run_id in seen or identity != expected:
            raise SystemExit("parent comparison ancestry is cyclic or belongs to another state")
        seen.add(run_id)
        if report.get("status") not in {"measurement_diagnostic", "measurement_resolved"}:
            raise SystemExit("parent comparison ancestry contains an incomplete measurement")
        chain.append(report.get("rows", []))
        link = report.get("comparison_parent")
        if not link:
            break
        ancestor_path = Path(output_parent)/link["run_id"]/"measurement.json"
        if not ancestor_path.is_file() or sha256_file(ancestor_path) != link.get("report_sha256"):
            raise SystemExit("parent comparison ancestry report hash does not match")
        ancestor = json.loads(ancestor_path.read_text(encoding="utf-8"))
        if ancestor.get("run_id") != link["run_id"]:
            raise SystemExit("parent comparison ancestry run id does not match")
        report = ancestor
    return [row for rows in reversed(chain) for row in rows]


def checkpoint_grid(output_dir: Path, grid_id: str, row: dict) -> dict:
    """Persist one immutable completed grid and atomically update its receipt."""
    if not grid_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in grid_id):
        raise ValueError("grid_id may contain only letters, digits, dash, and underscore")
    directory = Path(output_dir)/"grid_checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory/f"{grid_id}.json"
    write_json(record_path, row, exclusive=True)
    manifest_path = directory/"checkpoints.json"
    entries = []
    if manifest_path.is_file():
        entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("grids", [])
    entries.append({"grid_id": grid_id, "path": record_path.name,
                    "sha256": sha256_file(record_path)})
    write_json(manifest_path, {"schema": 1, "grids": entries})
    return entries[-1]


def interruption_receipt(output_dir: Path, *, run_id: str, completed_grids: list[str],
                         error: BaseException) -> dict:
    """Write a resumable receipt while leaving the interrupted run incomplete."""
    receipt = {
        "schema": 1,
        "run_id": run_id,
        "status": "interrupted",
        "completed_grids": list(completed_grids),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    write_json(Path(output_dir)/"interruption.json", receipt)
    return receipt


def reserve_run_directory(parent: Path, run_id: str | None = None) -> tuple[str, Path]:
    """Reserve a never-reused run directory; fail on collisions."""
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_id = f"{stamp}-{uuid.uuid4().hex[:12]}"
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ ." for ch in run_id):
        raise ValueError("run_id may contain only letters, digits, spaces, dot, dash, and underscore")
    destination = Path(parent) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return run_id, destination


def write_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    """Write valid JSON atomically. Exclusive mode refuses to replace a file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with target.open("x", encoding="utf-8") as output:
            output.write(payload)
        return
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def source_metadata(module_file: str | Path, package: str, version: str | None) -> dict:
    """Identify imported source without recording its private local path."""
    source = Path(module_file).resolve()
    base = {
        "repository": package,
        "version": version,
        "distribution_version": version,
        "version_source": "installed_distribution",
        "commit": None,
        "tracked_tree_clean": None,
        "tracked_diff_sha256": None,
    }
    root = subprocess.run(
        ["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"],
        text=True, capture_output=True,
    )
    if root.returncode:
        return base
    project_metadata = Path(root.stdout.strip())/"pyproject.toml"
    if project_metadata.is_file():
        try:
            project = tomllib.loads(project_metadata.read_text(encoding="utf-8")).get("project", {})
            source_version = project.get("version")
        except (OSError, tomllib.TOMLDecodeError):
            source_version = None
        if isinstance(source_version, str):
            base["version"] = source_version
            base["version_source"] = "checkout_pyproject"
    commit = subprocess.run(
        ["git", "-C", root.stdout.strip(), "rev-parse", "HEAD"],
        text=True, capture_output=True,
    )
    status = subprocess.run(
        ["git", "-C", root.stdout.strip(), "status", "--porcelain", "--untracked-files=no"],
        text=True, capture_output=True,
    )
    diff = subprocess.run(
        ["git", "-C", root.stdout.strip(), "diff", "--binary", "HEAD"],
        capture_output=True,
    )
    base.update(
        commit=commit.stdout.strip() if commit.returncode == 0 else None,
        tracked_tree_clean=(status.stdout == "" if status.returncode == 0 else None),
        tracked_diff_sha256=(hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None),
    )
    return base


def controls_record(requested: dict, effective: dict) -> dict:
    """Keep caller intent next to the arrays and settings VMEX actually receives."""
    return {"requested": requested, "effective": effective}


def single_source(records: list[dict], field: str = "source_commit") -> str:
    """Return the common nonempty source id or reject an ambiguous pool."""
    values = {row.get(field) for row in records}
    if not records or None in values or "" in values or len(values) != 1:
        raise ValueError(f"records do not have one complete {field}: {sorted(map(str, values))}")
    return str(next(iter(values)))


def validate_point_cloud(point_cloud: dict) -> None:
    """Check that reference positions carry no solver-specific observations."""
    contaminated = SOLVER_SAMPLE_FIELDS.intersection(point_cloud)
    if contaminated:
        raise ValueError(f"reference point cloud contains solver fields: {sorted(contaminated)}")
    if "s" in point_cloud or "vmex_s" in point_cloud:
        raise ValueError("use explicit s_reference in the solver-independent point cloud")
    if "s_reference" not in point_cloud:
        raise ValueError("reference point cloud requires s_reference")


def desc_native_flux_label(rho: np.ndarray) -> np.ndarray:
    """DESC's normalized radial flux label, computed from its own rho coordinate."""
    rho = np.asarray(rho, dtype=float)
    if not np.isfinite(rho).all() or np.any(rho < 0):
        raise ValueError("DESC rho must be finite and nonnegative")
    return rho**2


def acceptance_state(
    *, solver_converged: bool,
    pointwise_thresholds_met: bool,
    measurement_resolved: bool,
    representation_resolved: bool,
    root_certified: bool,
) -> dict:
    gates = {
        "solver_converged": bool(solver_converged),
        "pointwise_thresholds_met": bool(pointwise_thresholds_met),
        "measurement_resolved": bool(measurement_resolved),
        "representation_resolved": bool(representation_resolved),
        "root_certified": bool(root_certified),
    }
    return {**gates, "accepted": all(gates.values())}


def failure_record(*, run_id: str, error: BaseException,
                   peak_rss_mib: float | None, requested: dict,
                   effective: dict, source: dict) -> dict:
    """Create finite failure metadata; missing memory remains explicitly unknown."""
    if peak_rss_mib is not None and (not np.isfinite(peak_rss_mib) or peak_rss_mib < 0):
        raise ValueError("peak_rss_mib must be a finite nonnegative value or None")
    return {
        "schema": 2,
        "run_id": run_id,
        "status": "solver_failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "host_peak_rss_mib": peak_rss_mib,
        "device_memory_peak_mib": None,
        "source": source,
        "controls": controls_record(requested, effective),
        **acceptance_state(
            solver_converged=False,
            pointwise_thresholds_met=False,
            measurement_resolved=False,
            representation_resolved=False,
            root_certified=False,
        ),
    }
