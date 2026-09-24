"""Small immutable-run and source-provenance helpers for benchmark records."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
