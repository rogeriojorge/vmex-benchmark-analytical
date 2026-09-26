"""Small immutable-run and source-provenance helpers for benchmark records."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

try:  # Python >= 3.11; the DESC environment is 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None



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
    if project_metadata.is_file() and tomllib is not None:
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



def capture_execution_source(output_dir: Path, files: list[str | Path], *,
                             packages: dict[str, str | Path] | None = None) -> dict:
    """Copy the exact runtime bytes of ``files`` into ``output_dir`` before a run.

    Also records the benchmark tree's commit, tracked-diff hash and the source
    identity of each imported package (``name -> module file``).  Write this
    before any expensive computation so an interrupted run still identifies the
    code it executed.
    """
    output_dir = Path(output_dir)
    snapshot = output_dir/"execution_source"
    snapshot.mkdir(exist_ok=False)
    rows = []
    for path in files:
        path = Path(path).resolve()
        data = path.read_bytes()
        target = snapshot/path.name
        if target.exists():
            raise FileExistsError(f"duplicate source basename {path.name}")
        target.write_bytes(data)
        rows.append({"name": path.name, "sha256": hashlib.sha256(data).hexdigest(),
                     "bytes": len(data)})
    benchmark = source_metadata(Path(__file__), "rogeriojorge/vmex-benchmark-analytical", None)
    record = {
        "schema": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "files": rows,
        "benchmark_tree": benchmark,
        "packages": {name: source_metadata(module, name, None)
                     for name, module in (packages or {}).items()},
    }
    write_json(output_dir/"execution_source.json", record, exclusive=True)
    return record

