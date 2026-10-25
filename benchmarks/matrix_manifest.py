"""Copy the small records of matrix runs into the repository and write the WOUT manifest.

For every run directory: receipt.json, volume_score_v1.json, fsq_history.json, the deck and the stdout log
are copied; the WOUT stays out of git and is listed in a manifest with its sha256, size and the office path
it was produced at (``~/vab-matrixG/<office-dir>/<run>/``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

KEEP = ("receipt.json", "volume_score_v1.json", "fsq_history.json", "stdout.log")


def publish(source, dest, office_dir):
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for run in sorted(p for p in Path(source).iterdir() if (p/"receipt.json").exists()):
        target = dest/run.name
        target.mkdir(exist_ok=True)
        for name in (*KEEP, *(p.name for p in run.glob("input.*"))):
            if (run/name).exists():
                shutil.copy2(run/name, target/name)
        for wout in run.glob("wout_*.nc"):
            rec = json.loads((run/"receipt.json").read_text())
            manifest.append({"run_id": run.name, "file": wout.name, "bytes": wout.stat().st_size,
                             "sha256": (rec.get("wout") or {}).get("sha256"),
                             "office_path": f"~/vab-matrixG/{office_dir}/{run.name}/{wout.name}"})
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--office-dir", required=True, help="runs or runs_wp")
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args(argv)
    manifest = publish(a.source, a.dest, a.office_dir)
    a.manifest.write_text(json.dumps({"schema": 1, "note": "WOUTs are not committed; re-fetch from office or re-run", "files": manifest}, indent=1)+"\n")
    print(len(manifest), "WOUTs listed")


if __name__ == "__main__":
    main()
