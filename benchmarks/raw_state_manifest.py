"""Consolidated manifest of every raw state (WOUT, .h5, .npz) behind the result index.

Merges the per-track manifests (matrix, work-precision, cross-build, DESC, Lopez, iotaF, mechanism), adds the committed
.npz states and the DESC equilibria that only the records name, and (with ``--office``) checks every office path with one
ssh call: existence, size and sha256 against the recorded value.  Writes results/raw_state_manifest.json.

    python benchmarks/raw_state_manifest.py --office      # rebuild and verify on the office box (reads files, writes nothing there)
    python benchmarks/raw_state_manifest.py --check       # fail if the committed manifest no longer matches the index/sources
    python benchmarks/raw_state_manifest.py --pack-list   # office paths of the research release tarball (see RELEASE.md); committed .npz stay in git
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"results/raw_state_manifest.json"
OFFICE_HOME = "/home/rjorge"
MAC = "Rogerio-macmini-2025"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def office_abs(path: str) -> str:
    return path.replace("~", OFFICE_HOME, 1) if path.startswith("~") else path


def load(rel):
    return json.loads((ROOT/rel).read_text())


def sources(index):
    """Yield dicts: name, kind, sha256, bytes, office_path|local_path|None, manifest, record_key, index_keys."""
    owners = {}
    for key, rec in index["records"].items():
        raw = rec.get("raw", {})
        for m in ([raw["manifest"]] if isinstance(raw.get("manifest"), str) else raw.get("manifest", [])):
            owners.setdefault(m, []).append(key)
    rows = []

    def add(manifest, name, sha, size, office=None, local=None, note=None):
        rows.append(dict(name=name, kind=Path(name).suffix.lstrip("."), sha256=sha, bytes=size, office_path=office, local_path=local,
                         manifest=manifest, index_keys=owners.get(manifest, []), note=note))

    for manifest in sorted(owners):
        data = load(manifest)
        if manifest.endswith("wout_lines.json"):
            continue  # copies of iotaF WOUTs, matched by sha below
        if "files" in data:
            for f in data["files"]:
                if "office_path" in f:
                    where = f["office_path"]
                    if "local Mac" in where:
                        add(manifest, f["file"], f["sha256"], f.get("bytes"), note="macOS arm64 scratch of the September/October campaign; not archived, regenerate")
                    else:
                        add(manifest, f["file"], f["sha256"], f.get("bytes"), office=office_abs(where))
                else:
                    add(manifest, f["file"], f["sha256"], f.get("bytes"), note=f"not archived; regenerate: {f.get('regenerate')}")
        elif "wouts" in data:
            for f in data["wouts"]:
                add(manifest, Path(f["office_path"]).name, f["sha256"], f.get("bytes"), office=f["office_path"])
        elif "states" in data:
            for f in data["states"]:
                add(manifest, Path(f["office_path"]).name, f["sha256"], f.get("bytes"), office=f["office_path"])
    return rows


def committed_npz(index):
    rows = []
    for key, rec in index["records"].items():
        pattern = rec.get("raw", {}).get("glob")
        if pattern:
            for p in sorted(ROOT.glob(pattern)):
                rows.append(dict(name=p.name, kind="npz", sha256=sha256_file(p), bytes=p.stat().st_size, office_path=None,
                                 local_path=str(p.relative_to(ROOT)), manifest=f"glob:{pattern}", index_keys=[key], note="committed in git"))
    return rows


def desc_extra(index):
    """DESC equilibria named only by their records (general ladders, published protocol)."""
    rows = []
    for p in sorted((ROOT/"results/desc/general").glob("*/record_v2.json")):
        rec = json.loads(p.read_text())
        art = rec["rescored_artifact"]
        case = p.parent.name
        rows.append(dict(name=art["path"], kind="h5", sha256=art["sha256"], bytes=None,
                         office_path=f"{OFFICE_HOME}/vab-c26-rerun/results/desc/general/{case}/{art['path']}", local_path=None,
                         manifest=str(p.relative_to(ROOT)), index_keys=["rec.desc.general"], note="rescored artifact of record_v2.json"))
    sha = load("results/desc/published_protocol/sheared_A/record.json")["artifacts"]["landreman_sheared_desc.h5"]
    rows.append(dict(name="landreman_sheared_desc.h5", kind="h5", sha256=sha, bytes=None,
                     office_path=f"{OFFICE_HOME}/vab-shearD/run_default/landreman_sheared_desc.h5", local_path=None,
                     manifest="results/desc/published_protocol/sheared_A/record.json", index_keys=["rec.desc.published"], note="published sheared-DESC driver output"))
    return rows


def office_scan(paths, extra_dirs):
    """One ssh call: size and sha256 of every given path, and of *.nc/*.h5 under extra_dirs."""
    dirs = " ".join(extra_dirs)
    script = ('while IFS= read -r p; do if [ -f "$p" ]; then echo "$p $(stat -c %s "$p") $(sha256sum "$p" | cut -d" " -f1)"; '
              'else echo "$p MISSING"; fi; done; for d in DIRS; do find "$d" -name "*.nc" -o -name "*.h5" 2>/dev/null | '
              'while IFS= read -r p; do echo "$p $(stat -c %s "$p") $(sha256sum "$p" | cut -d" " -f1)"; done; done').replace("DIRS", dirs)
    result = subprocess.run(["ssh", "office", "nice -n 10 bash -c " + shlex.quote(script)], input="\n".join(paths)+"\n", capture_output=True, text=True, check=True)
    found = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        found[parts[0]] = None if parts[1] == "MISSING" else (int(parts[1]), parts[2])
    return found


def build(office: bool):
    index = load("results/index.json")
    rows = sources(index) + committed_npz(index) + desc_extra(index)
    scan = {}
    if office:
        paths = sorted({r["office_path"] for r in rows if r["office_path"]})
        scan = office_scan(paths, [f"{OFFICE_HOME}/vab-mechK", f"{OFFICE_HOME}/vab-iotaF", f"{OFFICE_HOME}/vab-lopezE",
                                   f"{OFFICE_HOME}/vab-c26-rerun", f"{OFFICE_HOME}/vab-c26", f"{OFFICE_HOME}/vab-desc"])
        by_sha = {}
        for p, v in scan.items():
            if v:
                by_sha.setdefault(v[1], []).append(p)
        for r in rows:
            if r["office_path"]:
                v = scan.get(r["office_path"])
                if v is None and r["sha256"] in by_sha:
                    r["office_path"] = by_sha[r["sha256"]][0]
                    r["bytes"] = r["bytes"] or scan[r["office_path"]][0]
                    r["office_status"] = "ok (found by sha256)"
                elif v is None:
                    r["office_status"] = "missing"
                else:
                    r["bytes"] = r["bytes"] or v[0]
                    r["office_status"] = "ok" if v[1] == r["sha256"] else f"sha256 differs ({v[1]})"
            elif r["local_path"] is None and r["sha256"] in by_sha:
                r["office_path"] = by_sha[r["sha256"]][0]
                r["bytes"] = r["bytes"] or scan[r["office_path"]][0]
                r["office_status"] = "ok (found by sha256)"
                r["note"] = None
            elif r["local_path"] is None:
                r["office_status"] = "not on office"
    else:
        old = {(r["name"], r["sha256"], r["manifest"]): r for r in json.loads(OUT.read_text())["states"]} if OUT.exists() else {}
        for r in rows:
            prev = old.get((r["name"], r["sha256"], r["manifest"]))
            if prev:
                for k in ("office_path", "office_status", "bytes", "note"):
                    r[k] = prev.get(k, r.get(k))
    rows.sort(key=lambda r: (r["manifest"], r["name"]))
    seen = set()
    states = []
    for r in rows:  # one entry per (name, sha256, manifest, location)
        k = (r["name"], r["sha256"], r["manifest"], r["office_path"], r["local_path"])
        if k not in seen:
            seen.add(k)
            states.append(r)
    total = sum(r["bytes"] or 0 for r in states)
    summary = {"states": len(states), "bytes": total,
               "office_ok": sum(str(r.get("office_status", "")).startswith("ok") for r in states),
               "committed_in_git": sum(r["local_path"] is not None for r in states),
               "not_archived": sum(r.get("office_status") in ("not on office",) for r in states)}
    return {"schema": 1, "hosts": {"office": "pop-os, ~/vab-* trees (read-only for this manifest)", "mac": MAC},
            "verified_on_office": office, "summary": summary,
            "note": "sha256 recorded when each state was written; office_status compares the file now on disk. 'not on office' states were "
                    "not archived and are regenerated with the command in the owning index record.",
            "states": states}


def pack_list(manifest):
    return sorted({r["office_path"] for r in manifest["states"] if r.get("office_status", "").startswith("ok") and r["office_path"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--office", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--pack-list", action="store_true")
    args = parser.parse_args(argv)
    if args.pack_list:
        print("\n".join(pack_list(json.loads(OUT.read_text()))))
        return 0
    manifest = build(args.office)
    text = json.dumps(manifest, indent=1, allow_nan=False)+"\n"
    if args.check:
        key = lambda states: {(r["name"], r["sha256"], r["manifest"], tuple(r["index_keys"])) for r in states}  # noqa: E731
        stale = not OUT.exists() or key(json.loads(OUT.read_text())["states"]) != key(manifest["states"])
        if stale:
            print("results/raw_state_manifest.json is out of date", file=sys.stderr)
        return int(stale)
    OUT.write_text(text)
    print(json.dumps(manifest["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
