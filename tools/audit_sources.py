"""Inventory local source checkouts; inventory is not semantic code review.

Usage: python tools/audit_sources.py /absolute/path/to/checkouts
Checkout directory names are in sources.json. This script never clones,
installs, modifies a checkout, or marks a file reviewed. Complete the review
ledger required by plan.md after reading source, tests and contracts.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if len(sys.argv) != 2:
    raise SystemExit(__doc__)
base = Path(sys.argv[1]).resolve()
manifest = json.loads((ROOT / "sources.json").read_text())
result = {"schema": 1, "inventory_only": True, "repositories": [], "files": []}
failed_required = False
for spec in manifest["repositories"]:
    repo = base/spec["name"]
    if not (repo / ".git").exists():
        result["repositories"].append({**spec, "status": "missing"})
        failed_required |= spec["required"]
        continue
    git = lambda *args: subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    sha = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    pinned = spec["pin"] is not None and sha == spec["pin"]
    result["repositories"].append({**spec, "actual_sha": sha, "dirty": dirty,
                                  "pin_matches": pinned, "status": "inventoried_not_reviewed"})
    failed_required |= spec["required"] and (dirty or not pinned)
    paths = subprocess.check_output(["git", "-C", str(repo), "ls-files", "-z"]).decode().split("\0")
    for name in paths:
        file = repo/name
        if not name or file.suffix not in {".py", ".md", ".rst", ".toml", ".yml", ".yaml"}:
            continue
        raw = file.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        row = dict(repository=spec["repo"], commit=sha, path=name,
                   sha256=hashlib.sha256(raw).hexdigest(), lines=len(text.splitlines()),
                   review_status="unreviewed", functions=[], imports=[])
        if file.suffix == ".py":
            try:
                tree = ast.parse(text)
                row["functions"] = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                row["imports"] = sorted({n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                                        {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
            except SyntaxError as exc:
                row["parse_error"] = str(exc)
        result["files"].append(row)
out = ROOT / "results/audit"
out.mkdir(parents=True, exist_ok=True)
(out / "inventory.json").write_text(json.dumps(result, indent=2)+"\n")
print(f"Inventoried {len(result['files'])} tracked text/source files; no file was marked reviewed.")
if failed_required:
    raise SystemExit("Required sources are missing, dirty, unpinned or at different commits. Inspect inventory.json.")
