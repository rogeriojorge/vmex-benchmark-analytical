"""Run receipts must never report an unknown or dirty source as a clean pinned tree."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

from evidence import source_metadata  # noqa: E402


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"})


def test_outside_git_is_unknown_not_clean(tmp_path):
    module = tmp_path/"pkg"/"__init__.py"
    module.parent.mkdir()
    module.write_text("")
    meta = source_metadata(module, "pkg", "1.0")
    assert meta["commit"] is None and meta["tracked_tree_clean"] is None
    assert meta["version_source"] == "installed_distribution"


def test_clean_and_dirty_checkouts(tmp_path):
    _git(tmp_path, "init", "-q")
    module = tmp_path/"pkg"/"__init__.py"
    module.parent.mkdir()
    module.write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "init")
    clean = source_metadata(module, "pkg", None)
    assert clean["tracked_tree_clean"] is True and len(clean["commit"]) == 40
    module.write_text("x = 2\n")
    dirty = source_metadata(module, "pkg", None)
    assert dirty["tracked_tree_clean"] is False and dirty["commit"] == clean["commit"]
    assert dirty["tracked_diff_sha256"] != clean["tracked_diff_sha256"]
