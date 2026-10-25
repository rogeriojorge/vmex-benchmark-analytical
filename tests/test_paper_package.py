"""The paper package: index integrity, figure regeneration from the index only, claim ledger and raw-state manifest."""
import hashlib
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import build_index  # noqa: E402
import paper_figures  # noqa: E402
import raw_state_manifest  # noqa: E402

INDEX = json.loads((ROOT/"results/index.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_index_is_current():
    assert build_index.main(["--check"]) == 0


def test_every_indexed_file_exists_with_matching_hash():
    n = 0
    for key, rec in INDEX["records"].items():
        assert rec["files"], key
        for f in rec["files"]:
            path = ROOT/f["path"]
            assert path.is_file(), f"{key}: {f['path']} missing"
            assert sha(path) == f["sha256"], f"{key}: {f['path']} changed"
            n += 1
    assert n > 300


def test_member_digests_match():
    for key, rec in INDEX["records"].items():
        if "members" in rec:
            members = build_index.expand(rec["members"]["patterns"])
            assert len(members) == rec["members"]["count"], key
            lines = "".join(f"{p.relative_to(ROOT)} {build_index.sha256_file(p)}\n" for p in members)
            assert hashlib.sha256(lines.encode()).hexdigest() == rec["members"]["digest"], key


def test_producers_exist_and_code_identities_are_declared():
    for script, digest in INDEX["producer_script_sha256"].items():
        assert sha(ROOT/script) == digest, f"{script} changed since the index was written"
    for key, rec in INDEX["records"].items():
        declared = " ".join(rec["code"])
        for name, values in rec["observed"].items():
            for value in values:
                assert value[:7] in declared or name == "jax" and value in declared, f"{key}: record reports {name} {value}, not declared"
        assert rec["label"] and rec["status"] and rec["producer"]["scripts"], key


def test_figures_and_claims_reference_known_records():
    assert len(INDEX["figures"]) == 6
    for fig in INDEX["figures"].values():
        for panel in fig["panels"].values():
            assert all(k in INDEX["records"] for k in panel["records"])
    for claim in INDEX["claims"].values():
        assert all(k in INDEX["records"] for k in claim["records"])


def test_figures_build_from_the_index_only(tmp_path):
    manifest = paper_figures.build(tmp_path)
    assert set(manifest["figures"]) == set(INDEX["figures"])
    for name, info in manifest["figures"].items():
        declared = {k for p in INDEX["figures"][name]["panels"].values() for k in p["records"]}
        assert set(info["records_used"]) <= declared, f"{name} read records it does not declare"
        assert info["inputs"], name
        for ext in ("png", "pdf"):
            assert (tmp_path/f"{name}.{ext}").stat().st_size > 10_000


def test_index_refuses_unlisted_and_modified_files(tmp_path):
    idx = paper_figures.Index()
    with pytest.raises(KeyError):
        idx.read("results/matrix/TABLES.md")
    bad = json.loads((ROOT/"results/index.json").read_text())
    first = next(iter(bad["records"].values()))["files"][0]
    first["sha256"] = "0"*64
    path = tmp_path/"index.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        paper_figures.Index(path).read(first["path"])


def test_claim_ledger_matches_index():
    text = (ROOT/"CLAIMS.md").read_text()
    for cid, claim in INDEX["claims"].items():
        assert re.search(rf"\b{cid}\b", text), f"{cid} missing from CLAIMS.md"
        row = next(line for line in text.splitlines() if re.match(rf"\|\s*{cid}\s*\|", line))
        for key in claim["records"]:
            assert f"`{key}`" in row, f"{cid}: CLAIMS.md row does not cite {key}"
    cited = set(re.findall(r"`((?:ref|proj|rec|lopez_mrx|lopez|sens|mech)\.[a-z_0-9.]+)`", text))
    assert cited <= set(INDEX["records"]), cited - set(INDEX["records"])


def test_raw_state_manifest_is_consistent():
    manifest = json.loads((ROOT/"results/raw_state_manifest.json").read_text())
    assert raw_state_manifest.main(["--check"]) == 0
    assert manifest["verified_on_office"]
    for r in manifest["states"]:
        assert re.fullmatch(r"[0-9a-f]{64}", r["sha256"])
        if r["local_path"]:
            assert sha(ROOT/r["local_path"]) == r["sha256"]
        else:
            assert r["office_path"] or r.get("office_status") == "not on office", r["name"]
        assert all(k in INDEX["records"] for k in r["index_keys"]), r["name"]
    assert any(r["index_keys"] for r in manifest["states"])
