# Research release

The git tree holds records, tables, scripts and figures (about 55 MB of results). The raw numerical states behind them (WOUT `.nc`, DESC `.h5`) stay outside git and are listed, with sha256 and size, in [`results/raw_state_manifest.json`](results/raw_state_manifest.json). Nothing has been uploaded and no GitHub release exists; this file says what a release would contain and how to pack it.

## Contents of the raw-state manifest

659 entries, 880 MB where the size is known. Built by `benchmarks/raw_state_manifest.py --office` on 2026-10-24 from the per-track manifests (`results/matrix/{wout,wp_wout,wp2_wout,xbuild_wout}_manifest.json`, `results/desc/matched/h5_manifest.json`, `results/iotaF/wout_manifest_iotaF.json`, `results/lopez/wout_manifest.json`, `results/mechanism/wout_manifest.json`), the committed `.npz`, and the DESC equilibria named only by their records.

| State of the file | Entries | How to get it |
|---|---|---|
| on office, sha256 and size verified against the manifest | 523 (521 distinct paths, 852 MB) | pack list below |
| committed in git (`.npz`: sensitivities, MRX Hamiltonian fields, projection seeds) | 60 | `git clone` |
| not archived: 73 mechanism WOUTs and 3 cross-build WOUTs | 76 | rerun the command in the manifest note; sha256 recorded when written |

An entry's `index_keys` name the index records (`results/index.json`) it belongs to. Identical WOUTs written by the two work-precision passes appear once per manifest. WOUTs of runs that were rescored in place (`--no-wout`) have a record but no entry.

## Pack the research tarball

On the office box, from a checkout of this branch (reads the files, writes only the tarball):

```sh
python benchmarks/raw_state_manifest.py --check                       # manifest still matches the index
python benchmarks/raw_state_manifest.py --pack-list > /tmp/raw_states.txt   # absolute office paths, 521 files
wc -l /tmp/raw_states.txt
tar -C / -czf vmex-bench-raw-states-<tag>.tar.gz -T <(sed 's#^/##' /tmp/raw_states.txt)
sha256sum vmex-bench-raw-states-<tag>.tar.gz > vmex-bench-raw-states-<tag>.sha256
tar -tzf vmex-bench-raw-states-<tag>.tar.gz | wc -l                   # expect 521
```

Paths inside the tarball keep their `home/rjorge/vab-*/...` prefix, which is exactly the `office_path` of the manifest, so a consumer verifies with:

```sh
python - <<'EOF'
import json, hashlib, pathlib
root = pathlib.Path("unpacked")
for s in json.load(open("results/raw_state_manifest.json"))["states"]:
    if s["office_path"] and s.get("office_status", "").startswith("ok"):
        assert hashlib.sha256((root/s["office_path"].lstrip("/")).read_bytes()).hexdigest() == s["sha256"], s["name"]
EOF
```

## Before publishing anything

1. Tag the benchmark commit that carries `results/index.json`, `results/raw_state_manifest.json` and `figures/paper/manifest.json`; the tag and the tarball sha256 are the immutable identifier.
2. Do not rewrite the mechanism claims' missing states as present: keep the 76 `not on office` entries in the release notes.
3. Rebuild and recheck from the tag in a clean environment: `build_index.py --check`, `paper_figures.py`, `pytest -q` (see [ENVIRONMENT.md](ENVIRONMENT.md), level 1).
