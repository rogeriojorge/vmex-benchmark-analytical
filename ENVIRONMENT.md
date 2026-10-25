# Environment and code identities

What produced each result, and how to rerun it. The identities below are what the records themselves report (checked against `results/index.json` by `tests/test_paper_package.py`); a campaign pin is not interchangeable with the README's historical `f16c300`.

## Code identities

| Code | Identity | Used for |
|---|---|---|
| VMEX | `caf09ee7faca2e1c74d139ef6910f05e02654514` (0.12.0, upstream main 2026-10-22, clean tree) | recovery matrix, work-precision, sensitivities, mechanisms, mesh-aligned projection certificates |
| VMEX | `3b73d6f88b832aaa6c63dc7574feb7de9e80d064` (0.11.x, includes #452) | cross-code runs (September campaign, Mac mini), Lopez D early and iotaF runs, Lopez WOUT read-out |
| VMEX | `fcf211f20692e557345fa41fe53d83b411ad53a2` (0.11.10) | sheared A and Issan projection certificates (plain `q*` records) |
| VMEX | `ec8641f3c02ae45b4d1b055483b5ba6fa44f3aeb` | early projection records (`results/projection`) |
| VMEC2000 | STELLOPT-built `70c637a56a535de7b72a5721d4294a7c1a6cf739`, office Linux, binary sha256 `1dec62d1c8026f6ed5dccc1e5199dd5320ee702c354153c296d81d3411f751e0` | matrix, work-precision, Lopez, operator comparison |
| VMEC2000 | STELLOPT `ee175502dec1766c60c2027e1df5c74c343d9794`, macOS arm64, dirty tracked tree (diff sha256 `cccd8338...`), binary sha256 `8ba1188465be8a76bcdf8ff481ab3e03ca6b125eab7a22958e92f39db710c42e` | cross-code runs, `results/matrix/xbuild` |
| VMEC2000 | STELLOPT `3e1439d` (the README's historical pin) | **unavailable**: not built on office or Mac; no record uses it |
| VMEC++ | 0.5.2 (pip) | cross-code runs only (Mac mini); no `LASYM` support |
| DESC | 0.17.3+32.gc9ec78e2d, JAX 0.6.2, Python 3.10.12 (office `~/desc-env`) | matched ladders, factor isolation, published protocol, general records v2 |
| DESC | 0.17.3+27.g4f48720be, same environment | first general records (rescored under +32 in `record_v2.json`) |
| MRX | fork `rogeriojorge/mrx` main `732f8249a8a1d39d0dabedc02e2ac0d743d3697a` (unmodified), JAX 0.11.0, equinox 0.13.8, diffrax 0.7.2, Python 3.13.7, `MRX_DTYPE=float64`, CPU | Lopez D vacuum solves (records carry no identity; it is logged in STATUS.md) |
| Published sheared-DESC driver | landreman/analytic_3d_equilibria blob `e54499a810da26518cff4f0d3f1940b45e8d1789`, vendored unchanged in `benchmarks/upstream` | published-protocol run |
| Lopez certificate | omlopez/cubicRootCertificate `certify.py` blob `008fe6730550649d18f627d3a4caaf8c1d0259b2` | rerun once, result in STATUS.md only |
| Benchmark | base `f904b3a0885dd24e0f9b6d71b5ffc625f132a938`; Stage 1-7 commits on `research/exact-equilibrium-verification`; receipts carry the benchmark commit where the run was inside a checkout (`matrix` receipts do not) | everything |

## Platforms and JAX

| Records | Host | JAX | Notes |
|---|---|---|---|
| matrix, work-precision, sensitivity, iotaF, Lopez, office eigen runs | office (pop-os, Linux x86-64, 36 cores, shared; 62 GB) | 0.11.2 (VMEX venv Python 3.12.13, solvax 0.31.0) | CPU, float64, 4 OpenMP threads, `JAX_PLATFORMS=cpu`, nice 10, one job at a time |
| mechanism runs, projection references, reference identities | Mac mini (macOS 26.6.2 arm64, Python 3.13.7, numpy 2.5.1, scipy 1.18.0) | 0.11.0 | CPU, float64 |
| DESC | office | 0.6.2 | CPU, 4 threads |
| MRX | Mac mini (`venv-mrxJ`) and office | 0.11.0 | CPU, float64 |
| cross-code (September) | Mac mini | 0.11.0 | VMEX 3b73d6f, VMEC2000 ee17550, VMEC++ 0.5.2 |

Every record names its own versions; `x64` must be enabled before JAX is first imported (`MRX_DTYPE=float64` for MRX). A local float32 run silently changes results.

## How to rerun

Three levels, cheapest first.

1. **Figures and checks from archived records** (seconds, no solver, no JAX):

   ```sh
   python -m pip install -r requirements.txt
   python benchmarks/build_index.py --check          # index matches the files on disk
   python benchmarks/paper_figures.py                # figures/paper/*.png, *.pdf, manifest.json
   python -m pytest -q                               # includes tests/test_paper_package.py
   ```

2. **Rescoring from raw states** (minutes to hours; needs the research release, see [RELEASE.md](RELEASE.md), and VMEX `caf09ee7` importable):

   ```sh
   python benchmarks/score_solved.py --help          # scores any VMEC-format WOUT with the certificate norms
   python benchmarks/certify_projection.py --case sheared_A --ns 65 --mpol 13 --ntor 12
   python benchmarks/desc_matched_tables.py          # DESC tables from results/desc/matched
   python benchmarks/matrix_tables.py                # matrix summary from results/matrix/runs
   python benchmarks/mech_tables.py                  # results/mechanism/TABLES.md
   ```

3. **Re-solving** (days; the commands are in each index record under `producer.command` and in the per-run receipts):

   | Family | Entry point |
   |---|---|
   | recovery matrix | `bash benchmarks/matrix_queue.sh <queue>` (queues in `results/matrix/queue_*.txt`) |
   | work-precision | `bash benchmarks/matrix_wp_queue.sh` |
   | DESC | `bash benchmarks/desc_queue.sh` (jobs from `desc_matched_jobs.py factor|ladder`) |
   | sensitivities | `python benchmarks/sensitivity_vmex.py ...` (commands in `results/sensitivity/RUNS.md`; integer 3-D NS 65 needs more than 52 GB) |
   | mechanisms | `python benchmarks/mech_solve.py --case <case> --ns <NS> --tcon0 <t> --baseline <vmec|exact> [--polar]`, `mech_linear.py`, `mech_eigen.py` (integer 3-D NS 17 eigenvalues: 14k dofs, about 13 minutes of dense geev on office) |
   | Lopez D | `lopez_surfaces.py`, `lopez_polar.py`, `lopez_solve.py`, `lopez_iotaF_queue.sh`; MRX: `lopez_mrx_run.py` with `PYTHONPATH` at the MRX clone and from a directory without an `mrx/` folder |

   Work-precision timings depend on machine load; rerunning on an idle box will not reproduce the recorded times, only the errors and iteration counts.

## Known identity gaps

- No record uses STELLOPT `3e1439d`. VMEC2000 results are from `70c637a` (office) and `ee17550` (macOS, dirty tree); stalled asymmetric 3-D states differ between the two builds (claim M4).
- The matrix receipts record no benchmark commit (the run directory was not a checkout); the producing scripts are hashed in `results/index.json`.
- `results/mechanism` WOUTs were written on the Mac mini and, apart from five on office, were not archived; the manifest gives the regenerating command per state.
