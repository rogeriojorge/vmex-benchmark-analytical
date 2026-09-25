# Continuation handoff for VMEX analytical benchmarks

This package reviews `rogeriojorge/vmex-benchmark-analytical` at `d5484d1e7a15c9cf10599c2b0f05bdf69e22e861`, compared with the preceding `575f13f` review. It is an overlay of instructions and independent review probes, not a replacement checkout or a completed solver benchmark.

## Read and adopt

Start with [the implementation plan](plan_next.md), [source-review findings](REVIEW_FINDINGS.md), and [references](REFERENCES.md). [AGENT_PROMPT.md](AGENT_PROMPT.md) is the copyable local-agent prompt. Preserve the existing plan/logbook and any work after the reviewed commit before adoption. The snapshot-based filename avoids colliding with separately circulated revision numbers.

The important change is the order of work: repair portable tests and report compatibility; fix native-knot alignment and interruption-safe scoring; then complete the bounded missing constraint experiment while starting actual axisymmetric responses and sheared-A work in parallel. The previously completed evidence repairs are retained rather than requested again.

## Included files

| File | Purpose |
|---|---|
| `plan_next.md` | Complete continuation contract, models, equations, phases, gates, figure plan and logbook instructions |
| `REVIEW_FINDINGS.md` | Twelve findings with source-confirmed scope, implications and minimal repairs |
| `REFERENCES.md` | Pinned project evidence and primary papers/documentation with specific applications |
| `AGENT_PROMPT.md` | Local implementation prompt, source/identity/publication constraints |
| `review_snapshot.json` | Reviewed pins, CI state, selected source hashes and explicit review limits |
| `tasks.json` | Compact dependency and deliverable index; the Markdown plan remains authoritative |
| `review_probes.py` | Six executable mathematical/source-contract probes, requiring only NumPy and SciPy |
| `results/review_probes.json` | Actual probe output and environment; no equilibrium solver execution |
| `SHA256SUMS` | Package integrity hashes |

Run the review probes with:

```sh
python review_probes.py
```

The probes check displaced-knot quadrature, NS-dependent grid counts, exact field-line flow and tangent maps, derivative-block scaling, radial regularity and streaming reductions. They do not import VMEX or reproduce the repository's CI/solver run. The repository reports 47 local passes, but the current public reference CI failed at pytest; those different environments must be reconciled locally.

No remote repository changes were made during this review. No VMEX, DESC, GVEC, free-boundary, kinetic or full repository test-suite run was performed here. The current source/record review is targeted, not a claim of whole-tree semantic certification.
