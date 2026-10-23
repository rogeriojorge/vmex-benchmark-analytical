# Status

Working branch `research/exact-equilibrium-verification`, from main `f904b3a`. One record of where the study stands; update it with each pushed milestone.

## Goal

A reproducible verification of equilibrium fields and their sensitivities against exact solutions (Landreman integer and sheared families, Issan asymmetric extensions, Solov'ev, López vacuum D), across VMEX, VMEC2000, VMEC++ and DESC, with matched physical norms, resolved quadrature, certified derivatives and at least one established numerical mechanism. Verification only: no claims about stability, confinement or experiments, and no claim to be the first exact-solution verification (Landreman 2026, Panici et al. 2023, Hudson et al. 2025 precede it).

## Pins

| Source | Identity |
|---|---|
| Benchmark base | `f904b3a0885dd24e0f9b6d71b5ffc625f132a938` |
| Archive | tag `archive-2026-09-25` -> `1d8c24d1f7328c72b1f116907fabd3e0048b3dc3` |
| VMEX main, 2026-10-23 | `caf09ee7faca2e1c74d139ef6910f05e02654514` (review observed `ade7afa`) |
| López certificate | `certify.py` blob `008fe6730550649d18f627d3a4caaf8c1d0259b2` |
| Sheared DESC protocol | landreman/analytic_3d_equilibria blob `e54499a810da26518cff4f0d3f1940b45e8d1789` |

VMEX PRs: #452, #453, #462, #578, #612, #622, #628 merged; #585 (B-spline), #626 (constraint bias), #630 (GPU sync) open.

## Stages

| Stage | Work | Exit | State |
|---|---|---|---|
| 1 | Fix confirmed defects: `tau` in reference report; axis/minor-radius tiling; `B_relative_mid` is a median; scorer grids hard-coded; helpers accept unsupported `tau`/phase/shift; sheared projector uses `edge*s`; vacuum zero denominators; `cross_code_run` provenance and start record; dead `--volume` path; README claims (capped DESC called converged, "degenerate neighbour" asserted, unresolved volume J promoted) | failed-before/passed-after test for each; corrected records under a new scorer version; old records kept | in progress |
| 2 | Certify references and projection: one common norm set; resolved quadrature; full-flux lambda seed for sheared cases; reproduce the published sheared-DESC protocol | projection error converges in radius and angle; full-field reconstruction passes; scoring uncertainty below 1/10 of the gate | planned |
| 3 | López D vacuum: certificate rerun, Cartesian evaluator with label-based root selection, chart checks, exact flux map `h(s)`, flux-uniform angle, vacuum metrics (curl, div, flux, circulation, energy) | resolved projection, then recovery with physical metrics; direction-only seed kept as negative control | planned |
| 4 | Matched recovery matrix: Solov'ev sym/asym; integer 3-D tau 0 and mirror pair; sheared continuation toward Issan; López D. Cold, continued and exact-seed protocols kept separate. Work-precision at matched error | at least one 3-D finite-pressure, one asymmetric and the vacuum case with resolved recovery or a quantified obstruction | planned |
| 5 | Sensitivities: nonzero `c` and null `delta` on current VMEX (root, tangent, transpose, FD ladder, continuum limit), then one 3-D observable; DESC where comparable | one nonzero and one null physical response accepted | planned |
| 6 | Mechanisms: near-axis current in solved states; asymmetric `iota = 2` plateau (input/projection refinement, signed phases, closures, Jacobian null directions) | discriminating experiments, not only a plateau | planned |
| 7 | Paper package: six figures from one result index, claim ledger, environment, raw-state release | figures regenerate from archived records | planned |

## Salvage from the interrupted September agents (local clones `~/local/vmex-bench-work/item1..5`, based on `418355d`)

- item1 free-boundary cut (s0 = 0.5): setup, MGRIDs, extender score. Free boundary is a follow-on, after Stage 4.
- item2 polishing logs: fold into Stage 6 near-axis study as one controlled comparator.
- item3 sheared-A cold-start logs (axis guess, DELT, MPOL 17): input to Stage 2/4 sheared work.
- item4 derivative drivers (JAX input map, VMEX implicit tangent): starting point for Stage 5.
- item5 cost campaign scripts: starting point for Stage 4 work-precision.

## Log

- 2026-10-23: baseline on `f904b3a`, 52 reference tests pass locally. Audit finding 1 reproduced: `integer_3d_tau` report volume 0.267104 versus case value 0.411234.
