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
| 1 | Fix confirmed defects: `tau` in reference report; axis/minor-radius tiling; `B_relative_mid` is a median; scorer grids hard-coded; helpers accept unsupported `tau`/phase/shift; sheared projector uses `edge*s`; vacuum zero denominators; `cross_code_run` provenance and start record; dead `--volume` path; README claims (capped DESC called converged, "degenerate neighbour" asserted, unresolved volume J promoted) | failed-before/passed-after test for each; corrected records under a new scorer version; old records kept | done except figures (wait for DESC reruns) |
| 2 | Certify references and projection: one common norm set; resolved quadrature; full-flux lambda seed for sheared cases; reproduce the published sheared-DESC protocol | projection error converges in radius and angle; full-field reconstruction passes; scoring uncertainty below 1/10 of the gate | sheared projection certified (track shearD): sheared A and sheared Issan; common norm set for solved states still open |
| 3 | López D vacuum: certificate rerun, Cartesian evaluator with label-based root selection, chart checks, exact flux map `h(s)`, flux-uniform angle, vacuum metrics (curl, div, flux, circulation, energy) | resolved projection, then recovery with physical metrics; direction-only seed kept as negative control | reference done: certificate PASS (343037/21733/depth 28), field + label root selection, sampled chart check, h(s), eta, tilted frame, decks; projection resolved; recovery fails (0.8% attractor, see log) |
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
- Stage 2, sheared projection (track shearD, VMEX `fcf211f2`, 0.11.10): the projector seeds lambda from the full exact field for non-integer families (`flux_lambda.py`, restored from the September archive); `seed` in `projection.json` records `full_flux_lambda`, `exact_field_lambda_zero` or `geometry_only_lambda_zero`. `certify_projection.py` scores the projected state's continuous interior field (B, J, force against the exact grad p) on exact-chart volume points with exact Jacobian weights. Gates: B <= 1e-5, J <= 1e-3, force/|grad p| <= 1e-3 (relative L2, whole volume). Records in `results/projection_certificate/`. The published sheared-DESC driver (vendored unchanged in `benchmarks/upstream`) was run on office: every reference and final-resolution check passes (`results/desc/published_protocol/sheared_A`).
- 2026-10-23: Stage 1 rescoring (35 WOUT, 11 DESC v2); native route unavailable for LASYM and for two unconverged sheared A states.
- 2026-10-23: López D reference and surfaces merged. Exact iota = +1 in the VMEC gauge; the paper frame needs ntor > 480, the tilted frame gives 7e-7 at (10, 72).
- 2026-10-23: sheared projection certified: gates met at NS >= 33, mpol/ntor >= 13/12 (A) and all NS (Issan, 17/24). Published sheared-DESC protocol passes all checks on office.
- 2026-10-23: López D projection resolved (NS^-2; floor 3e-5 at (10,72), 7e-6 at (12,96); alpha gauge 1.2% as control). Recovery fails in VMEX and VMEC2000 alike: fsq limit-cycles 1e-6..1e-4 and every protocol settles near 0.8% B error with a displaced axis. Hypothesis to test next: iota = 1 exactly (all field lines closed) makes the nested-surface problem degenerate.
- 2026-10-23 (track iotaF): iota = 1 hypothesis refuted; every old exact-seed run was a silent cold start (eta-gauge seed violates the m = 1 polar constraint, hot_restart shear, Jacobian flag, axis re-guess). Polar-gauge seed (`lopez_polar.py`) stays on the exact state (B 1.4e-3, axis 0.6 % of a); cold iota scan (0.8-1.618, sheared) stalls at fsq 3e-6 for every iota; energy never below exact (scorer); Solov'ev LASYM control converges. See `results/iotaF/SUMMARY.md`.
