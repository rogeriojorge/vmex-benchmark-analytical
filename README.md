# VMEX analytical benchmarks

Exact toroidal equilibria, physical-field checks and a staged benchmark of VMEX's derivatives, diagnostics and free-boundary interfaces.

The primary references are [Landreman's analytical equilibria](https://arxiv.org/abs/2609.26742), their [supplementary implementation](https://github.com/landreman/analytic_3d_equilibria), and an explicitly derived asymmetric Solov'ev case. The implementation plan and continuing logbook are in [plan.md](plan.md). Start a local implementation session with [AGENT_PROMPT.md](AGENT_PROMPT.md).

## Results currently included

These are **analytical reference results, not VMEX solver results**. The reference suite passed 29 tests, and sampled force identities passed on 14 configurations. The largest sampled force RMS divided by pressure-gradient RMS was 1.66e-15. Tests also cover an independent Grad-Shafranov identity, volume and toroidal-flux integrals, coordinate covariance, current integration, flux inversion, field Jacobians and sensitivity cancellations.

| Reference | Volume-averaged beta | Axis iota, counterclockwise R-Z convention |
|---|---:|---:|
| Integer 3-D, epsilon=0.5, delta=1/64 | 3.50877193% | -2.00000000 |
| Sheared A | 19.66235650% | -5.68886661 |
| Sheared B | 7.46725404% | -4.36231483 |
| Sheared C | 14.09668185% | -3.88989400 |

The paper uses the opposite poloidal orientation. The signs above are deliberate; the local VMEX parser/field conversion still needs its own physical-vector check. The exact field's derivative with respect to the integer family's outer label is zero at a fixed interior Cartesian point. The sheared transform's lambda derivative also vanishes; measured explicit-reference values and Taylor curves are in [results/reference/derivatives.json](results/reference/derivatives.json).

![Exact boundaries](figures/reference_geometry.png)

![Asymmetric Solovev reference](figures/solovev_asymmetry.png)

## Run the reference checks

Run from the repository root in a dedicated environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python benchmarks/verify_reference.py
python benchmarks/reference_derivatives.py
python benchmarks/build_inputs.py
python benchmarks/plot_results.py
```

The scripts use float64. Recorded package versions and machine information are in [results/reference/identities.json](results/reference/identities.json); test and command evidence is in [results/reference/execution.json](results/reference/execution.json). Exact local versions, rather than an assumed latest release, must accompany later solver results. Python 3.12 or newer is a practical starting point for the separate current-VMEX environment; resolve its actual dependency constraints during phase P0.

![Reference Taylor test](figures/reference_derivatives.png)

## First measured VMEX recovery

At the pinned VMEX baseline, the axisymmetric integer case converged with prescribed iota at three cold radial resolutions. Errors use 96 native Cartesian field samples on a 3×8×4 physical-volume quadrature. They compare the solved state with the independent exact field; `J` is the curl of native `B`, and the pressure gradient uses VMEX's inverted flux coordinate and parsed pressure profile.

| NS | B relative L2 | J relative L2 | Force RMS / exact pressure-gradient RMS |
|---:|---:|---:|---:|
| 33 | 5.76e-5 | 2.58e-2 | 2.88e-1 |
| 65 | 7.90e-6 | 2.49e-3 | 2.76e-2 |
| 129 | 6.51e-7 | 8.24e-5 | 8.18e-4 |

The `NS=129` prescribed-iota run meets the initial single-run B, J and force targets. Its largest normalized discrete force component was 9.66e-15. The current-prescribed `NS=129` run also meets them: B error 6.89e-7, J error 8.25e-5 and force ratio 8.18e-4. The near-axis sample was the main source of the larger `NS=65` current/force error. A current-prescribed multigrid run at `NS=65` gave B error 7.97e-6, J error 2.49e-3 and force ratio 2.75e-2, with current and force still above their planned targets. This is one fixed-boundary physical case; independent angular refinement and the other required geometries remain open. All run records and native sample arrays are in [results/vmex](results/vmex).

The separate exact symmetric Solov’ev projection into VMEX’s continuous basis had sampled B relative L2 error from 1.59e-12 to 1.06e-12 over 2, 4 and 8 degree-5 spline spans. It used no nonlinear solve and its strong-force certificate showed a finite floor. See [the projection record](results/projection/solovev_symmetric.json); this basis check is separate from the solved state’s native field interpolation.

For the genuinely asymmetric Solov’ev input, the discrete solve converged at `NS=33, 65, 129`. VMEX’s live Cartesian interior field API currently rejects LASYM. Its WOUT surface field route does accept LASYM: the worst of five area-weighted surface B errors decreased from 1.26e-4 to 3.15e-5 to 7.86e-6. These are surface B checks only. A continuous fitted-state lift was also tried, but fixed span caps made its errors much larger and could reverse their refinement trend; failed trials and arrays remain in [results/vmex](results/vmex). No LASYM volume current/force recovery is certified.

A same-deck [LASYM WOUT restart round trip](results/vmex/solovev_asymmetric_iota_ns129/lasym_roundtrip.json) reconstructed a native state and regenerated WOUT at NS=33, 65 and 129. On five surfaces per resolution, the largest geometry difference from the original WOUT was 1.4e-17 m and the largest relative surface B difference was 3.42e-14. This tests serialization and surface reconstruction; the live Cartesian LASYM volume path remains unavailable.

The first cold 3-D integer prescribed-iota attempt at NS=33 [did not converge](results/vmex/integer_3d_iota_ns33_niter3000/attempt_history.json). VMEX reported an initial Jacobian sign change, improved its axis guess, and ended with `MORE ITERATIONS REQUIRED` at both 3,000 and 30,000 iterations. The final reported normalized force components at 30,000 were 1.15e-10, 8.18e-11 and 4.54e-11; no solved field was scored. This is a failed initialization/solver attempt, not a test of the analytical field's validity.

For the same integer 3-D case, direct physical-field reconstruction shows that the supplied geometric angle has a vanishing lambda gradient to roughly 1e-10 of the flux scale on three sampled surfaces; its measured transform is -2. An [exact-geometry VMEX state projection](results/projection/integer_3d_vmex_ns129/native_scores.json) then reaches native volume B/J/force errors of 2.42e-7 / 7.14e-6 / 8.15e-5 at NS=129, with no nonlinear solve. Its sampled WOUT surface B error is 8.86e-6 at that resolution.

Strict warm recovery from the projected NS=33 state [still failed](results/vmex/integer_3d_iota_ns33_niter3000_projected/attempt_summary.json) at 3,000 iterations. Explicitly looser `FTOL=1e-10` roots returned at NS=33,65,129, but their native physical errors remained above the initial targets; at NS=129 they were B 1.30e-4, J 2.92e-3 and force ratio 1.74e-2. [The measured comparison](figures/vmex_integer_3d_projection.png) keeps projection and loose-root results separate. The cause of the solver trajectory’s physical displacement remains unresolved.

A [same-grid invariant-force comparison](results/vmex/integer_3d_raw_residual_comparison.json) found projected NS=129 VMEX `(FSQR, FSQZ)=(1.47, 5.34)` despite its small independent physical errors. Reconstructed loose-root WOUTs returned components near `1e-10`, agreeing with saved solver values to within `6.5e-15`. This is a discrete residual versus physical-field discrepancy, not accepted recovery or a diagnosed cause.

The comparison was repeated with VMEX's actual warm-start boundary transfer and constraint-baseline rebinding; its reported components changed only at roundoff. A [bounded NS33 warm run](results/vmex/integer_3d_short_warm_ns33_ftol1e-4/probe.json) reached `FTOL=1e-4` after 65 iterations, but native B/J/force errors had increased to `1.51e-3 / 0.326 / 3.72` from the projected seed's `2.59e-5 / 3.55e-3 / 3.78e-2`. [Its saved trajectory](figures/vmex_integer_3d_trajectory.png) shows the discrete force reduction alongside those independent physical scores. This remains diagnostic only.

The [projected NS33 force localization](results/vmex/integer_3d_force_localization_ns33.json) places 86% of R and 86% of Z invariant-force sums in radial rows 16–31 of 0–32, and 85% of R and 87% of Z in poloidal modes 3–6. The checked spectral block sums reproduce VMEX's scalar residuals. This rules out a force concentrated solely at the axis or fixed boundary; the cause remains open.

A [constraint-switch diagnostic](results/vmex/integer_3d_constraint_switch.json) reevaluated each saved state with VMEX's spectral-condensation strength set to zero. On the NS33 projection, R/Z invariant residuals fell from `0.254/0.919` to `8.51e-7/9.99e-7`; at NS129 they fell from `1.47/5.34` to `5.17e-8/6.18e-8`. The state and independent physical scores did not change in this evaluation. The large projected VMEX residual is therefore dominated by its coordinate constraint. Whether a physical-field-preserving coordinate remap can satisfy that constraint and recover the exact state remains untested.

![Integer 3-D projection and loose-root comparison](figures/vmex_integer_3d_projection.png)

![Integer 3-D short warm trajectory](figures/vmex_integer_3d_trajectory.png)

![Integer 3-D projected force localization](figures/vmex_integer_force_localization.png)

![Integer 3-D constraint switch](figures/vmex_integer_constraint_switch.png)

![Measured asymmetric Solov'ev surface B refinement](figures/vmex_lasym_surface_B.png)

![Measured axisymmetric VMEX recovery](figures/vmex_axisymmetric_recovery.png)

## Candidate VMEX inputs

`inputs/` contains 28 generated INDATA candidates: 14 configurations, each with prescribed iota and prescribed current. The current profile is generated from an independent Ampere integral and converted to VMEX's derivative-profile convention. Pressure, flux and geometric scales are recorded in [inputs/manifest.json](inputs/manifest.json).

The regenerated boundary candidates use `(MPOL, NTOR) = (17, 96)` for sheared B and `(25, 100)` for sheared C. Their independent sampled Fourier-fit maxima are 8.59e-9 m and 3.34e-7 m at a 1 m length scale, below the 1e-6 m smoke gate. VMEX's pinned parser and setup accepted all 28 candidates: their sign map has `signgs=-1` and no unexpected theta flip, and sampled pressure, iota or normalized current agree with the independent reference fits. The details are in [results/inputs/vmex_parser.json](results/inputs/vmex_parser.json). These checks do not certify the final output error budget or an equilibrium solve.

The [sheared-chart check](results/reference/sheared_chart.json) sampled all six sheared variants on five radii, 64 poloidal labels and 513 toroidal nodes, plus 1024 independent physical-angle inversions per case. Every sampled angle map was monotone; the smallest normalized angular slope was 0.436 for sheared C. A [local implicit-root JVP check](results/reference/sheared_derivative.json) compared physical-position directional derivatives with independent central differences at four steps for two interior points each in A, B and C. Its smallest directional errors per point were 7.38e-13 to 1.95e-12 in reference length units. The branch-boundary and full input-map derivatives remain untested.

A [near-domain probe](results/reference/sheared_domain_edges.json) found that the smooth analytical domain does not alone guarantee a single-valued physical-angle chart. In sheared C with `S - asin(sqrt(2 edge))` of 0.001, 0.01 and 0.1, sampled minimum angle slopes were -3.22, -2.22 and -0.209, with three chart crossings at selected physical angles. The new sampled graph guard rejects these inputs before deck generation. The tested margins 0.2 and above had positive sampled slopes; their graph validity outside the sample is unproved.

![Boundary input fit error](figures/boundary_fit.png)

After installing the pinned VMEX and its dependencies in a separate local environment:

```sh
python benchmarks/run_vmex.py inputs/input.integer_axisymmetric_iota
python benchmarks/run_vmex.py inputs/input.integer_axisymmetric_current
python benchmarks/run_gradient_smoke.py inputs/input.integer_axisymmetric_iota
```

The forward runner now saves native Cartesian samples and physical scores after a converged solve. The gradient smoke remains unrun; it varies only PHIEDGE at fixed boundary and profiles and does not differentiate the complete analytical family. Complete the other recovery and derivative checks in phases P2-P4 before broader claims.

The shared scorer accepts an NPZ file of **native physical samples**, with its contract in the script and plan:

```sh
python benchmarks/score_samples.py samples.npz scores.json
```

It compares Cartesian B, J and grad p against the reference at the same points. Its `accepted` field remains unset until the experiment's resolution and error gates have been applied.

## Study coverage

The [execution matrix](benchmark_matrix.json) records implemented, planned and blocked experiments. The plan covers axisymmetric and non-axisymmetric equilibria, LASYM=F/T, genuine asymmetric Solov'ev physics, exact coordinate changes, both current/transform closures, field derivatives, implicit responses, file/restart paths, Boozer and bounce diagnostics, polishing, free-boundary operators and coupled roots, source fitting, optimization and scoped adjacent-code integrations.

An exact interior field is not automatically an exact free-boundary solution. The free-boundary plan distinguishes exact vacuum/operator tests, independently converged numerical coupled equilibria, and approximate exterior fits to an exact interior target. Genuinely symmetry-broken 3-D extensions outside the exact families also require numerical references. These distinctions are part of the benchmark, not missing labels to be filled with assumed answers.

The fixed-boundary VMEX runs described above are the only nonlinear recovery measurements so far. No DESC, free-boundary, GPU or kinetic benchmark has run yet. The [public benchmark repository](https://github.com/rogeriojorge/vmex-benchmark-analytical) was created using the verified owner's authentication. [The source review](docs/SOURCE_REVIEW.md) identifies inspected paths and the full local semantic audit still needed; an inventory and focused parser review are not a whole-source semantic review.

## Local source inventory and publication

Resolve the used pins in `sources.json`, locate checkouts outside this repository, then run:

```sh
python tools/audit_sources.py /absolute/path/to/checkouts
sh tools/publish.sh
# Read the staged diff before authorizing publication:
PUBLISH=1 sh tools/publish.sh
```

The inventory tool does not mark files reviewed. The publication helper verifies the authenticated owner and configures only local Git identity; it does not force-push or rewrite history. Preserve third-party notices. Update the phase table and append a logbook entry after each meaningful experiment.

## License and references

Original code: MIT. See [NOTICE.md](NOTICE.md) for attribution and [plan.md](plan.md#references) for the scientific and implementation references. Large future state files should be checksummed release assets, while code, small numerical summaries and figure generators remain in Git.
