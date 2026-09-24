# Sources, methods and their use in this handoff

Access/review date: 2026-09-24. Source commits are frozen where the handoff relies on recorded software behavior. A paper, repository README, source inspection and an executed numerical experiment provide different kinds of evidence. The entries below do not imply that every referenced implementation was installed or every line reviewed.

## R01. Landreman analytical equilibria

M. Landreman, *Analytic toroidal 3D MHD equilibria and steady Euler flows with invariant surfaces*, arXiv:2609.26742.

- https://arxiv.org/abs/2609.26742
- https://arxiv.org/html/2609.26742v1

Use the exact Cartesian fields and pressure surfaces as independent references. The paper covers the constant-transform and sheared-transform constructions. Its statements about exact solutions do not establish convergence of a particular finite numerical representation, its coordinate gauge, a kinetic model or a vacuum exterior.

## R02. Analytical supplementary implementation

- https://github.com/landreman/analytic_3d_equilibria/tree/4c0b690ddebdc71811c88223eb9f44a98ab64222
- Integer DESC script: `iota_2/20260910-01_analytic_3D_equilibrium_iota_2_desc.py`
- Sheared DESC script: `sheared_iota/20260912-02_analytic_3D_equilibrium_with_sheared_iota_desc.py`

Use its independent Cartesian evaluators, flux/current quadratures, coordinate conversions and solved-DESC comparison as a reference implementation. Preserve pressure and poloidal-orientation conventions. The extra released diagonal stretch in this project is a derived extension, not a claim that every formula in it was supplied by this repository.

## R03. Reviewed benchmark and immutable evidence

Root: https://github.com/rogeriojorge/vmex-benchmark-analytical/tree/575f13f67b346118c3d7f05cc60cc6e29131359a

All paths in this section refer to that commit:

| Path | Review purpose |
|---|---|
| `README.md` | Claims, measured values, limitations and run commands |
| `plan.md` | Scientific contract, phase table, dated corrections and next action |
| `sources.json` | Actual baseline pins versus unresolved optional dependencies |
| `benchmark_matrix.json` | Evidence/status classification and unrun work |
| `docs/SOURCE_REVIEW.md` | Semantic-review scope; documentation/source discrepancies |
| `benchmarks/analytic.py` | Exact references, coordinate maps, validation and chart guard |
| `benchmarks/native_samples.py` | Legacy grid, physical weights, field/current/grad-p paths |
| `benchmarks/score_samples.py` | Norm definitions, domain/scaling checks and flux labels |
| `benchmarks/run_vmex.py` | Overrides, initialization, failure handling, source metadata and results |
| `benchmarks/project_integer_vmex.py` | Exact-state Fourier projection, m=1 conversion and no-solve status |
| `benchmarks/probe_integer_gauge.py` | Gauge envelope, Clebsch compensation, selection and finite projection |
| `benchmarks/run_desc_coordinate.py` | DESC chart, fits, inversion, labels, source provenance and termination |
| `benchmarks/plot_tcon_ladder.py` | Mutation of primary reports, hard-coded metadata and derived figures |
| `tests/test_reference.py` | Actual scope of the 29 reference tests and duplicated physical control |
| `results/vmex/tcon_resolution_ladder/summary.json` | Five projected-start cells and the explicit NS129 zero-strength gap |

Construct a permanent source link as `https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/<path>`.

PR state and implementation history:
- https://github.com/rogeriojorge/vmex-benchmark-analytical/pull/1
- https://github.com/rogeriojorge/vmex-benchmark-analytical/commit/a3df653eac7e4c2d44465e1f61b9d78691820285
- https://github.com/rogeriojorge/vmex-benchmark-analytical/commit/575f13f67b346118c3d7f05cc60cc6e29131359a

The PR's actual state is merged, regardless of stale prose in its body. Recorded results remain measurements from the historical environment. This review did not rerun them.

## R04. VMEX source and numerical contracts

- Historical baseline: https://github.com/uwplasma/vmex/tree/b5f5267efc0795c4a49a224e321e9b370975c14c
- Current snapshot inspected: https://github.com/uwplasma/vmex/tree/4632dad8261ca72756819c1c5fcd2e2ec022aeaa
- Dependency planning commit: https://github.com/uwplasma/vmex/commit/4632dad8261ca72756819c1c5fcd2e2ec022aeaa

Relevant paths are `vmex/core/implicit.py`, `multigrid.py`, `residuals.py`, `strong_force.py`, `extender.py`, `freeboundary_implicit.py`, `polish_driver.py`, `restart.py`, `profiles.py`, `setup.py`, `tracing.py`, and the associated tests. The benchmark's source-review ledger records which ranges were actually inspected. Read the exact reachable implementations before patching them; this handoff does not claim a completed whole-tree semantic audit.

The fixed-boundary implicit documentation makes the frozen/gauge-defined residual and root anchoring particularly important. Current free-boundary source includes anchoring even where older narrative documentation says otherwise. Current dependency planning motivates separate, coherent VMEX and DESC environments. None of these source facts establishes that a benchmark has passed.

## R05. A deterministic residual is different from a legacy iteration policy

- https://github.com/proximafusion/vmecpp/issues/624
- https://github.com/proximafusion/vmecpp/pull/626

The maintainers' resolved issue describes previous-residual-dependent m=1 force projection in a reusable 3-D operator. The original stale-buffer/TCON diagnosis was explicitly corrected. The policy persisted with TCON0=0. The resolution separates deterministic evaluation from legacy iteration behavior.

Application here: test high-to-low and low-to-high residual histories, repeated evaluation, fresh/reused runtime, and the actual differentiated residual. Do not infer the same defect in VMEX without reproducing it. Pin a comparator revision containing the fix and inspect the exact policy before interpreting cross-code differences.

## R06. Spectral condensation

S. P. Hirshman and J. Breslau, *Explicit spectrally optimized Fourier series for nested magnetic surfaces*, Physics of Plasmas 5 (1998), DOI 10.1063/1.872954.

- https://doi.org/10.1063/1.872954
- Institutional publication record: https://collaborate.princeton.edu/en/publications/explicit-spectrally-optimized-fourier-series-for-nested-magnetic-/

The publication record describes an explicit solution of the spectral-condensation equation. This review used the accessible bibliographic/abstract record; retrieve the full paper before implementing its detailed formula. Use it as a concrete comparator for gauge initialization and spectral-width objectives, not as evidence that the present constraint is harmless or should be removed.

## R07. Regular interior coordinates for strongly shaped toroids

Z. Tecchiolli, S. Hudson, J. Loizu, R. Koeberl, F. Hindenlang and B. De Lucca, *Constructing nested coordinates inside strongly shaped toroids using an action principle*, arXiv:2405.08173 (2024).

- https://arxiv.org/abs/2405.08173
- https://arxiv.org/html/2405.08173v1

The proposed coordinate action uses squared Jacobian and radial-length information and is tested with a Fourier-Zernike mapping. Application here: bounded regular-chart initialization or a comparison against boundary interpolation, particularly near the difficult sheared chart. The paper motivates the method; it does not certify a new implementation or prove that an arbitrary chart optimization preserves the finite VMEX field.

## R08. Independent spectral equilibrium representation

D. Panici, R. Conlin, D. W. Dudt, K. Unalmis and E. Kolemen, *The DESC stellarator code suite. Part 1. Quick and accurate equilibria computations*, Journal of Plasma Physics (2023), DOI 10.1017/S0022377823000272; arXiv:2203.17173.

- https://arxiv.org/abs/2203.17173
- https://doi.org/10.1017/S0022377823000272
- Reviewed code pin: https://github.com/PlasmaControl/DESC/tree/4f48720beac3d4169e9165923d730445118bc2de

This is the correct Part 1 title; an earlier project reference used a different title. The Fourier-Zernike radial/angular representation is useful for independent native-field/current comparisons. Benchmark the actual finite boundary and physical norm; do not transfer reported performance rankings from another study to these cases.

## R09. Perturbation and continuation

R. Conlin, D. W. Dudt, D. Panici and E. Kolemen, *The DESC stellarator code suite. Part 2. Perturbation and continuation methods*, arXiv:2203.15927; Journal of Plasma Physics.

- https://arxiv.org/abs/2203.15927
- https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/desc-stellarator-code-suite-part-2-perturbation-and-continuation-methods/5766F6B713EC93D438A35705F2C1E861

Application here: continuous exact-family changes and nearby physical perturbations, with the selected branch and parameter constraints explicitly tracked. Perturbative prediction is an initial guess, not a substitute for reconverging and independently scoring the final state.

## R10. GVEC as a separately scoped comparator

- Official documentation: https://gvec.readthedocs.io/v1.3.x/
- Versioned software record: https://zenodo.org/records/15026781

The documentation describes Fourier angular bases, arbitrary-degree radial B-splines and alternative geometry mappings. The cited Zenodo record is a particular release, not a claim to be the latest. Use a pinned, small native comparison to study radial order and chart choices. Validate input profiles, angles and flux normalization first. This review did not build or run GVEC.

## R11. High-order singular surface quadrature

D. Malhotra, A. J. Cerfon, M. O'Neil and E. Toler, *Efficient high-order singular quadrature schemes in magnetic fusion*, arXiv:1909.07417; DOI 10.1088/1361-6587/ab57f4.

- https://arxiv.org/abs/1909.07417
- https://doi.org/10.1088/1361-6587/ab57f4

Use the high-order surface-integral approach as an independent reference for exterior field/source decomposition and target-distance convergence. A small boundary-normal error alone does not certify all vector components or the near-surface derivative.

## R12. An axisymmetric virtual-casing reference

E. Toler, A. Cerfon and D. Malhotra, *Direct, simple, and efficient computation of all components of the virtual-casing magnetic field in axisymmetric geometries with Kapur-Rokhlin quadrature*, arXiv:2404.02799 (2024).

- https://arxiv.org/abs/2404.02799

This provides a useful separately implemented axisymmetric reference before the full 3-D exterior study. Match principal-value and one-sided limiting conventions. The paper supplies a method for field evaluation, not an exact free-boundary coil realization for every analytical interior.

## R13. Equilibrium shape derivatives

E. J. Paul, T. Antonsen, M. Landreman and W. A. Cooper, *Adjoint approach to calculating shape gradients for three-dimensional magnetic confinement equilibria. Part II: Applications*, arXiv:1910.14144.

- https://arxiv.org/abs/1910.14144

Use the continuum shape-derivative viewpoint to distinguish physical boundary displacement from a coordinate relabelling. It is complementary to testing VMEX's discrete implicit operator, not an automatic identity for its constrained finite representation.

## R14. Gradient-based physical optimization

E. J. Paul, M. Landreman and T. Antonsen, *Gradient-based optimization of 3D MHD equilibria*, arXiv:2012.10028.

- https://arxiv.org/abs/2012.10028

Relevant precedents include explicit shape/physics objectives and geometric regularization. Apply that discipline to the final physical optimization; first establish the analytical-family response and independent final error. Do not make a novelty claim for using an adjoint in stellarator optimization.

## R15. Actual JAX transformation support

- https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html
- https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html

`custom_vjp` defines a custom reverse rule and precludes direct forward-mode differentiation. Public grad/JVP/VJP/Hessian support must be tested separately. A residual-based tangent solve can be useful even when the public solve wrapper is reverse-only; label it accurately.

## R16. Transparent VMEC implementation context

- VMEC++: https://github.com/proximafusion/vmecpp
- Educational VMEC documentation: https://educationalvmec.readthedocs.io/en/latest/

Use documented intermediate physical and constraint stages to understand, not merely compare, a final FSQ. Verify the exact revision and conventions before importing a formula. Shared VMEC discretization makes these strong implementation comparators but not wholly independent continuum oracles.

## R17. New alternative parameterizations: context, not a first deliverable

A recent search also located *VMEC-comparable magnetohydrodynamic equilibria by physics-informed neural networks*, DOI 10.1017/S0022377826101548 (2026).

- https://doi.org/10.1017/S0022377826101548

This is a possible later parameterization comparison. The present review does not establish its accuracy, cost or suitability for these analytical families. Do not replace the small established solver/scorer study with training a neural equilibrium model merely because a recent method exists. The immediate discriminating tests do not require it.

## R18. Adjacent implementation scope

| Repository | Proposed use; pin when reached |
|---|---|
| https://github.com/uwplasma/SOLVAX | Imported tangent/transpose operators, factor reuse and true residuals |
| https://github.com/uwplasma/booz_xform_jax | Physical Boozer reconstruction, symmetry and parameter response |
| https://github.com/hiddenSymmetries/booz_xform | Independent transform comparator |
| https://github.com/uwplasma/virtual_casing_jax | Current-source/exterior quadrature and fixed-plan derivatives |
| https://github.com/hiddenSymmetries/virtual-casing | Independent source-field implementation |
| https://github.com/uwplasma/ESSOS | Coil kernels, tracing and field interface |
| https://github.com/uwplasma/NEO_JAX | Effective-ripple geometry/integration checks |
| https://github.com/uwplasma/DKX | Scoped kinetic geometry and normalization handoff |
| https://github.com/uwplasma/GKX | Geometry/curvature/drift data and optional isolated eigenpair response |
| https://github.com/uwplasma/pyQSC_JAX/pull/2 | Optional compatible near-axis field jets, at a pinned draft head |

These are task-specific interfaces, not claims of a completed source audit or a completed benchmark of the whole organization.

## Review coverage and unresolved literature work

Source-level conclusions in this package come from the named source paths and reports. Method selections draw on primary papers, official documentation, the authors' supplement and maintainer issue/PR records. Full derivations of a selected external algorithm must still be checked against its paper before implementation. Several candidate methods were assessed at abstract/documentation level only; no new code-comparison performance claim is made from those sources.

The handoff deliberately prioritizes resolving the actual measured discrepancy over accumulating a larger bibliography. Establishing novelty of a final coordinate/regularity result will require a focused comparison with the full spectral-condensation and equilibrium-regularity literature after the numerical outcome is known.
