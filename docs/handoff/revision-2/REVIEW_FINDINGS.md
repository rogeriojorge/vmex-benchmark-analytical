# Review of the analytical benchmark repository

Review date: 2026-09-24. Reviewed benchmark snapshot: `575f13f67b346118c3d7f05cc60cc6e29131359a`. This review concerns the committed source and saved evidence. No new VMEX, DESC, GPU, free-boundary, or kinetic solve was executed for this review. Three small independent mathematical probes were executed; their scope is recorded in `audit_probe_results.json`.

## Assessment

The work has moved beyond preparation. It has implemented input conversion, projected analytical states, performed nonlinear recovery attempts, compared native VMEX and DESC fields, investigated the coordinate constraint, and retained unsuccessful runs. The distinction between an exact-state projection and a solved equilibrium is particularly valuable. The experiment that disables the constraint on the *same state* isolates a large contribution to the reported discrete residual. It does not, on its own, establish the cause of all physical-field errors or justify changing VMEX's default constraint.

The immediate bottleneck is now the reliability and interpretation of the measurements, not a lack of additional configurations. Several defects in the benchmark harness should be repaired before extending the matrix. Existing numerical arrays should be retained, with amended metadata and stronger rescoring where possible. There is no basis for discarding all previous work or for claiming that its physical arrays are fabricated.

## Evidence already obtained

| Evidence | Supported conclusion | Limit |
|---|---|---|
| 29 reference tests and 14 configurations | Several independent algebraic, quadrature, symmetry and normalization checks pass in the recorded environment | These tests do not cover the new execution/reporting paths comprehensively |
| 28 parsed input decks | Both closures are converted and accepted by the pinned parser/setup; stronger B/C boundaries have been refined | Input smoke tolerance is not the final solution error budget |
| Axisymmetric integer recovery through NS=129 | Small sampled field/current/force errors occur for both closure choices | The 96-point quadrature has not been independently converged |
| Asymmetric Solov'ev WOUT surface refinement and round trip | LASYM surface values and serialization are exercised | Native Cartesian LASYM current/force remains unverified |
| Integer 3-D exact-state projection | The finite representation can approximate the analytical field substantially better than the tested solved states | Projection is not nonlinear recovery |
| Same-state TCON0 switch | Large projected R/Z residuals are dominated by the coordinate-constraint contribution | A discrete coordinate residual is not a physical force norm |
| Nine TCON ladder solves | All met the recorded discrete stop and all missed at least one sampled physical threshold | No complete radial/angular/tolerance or quadrature study; NS129 projected zero remains unrun |
| DESC base/remapped comparison | Finite representation and nonlinear recovery depend on coordinates; an independent radial representation is now available | Different boundaries/resolutions; one M8 remapped terminal state is unconverged |
| Local sheared-map derivatives | The selected implicit angle inversion differentiates accurately at tested interior points | Whole input-map derivatives, all branches and all radii are not certified |

The axisymmetric integer and the present symmetric Solov'ev control are explicitly checked to give the same magnetic field by `test_two_independent_axisymmetric_field_formulas`. They are useful independent formulas, but should not be counted as two independent physical equilibrium cases. Add a different Solov'ev parameter set when claiming breadth.

## Findings requiring action

### F01. Plotting modifies primary evidence -- confirmed, highest priority

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/benchmarks/plot_tcon_ladder.py).

`benchmarks/plot_tcon_ladder.py`, notably its `CASES`, `RESOLUTION_CASES` and first reporting loop, supplies iteration counts from constants, assigns package versions and accelerator text, reconstructs commands, modifies `accepted`, and writes both `forward.json` and `native_scores.json`. It then hashes the modified reports. Some batch-comparison measurements also appear as literals rather than references to immutable observations.

**Consequence:** a figure-generation command can change the purported measurement record. A checksum of the modified file cannot establish what the runner originally observed. The three-threshold acceptance expression is also insufficient for future certification, even though the currently listed cases fail it.

**Repair:** archive the reviewed bytes, make renderers read-only, and generate only derived summaries/figures. Move recoverable historical corrections to a separate amendment record with field, old value, new value, reason and evidence. Unknown versions or iteration counts remain unknown. New runs must collect observations from the executing environment. Hash inputs, raw outputs and executable source separately. Add a test that plotting leaves every input byte unchanged and that a numerically small but unconverged state cannot be accepted.

### F02. The common 96-point grid is an initial diagnostic, not a certificate -- confirmed design limitation

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/benchmarks/native_samples.py).

`native_samples.reference_points` defaults to three Gauss-Legendre radial nodes, eight poloidal nodes and four toroidal nodes per field period. The radial nodes are approximately 0.1127, 0.5 and 0.8873 in normalized toroidal flux. This does not directly sample the innermost or outermost eleven percent of the radial label. Periodic modes can vanish at every angular node: `sin(4*theta)` does so on eight equally spaced poloidal nodes. The declared VMEX angular truncation in the TCON summary is MPOL=13, NTOR=12.

**Repair:** retain the 96-point grid as `legacy96`. Add independent shifted and refined integration grids, explicit near-axis/edge diagnostics, and tests for aliasing. Establish convergence of each reported norm on a fixed state before spending effort on more equilibrium resolutions. Do not assume that matching the number of solution modes is enough: metric factors and inverse Jacobians generate additional spectral content.

The standalone probe measures an RMS of about `5.1e-16` for that mode on eight points, versus `0.70710678` on a resolved shifted grid. This is a mathematical demonstration, not a measurement of hidden error in any saved VMEX state.

### F03. One-period and full-torus volume conventions are mixed -- source-level inconsistency

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/benchmarks/native_samples.py).

The reference sampler integrates physical phi over `2*pi/NFP`. The LASYM fitted-state sampler uses a field-period coordinate and applies an additional NFP factor. Under the documented `strong_force` coordinate convention these represent different volume conventions. Relative RMS ratios can be unchanged by a common constant multiplier, while `quadrature_volume_m3` changes by NFP.

**Repair:** verify the pinned Jacobian definition, then make the sample contract explicit: integration region, angular variable, Jacobian convention and replication factor. Test the integral of one against an independently known volume for NFP=1, 2 and 3. Never correct historical ratios unnecessarily; amend absolute integrals and metadata only after checking the actual arrays.

### F04. Physical-coordinate inversion lacks a complete acceptance test -- confirmed omission

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/benchmarks/native_samples.py).

The native sampler checks finite values but does not itself record forward reconstruction error, inverse-Jacobian consistency, or a numerical-domain inclusion certificate. Finite coordinates can still be inaccurate or on an unintended branch.

**Repair:** check `X(q(x))-x`, valid native radial range, orientation, and `D_q X * D_x q - I`. Compare direct forward-chart fields and derivatives with the Cartesian inversion route on the same state. Isolate interpolation and inversion error before blaming the equilibrium iteration. Keep failure points; do not silently discard them or shrink the scoring region until a run passes.

### F05. A DESC score can inherit a VMEX flux label -- confirmed data-flow defect

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/benchmarks/run_desc_coordinate.py).

`run_desc_coordinate.sample_score` copies the full input reference dictionary, replaces B, J and grad-p, but leaves `vmex_s` in it. The shared scorer can therefore attach a VMEX flux-label discrepancy to a DESC record without using DESC's newly inverted `rho**2`.

**Repair:** store the common point cloud separately from solver observations. Adopt neutral keys `s_reference` and `s_native`, with `s_native` supplied by the solver under test. Never copy solver-specific state into another solver's record. Test deliberately different native labels. The field/current arrays need not be invalid merely because the inherited label diagnostic is invalid.

The same driver obtains `source_commit` with `git rev-parse HEAD` in the working directory, not from the imported DESC source directory. Preserve a benchmark commit and a distinct DESC source commit, each resolved from the relevant path. Its forced GPU selection and separate permeability constant should also become explicit, consistent run settings.

### F06. Effective controls and failed-run memory need correction -- confirmed code paths

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/benchmarks/run_vmex.py).

`run_vmex.py` only applies FTOL/NITER overrides when an NS override is present, while reports can still show the override/default scalar rather than the actual input arrays. Early peak RSS is sampled before solving and reused on failure paths. Repeated commands reuse the same output directory.

**Repair:** report actual per-stage controls from the resolved input/runtime; either apply explicit overrides consistently or reject ambiguous use. Sample RSS after each stage and distinguish process high-water mark from stage-local memory and device memory. Preserve separate immutable attempt IDs. Save terminal states when available, but do not convert their diagnostic scores into convergence. Hash the seed, effective input, benchmark source and installed solver source, including dirty patches.

### F07. The m=3 gauge scan introduces a nonanalytic axis chart -- derived source-level limitation

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/benchmarks/probe_integer_gauge.py).

`probe_integer_gauge.project_shift` uses `u=A*s*(1-s)*sin(m*theta-n*NFP*phi)` for both m=2 and m=3. With `rho=sqrt(s)`, the m=3 choice has `u~rho**2 sin(3 theta)`. For a circular transverse map, its first position variation contains `rho**3 cos(4 theta)`. An analytic polar map instead requires an m=4 harmonic to start at rho**4 with the usual parity.

This does not invalidate the physical analytical equilibrium, nor the selected m=2 remap. It makes m=3 unsuitable for a clean axis-regular spectral comparison. The probe in this bundle measures the induced order as approximately 3.000; a rho**3 gauge envelope yields the required order of approximately 4.000.

**Repair:** use axis-regular scalar gauges, for example `rho**m*(1-rho**2)**2*P(rho**2)` for m>=1, then prove/test the full mapped R/Z/lambda regularity. Check invertibility over the volume. Do not minimize FSQR+FSQZ by using charts outside the representation's regularity class.

### F08. Flux-label displacement is not a pure poloidal gauge displacement -- interpretation correction

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/results/vmex/tcon_resolution_ladder/summary.json).

With `s=Phi_t/Phi_edge` fixed by oriented toroidal flux, a relabelling of the poloidal angle leaves s unchanged at a given physical point. A measured `s_native-s_reference` can indicate different physical surfaces, an inversion problem, different flux normalization, or an improperly transformed input. Calling it coordinate drift without resolving those alternatives obscures the physical question.

**Repair:** record poloidal gauge displacement and flux-surface displacement as different metrics. Pure-gauge tests must have zero Eulerian field response and zero normalized-flux response, within projection/inversion error. Compare pressure as well as B and J.

### F09. The sheared graph guard is a chart guard, not an existence test -- scope correction

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/benchmarks/analytic.py).

The generation guard samples the edge and tests monotonicity of physical phi while holding the chosen auxiliary poloidal label fixed. The separate diagnostic samples additional radii. A fold in this one parameterization invalidates its quadrant inversion; it does not prove that no other cylindrical-angle parameterization, or no physical equilibrium, exists.

**Repair:** name the guard accordingly, validate all radii used, record a derivative margin, and reject unsupported inversions. Before making an existence claim, examine the full surface projection and possible changes of poloidal chart. Differentiate only a selected regular root branch. The complete pressure/current/flux/boundary input map remains a separate task.

### F10. Source inventory is not semantic review; add tests for the new harness

[Source at the reviewed commit](https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/575f13f67b346118c3d7f05cc60cc6e29131359a/tests/test_reference.py).

The ledger honestly records hundreds of inventoried files but only partial semantic reviews. Preserve that distinction. The reference suite should not be used as evidence that the newer report generation, solver dispatch, DESC labels or failed-run handling is correct.

Prioritize reachable source slices for the next experiment: boundary/profile setup, native/physical basis conversion, m=1 policy, force/constraint stages, root anchoring, interpolation/inversion and serializers. The wider module review can continue alongside experiments. Do not impose an all-organization audit as a serial prerequisite for the next useful physical test.

## Findings that must not be promoted into claims

- The TCON switch does not prove that the constraint should be removed globally.
- A low sampled error is not a converged continuum norm.
- A history-dependent-force defect documented in VMEC++ is not evidence that VMEX has the same defect. It motivates a reproducible operator-purity test.
- A small singular value of an unscaled coordinate Jacobian is not by itself a physical instability or an equilibrium singularity.
- A low DESC objective does not remove its boundary-fit error, and an iteration-capped state is not a recovered solution.
- Failed LASYM fitted reconstruction is not the same failure as an incorrect LASYM nonlinear equilibrium.
- A prescribed iota profile is not an independently measured transform.
- Seeing two devices in metadata is not evidence that a solve used both devices.
- Exact interior force balance does not supply a vacuum exterior, a coil realization, ideal stability, omnigeneity or an exact kinetic transport coefficient.

## Recommended disposition

Preserve the present release of evidence as a historical diagnostic dataset. Amend the harness first, rescore saved states, close the NS129 zero-constraint gap, then separate constraint/gauge effects from physical recovery and derivative convergence. Run the axisymmetric sensitivity and sheared-A work in parallel so that the full program does not stall on one rational-transform example. The new schedule and mathematical contracts are in `plan_revision_2.md`; file and literature references are in `REFERENCES.md`.
