# Review of d5484d1: what changed and what to repair next

**Reviewed repository:** `rogeriojorge/vmex-benchmark-analytical`  
**Commit:** `d5484d1e7a15c9cf10599c2b0f05bdf69e22e861`  
**Previous reviewed commit:** `575f13f67b346118c3d7f05cc60cc6e29131359a`  
**Review date:** 2026-09-24

Source links and primary-method references are in [REFERENCES.md](REFERENCES.md). Findings below distinguish code-confirmed issues, recorded measurements, mathematical deductions, and untested implementation leads. No VMEX/DESC solve or repository test-suite rerun was performed during this review. Six independent probes were executed; their exact scope and results are in [results/review_probes.json](results/review_probes.json).

## Progress to retain

The two new commits implement substantial parts of the preceding handoff. The TCON plotter no longer modifies primary reports. Historical amendments identify the metadata previously injected by plotting and exclude it from new summaries. The runner reserves unique directories and records requested/effective controls. Reference point clouds and solver labels have been separated, including DESC's own `rho**2`. The original history and handoff are archived rather than erased. Do not describe those repairs as still absent. [S1, S2, S6]

The new independent NumPy complex-step reference reproduces the legacy samples closely and corrects full-torus volume normalization. The refined grids show that the earlier 96-point current/force norms were not representative of the denser samples. A pressure audit distinguishes a negligible input-polynomial error from native/reference flux-label displacement. These are useful findings even though radial quadrature remains unresolved. [S2-S4]

The complete axisymmetric input-map finite differences vary boundary, pressure, current convention, flux, volume and beta together. The corrected Ampere comparison separately tests raw AC coefficients and CURTOR normalization. This is a better starting point for actual solver responses than a PHIEDGE-only smoke test. It is not yet an equilibrium adjoint test. [S5]

## F01. Reference-only CI now imports a solver it does not install

**Status:** dependency defect confirmed by source; current CI failure independently observed. Detailed CI traceback was not available through the review connector.

`tests/test_measurement_reference.py` imports `_composite_spread` and `_parent_comparison_rows` from `verify_measurement.py`. Importing that driver imports `vmex` and private VMEX field routines. `requirements.txt` contains NumPy, SciPy, JAX, Matplotlib, SymPy and pytest, not VMEX. The reference workflow installs that file and runs all tests. At the reviewed SHA, GitHub Actions run `36061177756` completed with failure in the pytest step; subsequent reference scripts were skipped. The reported 47 local passes and this clean-environment failure are not contradictory: they are different environments. [S3, S7]

**Repair:** move the pure acceptance/ancestry helpers into the existing solver-independent measurement/evidence modules. Keep VMEX imports behind the integration driver boundary. Do not solve this by making a large solver stack mandatory for every reference test, or by skipping the entire pure-reference module. Run the reference suite in a fresh environment with only declared reference requirements. Add one separately pinned integration job or explicit local command for solver tests.

## F02. The new runner and measurement driver disagree on report schema

**Status:** code-confirmed integration defect.

The new `run_vmex.py` writes schema 2 with `source.commit`, `solver_converged`, and `controls.effective`. `verify_measurement.py` still reads top-level `vmex_commit`, `converged`, and legacy scalar controls when attaching a sibling `forward.json`. Its source comparison therefore cannot recognize a correctly produced schema-2 run. This silently downgrades lineage and convergence to unknown. [S3, S6]

**Repair:** a small pure reader that normalizes the two explicitly supported schemas into one internal record. Keep the original bytes. Reject contradictions; retain unknown fields for historical records instead of guessing. Test both schemas, a capped state, a wrong source pin, a dirty source, and absent metadata. The scorer's hard-coded `root_certified=False` is an honest current limitation, not a value to flip after fixing the schema.

The runner's `_stage_controls` also uses `np.resize` on short stage arrays, which repeats a sequence. Verify that this matches the pinned solver's intended padding policy before using uneven NS/FTOL/NITER arrays. Prefer an explicit validated stage specification over silently cycling controls. This is a review lead, not a reproduced solver error.

## F03. The proposed composite quadrature is not generally aligned with native knots

**Status:** code-confirmed coordinate mismatch; its quantitative contribution to the NS33 discrepancy is not yet measured.

`reference_grid` partitions reference normalized flux and maps it to positions using the exact geometry. `_evaluate_grid` then inverts these positions to the numerical state's normalized flux. The interpolation knots in the historical native field evaluator are at `s_native = j/(NS-1)`. On displaced surfaces, `s_native(x_reference(s_reference,theta,phi))` differs from `s_reference`. A reference-grid breakpoint at `j/(NS-1)` therefore does not generally lie at a native spline breakpoint. Nevertheless `_composite_spread` labels the partition as native-knot aligned. [S3, S8]

The existing grid test proves coverage of each *reference* interval and correct full-torus volume. It cannot prove native-knot alignment because it never supplies a displaced numerical map.

**Repair:** either integrate directly on native cells with the native physical Jacobian and clearly label the native domain, or find the crossings of native knots along the reference integration curves and split there. Do not quietly change the integration domain to improve a score. Test a controlled displaced chart before an expensive VMEX run.

**Executed illustration:** the probe uses `s_native=s+0.12*s*(1-s)` and a knot at native flux 3/8. Its reference crossing is approximately 0.34778049, not 0.375. A piecewise-polynomial integral has error about 7.75e-7 with order-3 quadrature on the uncorrected reference partition, versus 6.94e-17 after adding the actual crossing. This demonstrates the mechanism, not that it explains all VMEX error.

## F04. Composite point counts are hard-coded to NS33

**Status:** code-confirmed; arithmetic reproduced by a standalone probe.

`COMPOSITE_GRIDS` contains `nradial=64,128` for orders 2 and 4. The driver replaces the number of cells with `state.R_cos.shape[0]-1` but does not update `nradial`. `reference_grid` requires `nradial=radial_cells*radial_order`. Thus the advertised same-contract extension to NS65 or NS129 fails its own grid validation. [S3]

**Repair:** derive the node count from actual breakpoints and order. Test NS33/65/129 and a nonuniform knot vector. The quadrature interface should take breakpoints and order, not redundant mutable counts. For adaptive spline states, use actual knots rather than NS alone.

## F05. The convergence predicate requires inaccurate coarse rules to pass forever

**Status:** code-confirmed logic limitation.

`_composite_spread` requires order 2 and order 4, and demands that each order's Gauss/midpoint spread satisfy the final one-tenth-target allowance. An inaccurate order-2 midpoint estimate continues to veto acceptance even after finer rules converge. There is no general order-8/16 progression. `_fine_spread` also selects hard-coded 64x64 angular grids, while the acceptance logic does not explicitly require a successful angular-resolution comparison. [S3]

**Repair:** judge the final two or three comparable refinements, not permanent accuracy of every coarse rule. Keep coarse discrepancies in the report. Independently track radial order, cell subdivision, poloidal/toroidal resolution and phase shifts. Shift agreement is a useful alias check, not a replacement for angular refinement.

Separate a diagnostic estimate of a clearly failed state from a publication-quality norm and from an accepted recovery. The current force error is of order two, compared with a target of 1e-3. Spending indefinitely to estimate that failure within 1e-4 absolute is not a prerequisite for investigating its source or testing a different case. The plan introduces bounded diagnostic tolerances without relaxing accepted-recovery gates.

## F06. Inversion and geometry diagnostics are reported but not all are acceptance gates

**Status:** code-confirmed acceptance-contract gap.

The measurement code computes position backward error, inverse-Jacobian identity error, determinant sign and condition estimates. The spread functions do not include those tests in `measurement_resolved`. The reported `reference_jacobian_orientation_consistent` is actually calculated from the native chart Jacobian. Finite-sample norms and a correct volume are not sufficient evidence that every coordinate inversion is valid. [S3]

**Repair:** reject nonfinite/invalid data before reductions, require signed orientation consistency and a dimensionless inverse residual, and set backward-error tolerances from physical scales and field sensitivity. Keep small geometry determinant due to the coordinate axis separate from an actual fold. Add negative tests for a failed inverse with plausible finite fields, NaNs, sign changes and mixed domains. JSON's NaN refusal is a last serialization guard, not numerical validation.

## F07. Chunked computation is not yet streaming, and interruption loses completed work

**Status:** code-confirmed and supported by an actual interrupted run.

`_evaluate_grid` appends numerous full field, tensor, coordinate and Jacobian arrays and concatenates them after the batches. This happens even when `keep_arrays=False`. The main loop retains completed rows in RAM and writes them only after the profile finishes or an ordinary `Exception` is caught. `KeyboardInterrupt` is not an `Exception`. The composite run was interrupted after 27:52 while processing `cell4_gauss`; the running report has no saved grid rows, and three earlier values survive only in a separate stdout receipt. [S3, S4]

The publication log also records a 55.16 MB `finest_samples.npz` committed to Git. Preserve it, but do not repeat this as the default storage pattern. [S2]

**Repair:** reduce sufficient statistics and radial-cell contributions during evaluation; save a bounded deterministic audit sample and explicitly requested full arrays only. Write each completed grid/cell artifact and its hash before starting the next one. Record interruption as a terminal receipt, then re-raise. A replacement run can reuse compatible completed artifacts without reusing the interrupted run ID or pretending stdout is certified data. Bound peak live arrays as well as process RSS; a batch-size setting alone is not a memory bound.

## F08. Route A/B agreement is primarily implementation consistency, not independent curl validation

**Status:** code-confirmed dependency overlap.

Route A first inverts the requested Cartesian points, then calls `set_points_flux` and the public stored-point B/gradB methods. Route B calls `_cartesian_derivative` using those same recovered coordinates. The public methods and the private route share the native field representation and derivative kernel. Their near-roundoff B/J agreement is therefore expected and does not independently verify that kernel. The inverse-versus-forward pressure-gradient comparison remains useful. [S3, S8]

**Repair:** retain A/B as a consistency test, add a small true direct-Cartesian call at the original requested positions, and add an independently implemented covariant curl or an independently evaluated spline/Fourier chart on a bounded point set. State shared dependencies in the result. The NumPy analytical oracle is already valuable; do not discard it.

## F09. Derivative order must respect the radial reconstruction's regularity

**Status:** mathematical deduction from inspected source; not a claim that every high-order VMEX query is wrong.

The historical native radial reconstruction uses not-a-knot cubic splines of regularized Fourier amplitudes. Away from the magnetic axis these amplitudes and the geometry are generically C2 at simple knots. The native magnetic field contains first geometry derivatives through the Jacobian, so B is generically only C1 across those knots. Its first derivative and current can be continuous, while second B derivatives can have jumps and third derivatives need a piecewise/distributional interpretation. A JAX array returned at a knot does not establish the existence of a classical derivative there. [S8, L3]

**Repair:** report interior-cell derivative errors and one-sided jumps separately. Audit the axis limit, radial interpolation and actual knot multiplicities. A smoother reconstruction or a high-order native state is a separate tested representation, not a silent replacement in the benchmark. For a sufficiently smooth simple-knot quintic geometry, C3 magnetic-field continuity is possible, but boundary/axis conditions and nonlinear Jacobians must still be checked.

The included local C2-map probe exhibits a finite jump in the second derivative of a divergence-free reconstructed field. It is a regularity illustration, not a toroidal MHD benchmark.

## F10. The complete input-map check needs signed, dimensionless block diagnostics

**Status:** code-confirmed limitation; reference work remains useful.

`full_input_derivatives.py` concatenates geometry coefficients, pressure in Pa, current in A, flux in Wb, dimensionless profiles, volume and beta into one vector. Its global relative norm is dimensionally uninformative and can be dominated by current/pressure entries. The returned report retains only the norm of the finest derivative in each block, not its signed components. It cannot serve directly as a saved tangent RHS. [S5]

**Repair:** record fixed output ordering, signed derivatives, physical scales, per-block convergence and absolute tests for known-zero channels. Keep profile degree and sample/fitting grids fixed within a derivative experiment. A verified finite-difference input tangent is an acceptable first RHS for a VMEX residual tangent solve; label it as such rather than waiting for a complete all-JAX conversion. Add the differentiable fixed-basis map afterward and compare both.

The scaling probe has a 50% error in a boundary derivative hidden by an unscaled combined relative error of 5e-10. This is a generic counterexample to the norm, not a measurement of the repository's boundary derivative error.

## F11. Provenance compatibility needs semantic checks, not only four matching strings

**Status:** code-confirmed limitation.

Parent-grid ancestry validates source commit, case name, state hash and input hash. It does not require compatible scoring code, dirty patches, normalization, parameter values or integration-domain definitions. Code hashes are recorded, which is good, but a changed evaluator can still enter the same convergence ladder. `source_metadata` identifies the nearest Git root; an installed package nested in a benchmark checkout need not be tracked by that repository. [S3, S6]

**Repair:** add an explicit measurement-contract hash covering domain, model, units, field route and grid semantics. Compare actual imported tracked files or package contents, not merely a nearest ancestor repository. Deliberate evaluator comparisons should have their own comparison records rather than being silently pooled as quadrature refinements.

## F12. Adjacent high-order recovery offers reusable methods, not a ready-made success

**Status:** current PR evidence reviewed; implementation not fully audited here.

VMEX PR 448 remains draft and unmerged. Its sparse support/coloring, exact knot insertion and constrained linear certificates are relevant to later refinement and polishing. Its current implementation scope is axisymmetric fixed-profile research. The reported provisional target crossing lacks the full independent point certificate, and measured process RSS is much larger than the explicit sparse-array estimate. [S10]

Most importantly, its `epsilon_B` is a magnetic-scale **force** measure. It is not this repository's `E_B=||B_h-B_exact||/||B_exact||`. Do not put them in the same error column. Benchmark this branch only as a pinned optional experiment with its stated scope; do not merge it as a prerequisite.

## Review scope

The review inspected the new core measurement and input-derivative implementations, their relevant tests, the current runner/evidence/plotting contracts, current plan/logbook and selected immutable records; it traced the historical native spline/field/inversion implementation; and it checked current release/PR/CI state. It did not semantically review every VMEX module, all adjacent repositories, every older plotting script, or PR 448's complete implementation. The local agent should continue the existing source ledger along the actual dependency paths, not label an inventory as semantic verification.
