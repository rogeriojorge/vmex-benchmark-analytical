# VMEX analytical benchmarks: continuation after d5484d1

**Owner:** `rogeriojorge`
**Repository:** `https://github.com/rogeriojorge/vmex-benchmark-analytical`
**Review date:** 2026-09-24
**Reviewed main:** `d5484d1e7a15c9cf10599c2b0f05bdf69e22e861`
**Preceding reviewed main:** `575f13f67b346118c3d7f05cc60cc6e29131359a`

This is a continuation of the existing implementation, not a replacement project. It retains the analytical models, case matrix, useful measurements, code and historical logbook. It supersedes the immediate instruction to rerun the interrupted composite calculation unchanged. Read the preserved [review findings](docs/handoff/review-d5484d1/REVIEW_FINDINGS.md) and [pinned references](docs/handoff/review-d5484d1/REFERENCES.md) with this plan. The full continuation package is archived under `docs/handoff/review-d5484d1/`. Source labels S1-S12 and literature labels L1-L13 refer to that reference file.

The review inspected current source and recorded evidence, but did not execute VMEX, DESC, GVEC, free-boundary or kinetic benchmarks. The six delivered review probes ran independently of those codes. Their outputs illustrate numerical mechanisms and contracts; they are not additional equilibrium results.

## 1. Adoption, scientific mandate and operating rules

The active plan/logbook at `d5484d1e7a15c9cf10599c2b0f05bdf69e22e861` was archived byte-for-byte as [plan_through_d5484d1.md](docs/history/plan_through_d5484d1.md) before adopting this continuation. Its SHA-256 is `0b34804db8c0741f8f94b0e6e1af8616119fbe002cc8bc4330bea54f96fa8736`. The handoff source files and checksums are preserved under `docs/handoff/review-d5484d1/`. Keep this as the single active contract; preserve earlier plans through links rather than rewriting their history.

The benchmark has four scientific goals:

1. Establish when a solved numerical equilibrium, rather than an analytical projection or small discrete residual, recovers the exact B, J, pressure surfaces and force balance.
2. Separate errors in analytical input conversion, finite representation, nonlinear recovery, output reconstruction, coordinate inversion and numerical measurement.
3. Verify spatial derivatives and physical equilibrium sensitivities, including the effects of coordinate constraints, initialization, symmetry and current/transform closure.
4. Use verified capabilities to study exact families and nearby equilibria, while demonstrating useful coverage of field, diagnostic, exterior, free-boundary and downstream interfaces.

Do not change the global TCON0 default, rewrite the equilibrium solver, or add a general workflow framework as the first response to the current discrepancy. Prefer a small falsifiable experiment, then a narrow implementation correction. Negative results are publishable evidence when the question and numerical uncertainty are resolved; they are not passing capabilities.

All new Git writes, pushes, comments and PRs must use the authenticated `rogeriojorge` account. Verify `gh api user --jq .login`, configure author and committer only in the project/worktree, and use the verified owner email or correct account noreply address. Do not add AI authors or co-author trailers. Preserve inherited merge history and third-party notices. Use a work branch for this continuation. Necessary VMEX/adjacent-library fixes belong in separate narrow PRs, with baseline/candidate comparisons; do not merge them or push to upstream default branches. Inspect the existing publication helper before using it: it previously pushed benchmark commits directly to main and staged a broad allowlist. Do not invoke it blindly.

Never publish credentials, private machine notes, shell environment dumps or connection details. GitHub API receipts and sanitized package/device metadata are sufficient. Keep source-code comments and prose factual. No badges or statements implying full validation from a passing unit suite.

## 2. What has changed since the previous review

There are two new commits: `9df03c2` records the implementation/results and `d5484d1` records publication. The R0 evidence repairs should be retained: read-only TCON rendering, historical metadata amendments, unique run directories, explicit requested/effective controls, and independent reference/native radial labels. The source now includes an independent NumPy/complex-step oracle, full-torus weighting, pressure decomposition, radial quadrature extensions and complete axisymmetric input-map finite differences. [S1-S6]

The reported local suite increased from 29 to 47 tests. This does not establish clean-environment reproducibility: the current GitHub reference workflow failed at pytest after installing the stated requirements. The new pure measurement tests import a driver that requires VMEX, which the reference environment does not install. Detailed CI logs were not accessible during review; inspect them locally and fix the source-confirmed dependency problem first. Do not repeat the statement 'all tests pass' without naming the environment. [S7]

The latest saved NS33 measurement still has unresolved radial quadrature. The legacy 96-point sample remains reproducible, but denser grids change the result substantially. Selected reported values for the same default-TCON projected-start solved state are:

| Radial/angular rule | E_B | E_J | E_gradp | E_F,p |
|---|---:|---:|---:|---:|
| legacy96 | 8.71e-4 | 4.39e-1 | separately recorded | 4.93 |
| 16x64x64 Gauss | 1.737e-3 | 1.628e-1 | 8.069e-3 | 1.744 |
| 64x64x64 Gauss | about 1.425e-3 | about 1.830e-1 | about 8.058e-3 | about 1.972 |
| 64x64x64 midpoint | about 1.380e-3 | about 1.941e-1 | about 8.056e-3 | about 2.101 |

These are observations from committed records, not new runs in this review. Retrieve full precision from the records. Angular shifts agree closely in the tested comparisons; that does not establish convergence under arbitrary angular refinement. The current physical gates remain E_B <= 1e-5, E_J <= 1e-3, E_F,p <= 1e-3. None of the above is close to passing.

The composite run was interrupted after 27:52, during `cell4_gauss`. Its three completed printed rows are preserved only as provisional stdout in `interruption.json`. Do not reuse its run ID or cite those rows as a completed dataset. Its existing definition also partitions the reference flux label, not the true native knot coordinate, and its node counts are hard-coded for NS33. [S3, S4]

Important current lineage:

| Item | Value |
|---|---|
| Latest completed radial64 run | `ns33-projected-default-gpu-radial64-v1` |
| Radial64 report SHA-256 | `ac448fd9be924b64119c98238b9317310d9a0c98c1e6d184197dfddba850a000` |
| Interrupted run | `ns33-projected-default-gpu-composite-v1` |
| Saved spectral-state SHA-256 | `3fefa156a0a7835a36503aa3ffeb6f72905c0743f446a8784403d4550687c48a` |
| Input SHA-256 | `7a7b0cdb0c31774fa0497efd99be3874c25bbf52a3f27dd7420fd6027d05deab` |
| Corrected input-map report | `results/reference/derivative_runs/axisym-input-map-20260924T201351.596655Z/derivatives.json` |
| Input-map report SHA-256 | `56bcf3769c394117e5586480827e55f66671e6a2af7c8d85fda80d512af569db` |

The existing pressure audit finds an input-profile error near roundoff but a measurable mismatch between native and exact flux labels. This is evidence about a finite reconstructed state, not proof of a continuum solver defect. Pure angle relabelling cannot change normalized toroidal flux at a fixed physical point.

No new nonlinear recovery or VMEX response derivative was completed in the latest block. The missing historical `NS=129, TCON0=0` projected-start solve remains missing. Do not lose track of it, but do not make every useful experiment wait for a high-precision norm of the already-failed NS33 state.

## 3. Source pins and scope

Keep historical and candidate environments distinct:

| Role | Pin |
|---|---|
| Historical VMEX | `b5f5267efc0795c4a49a224e321e9b370975c14c` |
| Current VMEX inspected here, release 0.11.2 | `926892ab7131a6bc0c5b61218d1f75e7b77bc401` |
| Earlier comparison pin in the active plan | `4632dad8261ca72756819c1c5fcd2e2ec022aeaa` |
| Analytical supplement | `4c0b690ddebdc71811c88223eb9f44a98ab64222` |
| Recorded DESC | `4f48720beac3d4169e9165923d730445118bc2de` |
| Recorded SOLVAX | `2e246a5d6093662f9b5f72c46f995cd7c4bbd479` |
| Recorded booz_xform_jax | `cd25084422de10b620bd86ede0bbd51ba06d7fa6` |
| Optional VMEX PR448 head, not main | `70bfe9da0372326633bae8054042f985255456c2` |
| Optional VMEC++ PR849 head, not main | `cec07e9e06f54af38851eb2cf0dcf46f75a1aea6` |

The new VMEX release addresses compilation-cache separation by machine. That change is relevant to timing reproducibility, not evidence of an equilibrium fix. PR448 is draft research, and PR849 is an unmerged comparator. Recheck their actual state before using them. Keep separate worktrees and record exact imported-source identity and relevant dirty patches. [S8-S11]

Continue the source review along the reachable implementation paths. Inventory, semantic review, executed tests and physical validation remain four separate fields. Review field/interpolation, residual/masks, profiles and derivative routines before claiming those capabilities; review optional libraries when a concrete integration reaches them. Do not stall all experiments behind reading every unrelated repository in the organization.

## 4. Models, units and conventions that remain fixed

### 4.1 Reference families

The integer family and diagonal-stretch extension use positive a,b,c and outer pressure label delta. Let

$$q=(x/a)^2+(y/b)^2,\qquad f=\sqrt{2q-q^2-4(z/c)^2},$$
$$\mathbf B=\left(\frac{2zx/c-(a/b)fy}{q},\frac{2zy/c+(b/a)fx}{q},c(1-q)\right).$$

With

$$u_a=-\frac{a^2-b^2}{4c^2},\quad H_a=\frac{a^2+b^2}{2}-\frac{(a^2-b^2)^2}{8c^2},$$
$$H=\frac{x^2+y^2+4z^2+|B|^2}{2},\quad \psi=\frac{H-H_a}{2c^2},\quad p=2c^2(\delta-\psi),$$

require `abs(u_a)+sqrt(delta)<1/2`. The domain must stay strictly inside the positive-radicand chart. The displayed paper family has a=sqrt(1+epsilon), b=sqrt(1-epsilon), c=1. The extension changes the pressure-surface center when c changes; it is not just a vertical stretch of an unchanged old boundary. [L1; preserved model derivation]

Useful exact quantities are

$$s=\psi/\delta,\quad V=2\pi^2abc\delta,\quad\Phi_t=\pi abc\delta,$$
$$\langle B^2\rangle=H_a+c^2\delta,\quad\langle p\rangle=c^2\delta,\quad
\beta_V=\frac{2c^2\delta}{H_a+c^2\delta}.$$

Use actual values from `cases.json`, including stored rounding. Do not replace rounded a,b by symbolic square roots when checking previously generated artifacts; that would change the reference problem and its exact volume.

For the sheared family, keep the explicit field and chart in `analytic.py` and the independent supplement. Its pressure label is psi=k^2/2, but normalized toroidal flux is s=Q_t(k)/Q_t(k_b). Pressure is `(delta-psi)/lambda^2` in reference units. Invert Q_t consistently, and obtain current/transform from independently refined flux and Ampere integrals. A smooth analytical torus need not be a single-valued graph in physical cylindrical angle. Check monotonicity over the relevant radii, not just one boundary sample; reject an invalid chart rather than clipping it into a deck.

For the independent axisymmetric control, retain

$$\psi=b(R^2-R_0^2)^2+gZ^2+2\chi\sqrt{bg}(R^2-R_0^2)Z,$$
$$p=8b(\psi_b-\psi),\qquad F^2=F_0^2-4g\psi,$$

with positive-definite quadratic form, R>0 and F^2>0. The cross term is genuinely up-down asymmetric for nonzero chi. A shifted symmetry plane or a rephased 3-D field is a covariance test, not a new generic asymmetric equilibrium.

### 4.2 Dimensionalization and closures

Reference formulae use mu0=1. With fixed scales L_* and B_*, use x=L_* xbar, B=B_* Bbar, p=B_*^2 pbar/mu0, J=B_* Jbar/(mu0 L_*), and magnetic flux B_* L_*^2 times reference flux. Current from a line integral scales as B_* L_*/mu0. Record the numerical mu0 used.

`NCURR=0` prescribes iota. `NCURR=1` prescribes current and makes iota an output. Generated raw `AC` coefficients describe the derivative of the chosen enclosed-current fit, while `CURTOR` specifies the dimensional edge-current normalization; confirm the actual VMEX profile semantics. The earlier incorrect normalized-versus-raw Ampere comparison has already been corrected. Do not reintroduce it.

Compare signed physical fields, not absolute iota alone. The repository's counterclockwise R-Z poloidal convention has negative transform for these examples; the paper/DESC orientation has the opposite sign. NFP appears both in Fourier phases and physical-angle integration. A one-period volume integral needs explicit replication; normalized ratios cancel a common missing factor but absolute integrals do not.

## 5. T0: restore portable tests and compatible evidence readers

**Immediate; no equilibrium solve needed. Maps to previous R0/P0.**

Move pure measurement gates/ancestry code out of the VMEX-importing driver into the existing pure module. Keep a clean reference environment containing only `requirements.txt`, and a separate solver environment with pinned VMEX and compatible dependencies. Run both intentionally. Do not silently skip pure tests on systems without VMEX. Add a regression that imports every reference-test dependency with VMEX absent. Retrieve the current CI traceback to identify any additional failures. [F01]

Implement a small normalization reader for old/new forward reports. For schema 1 read the documented legacy fields; for schema 2 read `source`, `controls.effective`, `solver_converged`, status and artifact hashes. Keep input copy, effective settings and state provenance distinct. Verify a scorer can consume a newly generated schema-2 run before the missing NS129 experiment. Preserve raw bytes and make amendments separate. [F02]

Do not label a boundary smoke threshold as full representation convergence. Keep fields such as `input_fit_smoke_passed`, `representation_resolved`, `measurement_resolved`, `solver_converged`, `root_certified`, `adjoint_certified` and `accepted` distinct. A projection can have a resolved physical score without ever being a solved equilibrium. The measurement program must not hard-code acceptance of a root it never tested.

Validate imported-source membership, not just the nearest Git root. Include hashes of imported untracked/installed source when applicable, and never record a benchmark checkout SHA as the solver's SHA because its virtual environment happens to be nested there. Parent-grid compatibility must include domain, units, coordinate conventions, field representation, relevant code/patch identity and dtype. Intentional cross-version comparisons have explicit comparison records rather than being treated as one grid-refinement ladder. [F11]

**Exit:** clean reference tests pass; old and new records normalize correctly; deliberately incompatible records are rejected; a minimal end-to-end producer/consumer test works; no historical raw file changes. Correct the README test statement to distinguish local and CI evidence.

## 6. T1: replace the expensive unresolved scorer with bounded, coordinate-aware integration

**Immediate numerical priority; maps to R1/P2.**

### 6.1 Keep three different questions separate

At physical positions x, compute

$$E_B^2=\frac{\int_\Omega|B_h-B_e|^2dV}{\int_\Omega|B_e|^2dV},\qquad
E_J^2=\frac{\int_\Omega|J_h-J_e|^2dV}{\int_\Omega|J_e|^2dV},$$
$$E_{F,p}^2=\frac{\int_\Omega|J_h\times B_h-\nabla p_h|^2dV}{\int_\Omega|\nabla p_e|^2dV}.$$

Also retain dimensional force RMS, magnetic-scale force RMS and pressure-gradient error. Every report identifies Omega and its volume. Use the exact reference, not numerical J or grad p, in the reference denominators. Zero-gradient vacuum fixtures need a fixed magnetic scale instead of division by zero.

The three questions are: how inaccurate is this fixed state; is that estimate adequately integrated; and does a sequence of solved states converge to the exact equilibrium? A resolved estimate of a bad state is not recovery, while an unresolved norm does not prevent a bounded diagnosis of obvious failure.

Use three explicitly labelled tolerances:

- **Diagnostic failure estimate:** a proposed initial allowance `max(0.1*target, 0.01*E)` for each norm, plus local/domain checks. It supports triage, not publication-level accepted recovery. Refine uncertainty if the decision is close.
- **Reported quantitative norm:** a stated tighter relative allowance, initially 1e-3 of E for a well-above-threshold error, with independent rule/partition checks and a reported empirical uncertainty.
- **Accepted recovery:** measurement contribution below 0.1 of the physical target, input/representation/root checks, and error plus its adopted uncertainty below the physical gate. State clearly that quadrature-difference estimates are empirical, not rigorous interval bounds.

Keep initial physical gates E_B<=1e-5, E_J<=1e-3, E_F,p<=1e-3. Targeted near-axis/edge point checks get separately stated norms and thresholds; do not apply a global volume-L2 threshold indiscriminately to every local maximum or equal-weight point cloud.

### 6.2 Integrate on actual interpolation intervals

The historical native field uses uniform knots in s_native. The analytical comparison grid uses s_reference. These coordinates differ on the current solved state. There are two legitimate paths:

**Native-domain path:** generate quadrature directly in q_h=(s_h,theta_h,phi), split at all native knots, evaluate x_h(q_h), B_h and J_h without inverse coordinates, evaluate the exact field at those same Cartesian positions, and weight by `abs(det D_q x_h)`. If rho rather than s is used, include the correct ds=2rho drho conversion. Label the domain Omega_h. Check that the analytical field is defined there. Report boundary/domain differences rather than silently discarding outside points.

**Common/reference-domain path:** retain x_e(s,theta,phi). For each angular curve, find all roots of

$$s_h(x_e(s,\theta,\phi))-s_{h,j}=0$$

and split the reference integral at those crossings. Prove or test the required monotonicity in each admitted region; handle or reject multiple crossings explicitly. Near distinct magnetic axes the mapping may not be one-to-one along a ray. Use a documented common interior and separately measured excluded shells only when necessary, not as a way to hide bad points. The reference determinant remains the volume weight on this path.

Start with native-domain integration for inexpensive error localization and a sparse common-point cross-check. For the fixed-boundary comparison, quantify the domain mismatch before interpreting a native-domain norm as equivalent to the previous reference-domain norm. Once accepted boundary fits are sufficiently accurate, verify both routes agree within the measurement budget. The primary cross-code ranking, if eventually made, must use the same physical domain and metric.

Do not rename the existing reference-cell rule 'native-knot aligned'. Add a synthetic displaced-chart test using the supplied probe. Derive total node count from breakpoints and order, so NS33/65/129 and nonuniform spline states work. Test cells containing every relevant geometry/lambda/profile knot; the union, not a single table's knots, is the partition. [F03, F04, L2]

### 6.3 Stream statistics and preserve partial work

For every cell or fixed-size batch, accumulate at least: sum(w); squared error and reference integrals for B,J,grad p,force; dimensional maxima; minimum signed Jacobian and condition indicators; inversion residuals; invalid sample counts; and local contributions by radial cell. Use pairwise or compensated float64 accumulation where sums span a large range. Do not retain all 3x3 tensors for millions of points when only their norm sums are needed.

Persist each completed cell group/grid to a small immutable JSON/NPZ record, with checksums and a stable grid ID, before the next group starts. Keep a bounded deterministic audit sample, selected worst-error points and optional explicitly requested full arrays. The default result should not create another 55 MB Git blob. Large full states remain checksummed release assets when needed; do not rewrite the existing Git history to remove old files.

The run manifest may point to completed immutable children and a terminal receipt. Catch KeyboardInterrupt to save status/known completed references, then re-raise; do not mask unexpected programming failures. Use a new run ID for resumption, with explicit reuse of compatible completed children. Add a tiny interruption test that completes one grid, interrupts the next, and proves the first is recoverable and byte-identical. [F07]

Prepare JAX kernels once per state/shape. Use fixed-size batches with a valid final-batch mask; test a non-divisible sample count. Measure compile, evaluation, oracle, transfer and reduction costs separately. Only optimize a kernel after checking its outputs against the baseline. A smaller batch does not by itself limit compilation-cache RSS; use process isolation and a measured budget where necessary.

### 6.4 Independent field and curl checks

Keep the current A/B comparison, but label its shared dependencies. On a small audit set, explicitly call B(x_original) and gradB(x_original) before coordinate seeding, and compare with x_h(q_h). Gate backward-error propagation using measured field sensitivities and a fixed length scale. Require finite inputs/outputs, correct signed chart orientation, and a dimensionless inverse-Jacobian residual. [F06, F08]

Add an independently coded covariant-curl evaluator at a modest resolution. With e_i=partial_i x, signed Jacobian Jq=e_1 dot (e_2 cross e_3), and B_i=B dot e_i,

$$\nabla\times B=\frac{1}{J_q}\left[(\partial_2B_3-\partial_3B_2)e_1+
(\partial_3B_1-\partial_1B_3)e_2+(\partial_1B_2-\partial_2B_1)e_3\right].$$

Use independent spline/Fourier evaluation, or independently differentiated covariant components, rather than calling `_cartesian_derivative` twice. An implementation using SciPy's not-a-knot spline on independently extracted regular amplitudes is a reasonable small CPU comparator. This checks evaluation of the same representation, not an independent equilibrium model. Separate that limitation from the exact analytical Cartesian oracle. Confirm the orientation and tensor ordering on simple fields before the 3-D case.

### 6.5 Refine the integrator without a coarse-rule veto

Permit orders 2,4,8,16 or adaptive subdivision. Compare the finest two or three compatible results and an independent rule. Do not require the coarse order-2 midpoint result to meet the final tolerance forever. Gate angular refinement separately from radial refinement, and retain phase-shift checks. Include the derivative-weighted Fourier tails when choosing angular grids; a small geometry tail alone does not bound current/force error.

The test suite must include a case in which coarse rules fail but fine rules converge, NS-independent point counts, a known angular alias, a failed inverse with finite-looking fields, a wrong determinant sign, and a mixed-contract parent ladder. Extract pure predicates into the reference environment rather than importing the solver driver.

### 6.6 Minimal fixed-state experiment set

Run the repaired measurement on: the saved NS33 default solved state; the NS129 exact projection (no solve); and the NS129 axisymmetric solved control. Add a zero-TCON state after the inexpensive source/measurement checks are stable. Do not run all old states at the largest grid first. For the bad NS33 state, stop at a defensible diagnostic estimate and publish its local error distribution; return for tighter quantitative norms only when they answer a remaining question.

**Exit:** real native-knot alignment or explicitly corrected reference crossings; clean portable tests; immutable partial data; bounded memory/cost; resolved control measurements; and a clearly labelled bad-state diagnosis. No claim that the currently interrupted composite result is completed.

## 7. T2: distinguish discrete force, coordinate choice and physical error

**Depends on T0; T1 controls are used as they become available. Maps to R2/P3.**

Before a new long solve, make the reusable residual an explicit mathematical object. Check `F(z_a)` after evaluating z_b, after z_c, with fresh/reused caches, and in direct/multigrid paths. Compare values and JVP/VJP results under identical masks and gauge semantics. A legacy iteration can intentionally depend on the preceding residual; that behavior must not silently enter the function differentiated as F(z). VMEC++ PR849 is a relevant comparison design, not a demonstrated VMEX defect. [S11]

Evaluate the same projected/solved states with and without the coordinate constraint, using the actual warm-start transfer and baseline rebinding. Compare complete raw force vectors, not differences of squared normalized FSQ values. If F=F0+C, then `||F||^2` contains `2 F0 dot C`; squared norms do not add or subtract as force vectors. Record radial/mode contributions, masks, normalizations and omitted degrees of freedom.

Independently measure physical force contributions on those same states. With dB=B_h-B_e, dJ=J_h-J_e and dgp=grad p_h-grad p_e,

$$F_h=dJ\times B_e+J_e\times dB+dJ\times dB-dgp.$$

This identity is exact for an exact reference equilibrium. Store the vector terms and their inner products or local squared contributions. Their norms need not sum to the total norm. This tells whether the discrepancy is dominated by current reconstruction, field displacement or pressure-gradient mismatch before changing the solver.

Where possible, project the independent continuum force into the actual VMEX test directions with the correct coordinate Jacobians, boundary terms, scaling and radial staggering. Do not compare a naive Cartesian FFT with a differently normalized covariant VMEC residual. Derive the map on an axisymmetric control first. Separate physical virtual work from an auxiliary coordinate penalty rather than assuming every force block is a direct Cartesian force.

### 7.1 Complete the missing historical cell without an unbounded dependency

After T0 and a bounded T1 control, run the missing NS129 projected `TCON0=0` case once at the historical pin. It remains diagnostic until measurement, representation and root gates are resolved. The intended controls are:

```sh
BENCH_FTOL=1e-10 BENCH_TCON0=0 BENCH_RUN_ID=integer3d-ns129-zero-NEW_UNIQUE_ID \
  python benchmarks/run_vmex.py inputs/input.integer_3d_iota 129 3000 \
  results/projection/integer_3d_vmex_ns129/seed.npz
```

Choose a genuinely new run ID and verify the imported historical source before this command. The schema-2 output is under `results/vmex/runs/`; update the measurement invocation accordingly. Preserve the original input/seed hashes and actual effective controls. Do not replace the matching historical default result with one from a newer solver.

Then use a small factorial study, not an unrestricted scan: at fixed angular representation vary NS; at fixed NS vary angular representation; at a selected well-behaved case vary root tolerance; only then compare constraint strengths. Include cold/projected/continuation starts where they distinguish hypotheses. Record coordinate Jacobian conditioning, high-mode content and physical label drift. A lower numerical constraint residual alone is not a success criterion.

### 7.2 Gauge-defined roots and TCON sensitivity

For a deterministic residual in an explicitly constrained space,

$$F_h(z,P(a),\tau)=0,\quad F_z z_a=-F_P P_a,\quad F_z z_\tau=-F_\tau.$$

Here tau is the constraint strength, not a physical parameter or the field-line integration variable. For Q, solve

$$F_z^T\lambda=Q_z^T,\qquad Q_a=Q_P P_a-\lambda^T F_P P_a,$$
$$\frac{dQ}{d\tau}=Q_\tau-\lambda^T F_\tau.$$

Implement tau dependence explicitly in a benchmark residual only after tracing how runtime/baseline/normalization depend on it. A centered finite-difference RHS is acceptable initially if its step convergence is reported; do not call it automatic differentiation. Do not assume tau is already included in the public implicit parameter object.

An auxiliary coordinate parameter should not alter a unique continuum physical branch. Test whether physical `dQ/dtau` and coordinate-remap sensitivity decrease with spatial refinement. At tau=0 the discrete system can lose a useful gauge constraint; a tiny diagonal regularization or a pseudoinverse cutoff is a changed derivative problem unless explicitly justified and tested. Use eliminated gauge coordinates or a bordered constrained solve, verify rank/scaling, and check residuals in the original operator.

Derive regular angle variations, including their compensating lambda, in the source's actual internal normalization. Preserve radial analyticity at the axis. Coordinate coefficient derivatives are not physical observables; compare Eulerian fields, pressure and flux labels at fixed Cartesian points. Test at least one genuine physical direction separately from a pure coordinate direction. Never rename a displaced pressure surface as an angle-gauge change.

**Exit:** a residual-purity test, a completed missing-cell record or diagnosed bounded failure, independent physical error decomposition, and at least one constrained response calculation with original-operator residuals. Decide whether an upstream issue is about the harness, field reconstruction, gauge policy, root accuracy, or a remaining physical branch question. Do not announce a global TCON default recommendation from one case.

## 8. T3: turn the verified input map into actual equilibrium sensitivities

**Can proceed in parallel with the bad-state diagnosis after T0 and an axisymmetric measurement control. Maps to R3/P4.**

The complete-map finite-difference experiment is useful and should not be repeated from scratch. Extend it to save signed derivatives and fixed output labels. Nondimensionalize each block using declared boundary length, pressure, current and flux scales. Report absolute errors for known-zero channels and per-block relative changes for nonzero channels. Do not use one unscaled norm of metres, pascals, amperes and webers as a convergence certificate. [S5, F10]

Use fixed polynomial degree, fit/quadrature grids and boundary modes throughout a derivative pair. Preserve raw AC versus CURTOR semantics. For parameter-dependent basis selection, freeze the selection for the local derivative and independently verify its approximation budget; a discontinuous fit-degree choice is not a smooth parameter map.

Start with a verified finite-difference `P_a` applied to the differentiated VMEX residual. This isolates the equilibrium derivative without requiring an all-JAX rewrite of the input generator. Then implement a small fixed-basis JVP and compare it with the saved signed FD vectors. Do not reinstate the previously removed untested helper without tests.

For a=b=1, useful analytical scalars are

$$\beta_V=\frac{2c^2\delta}{1+c^2\delta},\qquad
\partial_c\beta_V=\frac{4c\delta}{(1+c^2\delta)^2},\qquad
\partial_\delta\beta_V=\frac{2c^2}{(1+c^2\delta)^2}.$$

Volume and flux derivatives are good conversion controls, but alone they mainly test boundary/input dependence. The first nontrivial solver objectives should include a signed interior-field projection at fixed Cartesian points and beta or magnetic energy with its correct domain dependence. Choose points contained in every perturbed configuration. Differentiate moving-domain integrals with their Jacobian and boundary motion; do not compare them with fixed-domain derivatives by mistake.

The strong cancellation check is

$$\left.\partial_\delta B(x)\right|_x=0$$

for the integer family, although the boundary, pressure profile and total flux change. At moving numerical coordinates x_h(q,delta), the Eulerian response is

$$\left.\partial_\delta B_h\right|_x=
\frac{d}{d\delta}B_h(x_h(q,\delta),\delta)-\nabla B_h\,\partial_\delta x_h.$$

Do not divide its error by the zero exact value. Use a fixed physical derivative scale. The nonzero c-direction is an essential partner: a code returning zero for every derivative should fail immediately.

For sheared A, differentiate the toroidal-flux inversion as well as the boundary, pressure and current profiles. Use u=k^2 to remove an artificial axis 0/0. If `Q(u,a)=s Q_b(a)`,

$$u_a=\frac{s\,dQ_b/da-\partial_a Q}{\partial_u Q}.$$

Physical-angle inversion also needs its implicit derivative on a validated branch. In this family lambda changes vertical geometry, pressure and flux, but not iota at fixed normalized toroidal flux when the other reference parameters are fixed. Test `d iota(s)/d lambda=0` with **prescribed current**, not by differentiating an imposed iota profile. Check pressure/flux scaling and nonzero field responses in the same direction.

Every response record contains: the root state actually used for Q; root residual and anchor changes; DOF/gauge definition; parameter/observable scaling; tangent and transpose solver true residuals; finite-difference step interval; and physical derivative error across at least two spatial resolutions. Verify bilinear JVP/VJP duality and a small independent dense solve at the smallest rung. Use SOLVAX factors and Krylov methods already available; do not assume an unscaled nonsymmetric operator is suitable for conjugate gradients.

A public `custom_vjp` wrapper does not establish `jax.jvp` support. Test each transformation actually used, and expose residual-level tangent solves when the wrapper is reverse-only. Higher derivatives require additional smoothness and traceable rules; they are a later experiment, not inferred from a first gradient. [L9, L12]

**Exit:** one nonzero and one null physical response on the axisymmetric family with convergence evidence, then the integer 3-D and sheared current-prescribed controls. Keep frozen-path FD consistency and independently reconverged branch response as separate comparisons.

## 9. T4: recover breadth without a Cartesian explosion of cases

**Maps to R4/P1-P4. Start sheared-A preparation alongside T2/T3.**

Use the existing case records rather than one script per flag. Mandatory mild controls are:

| Group | Required comparisons | Independent evidence |
|---|---|---|
| Axisymmetric integer | NCURR=0/1; LASYM=F/T | Exact B/J/p/flux and physical responses |
| Symmetric Solov'ev | Native representation and solved control | Independent Grad-Shafranov identity |
| Asymmetric Solov'ev | LASYM=T, NCURR=0/1 | Genuine up-down-asymmetric exact field |
| Integer 3-D | Both closures; F/T same-physics control; selected stretch/rephase | Exact field, labels, flow and null response |
| Sheared A | Both closures; F/T; continuation from easier shape | Exact field plus independently refined flux/current quadrature |
| Sheared B/C | Selected higher-resolution stress runs | Resolved boundary/chart before attempting a solve |
| Generic 3-D asymmetric perturbation | LASYM=T after reference cases | Independently converged numerical reference, not an exact field |

Do not count symmetric F/T duplication as another physical configuration. A rigid shift or toroidal rephase exercises asymmetric harmonics and covariance but does not constitute a generic symmetry-breaking equilibrium. Sheared B/C already have improved input fits; do not restore obsolete unresolved low-resolution decks. Their large toroidal mode counts make them unsuitable first derivative cases.

At the historical pin, live Cartesian LASYM fields are not supported. Removing a rejection guard is not an implementation. Trace all R/Z/lambda sine/cosine blocks, native Clebsch reconstruction, inverse coordinates, derivative tensor ordering, WOUT conversion and restart. Build one narrow upstream patch only after tests on a symmetric state represented with LASYM=T and on the exact asymmetric Solov'ev field. Until then, label WOUT surface values and fitted-volume reconstruction as distinct, limited routes.

For each family first test the input, then project the exact state, then solve from a verified seed, then try cold/continuation starts. A projection passing the physical score does not prove that the nonlinear solver recovers it. An equilibrium need not be an attracting state of every relaxation algorithm; inspect residual, geometry and constrained curvature before assigning a branch or stability explanation to a failed solve.

Use a few matched native DESC runs and, where practical, VMEC2000/VMEC++ force/solve comparisons. Compare physical errors at equivalent input/domain accuracy, not nominally equal mode counts. GVEC is valuable for arbitrary-degree radial splines and mapping control after its input conventions are verified. The ordinary `itpplasma/benchmark_vmec` adapters may save effort, but inspect the specific conversions and do not import its entire framework. SPEC-type stepped-pressure results are a different model and require an explicit limit before comparison with smooth ideal-MHD references. [L5-L8, L13, S12]

**Exit:** resolved mild-case recovery or a precise bounded failure for each required class; current-predicted transform rather than merely imposed iota; and a capability table separating unsupported field paths from nonlinear-solver behavior.

## 10. T5: use exact fields to test more modules now

**Partly independent of nonlinear recovery. Maps to R5/P5/P9.**

### 10.1 An exact field-line flow and its tangent map

Landreman's integer field obeys `(B dot grad)B=-diag(1,1,4)x`. The diagonal-stretch extension preserves that identity. Set Omega=diag(1,1,2) and define the dimensionless field-line parameter t by `dx/dt=B(x)`. Then

$$x(t)=\cos(\Omega t)x_0+\Omega^{-1}\sin(\Omega t)B(x_0),$$
$$B(x(t))=-\Omega\sin(\Omega t)x_0+\cos(\Omega t)B(x_0),$$
$$D_{x_0}x(t)=\cos(\Omega t)+\Omega^{-1}\sin(\Omega t)\nabla B(x_0).$$

The flow closes after 2pi, preserves volume, and has identity full-period tangent map on its smooth domain. These formulae follow from the paper's oscillator construction; they are not a claimed new equilibrium result. [L1]

Use them to test VMEX's generic field API, Cartesian/cylindrical transforms, gradB ordering, field-line tracing, starting-point JVP/VJP and ESSOS field handoff. Compare trajectories at equal field-line parameter, or compare equivalent Poincare sections if the numerical tracer uses physical toroidal angle. Do not equate t with cylindrical phi. With dimensional `dx/dtau=B_SI`, the dimensionless parameter is t=B_* tau/L_*; a tracer using unit B has a different arc-length parameter.

Test axisymmetric, 3-D, stretched and rigidly transformed cases. Record step/order convergence and distinguish exact-callable errors from errors after VMEX state reconstruction. Full charged-particle trajectories are different equations: this exact flow is not their reference. The delivered probe checks field-line equations, group composition, closure, tangent map, volume preservation and a DOP853 comparison without VMEX.

### 10.2 Spatial derivatives and continuity

Benchmark orders 0-3 only with the derivative semantics appropriate to the actual reconstruction. For the historical cubic geometry, B is generically C1 across native knots; higher derivatives can be piecewise rather than classical at a knot. Record one-sided jump norms, cell-interior errors, axis limits and dependence on reconstruction order. Do not evaluate only midcells and conclude global smoothness. Do not use a C1 monotone interpolant as an automatic cure when the target requires higher derivatives. [F09, L3]

A higher-degree representation comparison should preserve the same boundary/profiles and report its fitting error. Exact spline knot insertion or Fourier zero-padding is preferable to refitting a WOUT during continuation, when the native representation supports it. It preserves the represented state but does not itself improve a physical equilibrium.

### 10.3 Diagnostics and downstream integrations

| Capability | Small useful experiment | Required qualification |
|---|---|---|
| WOUT, restart, input/output | Value/tensor/flux round trip on common points; resolution transfer | Serialization parity is not continuum correctness |
| Boozer transform / booz_xform_jax | Reconstruct physical B and surface averages; compare independent transform and selected spectrum derivatives | Rational closed lines have gauge/sampling subtleties; exact MHD is not automatically QS/QI |
| Bounce action / wells | Independent bracketed turning points and endpoint-regularized quadrature; fixed-topology parameter derivative | Well creation/merger and marginal trapping can invalidate ordinary derivatives |
| SOLVAX | Original-operator residual, dense small reference, transpose duality, factor reuse | Finite iterative tolerances and gauge rank affect response accuracy |
| ESSOS | Exact field-line flow, field conversion, chosen orbit invariants and step refinement | Magnetic moment is not an exact invariant of full-particle motion |
| NEO_JAX | Geometry/normalization and ripple quadrature on suitable nondegenerate surfaces | Exact MHD does not supply a known kinetic coefficient by itself |
| DKX | Metric/current/profile units, conservation checks and one matched kinetic reference | Total MHD current is not synonymous with bootstrap current |
| GKX | Metric, curvature and drift checks; one isolated eigenpair response if supported | Exact equilibrium does not determine an analytic growth rate; degeneracies need branch treatment |
| pyQSC_JAX | Axis/field-jet and valid radius-order asymptotics | Restricted QS near-axis ansatz cannot represent an arbitrary exact 3-D field |
| VMEX mirror path | Separate harmonic vacuum/open-geometry fixture and appropriate paraxial checks | Toroidal references do not validate open topology or a finite-beta mirror closure |

Resolve and record optional library pins only when their experiment runs. Reuse direct functions and existing adapters; one small shared adapter is preferable to copying physics throughout the benchmark. Missing dependencies are unavailable, not passed. Independent diagnostic tests can run on exact callables or labelled projections while nonlinear recovery remains open.

**Exit:** an expanded capability matrix backed by a few well-defined executed tests, especially exact-flow and field derivative controls, rather than a list of modules that merely imported successfully.

## 11. T6: exterior operators, coupled free boundary and bounded polishing

**Maps to R6/R7/P6/P7. No claim of exact global exterior from an interior formula.**

### 11.1 Exterior and free-boundary evidence

Retain three distinct levels: exact operator fixtures; independently converged coupled plasma/vacuum equilibria; and approximate exterior-source fits to an exact interior target.

Start with harmonic potentials, uniform/linear vacuum fields and the toroidal circulation field where mathematically appropriate. Test MGRID interpolation and derivatives, Neumann compatibility, source circulation, virtual-casing decomposition and target-distance convergence. The pure toroidal field does not select a unique plasma boundary and is not an invertible free-boundary adjoint fixture.

Use an independent axisymmetric singular quadrature, such as the appropriate Kapur-Rokhlin formulation, and a separately refined 3-D surface-integral comparator. Validate singular and near-singular target limits separately. Freeze a numerical integration plan for a derivative only after verifying it remains adequate under the perturbation; a fixed plan is not automatically a correct plan. [L10, L11]

For coupled roots, include an axisymmetric and a mild 3-D case, symmetric-basis controls and a genuine asymmetric configuration when supported. Prescribed external currents/fluxes, circulation choices, pressure boundary conditions and gauge must match between the objective and its derivative. Check normal B and total normal stress. On a flux boundary with B_n=0, require the appropriate jump of `p+B^2/(2 mu0)`; a tangential-field jump implies a surface current `K=n cross (B_out-B_in)/mu0` and must be admitted explicitly rather than ignored.

Do not continue the analytical interior field outside the plasma and call it a vacuum reference: its curl generally remains nonzero there. An exterior source fit is an approximate coupled construction until its vacuum/model/matching errors have been independently controlled. Coil realization and engineering constraints are separate questions from interior equilibrium existence.

Test the actual current `freeboundary_implicit` implementation, not a stale narrative that says anchoring is absent. Require the coupled state used for Q, the linearization and the adjoint to be the same anchored root, with strict failure status and true linear residual. Best-effort derivatives remain diagnostic. Do not assume a reverse API supplies a forward API.

### 11.2 High-order polish and PR448

The optional recovery branch has promising exact-transfer and sparse-Jacobian machinery but remains a separate, axisymmetric fixed-profile experiment. Its independent and provisional certificate levels, actual RSS, and unsupported cases must remain explicit. In particular, map its magnetic-normalized force `epsilon_B` to this study's force metric, not to analytical field error E_B. Its current target crossing is not independently closed. [S10]

First apply any candidate to a small exact Solov'ev/integer control with B/J/force truth. Preserve the state in its native high-order basis, use exact knot insertion and angular padding when available, and compare the same physical normalization before/after. Do not attribute a WOUT reconstruction change to a nonlinear physical improvement.

For a least-squares residual r(z,a), stationarity is `g=r_z^T W r=0`. Its exact derivative contains

$$g_z=r_z^T W r_z+\sum_i (Wr)_i\,\nabla_z^2r_i,$$

for fixed W, plus the appropriate terms when weights or constraints depend on parameters. Gauss-Newton alone omits the residual-weighted second derivatives. A force norm, stationarity norm, gauge feasibility, positive geometry and an exact stationary response are separate gates. With equality constraints use the differentiated KKT system, including constraint motion, rather than differentiating an unconstrained normal equation.

Use compact radial support, compressed JVPs and sparse/direct or matrix-free solves only after a small dense comparison verifies the pattern and products. Avoid squaring condition numbers unnecessarily through poorly scaled normal equations. Explicit-array admission is not an RSS guarantee; benchmark compilation/cache memory separately. A bounded negative outcome closes this experiment honestly; it does not justify importing a large research branch into the primary baseline.

## 12. T7: optimization and scientific extensions

**Maps to R8/P8. Direct-reference work can run early; solver-driven claims require T3.**

### 12.1 Optimize within exact families

Use the explicit fields, maps and quadratures to optimize exact-family parameters directly. VMEX should verify selected states and sensitivities, not repeatedly solve for a field already available in closed form. Hold edge pressure at zero, eliminate arbitrary length/field rescalings, and constrain volume/aspect ratio, current, elongation, domain margin and chart validity. Raising pressure by an arbitrary constant is not a useful beta optimization.

Start with a two- or three-parameter study, retaining all starts, constraints and objective histories. Appropriate objectives include field-strength variation, current concentration, a verified stability diagnostic or a constrained beta/shape tradeoff. Use dimensionless residuals and declare weights. Any confinement or stability improvement needs a relevant diagnostic calculation; exact force balance alone does not establish it.

The sheared family's lambda direction is particularly useful: iota is unchanged while geometry and pressure change. It isolates effects that a generic pressure scan entangles. Evaluate direct-reference derivatives and compare selected VMEX responses after the current-prescribed null test works.

### 12.2 Optimize coordinates without changing physical surfaces

A coordinate remap can reduce Fourier width and improve conditioning. Use regular radial amplitudes and a strictly monotone poloidal map, with the required lambda compensation. Choose a few modes rather than a new universal coordinate framework. Minimize a derivative-weighted spectral tail and/or a scaled condition indicator, subject to positive geometry and held-out field invariance. Distinguish the coordinate objective from physical objectives.

The action-based toroidal coordinate method [L7] and GVEC mappings [L8] are alternatives for difficult initialization. Evaluate them first on unchanged analytical surfaces. If a remap changes the represented finite field, measure that projection error rather than saying the remap is physically exact after truncation. Fit and judge on different grids. An optimized chart that only improves its training-grid force is not a success.

### 12.3 Tangent versus transverse physical response

After exact-family responses are verified, add a small number of boundary/profile directions outside the family. Use continuation and reconvergence, track the branch, and compare them with exact-family tangents and pure coordinate directions. For

$$J=\sigma B+\frac{B\times\nabla p}{B^2},$$

current continuity gives

$$B\cdot\nabla\sigma=-\nabla\cdot\left(\frac{B\times\nabla p}{B^2}\right).$$

On a closed line, the line integral of the right-hand side divided by |B| must vanish. Measure this compatibility residual and current convergence before interpreting a large response near rational transform as numerical failure or as physical nonexistence. Finite resolution and a finite set of resonance tests do not prove a smooth continuum branch. Conversely, integer transform alone does not prove that the whole constrained discrete Jacobian is singular.

A useful scientific result is a resolved difference between exact-family tangent, coordinate and transverse sensitivities, with numerical errors controlled. Report null/negative tradeoffs as well as improvements.

### 12.4 Goal-oriented refinement rather than uniform growth

Once a derivative-capable root is available, form a small enriched-space defect. Let z_H^0 be a consistent transfer of z_h and r_H=F_H(z_H^0). A first correction solves A_H delta z=-r_H. For an observable, the adjoint prediction is

$$\Delta Q\simeq Q_H(z_H^0)-Q_h(z_h)-\lambda_H^T r_H,\qquad
A_H^T\lambda_H=Q_{H,z}^T.$$

Use compatible gauges, scales and parameter constraints. Calibrate this estimate against an actual corrected/enriched solution; do not advertise it as a rigorous bound imported from finite-element theory. Local products of adjoint weights and residual contributions can prioritize radial spans/modes affecting Q, complementing pure force-based refinement. Exact knot insertion preserves a high-order state during this experiment; projection to another representation does not do so automatically. [L4]

The first useful objectives are already available: beta, a signed interior field functional, or current-prescribed iota. Keep this extension small. Only then consider validated boundary/coil optimization with held-out physics and geometry constraints.

## 13. T8: figures, performance and a usable repository

Generate figures only from persisted records. Plotting may produce derived summaries and manifests, never edit raw observations or fill missing metadata. Label historical96, reference-domain, native-domain, analytical projection, solved, terminal and provisional data distinctly.

The next figure set should answer specific questions:

| Figure | Required data | Purpose |
|---|---|---|
| Measurement partition | Native knot crossings versus reference knots on a few angular rays | Demonstrate actual interval alignment |
| Radial error density | Cellwise B/J/grad-p/force integrals and uncertainty | Locate the physical/reconstruction error |
| Measurement cost | Error-estimate spread versus cost/RSS for old and streamed routes | Show bounded reliable scoring rather than just a faster kernel |
| Representation continuity | One-sided derivative jumps and cell-interior errors | State what orders 0-3 mean on each representation |
| Constraint/root comparison | Completed NS129 cell, controls, pure residual and physical scores | Separate constraint choice from accuracy |
| Response tests | Signed tangent/adjoint/FD/analytical values and step/resolution studies | Verify physical derivatives, including nulls |
| Exact-flow checks | Trajectory/tangent/closure errors versus step and representation | Exercise interfaces with an exact time-parametrized reference |
| Broader recovery | Axisym, LASYM, integer and sheared mild-case results | Show real breadth without collapsing unlike evidence |
| Exterior/polish studies | Matching residuals, source-distance error, stationarity, memory | Keep model and certificate levels visible |
| Optimization/goal refinement | Histories, constraints, held-out metrics and predicted/actual changes | Demonstrate a resolved scientific improvement or limit |

Keep the README short enough to navigate. Lead with the latest accepted results and a short limitations table; put detailed failed-run narratives and old numbers in a linked results note. Show current test/CI status honestly. Include exact minimal commands for reference tests, one forward solve, one response, and figure regeneration. Do not copy the entire logbook into the README.

Timing comparisons must include source versions, actual device used, thread settings, independent compile/warm evaluation, root anchoring, reference/input preparation, scoring, derivatives and I/O. Use repeated measurements and report spread. Compare cost to a stated physical/derivative accuracy, not equal FTOL or nominal grid size. Record failures and budget overruns. Device enumeration is not evidence of multi-device execution. Host RSS and device memory are different metrics.

Use short pilots to set a per-process time/RSS budget before increasing resolution. When a pilot spends most time on certificate evaluation, repair/reuse that evaluation rather than launching a longer version blindly. Preserve useful completed work on interruption. Isolate cache experiments in benchmark-owned directories, never delete the user's global cache.

## 14. Minimal implementation layout and task order

Prefer edits to the existing modules:

- `evidence.py`: explicit supported record schemas, compatibility checks, child-artifact receipts.
- `measurement.py`: pure quadrature partitions, reductions, scales and acceptance logic; no mandatory VMEX import.
- `verify_measurement.py`: solver-specific evaluation and a thin driver; remove duplicated large-array accumulation and hard-coded grids.
- `full_input_derivatives.py`: signed block-scaled tangents and fixed-basis JVP tests.
- `run_vmex.py`: tested effective settings, source identity and the new-schema integration path.

Add one `native_evaluation.py` only if it removes real duplication between measurement and derivatives. Add one response driver and one exact-flow integration driver when their experiments are ready. Do not create a file for every grid, flag, plot panel or failed attempt. Keep analytic formulas in the shared reference module with independent tests.

A practical first sequence is:

1. T0 clean-reference CI/import boundary and schema-reader tests.
2. T1 pure displaced-knot and interruption tests, NS-generic counts, streamed sufficient statistics.
3. One small native-cell/inverse/covariant-curl comparison; gate geometry and labels.
4. Bounded scoring of the NS33 failure plus the NS129 projection/axisymmetric controls.
5. The missing historical NS129 zero-constraint solve and residual-history checks.
6. Signed complete input tangents and the first actual axisymmetric root response; sheared-A input/projection/recovery in parallel.
7. Exact-flow/interface tests and selected LASYM implementation; then downstream/exterior and bounded polish experiments.
8. Direct-family optimization, transverse continuation, goal-oriented refinement and final performance comparison.

No phase requires every optional library to be installed. A reached but unsupported capability remains explicitly unavailable/blocked with its reason and next reproducible action. Core completion requires actual solved mild axisymmetric and 3-D examples, meaningful LASYM coverage, both closures, measured spatial/response errors and at least operator-level plus coupled free-boundary evidence. An unresolved case stays unresolved; do not claim a completed all-round benchmark by counting planned cells.

## 15. Current status and continuing logbook

| Task | Status at this handoff | Next evidence needed |
|---|---|---|
| T0 portable evidence/tests | Completed for this block: clean Python 3.12/VMEX-absent suite passed 55 tests; post-edit Python 3.11 suite also passes 55; old/new report reader and schema-2 consumer exercised | Keep the import boundary and immutable evidence tests in CI |
| T1 measurement | Bounded native-knot and streamed-checkpoint work completed; 96-point controls remain unresolved measurements; interrupted composite was not repeated | Use exact native-knot roots; design a cheaper resolution ladder before any larger run |
| T2 constraint/root | NS129 projected TCON0=0 and same-seed TCON0=1 comparison completed at historical pin; residual-history and force-decomposition diagnostics completed | Resolve spatial scorer/root certificates before any default recommendation |
| T3 sensitivities | NS17 dense solve certifies the linear response; NS65 frozen-path FD converges; branch inversion is stable; JVP of the field map along the measured branch-state FD matches branch B-FD to 0.70%, while residual-level JVP remains 5.92e-3 away | Analyze the active m=1 root-response difference and test any candidate gauge alignment against physical B before assigning a cause; derivative remains uncertified |
| T4 breadth | Sheared-A cold and lambda-zero geometry-seeded NS17 current solves both capped at 10,000; matched 96-point plot confirms the projected start is worse on B/J/gradp/flux labels; no recovery accepted | Derive and validate a nonzero straight-field lambda seed from exact field-line flow before another bounded recovery solve |
| T5 diagnostics | Broad program mostly pending | Exact-flow and bounded field-derivative/interface tests |
| T6 exterior/polish | Mostly pending; separate PR448 research context | Exact operator fixtures, strict coupled case, bounded optional polish |
| T7 optimization | Planned | Direct-family/chart pilot, then verified solver-driven extensions |
| T8 presentation/cost | Many historical figures; new cost bottleneck established | Read-only updated figures, complete resource accounting |

For every work block append: question; relevant task; benchmark and imported-source hashes; branch/PR state; command and effective configuration; parent/input/state/grid/contract hashes; measurements with uncertainty; failed or interrupted attempts; artifacts; code/tests actually inspected or run; and the exact next action. Distinguish 'not run', 'failed', 'unavailable', 'diagnostic', 'resolved measurement', 'accepted recovery', and 'accepted derivative'. Do not infer a terminal state from an old paragraph saying a process was running; use the newest receipt and actual process state locally.

### Review entry: continuation from d5484d1

The review confirms two commits beyond the earlier snapshot and retains the evidence repairs. New priorities are clean reference tests, compatible schema readers, real native-knot alignment, progressive convergence logic, interruption-safe streaming, signed/scaled input tangents and explicit reconstruction regularity. Current GitHub reference CI failed at pytest; the source contains a sufficient missing-dependency path, but detailed failure logs were not retrieved. The current VMEX release and open PR448/VMEC++ PR849 were checked as separate sources, not imported as successful benchmark results.

Six standalone probes passed: displaced-knot quadrature; NS-dependent grid-count arithmetic; exact integer-family flow/tangent/volume preservation; dimensionally scaled derivative comparison; derivative regularity of a C2-map field; and streaming sufficient-statistic equivalence. No new repository solver or full benchmark suite was executed by this review. No remote repository writes were made.

**Exact first action:** preserve the current plan/logbook and compare the working tree with d5484d1. Fix the solver-independent test imports and the schema reader, then write a cheap test that exposes the reference/native knot distinction and preserves a completed grid on interruption. Do not start by repeating the 27-minute composite profile unchanged.


### Adoption and checkout comparison, 2026-09-24

The actual checkout and `origin/main` were both exactly `d5484d1e7a15c9cf10599c2b0f05bdf69e22e861` at handoff review; the working tree was clean. The two publication commits after `575f13f` are present. All eight selected reviewed source-blob IDs in `review_snapshot.json` match the checkout. No later local/remote work was overwritten. A new continuation branch `t0-t1-d5484d1` was created from that reviewed tip. The archived preceding active plan hash matches `git show d5484d1:plan.md` byte-for-byte; the complete prior plan/logbook and prior historical plan remain in `docs/history/`.

The handoff package's ten files were extracted, each checksum passed, and the full package was copied into `docs/handoff/review-d5484d1/` with its original `SHA256SUMS`. The six review probes and their recorded outputs were inspected; they are standalone numerical illustrations, not solver measurements. `gh run view 36061177756 --log-failed` retrieved the reference CI traceback: collection fails because `tests/test_measurement_reference.py` imports `_composite_spread` and `_parent_comparison_rows` from `verify_measurement.py`, whose module-level `import vmex` fails in the declared reference environment. The review's source and commit pins are preserved, and historical/current VMEX remain distinct.

**Identity and branch state:** GitHub login is `rogeriojorge`; local Git author/committer remain configured for the verified account-derived noreply address. This is a local work branch based on reviewed public main; no commit, push, PR or upstream change has been made for this continuation.

**Exact next action:** Implement T0 by moving pure composite acceptance and ancestry/report-normalization helpers into the existing VMEX-independent modules. Add schema-1/schema-2, source/status/effective-controls/capped-state regressions. Then create a fresh environment from `requirements.txt`, confirm the VMEX import boundary is absent, and run the reference suite.


### Continuation work block, 2026-09-24: T0-T3 results and axisymmetric response discriminator

**Branch/state:** local continuation branch `t0-t1-d5484d1`, based on reviewed commit `d5484d1e7a15c9cf10599c2b0f05bdf69e22e861`; no continuation commit, push, PR, or upstream write has yet been made. Historical VMEX source remains `b5f5267efc0795c4a49a224e321e9b370975c14c`, separate from the recorded current snapshot. GitHub owner and local project Git identity were checked earlier in this continuation; no AI author/co-author has been configured.

**T0 evidence/import boundary:** Extracted handoff members and checksums are preserved in `docs/handoff/review-d5484d1/`; preceding plan bytes are preserved at `docs/history/plan_through_d5484d1.md` with SHA-256 `0b34804db8c0741f8f94b0e6e1af8616119fbe002cc8bc4330bea54f96fa8736`. Moved schema/evidence helpers into solver-independent modules and adjusted reference tests so their collection path no longer imports VMEX. The clean Python 3.12 environment with VMEX absent passed 55 tests in 14.17 s; `verify_reference.py` passed all 14 reference checks. `reference_derivatives.py` passed and explicitly reported `vmex_executed=false`. The schema-2 consumer path accepted a generated report; a deliberate one-iteration cap was retained as a diagnostic, not a valid measurement. The previous reference outputs were restored byte-for-byte after the isolated rerun. Latest recorded source hashes at the time of this block: `benchmarks/evidence.py` `a4cc14b249d98103cd5da7afbd9fb72e32ac865712e44f8dab2df9410c480371`; `benchmarks/measurement.py` `91b042394d3df7f1d480a61a88904553993e6d1eb76b230256e77bf72e458e48`; `benchmarks/verify_measurement.py` `db803fa1f36e93d3cc857ec805d38cf01fe43f1b8f1c997bddd8c9dea6aacd9e`.

**T1 measurement mechanics:** Actual radial knots were found by directly bisecting the saved NS33 fixed ray against historical VMEX. The report `results/audit/native_knot_alignment_ns33_exact_roots.json` has maximum root residual `3.55e-15`, maximum knot displacement from uniform reference knots `0.00359461797`, and SHA-256 `b87e3dbba9e844733a3044ec46be0f498a54f0112df2ac09e5b64358b5ab6c6b`. Radial64 interpolation crossing error was `2.77e-6`; radial32 differed from direct roots by `8.81e-6`. Whole-volume Gauss points are not native-knot aligned; future aligned integration must split each native cell and use cell-local Gauss nodes. The bounded `measurement_t0` smoke took 8.215 s warm, peaked at 1071.78 MiB, and retained at most 96 audit samples; it did not resolve the NS33 measurement. Streamed sufficient statistics and immutable completed-grid checkpoints were exercised without rerunning the interrupted 27-minute composite unchanged. Thus T1 repair mechanics are demonstrated, but physical score convergence remains open.

**T2 zero-strength and reusable residual:** At historical VMEX, the same projected NS129 input/seed converged in 50 iterations for TCON0=0 versus 420 at TCON0=1. On the matched 96-point grid, zero-strength B/J/gradp/force ratios were `1.0795e-6 / 1.8027e-5 / 3.0036e-6 / 1.9413e-4`, versus `1.2991e-4 / 2.9206e-3 / 9.7036e-3 / 1.7449e-2`. Both are diagnostic legacy-grid scores only: scorer resolution, independently certified root, and representation gates are not satisfied. Do not change the default on this evidence. The zero run's `forward.json` SHA-256 is `ff81f475982fb78d875fca3bc789c3bfea5aa8d5e70e33f2cd069851f6f67134`. Reused preconditioned single-grid residual value/JVP/VJP were exactly history independent in the saved check: `results/audit/residual_history_ns33_20260924.json`, SHA-256 `cfa6708746909832331be9872a84b1bb61dae44cbeaf6f9f50d0044c8bef5b9c`; elapsed 10.053 s, peak RSS 1102.25 MiB. Raw and multigrid formulations are still untested. The first nested-JIT setup attempt failed because runtime compiler options cannot be set from a nested trace; runtime was then materialized before tracing, and both the failure and corrected method are retained in the run record. Physical force identity defect maximum was `3.8614e-10 N/m^3`; weighted RMS contributions were dJ×B `68.776`, J×dB `3.316`, dJ×dB `9.78e-5`, and pressure-gradient variation `1.0656 N/m^3`. This localizes the dominant term in that state; it is not a root certificate.

**T3 axisymmetric tangent discriminator:** Added `benchmarks/axisymmetric_root_response.py` to transfer central signed input tangents in declared block scales into the historical VMEX residual response, evaluate fixed-Cartesian field/beta/iota, test the transpose pairing, and compare frozen-path FD separately from independently reconverged branch FD. At NS65 the anchored root residual was `5.70e-14`, anchor shift exactly zero, both tangent solves reported converged with residuals `6.54e-14` and `3.67e-12`, and JVP/VJP relative pairing errors were `4.45e-11` and `8.18e-12`. Nonzero `c` field derivative relative error against the exact field was `8.11e-5`; beta derivative absolute error was `1.73e-8`. The null `delta` exact B derivative is zero; its residual JVP has fixed-scale magnitude `6.194e-3`. Crucially, on the frozen linear path its centered FD difference from the JVP falls from `0.1387` at step `1e-3` to `1.268e-6` at `3e-6`, indicating the field/JVP path is consistent as the step shrinks. The independently reconverged branch FD remains separated from that JVP by `5.92e-3` at step `1e-4` (values `6.14e-3` and `6.03e-3` at `1e-3` and `3e-4`). This is an unresolved branch-vs-linear-response result; the derivative is not accepted. NS65 run report SHA-256 `4151a9a9c8ff25ad6d349f457eeeebc4d260a8f294a849bbddb64b49a2c7150a`; arrays SHA-256 `3fa34a565c46402336870a4b6e78dd63d75b8ed0a7c6963fabba9f41fedf2f5c`; run elapsed 92.80 s, peak RSS 4479.59 MiB, on CPU. `benchmarks/axisymmetric_root_response.py` SHA-256 `2b4b99535f3c518f46e96b8b18ae9b5a72c4881f332263bee2d3b0f6e16dae68`. Several independent NS17/33/65 records remain preserved. Branch finite differences through `1e-4` have not established asymptotic convergence for `delta`, and a dense smallest-rung linear solve is still required by T3's exit gate.

**Artifacts/code/tests inspected:** Axisymmetric reports and arrays listed above; earlier response rungs under `results/vmex/response_runs/`; corresponding run receipts and perturbed decks; test suite and schema consumer; native-knot roots and measurements; source pins. No README axis-response figure exists yet; any figure must be generated from these saved records and visually inspected before inclusion.

**T4 launch:** The generated sheared-A current-prescribed deck is in `inputs/input.sheared_A_current`; its manifest records boundary maximum fit error `5.11801940894e-7 m` and pressure/iota/current profile fit errors between `9.49e-12` and `6.38e-11`, so it passes the existing representation gate. Analytic global chart validation and implicit physical-angle derivative checks each report six passing samples in `results/reference/sheared_chart.json` and `results/reference/sheared_derivative.json`; these are analytic-reference checks, not VMEX recovery. A bounded historical NS17, `FTOL=1e-10`, current-prescribed sheared-A solve was launched as run `sheared-a-ns17-historical-current-20260924` using the deck's default TCON0 and 10,000-iteration cap. Its outcome and native score are not known at this log entry; inspect its immutable forward report when the process exits.

**Exact next action:** Inspect the completed bounded sheared-A report and saved sample arrays for convergence/geometry/native physical scores. Then diagnose the NS65 `delta` reconverged-branch discrepancy (including finer centered branch steps and a state/field consistency check) without relabeling the residual-level tangent as accepted. In the same work block generate a saved-data axisymmetric response figure only if it makes the unresolved distinction clear; visually inspect it, update README/manifest if appropriate, then refresh this task table/logbook. Before commit/push, rerun the relevant tests, remove trailing whitespace, inspect all staged paths for private machine details, confirm authenticated account identity, and review the exact outgoing diff.


### Continuation work block, 2026-09-24: sheared-A bounded recovery and response figure

**T4 sheared-A result:** The first historical cold NS17 `NCURR=1` run at effective `FTOL=1e-10`, default `TCON0=1`, and NITER=10,000 ended with `MORE ITERATIONS REQUIRED`; receipt status is `solver_failed`, elapsed solve time 24.635 s, and peak RSS 670.03 MiB. Its forward report SHA-256 is `9817231581454e3531b86f2c7174d5836b489ebf7f2f559e378d4dbc9eedd6aa`. A second run used the same source `b5f5267efc0795c4a49a224e321e9b370975c14c`, input hash `def5ce4ef00329b7c7c7c69c1c60821d2e6d1c8d7db5c3694c8ed17d6de78897`, stage controls and default constraint, with `--keep-terminal` to save and physically score the capped result. It ran 10,000 iterations, returned `solver_converged=false`, `IER=2`, `FSQR/FSQZ/FSQL = 3.5353e-7 / 3.5571e-7 / 3.9462e-7`, 24.804 s solve, 32.005 s scoring, and peak RSS 2681.56 MiB on CPU. Across 96 native samples the diagnostic scores were `E_B=6.8862e-3`, `E_J=4.6726e-2`, `E_gradp=8.5662e-2`, force-over-pressure scale `1.7379e-1`; maximum native-vs-reference flux-label difference was `5.7767e-2`, RMS `2.7146e-2`. No pointwise threshold passed; scorer resolution and root certification are false, so this is not a recovered equilibrium. The solver showed recurrent residual/step oscillations after the starting-axis Jacobian sign changed. The captured full console log hash is `794cd5b69dff11bd2d39cce07eaa1f335721e0f47159763f96c6c938d3e16910`. Terminal `forward.json` SHA-256 `74bcd258bc1f0cacb5f3fccfcd84ee638223781c75d1796f21c6b01d8e00b061`; WOUT SHA-256 `423746f6a85c62a746b6d2fadfda9f152086cf16b6187f5456fe7df699078502`; score SHA-256 `657b3a53b82b4779a56cb73ad3ba8c82180cf870c9c3e819da781fdbcd964765`. No continuation seed/projection was created yet. The fit/chart checks establish admissible input coordinates but do not guarantee solver attraction or physical recovery. Keep the first failed receipt and terminal-scored run separate.

**T3 response figure:** Added `benchmarks/plot_axisymmetric_response.py`, which reads the saved NS65 response report only. The resulting `figures/vmex_axisymmetric_response_consistency.png` was inspected visually. Its two panels show frozen-path FD/JVP consistency improving as the finite-difference step shrinks, while the reconverged `delta` branch mismatch stays near `6e-3`; the plot does not imply derivative acceptance. Figure SHA-256 `fe6e9261f95813bd270dbde6904f282f65a6c93f1e28db41d2826bc995bb3df2`; manifest `results/vmex/axisymmetric_response_figure_manifest.json` binds its script and source report hashes. README now presents this result with its explicit unresolved status.

**Branch/source/publication state:** Still local on `t0-t1-d5484d1`; no commit/push/PR for this continuation and no upstream changes. VMEX source checkout was clean at the historical pin. No global TCON0 default was changed.

**Exact next action:** At NS17, construct an independent dense solve on the same projected active DOF subspace used by `implicit_state_tangent_multi_rhs` and certify its two input-response vectors against the Krylov result. In parallel with that smaller matrix test, inspect sheared-A chart-to-VMEX angle mapping and produce a saved exact-state projection/seed with a residual and native field score before another recovery solve. For the NS65 null-direction mismatch, project the branch-state FD minus tangent onto mode blocks (especially the axisymmetric m=1 geometry/lambda combination) and only then choose a gauge-aligned comparison; do not label it a gauge effect without that check. After these actions, update the task table and append their records before broadening resolutions.


### Continuation work block, 2026-09-24: independent dense response certification

**Question/task:** Is the NS17 residual-level tangent solve independently reproducible without the block-factor/Krylov solver? This is a linear-solve certificate only; it does not resolve the NS65 reconverged-root disagreement.

**Method and pin:** At the same integer-axisymmetric `NCURR=1`, historical source pin `b5f5267efc0795c4a49a224e321e9b370975c14c`, constructed the explicit Jacobian of the preconditioned residual on the active projected DOF subspace and solved both `c` and `delta` RHS with NumPy's dense solver. The matrix uses 568 active unknowns, formed by chunked (8-column) forward-mode products. Both input tangents use the same centered `3e-4` input-map step as the NS17 response record. Run command: `PYTHONPATH=benchmarks:<historical VMEX checkout> python3 benchmarks/certify_dense_axisymmetric_response.py`; script SHA-256 `7795d8a87171743357e0b17f6d120e76c75bfd2a542cb770b749cf9c0e56bbd0`.

**Result:** The base fixed-point residual was `5.96e-15` and anchor shift zero. Dense matrix singular-value range was `3.6826e-5` to `140.8166`, 2-norm condition estimate `3.824e6`. Dense linear residuals were `4.08e-16` (`c`) and `4.32e-15` (`delta`); reconstructed full-state tangents matched the residual block/Krylov tangents at relative L2 `4.84e-12` and `6.09e-12`. The response solver reported residuals `7.11e-15` and `4.46e-13`, each below its recorded tolerance. This passes the requested smallest-rung independent linear solve check, while leaving the separate reconverged-branch mismatch and NS65 dense check open. Report `results/audit/dense_axisymmetric_response/dense-axisym-response-20260924T234912.082849Z/dense_response.json` SHA-256 `e517857c8f61367e7bc1d8b866ddd2f6ebd53ba66bdb5cccd6b73eea23c0e40a`; matrix/RHS/solution arrays SHA-256 `8b8931bdf3845a2275178c6619b36b8bb387cef0a18092dc95aad149e602a788`. It took 58.48 s and peaked at 3127.28 MiB RSS on CPU. An initial identical matrix run and its report/arrays are retained under a separate timestamped directory; the final report additionally records script/runtime hashes and settings.

**Branch state decomposition:** Existing NS65 report `axisym-root-response-20260924T232219.753772Z` was inspected without additional solves. For the `delta` branch at step `1e-4`, state-FD minus tangent L2 was `6.72e-4` in `R_cos`, `6.70e-4` in `Z_sin`, and `3.39e-3` in `L_sin`. In each field the largest discrepancy is the axisymmetric m=1 coefficient: `6.60e-4`, `6.70e-4`, and `3.38e-3`, respectively. This localizes the state difference to an m=1 geometry/lambda combination but does not prove it is a removable gauge. The physical branch-FD/JVP discrepancy remains `5.92e-3` in the fixed B scale. A follow-up must check physical field reconstruction validity and derive/validate any gauge alignment before attributing the difference.

**Artifacts/code/tests:** Final dense script `benchmarks/certify_dense_axisymmetric_response.py`, report and NPZ above; previous run artifacts remain immutable. This run did not execute the general test suite. Script is deterministic for this source/configuration, writes a unique run directory, records the solver/source and artifact hashes, and writes the explicit matrix for independent review.

**Branch/PR state:** Local branch `t0-t1-d5484d1`, no continuation commit or push yet; no PRs or upstream changes.

**Exact next action:** Construct a clearly labeled sheared-A `lambda=0` geometry-projection seed on NS17 from the exact surfaces sampled at normalized toroidal-flux nodes. Before solving from it, record its input-edge mismatch, positive-volume/native-inversion checks, VMEX force residuals, and 96-point projected physical score. Use the score to decide whether it is a defensible continuation seed. Separately audit the saved NS65 branch state through the physical-coordinate inversion at 8/12/24 Newton steps and record validity; do not infer a gauge from the m=1 concentration alone.


### Continuation work block, 2026-09-24: sheared-A exact-geometry projection

**Question/task:** Can the NS17 sheared-A case start from the exact sampled boundary/interior geometry with its fitted current/pressure profiles? This projection deliberately uses `lambda=0`; it tests the geometry and label path only and is not presumed to encode the exact straight-field coordinate map.

**Method/pin:** Added `benchmarks/project_sheared_vmex.py` at historical VMEX `b5f5267efc0795c4a49a224e321e9b370975c14c`. It samples `surface(case, label_at_s(s), theta, phi)` on the actual VMEX full radial grid and a `64x64` angular mesh, Fourier projects R/Z, applies VMEX's native m=1 transform, leaves lambda zero, evaluates VMEX's force residual, and runs the standard native Cartesian scorer. The report preserves the seed and point/observation arrays. Command: `PYTHONPATH=benchmarks:<historical VMEX checkout> python3 benchmarks/project_sheared_vmex.py`; script SHA-256 `e6dadb28b0ca95dcad8c5b10cbe591da2a0cd8e458723908a5a09f795e97cacf`.

**Projection result:** NS17, MPOL=13, NTOR=12, NFP=2. The projected R/Z edge agrees with the processed deck boundary to max coefficient errors of `5.13e-17 m` or less; largest field block coefficient L2 mismatch is `1.22e-16 m`. All 96 held-out native field inversions were finite and reference volume weights positive. Native/reference normalized flux-label max/RMS differences were `2.679e-5 / 9.258e-6`. Yet with lambda zero, field/current relative errors are `1.3406e-1 / 2.6528e-1`, gradp relative error `3.5814e-4`, and force RMS divided by the pressure-gradient scale `1.2092`; preconditioned invariant residuals are `FSQR/FSQZ/FSQL = 0.22894 / 0.11049 / 0.02972`. The projection is therefore a strong geometry/label starting representation but a poor field-equilibrium state. It does not qualify as an accepted recovery or an exact full-state projection. Report `results/projection/sheared_A/sheared-A-projection-20260924T235541.410475Z/projection.json` SHA-256 `16be101e15eff97977b56c2304c5459a6fb91dc7d47a68e46061a5d065ea837d`; seed SHA-256 is recorded in that report. The complete run took 48.334 s and peaked at 2424.42 MiB RSS on CPU.

**Current branch/run state:** Continuation branch remains local and unpushed. A bounded NS17, `FTOL=1e-10`, `NITER=10000`, default `TCON0`, historical-source solve from this saved seed has been launched with terminal-state preservation enabled as `sheared-a-ns17-projected-terminal-20260924`. Its outcome is not known at this log entry; inspect the forward report and console log after exit. The cold solve and lambda-zero projection remain separate evidence records.

**Exact next action:** Inspect the projected-seed solve outcome and native score. Compare it with the cold-start terminal diagnostic without treating the different initial states as evidence of the same branch. If it caps, use its best terminal state as a continuation candidate only after recording geometry, residual, flux label, and native scores; do not raise NITER blindly. Then run the saved NS65 field-coordinate inversion audit at 8, 12 and 24 Newton iterations.


### Continuation work block, 2026-09-24: projected-seed sheared-A recovery

**Run:** Historical VMEX `b5f5267efc0795c4a49a224e321e9b370975c14c`, same sheared-A `NCURR=1` input hash `def5ce4ef00329b7c7c7c69c1c60821d2e6d1c8d7db5c3694c8ed17d6de78897`, NS17, FTOL `1e-10`, 10,000 iteration cap, default `TCON0=1`, using the saved lambda-zero geometry projection seed SHA-256 `da31b4218a55654d702434da1329febfc04dd2587c10a2ab42270348acb07413`. Full command: `PYTHONPATH=benchmarks:<historical VMEX checkout> BENCH_FTOL=1e-10 BENCH_RUN_ID=sheared-a-ns17-projected-terminal-20260924 python3 benchmarks/run_vmex.py inputs/input.sheared_A_current 17 10000 results/projection/sheared_A/sheared-A-projection-20260924T235541.410475Z/seed_ns17.npz --keep-terminal`.

**Result:** Capped at 10,000 iterations with `solver_converged=false`, `IER=2`, `FSQR/FSQZ/FSQL = 2.3044e-6 / 2.3292e-6 / 2.5086e-6`. Solve took 24.900 s, native scoring 32.543 s, peak RSS 2644.91 MiB on CPU. The 96-point terminal scores were `E_B=1.6902e-2`, `E_J=7.1452e-2`, `E_gradp=2.2697e-1`, force-over-pressure scale `1.7339e-1`, and max/RMS native-vs-reference flux-label difference `1.7484e-1 / 6.8342e-2`. Compared with the cold terminal state, the exact-geometry-start terminal state is worse in B, J, gradp, and flux label, with comparable force ratio. The cold terminal score was `6.8862e-3 / 4.6726e-2 / 8.5662e-2 / 1.7379e-1` and max label error `5.7767e-2`. The exact projection itself had near-machine-precision edge agreement and small label error, confirming that the solver moved away from the accurate geometry/label representation under the tested relaxation. Neither solve converged or passed a physical threshold; neither is accepted. Report `forward.json` SHA-256 `651c7ac96ee80cc044e55fd0b8984ad3b92a81362a594b4a278135554d52fc69`; solver console log SHA-256 `1950962e90d71ba241fee2bb688b35a415111a9be551cd7e932310a18d6b6810`; WOUT SHA-256 `38b8e6c0b7cc6b902aa61b60d746e1f497dacff3ab55f15bd7199cba58ddab13`; native score JSON SHA-256 `fefbfc57a0b3305844f481bb8a89b00be03707627211dd8addafb7763ac929f7`.

**Interpretation:** A lambda-zero projection is insufficient to represent the sheared physical field and is not a useful replacement for the VMEX cold start under this solver. The failed projected solve plus cold solve argues for constructing/validating the straight-field lambda map or a controlled continuation path before another attempt. It does not prove a global solver basin limitation. Preserve both diagnostics separately.

**Branch/PR state:** Still local on `t0-t1-d5484d1`; no commits, push, PR or upstream write yet. No global constraint default changed.

**Exact next action:** Extend the NS65 centered reconverged branch FD through steps `3e-5` and `1e-5`, saving each perturbed state and root residual. Compare branch-state FD minus tangent by active m-mode blocks; keep coordinate inversion ruled out as the field-evaluation cause. Then derive the sheared-A VMEX lambda map from the exact field-line flow, validate straightness and edge consistency, and only launch a new recovery if its saved projection score and force residual improve. Update README with a matched sheared-A diagnostic figure from saved reports, clearly labeling all rows as capped/projection evidence.


### Continuation work block, 2026-09-24: NS65 branch field-inversion audit

**Question/task:** Does the `delta` reconverged-branch/JVP gap come from insufficient Newton convergence when VMEX field spectra are evaluated at fixed Cartesian points?

**Failure preserved:** The first audit version called `runtime_from_params` inside a JIT trace and hit JAX's restriction on nested compiler options. The attempt is recorded at `results/audit/axisym_branch_inversion/axisym-branch-inversion-20260925T000042.117212Z/failure.json`, SHA-256 `19d354e4c3608163e8e165dc10438c0c761d0b6ef4c634c0b0577a57e9d5e307`. The corrected evaluator materializes each parameter runtime before tracing and passes it into the jitted inversion/field path; the failed numerical report was not reused.

**Corrected audit/pin:** Reused the saved NS65 base and perturbed branch states from report `axisym-root-response-20260924T232219.753772Z`, source pin `b5f5267efc0795c4a49a224e321e9b370975c14c`, four fixed Cartesian points, and centered steps `1e-3`, `3e-4`, `1e-4`. Compared 8, 12, and 24 Newton iteration caps. Command: `PYTHONPATH=benchmarks:<historical VMEX checkout> python3 benchmarks/audit_axisymmetric_branch_inversion.py`; script SHA-256 `8061994754c1c918c8b058ee24fd575ca297d56adedbc142a09bd0d2bc2bd107`.

**Result:** Base and every plus/minus perturbed branch state had all 4/4 points valid at all three iteration settings. The centered branch B finite differences were identical for 8/12/24 iterations (L2/fixed-scale `8.1675e-4`, `8.2600e-4`, `9.0324e-4` at the three steps); their difference from the saved JVP remained `6.1424e-3`, `6.0337e-3`, `5.9151e-3`. Increasing inversion iterations from 8 to 24 changed each branch FD by exactly zero in the recorded float64 arrays. The field-inversion tolerance is therefore not responsible for the observed mismatch at these points. This does not explain the underlying reconverged-branch response and does not certify either derivative. Report `results/audit/axisym_branch_inversion/axisym-branch-inversion-20260925T000119.473037Z/inversion_audit.json` SHA-256 `fed4fce7f7040cdd9cd65d277f526c2726b6e28f3713ca7c449b173fef383213`; saved audit arrays SHA-256 `5b9209aa0ff462e1322059eaa96f707c515ac07c728539a0fd7cf67c9300d2e1`. Elapsed 5.33 s, peak RSS 903.06 MiB CPU. The max (24 vs 12 iteration) field FD difference is exactly `0.0` at each tested step.

**Branch/PR state:** Local continuation branch only; no commits or remote writes; no upstream changes.

**Exact next action:** Run only the missing finer `delta` branch points at NS65 (`h=3e-5` and `1e-5`) from independent cold starts, reuse the saved base state/points, and record root residual plus state-field FD. If the finite difference remains distinct from the certified residual tangent, inspect the constrained m=1 active block and the family of discrete equilibria; do not remove or weaken TCON0 based on this result. Then resume the sheared-A straight-field lambda construction.


### Continuation work block, 2026-09-24: finer NS65 null-direction branch finite differences

**Question/task:** Does the `delta` independently reconverged field response move toward the residual tangent as the centered step shrinks below `1e-4`?

**Method/pin:** Historical VMEX `b5f5267efc0795c4a49a224e321e9b370975c14c`; same generated exact-family input map and base NS65 root/state/points as the response report. Independently cold solved plus/minus roots at `h=3e-5` and `h=1e-5`, anchored and field-evaluated at the same four fixed Cartesian points. Both variant decks, state arrays, root residuals and script hash are saved. Command: `PYTHONPATH=benchmarks:<historical VMEX checkout> python3 benchmarks/probe_axisymmetric_branch_fine_steps.py`; script SHA-256 `b33691521a4dc4ac845907ec88da2f9544c7f09fa7720a1fc72d2e9cb5c26fc4`.

**Result:** All four branch roots had residuals between `5.84e-14` and `8.06e-14`, with solver iterations 474 on every perturbation and zero anchor shift. All four fixed points remained valid. At `h=3e-5` the branch B FD L2/fixed scale is `9.7701e-4` and its difference from the residual JVP is `5.9163e-3`; at `h=1e-5` these are `9.8674e-4` and `5.9172e-3`. The branch FD changed from the previous `h=1e-4` value by only `1.93e-4` and `2.13e-4`, respectively, while the much larger branch/JVP gap persisted. Thus the tested smaller centered steps do not remove the mismatch; they do not prove the limiting derivative exists. State-FD minus residual-tangent L2 at `h=1e-5` is `7.52e-4` in `R_cos`, `7.52e-4` in `Z_sin`, and `3.385e-3` in `L_sin`; the largest component of each is m=1 (`7.42e-4`, `7.52e-4`, `3.383e-3`). This remains a mode-localized observation, not proof that m=1 can be removed as gauge. At `h=1e-5` the saved plus/minus inputs are `input_delta_ns65_h1e-05_plus.indata` SHA-256 `ae22f8748d0436eeb40a522da01393195c4eb23d49e47414a8a4924e7a9aea58` and minus SHA-256 `104574b5c7c757a9da9e527e46c491b3b69add53b798eb2ebe23c0a29d962e8b`; the `h=3e-5` inputs are hashed in the report. Fine-step report `results/audit/axisym_branch_fine_steps/axisym-branch-fine-20260925T000429.848273Z/fine_branch_probe.json` SHA-256 `635b2929aa495b40f2b42a1297fffe67f400b47086d773609b26ed5837175080`; full state/field arrays SHA-256 `93213116c568576c59764097d2ce5ebe6dc095f30b4b9b04ead944190d115e1e`. Total elapsed 20.81 s, peak RSS 1451.86 MiB CPU. A first identical fine-step run without saved input decks remains separately archived and was superseded only for reproducibility completeness, not numerical data.

**Assessment:** Together with the field-inversion audit (all valid and bitwise stable at 8/12/24 iterations), this narrows the disagreement to the discrete equilibrium branch/state response, not a field coordinate Newton cap. The residual-level tangent itself is independently dense-certified at NS17, but its physical null-direction prediction remains not validated against the NS65 re-solved branch. Do not change TCON0 or report an accepted derivative.

**Branch/PR state:** Local only; no continuation commit/push or PR, no upstream changes.

**Exact next action:** Implement a small saved-data analysis that identifies the constrained/active m=1 eigen-combination between each branch-state FD and residual tangent, verifies whether applying the corresponding transformation preserves physical B to tolerance, and records both before/after field differences. In parallel, derive the sheared-A lambda coordinate from the exact field-line flow and compare projected field/current before any more recovery solves. The README should get the sheared-A matched diagnostic figure from its three saved states, all labeled as projections/capped outputs.


### Continuation work block, 2026-09-24: smaller-step NS65 branch response

**Question/task:** Does the null-direction field branch derivative approach the residual-level tangent at smaller steps once inversion accuracy is already ruled out?

**Runs/pin:** Historical VMEX source `b5f5267efc0795c4a49a224e321e9b370975c14c`; NS65; same base state, four Cartesian points, and fixed-current exact input family as the saved response report. Four independent cold root solves cover plus/minus at `h=3e-5` and `1e-5`. All perturbed inputs are saved with hashes. Command: `PYTHONPATH=benchmarks:<historical VMEX checkout> python3 benchmarks/probe_axisymmetric_branch_fine_steps.py`; code SHA-256 `b33691521a4dc4ac845907ec88da2f9544c7f09fa7720a1fc72d2e9cb5c26fc4`.

**Measurements:** Branch B-FD L2 over the fixed `1 T/m` scale is `9.77009e-4` at `3e-5` and `9.86737e-4` at `1e-5`; B-FD minus residual-JVP is `5.91633e-3` and `5.91722e-3`. Relative to `h=1e-4`, branch FD changes `1.927e-4` and `2.134e-4`. All plus/minus anchored residuals lie between `5.84e-14` and `8.06e-14`, with 474 solver iterations each, zero anchor shift, and all four points valid. At `h=1e-5`, state-FD minus tangent L2 is `7.525e-4` in Rcos, `7.519e-4` in Zsin and `3.385e-3` in Lsin; m=1 accounts for `7.415e-4`, `7.518e-4` and `3.383e-3`. These small-step independent roots reproduce the mismatch; they do not establish a derivative limit or prove a gauge interpretation. Fine-step report `results/audit/axisym_branch_fine_steps/axisym-branch-fine-20260925T000429.848273Z/fine_branch_probe.json` SHA-256 `635b2929aa495b40f2b42a1297fffe67f400b47086d773609b26ed5837175080`; state/field arrays SHA-256 `93213116c568576c59764097d2ce5ebe6dc095f30b4b9b04ead944190d115e1e`. The run took 20.81 s and peaked at 1451.86 MiB RSS on CPU.

At `h=1e-5`, plus/minus deck SHA-256 values are `ae22f8748d0436eeb40a522da01393195c4eb23d49e47414a8a4924e7a9aea58` and `104574b5c7c757a9da9e527e46c491b3b69add53b798eb2ebe23c0a29d962e8b`; all four deck hashes are in the report. The first run with identical numerical results but without saved decks is preserved separately; the final run adds input artifacts and hashes.

**Assessment:** The remaining disagreement is stable under tighter coordinate inversion and smaller centered parameter steps. Root anchoring is not the cause at the measured tolerances. The mismatch is now sufficiently discriminating to separate the field map's derivative along the actual branch-state finite difference from the response obtained by differentiating the equilibrium residual.

**Exact next action:** At h=1e-5, compute `jax.jvp` of the physical field reconstruction using the measured branch state FD as its state tangent and the same signed input tangent. Compare this to the actual branch B-FD, then separately report the contributions from Rcos, Zsin, Lsin and direct input/profile terms. This determines whether the coordinate response difference explains the physical gap before attempting any mode alignment. Keep the branch derivative uncertified. In the same handoff block, generate the matched sheared-A projection/cold/capped-seed figure from the three saved reports and update README/manifest.


### Continuation work block, 2026-09-25: axisymmetric branch field-map discriminator and sheared-A figure

**Question/task:** Does the NS65 discrepancy between the null-direction residual-level tangent and independently reconverged branch B-FD come from the physical-field reconstruction, or from the root/state response? In parallel, make the three sheared-A projection/capped-state results reviewable on one verified sample grid.

**Axisymmetric method/pin:** At historical VMEX `b5f5267efc0795c4a49a224e321e9b370975c14`, applied `jax.jvp` to the physical Cartesian field reconstruction at the saved NS65 base state using the centered branch-state difference at `h=1e-5` plus the same centered signed input/profile tangent. Decomposed the response into `R_cos`, `Z_sin`, `L_sin` and direct input/profile contributions. The two independently saved branch roots had anchored residuals at most `8.0628e-14`; component-sum closure was `9.63e-13` in the fixed field scale. Command: `PYTHONPATH=benchmarks:<historical VMEX checkout> python3 benchmarks/decompose_axisymmetric_branch_jvp.py`; final script SHA-256 `8d285ee058982d0162544249d9dfacc35919ef45fcc636e13f35642259c8ac42`.

**Axisymmetric result:** The measured branch B-FD norm was `9.86737e-4` in the fixed `1 T/m` scale; field-reconstruction JVP using the measured branch state FD was `9.88037e-4`, differing from branch B-FD by `6.88258e-6` (relative `6.9751e-3`). The residual-level JVP norm was `6.19395e-3`; its direct difference from branch B-FD is `5.91722e-3`. Thus the field map reproduces the observed physical response along the measured state direction to about 0.70%; the unresolved discrepancy is in the equilibrium-root/state response, not the local field reconstruction. Direct input/profile and geometry-block terms are individually large and cancel: their fixed-scale norms are `130.724` (`direct`), `70.003` (`R_cos`), `63.056` (`Z_sin`) and `0.576` (`L_sin`). This is diagnostic only; it neither identifies the cause within the root response nor certifies a derivative. Report `results/audit/axisym_branch_jvp_decomposition/axisym-branch-jvp-20260925T001031.029246Z/branch_jvp_decomposition.json` SHA-256 `5c3b03be949df4afb67d9024c025a4ac4ea50001528d0e1f970c8175dd3bb64c`; component arrays SHA-256 `df6450082fe85be179b1def90cf804f461469babe625ad59c879d315347257f5`. The run took 4.00 s and peaked at 732.50 MiB RSS on CPU.

The first invocation completed the numerical work and wrote its report/arrays but exited nonzero in a post-write print expression that attempted a boolean check on a NumPy array. That immutable initial record remains under `axisym-branch-jvp-20260925T000812.228804Z`; report SHA-256 `ab904fbe69a5f1ed00ab2a7024ab7f85f77d1fb55a0f334c5ef4bf0b9dca7c14`, arrays SHA-256 `df6450082fe85be179b1def90cf804f461469babe625ad59c879d315347257f5`. The reporting expression was fixed and the full analysis rerun successfully at the final run ID above. The failure did not change numerical inputs or require repeating the root experiment.

**Sheared-A figure:** Added `benchmarks/plot_sheared_recovery_diagnostic.py`, which verifies the historical source pin, identical processed-input hash, NS17 and 10,000-iteration caps, 96-sample score type, and byte-identical point-cloud hashes before plotting the exact R/Z, lambda-zero projection and the cold/projected terminal states. The figure `figures/vmex_sheared_a_ns17_diagnostic.png` was visually inspected. It shows B/J/grad-p/force-over-pressure and maximum flux-label errors on log scales, with the separate lambda-zero projection and capped terminal states explicitly labeled. Figure SHA-256 `5f3864af5f33230a4e50d4c38af6b321578568a80bcd55f1ccb435f0132e7d6e`; manifest `results/vmex/sheared_a_ns17_figure_manifest.json` binds its script and all source report/grid hashes. README links the figure and preserves the non-recovery interpretation.

**Failure/attempt notes:** The first figure rendering exposed an invalid escaped MathText command in the pressure-gradient panel title; the label was corrected and the figure regenerated from the same immutable source reports. No solver rerun was needed.

**Validation after edits:** `python3 -m pytest -q` passed all 55 tests in 16.44 s on Python 3.11.14 (VMEX was discoverable in this environment); `python3 benchmarks/verify_reference.py` passed all 14 reference cases; `python3 -m compileall -q benchmarks tests` and `git diff --check` passed. The earlier clean Python 3.12 environment with VMEX absent also passed all 55 tests for this block's T0 implementation. All nine files in the supplied review handoff pass its package `SHA256SUMS`. The inherited root `HANDOFF_SHA256SUMS` was refreshed for its 62 covered files and all current hashes now validate.

The final staged review contains 219 paths, with no file larger than 1.76 MiB. The staged-content scan found no home-directory paths, local checkout/environment names or host aliases. `git diff --cached --check` is clean for all new work except ten intentional Markdown hard-break spaces in the supplied handoff and byte-for-byte archived plan; the exact source package checksums pass, so those documents are preserved unchanged. Local Git author and committer resolve to `rogeriojorge <6816712+rogeriojorge@users.noreply.github.com>`, and GitHub API authentication resolves to login `rogeriojorge`. `tools/publish.sh` is not used because its publication path pushes directly to `main`; the reviewed branch will be pushed only to `t0-t1-d5484d1`.

**Branch/PR state:** Local continuation branch `t0-t1-d5484d1`; no continuation commit/push or PR yet; no upstream PR or merge.

**Exact next action:** Use the saved branch-state/tangent arrays to characterize the active m=1 response difference, and test any candidate alignment by re-evaluating physical B before calling it gauge. In parallel, derive the sheared-A straight-field lambda map from exact field-line flow, verify straightness and edge consistency, and require improved saved projection/force scores before launching another recovery. Then rerun the VMEX-absent core suite, inspect the exact staged diff and privacy scan, and commit/push the sanitized continuation branch.
