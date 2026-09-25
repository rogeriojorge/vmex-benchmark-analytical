# VMEX analytical benchmarks: continuation after d5484d1

**Owner:** `rogeriojorge`  
**Repository:** `https://github.com/rogeriojorge/vmex-benchmark-analytical`  
**Review date:** 2026-09-24  
**Reviewed main:** `d5484d1e7a15c9cf10599c2b0f05bdf69e22e861`  
**Preceding reviewed main:** `575f13f67b346118c3d7f05cc60cc6e29131359a`

This is a continuation of the existing implementation, not a replacement project. It retains the analytical models, case matrix, useful measurements, code and historical logbook. It supersedes the immediate instruction to rerun the interrupted composite calculation unchanged. Read [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md) and [REFERENCES.md](REFERENCES.md) with this plan. Source labels S1-S12 and literature labels L1-L13 refer to that reference file.

The review inspected current source and recorded evidence, but did not execute VMEX, DESC, GVEC, free-boundary or kinetic benchmarks. The six delivered review probes ran independently of those codes. Their outputs illustrate numerical mechanisms and contracts; they are not additional equilibrium results.

## 1. Adoption, scientific mandate and operating rules

Preserve the exact current plan and all logbook entries before changing the active plan. For the reviewed object, a suitable archive is `docs/history/plan_through_d5484d1.md`, obtained with `git show d5484d1e7a15c9cf10599c2b0f05bdf69e22e861:plan.md`. If the working tree or remote has advanced, preserve and reconcile those additions as well. Do not reset local work or replace a newer plan solely because this handoff has a later filename. Keep one active contract, historical links and a current status table.

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
| T0 portable evidence/tests | R0 repairs retained; CI/import and schema mismatch found | Clean reference test run and new-run consumer integration |
| T1 measurement | Independent oracle and pressure audit useful; radial norm unresolved | Actual native-knot partition, streaming/checkpoints, control measurements |
| T2 constraint/root | Historical ladder partial; missing NS129 zero cell | Pure residual test, completed cell and physical decomposition |
| T3 sensitivities | Complete input-map FD recorded; no solver response | Signed/scaled input tangent and anchored axisymmetric response |
| T4 breadth | Earlier projections/axisym/surface-LASYM/DESC diagnostics remain | Mild sheared-A recovery and full LASYM field path |
| T5 diagnostics | Broad program mostly pending | Exact-flow and bounded field-derivative/interface tests |
| T6 exterior/polish | Mostly pending; separate PR448 research context | Exact operator fixtures, strict coupled case, bounded optional polish |
| T7 optimization | Planned | Direct-family/chart pilot, then verified solver-driven extensions |
| T8 presentation/cost | Many historical figures; new cost bottleneck established | Read-only updated figures, complete resource accounting |

For every work block append: question; relevant task; benchmark and imported-source hashes; branch/PR state; command and effective configuration; parent/input/state/grid/contract hashes; measurements with uncertainty; failed or interrupted attempts; artifacts; code/tests actually inspected or run; and the exact next action. Distinguish 'not run', 'failed', 'unavailable', 'diagnostic', 'resolved measurement', 'accepted recovery', and 'accepted derivative'. Do not infer a terminal state from an old paragraph saying a process was running; use the newest receipt and actual process state locally.

### Review entry: continuation from d5484d1

The review confirms two commits beyond the earlier snapshot and retains the evidence repairs. New priorities are clean reference tests, compatible schema readers, real native-knot alignment, progressive convergence logic, interruption-safe streaming, signed/scaled input tangents and explicit reconstruction regularity. Current GitHub reference CI failed at pytest; the source contains a sufficient missing-dependency path, but detailed failure logs were not retrieved. The current VMEX release and open PR448/VMEC++ PR849 were checked as separate sources, not imported as successful benchmark results.

Six standalone probes passed: displaced-knot quadrature; NS-dependent grid-count arithmetic; exact integer-family flow/tangent/volume preservation; dimensionally scaled derivative comparison; derivative regularity of a C2-map field; and streaming sufficient-statistic equivalence. No new repository solver or full benchmark suite was executed by this review. No remote repository writes were made.

**Exact first action:** preserve the current plan/logbook and compare the working tree with d5484d1. Fix the solver-independent test imports and the schema reader, then write a cheap test that exposes the reference/native knot distinction and preserves a completed grid on interruption. Do not start by repeating the 27-minute composite profile unchanged.
