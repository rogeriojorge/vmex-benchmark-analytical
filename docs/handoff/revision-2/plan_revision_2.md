# Analytical benchmarks for VMEX: revised implementation plan

**Revision 2 - 2026-09-24**  
**Owner:** `rogeriojorge`  
**Existing public repository:** `https://github.com/rogeriojorge/vmex-benchmark-analytical`  
**Reviewed benchmark commit:** `575f13f67b346118c3d7f05cc60cc6e29131359a`

This is a continuation of the existing project, not a request to recreate it. The earlier analytical definitions, case files, useful scripts, raw arrays and logbook remain part of the project. This revision changes the priority order, measurement contracts and acceptance criteria in light of the recorded experiments and source review. It does not discard the broad benchmark scope.

The review used committed source, reports, PR metadata and primary literature. It did not rerun VMEX or DESC. Numerical solver results below are reported observations from the reviewed snapshot, not new measurements by the reviewer. Three standalone mathematical probes were run; they do not establish the errors of any saved equilibrium.

## 1. Adoption, history and immediate instructions

Read `REVIEW_FINDINGS.md` and `REFERENCES.md` before implementing. Compare the current repository head with the reviewed commit. Preserve subsequent user work and record any intervening changes. Do not reset a working tree to this review snapshot.

Archive the previous plan, including its entire logbook, before installing this revision as the active `plan.md`. A suitable tracked location is `docs/history/plan_through_575f13f.md`, copied from that exact Git object. If the current plan has later logbook entries, preserve those too and carry their unresolved actions into the new current-status section. Avoid duplicate copies of source code or numerical result trees. Retain the historical plan link at the top of the new active plan.

PR #1 is merged at the reviewed snapshot. Its description contains stale wording saying it is unmerged. Read actual GitHub state instead of repeating that sentence. The benchmark repository already exists; do not run repository-creation steps again.

The next implementation block has five deliverables:

1. Make figure generation read-only and repair run metadata/label contracts, with regression tests.
2. Create a converged independent scorer, retaining the old 96-point grid only for continuity.
3. Finish the one missing projected NS129, TCON0=0 historical-baseline cell, keeping the matching default result and immutable run provenance.
4. Test the force operator's history independence and separate physical, coordinate and reconstruction effects.
5. Begin an axisymmetric exact-family derivative test and a sheared-A recovery path in parallel, rather than making every phase wait for the integer 3-D diagnosis.

Do not begin a wholesale coordinate-system rewrite, a global TCON0 default change, a large kinetic campaign, or a new optimization framework during this block.

## 2. Questions, scope and scientific outcome

The project should answer these questions with independently checked data:

- Does the numerical equilibrium recover a known physical field, its current and pressure gradient, not only a small discrete force residual?
- Which errors come from the input approximation, state representation, nonlinear root, field reconstruction, inverse coordinates and measurement grid?
- Do physical equilibrium sensitivities approach the derivatives of the exact family? How much do coordinate constraints and initialization affect them at finite resolution?
- Which public modules and downstream interfaces preserve those properties for both symmetry settings, both closure choices and the appropriate boundary conditions?
- What can be optimized within an exact solution family, and which nearby physical deformations remain regular and accurately differentiable?

A useful result can be a demonstrated limitation with a resolved cause. Failure to converge a particular family is not a reason to abandon the entire benchmark. Conversely, an unresolved discrepancy cannot be converted into a successful capability by changing a caption or a tolerance.

The most promising extension beyond reproducing Landreman is **a quantitative study of physical sensitivity under coordinate freedom**. Exact-family tangent directions, pure coordinate changes and transverse physical perturbations provide three different controls. They must not be mixed.

## 3. Source baselines and reproducibility

Keep the following historical pins immutable:

| Role | Repository | Pin |
|---|---|---|
| Historical benchmark | rogeriojorge/vmex-benchmark-analytical | `575f13f67b346118c3d7f05cc60cc6e29131359a` |
| Historical primary solver | uwplasma/vmex | `b5f5267efc0795c4a49a224e321e9b370975c14c` |
| Analytical supplement | landreman/analytic_3d_equilibria | `4c0b690ddebdc71811c88223eb9f44a98ab64222` |
| Recorded DESC | PlasmaControl/DESC | `4f48720beac3d4169e9165923d730445118bc2de` |
| Recorded SOLVAX | uwplasma/SOLVAX | `2e246a5d6093662f9b5f72c46f995cd7c4bbd479` |
| Recorded Boozer transform | uwplasma/booz_xform_jax | `cd25084422de10b620bd86ede0bbd51ba06d7fa6` |

The VMEX main head inspected in this review was `4632dad8261ca72756819c1c5fcd2e2ec022aeaa`. Treat it as a second snapshot, not an invisible replacement of the historical baseline. Resolve a later head explicitly if needed. The last inspected commit concerns dependency-floor planning; it does not itself establish that any numerical discrepancy is fixed.

Create baseline and candidate worktrees outside this repository. Record imported module paths, Git commits, dirty patches, Python/package versions, float precision, platform, device actually used, thread settings and relevant JAX environment variables. `git rev-parse HEAD` in the benchmark working directory does not identify an imported solver's revision. A package version string does not replace a source hash.

Keep DESC in its compatible environment. The current VMEX dependency discussion records a real old-Equinox/new-JAX import problem and conflicts with some DESC dependencies [R04]. Run `pip check` and import/solve smoke tests. Do not fix one environment by blindly upgrading all dependencies in another. Record CPU and accelerator environments separately; do not infer parallel execution from the list of available devices.

Resolve null optional-library pins only as their phases are reached. Review the source actually imported by the chosen capability. An inventory, a semantic review, an executed test and a physical validation are four separate ledger fields. Prioritize the reachable source slices needed below while the wider VMEX review continues.

## 4. Present measured state

The recorded reference suite has 29 passing tests, 14 configurations and 28 parser-checked decks. Sheared B/C boundaries have been refined beyond the original handoff. Their latest input fit errors clear a smoke gate, not a final solution accuracy budget. The explicit local sheared-map derivative checks do not yet differentiate the complete boundary/profile/flux conversion.

The table below reproduces selected reported values. All volume scores use the legacy 96-point grid and require stronger measurement certification.

| Case / representation | NS | TCON0 | E_B | E_J | E_F,p |
|---|---:|---:|---:|---:|---:|
| Axisymmetric integer, prescribed iota, solved | 129 | deck value | 6.51e-7 | 8.24e-5 | 8.18e-4 |
| Axisymmetric integer, prescribed current, solved | 129 | deck value | 6.89e-7 | 8.25e-5 | 8.18e-4 |
| Integer 3-D, projected exact state, no solve | 129 | not a root test | 2.42e-7 | 7.14e-6 | 8.15e-5 |
| Integer 3-D, projected start, solved | 33 | 1 | 8.71e-4 | 4.39e-1 | 4.93 |
| Integer 3-D, projected start, solved | 33 | 0 | 4.48e-5 | 1.73e-2 | 1.82e-1 |
| Integer 3-D, projected start, solved | 65 | 1 | 7.06e-4 | 1.49e-1 | 1.70 |
| Integer 3-D, projected start, solved | 65 | 0 | 8.26e-6 | 1.51e-3 | 1.67e-2 |
| Integer 3-D, projected start, solved | 129 | 1 | 1.30e-4 | 2.92e-3 | 1.74e-2 |
| Integer 3-D, projected start | 129 | 0 | **not run** | **not run** | **not run** |

The reported TCON ladder has nine measured solved states, including cold starts and intermediate strength at NS33. All meet their recorded discrete FTOL=1e-10 stopping test; none passes all three physical thresholds. This statement does not include the separately successful axisymmetric control.

Asymmetric Solov'ev WOUT surface B errors decrease with radial refinement, and a WOUT/restart round trip preserves surface values. The native Cartesian LASYM path is unsupported at the historical pin. A fitted continuous reconstruction is a separate path whose error must be measured, not silently substituted.

DESC now supplies an independent native comparison. Its base M=N=8, L=10 solved integer-3D state has reported E_B=1.05e-5, E_J=1.48e-4 and E_F,p=1.91e-5. The remapped M8 run is iteration-capped and remains unconverged. Its boundary differs from the analytical boundary by a nonzero fit error, so the existing table does not rank solvers at equivalent accuracy.

The complete-family solver derivatives, free boundary, polishing responses and physical optimization remain open. The historical logbook should remain available for all intermediate corrections, failed launches and source-review limits.

## 5. R0: repair evidence handling before new results

**Depends on:** adoption of this plan. **Maps to old phases:** P0/P10. **Priority:** immediate.

Implement the confirmed harness repairs in `REVIEW_FINDINGS.md`:

- Remove writes to raw reports and sample scores from plotting scripts. Read observations from records, never from a hard-coded iteration/version table.
- Preserve existing bytes and record historical amendments separately. Verify each recoverable field against logs, state metadata or environment records. Leave unrecoverable facts unknown.
- Give each run an immutable ID and parent ID. Save the effective input and source/seed hashes. Repetition produces another run, not an overwrite.
- Store a solver-independent point cloud separately from solver observations. Replace inherited `vmex_s` in DESC with actual `s_native=rho_DESC**2`; keep `s_reference` separate.
- Resolve source commits from imported packages, and report actual per-stage NS/FTOL/NITER/TCON0 settings. Correct failed-run RSS collection and distinguish host/device memory.
- Separate `solver_converged`, `pointwise_thresholds_met`, `measurement_resolved`, `representation_resolved`, `root_certified`, and `accepted`. A terminal state may have a physical score while `solver_converged=false`.

Use a small JSON contract, not a workflow engine. A practical record contains identity, effective configuration, source/environment, status/certificates, metrics, timings and artifact hashes. Save raw nonlinear reports, sample arrays and optional native state independently of derived summaries. Figure manifests name the exact generator commit and their read-only inputs.

Required regression tests: rendering does not mutate inputs; a capped state cannot be accepted; mixed source IDs cannot be silently pooled; requested/effective control differences are represented; reference samples contain no solver-specific state; DESC labels are independently assigned; failed-run records are finite/valid without pretending missing memory was measured.

**Exit:** tests exercise these paths, all existing JSON parses, a historical amendment table exists, and figures can be regenerated without modifying raw records. Physical results with incomplete metadata are retained as historical diagnostics, not silently deleted.

## 6. R1: independent physical scoring and an error budget

**Depends on:** R0 for accepted reports. **Maps to:** P1/P2/P10. **Priority:** immediate.

### 6.1 Quantities and normalization

At identical physical positions define

$$
E_B=\frac{\|\mathbf B_h-\mathbf B_e\|_{L^2}}{\|\mathbf B_e\|_{L^2}},\qquad
E_J=\frac{\|\mathbf J_h-\mathbf J_e\|_{L^2}}{\|\mathbf J_e\|_{L^2}},
$$
$$
E_{F,p}=\frac{\|\mathbf J_h\times\mathbf B_h-\nabla p_h\|_{L^2}}
{\|\nabla p_e\|_{L^2}},\qquad
E_{F,B}=\frac{\mathrm{RMS}(|\mathbf J_h\times\mathbf B_h-\nabla p_h|)}
{B_*^2/(\mu_0 L_*)}.
$$

Also record dimensional RMS, maximum sampled error, grad-p error, divergence, pressure variation and flux-surface mismatch. For vacuum or nearly zero reference current, do not divide by a vanishing reference norm. Use stated fixed magnetic scales. Maximum sampled values are not proven continuum suprema.

Adopt one permeability constant and record it. Length, field, pressure, current and flux scale as

$$
\mathbf x=L_*\bar{\mathbf x},\quad \mathbf B=B_*\bar{\mathbf B},\quad
p=B_*^2\bar p/\mu_0,\quad \mathbf J=B_*\bar{\mathbf J}/(\mu_0L_*),\quad
\Phi=B_*L_*^2\bar\Phi.
$$

The physical toroidal angle is phi. If zeta=NFP*phi is used, record that fact in every derivative and volume weight. Choose full-torus weights as the public contract; one-period integration must have an explicit replication factor. Verify the integral of one against exact/reference volume before trusting absolute integrals.

### 6.2 Three sampling roles

Keep `legacy96` unchanged for historical comparison. Do not relabel it a certified quadrature.

Create a volume integration ladder on a fixed state. For mild cases, start with roughly 8 radial by 32 poloidal by 32 toroidal nodes, then double separately where the error changes. These are starting experiments, not guaranteed sufficient resolutions. Stream batches. Use independent nonzero angular shifts and alternative radial nodes to detect accidental alignment with radial knots or periodic modes. Strongly shaped B/C inputs with toroidal modes near 100 require much higher angular resolution than the mild cases; begin from the actual represented bandwidth and verify convergence of nonlinear quantities.

Add targeted near-axis and near-edge samples, including several small positive rho values, radial cell interiors and one-sided approaches to interpolation knots. The axis itself requires the regular limiting formulas, not an ill-conditioned polar inverse. Report subregion errors separately; do not remove a difficult region from a previously declared norm without stating the change.

Use an independent scattered point set as a defect detector. It does not replace physical-volume quadrature unless its measure and statistical error are defined. A training grid used to optimize a chart is not its final verification grid.

### 6.3 Two field-evaluation routes

Route A evaluates the current native Cartesian API, including coordinate inversion. Route B evaluates the same native representation through a forward chart and its derivatives. Let q=(s,theta,phi), and write

$$
\mathbf X(q),\qquad C=\partial\mathbf X/\partial q,\qquad
\mathbf b(q)=\mathbf B_h(\mathbf X(q)).
$$

Then

$$
D_x\mathbf B_h=(D_q\mathbf b)C^{-1},\qquad
\nabla_x p_h=C^{-T}(p_s,0,0)^T,\qquad
\mathbf J_h=\nabla_x\times\mathbf B_h/\mu_0.
$$

Use a stable 3x3 solve, not an explicit large inverse. Match coordinate definitions and radial staggering; a high-order fitted lift is not Route B unless labelled as a different representation. Validate with analytical fields and compare A/B at common physical points. Record geometry reconstruction error `X(q(x))-x`, inverse consistency `C D_xq-I`, Jacobian orientation and native radial inclusion.

This forward-chart route is also the highest-value performance improvement: the existing scorer repeats costly inverse-coordinate differentiation. Reuse coordinate results and one field Jacobian, jit functions outside the batch loop, stream reductions, and avoid recompilation from newly created closures. Validate scalar and batched results before reporting speed. Measure whether a direct coordinate evaluator is actually faster; do not promise a factor in advance.

For public spatial derivatives, test B and its first derivative first, then second and third derivatives on smooth interior regions. State the continuity class of the interpolation; derivatives at knots cannot be claimed where they do not exist classically.

### 6.4 Input and measurement error budgets

Separate input boundary/profile error, analytical-reference integration error, projection error, nonlinear/root error, reconstruction error, inverse error and quadrature error. Do not add relative errors as if this were a rigorous bound for nonlinear force; measure each controlled difference directly.

Retain the initial mild-case targets E_B<=1e-5, E_J<=1e-3 and E_F,p<=1e-3. These remain initial acceptance levels, not a universal specification for every downstream observable. Aim to keep reference and measurement uncertainty below one tenth of the relevant target. Require agreement between independently shifted/refined scores within a small fraction of the target and a stable trend in the reported value. Record cases where roundoff prevents a requested target rather than inventing convergence orders.

The production root tolerance must be tightened until relevant physical values and derivatives stop changing appreciably. A small squared normalized FSQ does not supply this bound. Compare fixed-state measurements before and after each refinement to avoid confusing a changed quadrature with a changed equilibrium.

**Exit:** at least the axisymmetric solved state, integer-3D exact projection and one problematic integer solved state have converged independent scores, checked volume normalization and an A/B reconstruction comparison. The remaining matrix inherits the tested contract.

## 7. R2: close the coordinate-constraint experiment without overinterpreting it

**Depends on:** R0; R1 for physical certification. **Maps to:** P2/P3/P4. **Priority:** immediate.

### 7.1 Complete the missing historical cell

First verify the local executable really uses the historical VMEX pin. The old command is:

```sh
BENCH_FTOL=1e-10 BENCH_TCON0=0 python benchmarks/run_vmex.py \
  inputs/input.integer_3d_iota 129 3000 \
  results/projection/integer_3d_vmex_ns129/seed.npz
```

The repaired runner may use clearer equivalent arguments, but must record the effective controls. Preserve an unmodified copy of the seed and its hash. Save nonlinear history, final native state, effective input and both legacy/new samples. Use the same saved default NS129 state as a comparison; rerun default only when source/environment or incomplete provenance requires it. Do not extrapolate the zero-strength answer from NS33/65.

The legacy experiment uses FTOL=1e-10 and is not automatically a derivative-ready root. Root anchoring and a tolerance ladder come later. A failed run remains a result with a documented reason and available terminal state.

### 7.2 Establish a deterministic residual before using Newton or adjoints

Read the exact code paths that construct the m=1 constrained subspace, select residual branches, rebind hot-start baselines and reuse preconditioner/cache data. Test

$$F(x),\ F(y),\ F(x)$$

with fresh and reused runtime objects. Exercise both directions across any residual-dependent threshold, repeated identical calls, symmetry settings, and TCON0=0/nonzero. Repeat for the residual actually differentiated, not only a displayed FSQ scalar. Then test JVP linearity, finite-difference consistency and the adjoint dot-product identity.

VMEC++ issue #624 and PR #626 show why this is necessary [R05]: a previous-residual m=1 policy was appropriate for legacy iteration parity but inappropriate for a reusable function F(x). The same issue is not established in VMEX. A passing VMEX purity test is a useful outcome; it prevents misdiagnosing a coordinate constraint as a stale-state operator defect.

Preserve iteration compatibility when testing a stricter stateless residual. Do not silently replace the original time-stepping policy and call the resulting path identical to VMEC2000.

### 7.3 Decompose the residual and the physical response

On each unchanged projected/solved state, record the force before and after the coordinate-constraint terms, basis conversion, m=1 projection, normalization and preconditioning. Document actual source formulas. Where the stages do not combine additively, do not infer a physical/constraint decomposition by subtracting differently normalized scalar norms.

Record mode and radial contributions in dimensional or consistently scaled units. The increase of a normalized FSQ with NS need not mean an increasing continuum force. Its denominator, radial scaling and active DOFs also change.

Measure pressure and the normalized toroidal-flux label at physical points. A pure poloidal coordinate change leaves s=Phi_t/Phi_edge unchanged. Distinguish a changed theta representative from a displaced flux surface or an inverse-map error.

Use independent virtual-work projections of physical force onto controlled smooth displacements as a check of the discrete force map. Respect VMEX's prescribed-profile and flux constraints. Do not substitute a naive magnetic-plus-pressure energy variation at fixed coefficients for the functional actually represented by the solver, particularly with prescribed pressure/GAMMA=0.

### 7.4 Refine separate controls, not only NS

After the missing cell, vary radial resolution, angular truncation, angular quadrature, nonlinear tolerance and constraint strength independently on a small number of states. Reuse projected starts and continuation; retain selected cold starts as basin-of-attraction controls. Inspect spectral tails of geometry, lambda, B and J. Exclude the major-radius constant mode when defining a shape-width metric; also measure derivative-weighted tails.

Continue within the exact integer family from epsilon=0 to epsilon=0.5 with a fixed 3-D mode set. This avoids switching the meaning of the state vector when a symmetry-breaking amplitude leaves zero. Compare prescribed-current and prescribed-transform closures for the same physical sequence. In parallel start sheared A, whose nonconstant transform provides a different diagnostic of the same numerical questions.

Repeat a minimal axisymmetric, projected integer and solved integer control on the separately pinned current VMEX. Change one software or environment dimension at a time where practical. A current-head improvement must have both a source explanation and matched measurements.

### 7.5 Decision rules

- If forward-chart and inverse-chart scores disagree, repair the field/reconstruction path before attributing the effect to force balance.
- If the reusable residual depends on history, isolate the policy in a narrow upstream PR and rerun the minimal experiment before broader conclusions.
- If errors decrease with angular resolution at fixed NS, fix angular representation/aliasing rather than extending radial ladders alone.
- If a regular gauge improves efficiency at the same physical error, implement it as an initialization or preconditioning option before altering equilibrium physics.
- If TCON0=0 introduces extra null directions, an unconstrained inverse is not a valid adjoint. Impose an explicit mathematically equivalent gauge or use the validated nonzero-strength representation.
- If all validated paths show a persistent physical discrepancy, preserve a minimal reproducer and use independent codes to distinguish discretization, branch selection and model restrictions.

**Exit:** the missing cell is resolved, the physical measurement is converged, operator purity has an outcome, and the leading discrepancy is either isolated with a reproducer or bounded with specific remaining alternatives. A global default recommendation is not an exit requirement.

## 8. R3: coordinate freedom, regular charts and derivative tests

**Depends on:** R1 and an appropriate deterministic root from R2 or an axisymmetric control. **Maps to:** P2/P4/P8.

### 8.1 Pure coordinate changes are a separate reference problem

Let theta_old = eta - u(s,eta,phi), and define

$$\mathbf X_{new}(s,\eta,\phi)=\mathbf X_{old}(s,\eta-u,\phi).$$

If the straight-field-line angle is theta+lambda, its dimensionless displacement transforms as

$$\lambda_{new}(s,\eta,\phi)=\lambda_{old}(s,\eta-u,\phi)-u.$$

For a zero-lambda base chart, the corresponding flux-scaled quantity is -Phi'(s)u, with the exact 2*pi, NFP, `phipf`, `lamscale` and internal-basis conventions checked against the source. Do not interchange a dimensionless angular displacement, a flux-scaled lambda and an internal solver coefficient.

Use regular gauges. For m>=1 a convenient family is

$$u=\rho^m(1-\rho^2)^2 P(\rho^2)\sin(m\eta-nN_{FP}\phi),\qquad \rho=\sqrt{s},$$

with cosine partners where LASYM is exercised. Remove redundant rigid relabellings or fix them explicitly. Verify the regularity of the full transformed geometry and lambda; the envelope is a sufficient starting design, not a substitute for checking the mapping. Enforce a positive lower bound on 1-u_eta and a fixed orientation throughout the volume.

The previous scan's m=3 envelope s(1-s) produces a nonanalytic axis chart. Retain those runs as historical representation stress tests, not as valid members of the clean regular-gauge comparison. The previously checked m=2 remap does not have that particular defect.

For any parameter a, the physical field derivative at a fixed Cartesian location is

$$\delta\mathbf B_E=\partial_a\mathbf b-(D_x\mathbf B)\partial_a\mathbf X.$$

For a pure gauge, delta B_E, delta p_E and delta s_E must vanish. Measure their convergence before and after numerical projection. Compare integrated observables using the appropriately transformed volume element. A derivative of a moving-grid field is not automatically an Eulerian derivative.

### 8.2 Implicit sensitivity of the equilibrium and its gauge

Write the actual reduced, gauge-defined residual as

$$F_h(z,P(a),\tau)=0,$$

where P contains the full boundary/profile/flux input map and tau is TCON0. After confirming the residual is deterministic and differentiable on the selected branch,

$$F_z z_a=-F_P P_a,\qquad F_z z_\tau=-F_\tau.$$

For an objective Q,

$$F_z^T\lambda=Q_z^T,\qquad
\frac{dQ}{da}=Q_a-\lambda^T F_P P_a,\qquad
\frac{dQ}{d\tau}=Q_\tau-\lambda^T F_\tau.$$

Here Q_a means the explicit derivative at fixed z, including direct dependence through P(a); it does not include z_a. The last derivative is a useful new experiment. In a unique smooth physical branch, an auxiliary coordinate choice should not change continuum gauge-invariant observables. Its finite-resolution response and its convergence can quantify how much a numerical gauge influences those observables. A nonzero finite-h response is not automatically a coding error; a zero response obtained by fixing the objective as an input is not evidence of recovery. At tau=0, check rank rather than assuming an invertible derivative.

The state space must be the space actually evolved by the root. Specify constrained/frozen m=1 combinations, boundary DOFs, axis regularity, lambda gauges and source-dependent particular solutions. A parameter-dependent gauge needs its derivative included. Do not invert a redundant full coordinate system, add an arbitrary diagonal regularizer and call that result the derivative of the original physical problem.

For a low-resolution diagnostic, compute scaled singular vectors or rank-revealing factorizations on the admissible state space. Classify each small-residual direction by its Eulerian physical field/pressure/surface response and its overlap with known gauge tangents. A raw singular value alone depends on coordinate units, residual normalization and preconditioning; it is not a stability result. For nonnormal operators distinguish left and right near-null vectors.

Reuse SOLVAX factorizations, transpose solves and true-residual checks. Test dense independent solves at small size and matrix-free solves at larger size. Do not form a large dense Jacobian merely to produce a condition-number figure.

### 8.3 Three derivative comparisons, all necessary

1. **Discrete definition:** JVP/VJP dot products, linear residuals, and frozen-path finite differences of the same gauge-defined residual.
2. **Independent reconvergence:** physical observables from perturbed inputs with branch tracking and separately converged roots.
3. **Analytical-family limit:** complete-family derivatives compared with the explicit reference and refined in representation and measurement resolution.

VMEX documents differences between frozen-path derivatives and cold re-solves [R04]. Neither path alone establishes the continuum response. For finite differences, use several step sizes, both signs, and a resolved interval between truncation and root/roundoff error. Save the actual perturbed equilibria and branch diagnostics, not only a best relative-error number.

Use a nonzero derivative, a null derivative and a vector-output response. Suggested initial nonzero-gradient target is 1e-3 relative on a resolved scalar, then tighten toward 1e-4 where practical. For zero references use absolute derivatives normalized by a declared characteristic parameter/field scale. Require a convergence trend, not a division by zero or selection of one favorable finite-difference step.

Probe `grad`, JVP, VJP, batching and higher derivatives separately. `custom_vjp` is not directly forward-differentiable [R15]. A reverse rule does not establish that callbacks, linear solves and root logic support Hessians. Where the public API is reverse-only, a residual-based tangent solve is a legitimate separately labelled implementation. Do not call it a successful public `jax.jvp` test.

### 8.4 Full analytical input maps

For the integer/stretch family use a,b,c,delta with

$$u_a=-\frac{a^2-b^2}{4c^2},\quad
H_a=\frac{a^2+b^2}{2}-\frac{(a^2-b^2)^2}{8c^2},\quad
|u_a|+\sqrt{\delta}<\frac12.$$

The pressure surfaces and scalars are

$$p(s)=\frac{B_*^2}{\mu_0}2c^2\delta(1-s),\quad
\Phi_{edge}=B_*L_*^2\pi abc\delta,\quad
V=L_*^3 2\pi^2abc\delta,$$
$$\beta_V=\frac{2c^2\delta}{H_a+c^2\delta}.$$

Changing c changes the pressure-surface center u_a; it is not just multiplication of the old Z coefficients. Use the existing explicit field and geometry from `analytic.py`, retain their independent checks and differentiate the entire input construction. For a current-prescribed run, differentiate the Ampere current profile and its conversion to the VMEX derivative-profile convention too.

For the original a=sqrt(1+epsilon), b=sqrt(1-epsilon), c=1 family, with D=1-epsilon**2/2+delta,

$$\partial_\epsilon\beta_V=2\delta\epsilon/D^2,\qquad
\partial_\delta\beta_V=2(1-\epsilon^2/2)/D^2.$$

At a fixed Cartesian point in the common interior, partial_delta B=0: delta selects the outer pressure surface, not a different local magnetic field. This is the principal cancellation test. At fixed computational coordinates include the motion of the physical sample point.

For the sheared family let Q(u,a) be toroidal flux with u=k**2, and

$$Q(u,a)=s Q_b(a),\qquad
\frac{\partial u}{\partial a}=\frac{s\,dQ_b/da-Q_a}{Q_u}.$$

Use distinct names such as `du_da` for this derivative and `label_center` for the integer family's center. The u=k**2 variable avoids a removable on-axis singularity in a k-based inversion. The physical-angle root satisfies G(t,a)=0 and t_a=-G_a/G_t on a chosen regular branch. Record lower bounds on the relevant denominators.

Differentiate boundary samples, quadratures, flux inversion, current and pressure profiles, and the coefficient fits. Freeze mode sets, quadrature plans and fit bases during a derivative test. Differentiate the fixed linear least-squares fit analytically or by a verified solve rule; do not differentiate an adaptive change of polynomial degree or rank as though it were smooth. Verify physical profiles on held-out nodes, not just coefficient derivatives.

For a lambda change at fixed epsilon,S,delta and physical scales, the sheared family obeys

$$\partial_\lambda\iota(s)=0,\quad
\partial_\lambda\Phi_{edge}=-\Phi_{edge}/\lambda,\quad
\partial_\lambda V=-V/\lambda,\quad
\partial_\lambda p(s)=-2p(s)/\lambda.$$

Measure the transform cancellation with **prescribed current**, updating that current consistently. Prescribing iota and observing it unchanged is not the desired test.

**Exit:** nonzero, pure-gauge and exact-family null responses are measured on appropriate controls; tangent/adjoint certificates and finite-difference intervals are saved; effects of gauge choice, TCON0 and resolution are separated. An unresolved integer-3D branch must not block an independently successful axisymmetric response result.

## 9. R4: recovery breadth and independent codes

**Depends on:** R1 for acceptance; R2/R3 supply reusable methods. **Maps to:** P1-P4.

Use representative combinations rather than an exhaustive product of every grid, start, hardware and flag. The required physical coverage remains:

| Geometry | Symmetry setting | Closures | Required evidence |
|---|---|---|---|
| Axisymmetric integer control | LASYM=F and T, same physical case | iota and current | Native B/J/force, transform prediction, derivatives, restart |
| A distinct symmetric Solov'ev parameter set | F and selected T repeat | both | Independent Grad-Shafranov recovery, not a duplicate physical case |
| Genuinely up-down-asymmetric Solov'ev | T | both | Full-basis volume fields and response after the native path is supported |
| Integer 3-D, including stretch | F and T controls | both | Constraint/gauge diagnosis and exact-family response |
| Sheared axisymmetric limit and sheared A | F and selected T | both | Flux inversion, current/transform prediction and lambda null |
| Rephased/translated exact 3-D field | T | selected both | Covariance with nonzero asymmetric coefficients, not new physical asymmetry |
| Sheared B/C | selected symmetry/closure | stress subset | Representation and chart conditioning, not the first recovery gate |
| Transverse symmetry-broken 3-D perturbation | T | physical choice stated | Independently converged numerical reference, not an exact-family claim |

For LASYM, trace all sine/cosine partners through geometry, Clebsch field, inversion and derivatives. Merely removing the current guard is not an implementation. Start with a same-state live surface/WOUT check, then exact-projection full-volume checks, then nonlinear recovery. Keep fitted-state reconstruction as an independently assessed alternative, not a hidden substitute.

For all closures, pressure, current and transform profiles refer to the same normalized toroidal flux. VMEX's polynomial current input represents the prescribed derivative-profile shape with total-current normalization; do not copy enclosed-current coefficients into it. Test an independent loop integral and an independent field-line/flux transform measurement.

### Cross-code order

**VMEC2000 and VMEC++.** Use the same finite boundary/profile/flux problem to check shared discretization and implementation. VMEC++'s corrected stateless operator is particularly useful for the residual study. Pin a revision that includes the relevant policy fix, and test its semantics. Native iteration parity and stateless root evaluation are distinct evidence. Educational VMEC can expose intermediate constraint and NESTOR stages without requiring another production solver [R16].

**DESC.** Continue the current native comparison with independently converged integration and matched inputs. Bound the remapped M8 solve; inspect root/stationarity and representation before simply increasing its iteration cap. Test current closure as well as imposed transform. Its Fourier-Zernike representation and direct-force approach provide a genuinely different radial discretization [R08]. Use perturbation/continuation as a method, not as proof of a more accurate solution [R09].

**GVEC.** Pin and attempt a small axisymmetric and sheared-A or mild 3-D case when the minimal adapter is justified. Its B-spline radial basis and alternative coordinate mappings provide useful independent controls [R10]. Do not write a universal converter or block the core study on a difficult build. Validate geometry, orientation, flux and profile conventions before comparing output.

Use two comparison lanes: (i) the same finite represented problem, to isolate solver differences; (ii) separately refined approximations to the exact continuum problem, to measure convergence to the reference. Code-specific boundary fits are nonzero forcing errors in lane (ii), not just labels on equivalent input.

SPEC or a stepped-pressure calculation is optional for a later topology question. It does not solve the same smooth nested-pressure discretization at finite interfaces. Do not score it as an interchangeable exact-current reference without a defined limiting experiment. An independent axisymmetric Grad-Shafranov code is similarly useful only after mapping toroidal-flux profiles to its poloidal-flux conventions.

**Exit:** the mandatory mild cases have either resolved recovery evidence or a scoped, reproduced limitation; comparator statements are based on equal physical problems/accuracy, not equal mode counts or a successful flag alone.

## 10. R5: diagnostics, adjacent libraries and a separate mirror control

**Depends on:** exact projected geometry can start immediately after R1; use solved equilibria only after their own acceptance. **Maps to:** P5/P9.

The analytical field is available even when a nonlinear solve is difficult. It can independently test field/geometry consumers now, with `analytic_projection` or `analytic_reference_sampled` labels. This avoids making the entire all-module program wait for a single recovery problem.

| Capability | Fixture and check | Important limit |
|---|---|---|
| SOLVAX | Small dense versus structured tangent/transpose solves, factor reuse, true residuals and derivative residuals | Testing imported operators does not validate every SOLVAX algorithm |
| Boozer / booz_xform_jax | Reconstruct physical B, currents/covariants and transform; compare an independent transform at matched gauges; F/T and NFP controls | Rational-iota gauge nonuniqueness is not a physical field error; avoid arbitrary small-denominator division |
| Field derivatives | Exact B and spatial jets, tensor ordering, units, knots/axis, live versus WOUT versus fitted routes | Smooth analytical fields do not make a piecewise interpolant arbitrarily differentiable |
| Bounce/action | Independent root-bracketed wells, endpoint-regularized quadrature, action derivatives, passing/trapped limits | Integer closed lines are not an ergodic surface sampler; well topology changes can destroy ordinary derivatives |
| Mercier / magnetic well / ballooning | Geometry identities, independent formulas or independently refined eigenproblems, sign conventions | Exact equilibrium does not supply exact stability or prove stability |
| NEO_JAX | Geometry and ripple integrals against independent integration on suitable sheared surfaces | No assumed zero ripple; distinguish epsilon_eff from its powers and rational-surface assumptions |
| ESSOS / tracing | B handoff, full-orbit energy, appropriate canonical momentum in an axisymmetric control, timestep and interpolation refinement | Full orbits do not conserve magnetic moment exactly; LASYM support must be checked, not bypassed |
| DKX | Geometry/current-sign normalization, a prescribed kinetic case and matched independent solver if available | Total MHD current is not automatically bootstrap current; ambipolar roots require kinetics |
| GKX | Metric, curvature and drift tensors; optional isolated eigenpair and left/right response | Exact MHD does not imply a known turbulent heat flux or growth rate |
| pyQSC_JAX draft | Compatible axis/field jets and radius-order study on an appropriate case | A restricted QS ansatz need not represent a generic exact non-QS equilibrium |
| Open mirror | Harmonic vacuum potential, divergence/curl identities, open flux surfaces, optional paraxial control | Closed toroidal references do not validate open topology or finite-beta mirror closure |

A simple independent mirror/vacuum field is generated by

$$\Phi=B_0z+a[z^3/3-z(x^2+y^2)/2],\qquad \mathbf B=\nabla\Phi,$$

so

$$B_x=-azx,\quad B_y=-azy,\quad B_z=B_0+a[z^2-(x^2+y^2)/2].$$

Its scalar potential is harmonic. Choose a domain with nonzero axial field and verify the open-surface construction and boundaries separately. It is not a finite-pressure exact mirror equilibrium. Use it to exercise the mirror field/coordinate lane without conflating models.

Do not require every optional package to be installed simultaneously. Record unavailable and not-applicable states explicitly, with a precise reason and next supported test. Review any source-level lead, such as particle-phase handling, with a minimal reproducer before calling it a confirmed defect.

**Exit:** each relevant reachable module has a measured scoped test or a justified explicit status. No broad claim that all physics in a downstream code is analytically verified.

## 11. R6: free boundary, exterior fields and strict coupled derivatives

**Depends on:** R1; R3 for physical derivative claims. Exact operator tests may run independently. **Maps to:** P6/P9.

Retain three evidence levels:

1. **Exact operator tests on prescribed surfaces:** harmonic potentials, independent coil kernels, NESTOR/MGRID and virtual-casing identities.
2. **Independent numerical coupled roots:** established coil/MGRID equilibria, including axisymmetric and 3-D, F/T controls and meaningful asymmetric cases where supported.
3. **Analytical interior with fitted external sources:** an approximate matching and coupled-equilibrium experiment with separately converged source and plasma errors.

For operator tests, use harmonic polynomials, simple coil/loop fields with independent high-accuracy quadrature, and layer-potential identities. Compare NESTOR and an independent high-order singular quadrature; use a target-distance ladder approaching the surface. Malhotra et al. and Toler et al. give directly relevant singular/axisymmetric methods [R11,R12]. Pin both virtual-casing implementations. Fix discrete quadrature plans during differentiation and refine them independently afterward.

A toroidal vacuum domain requires appropriate circulation/flux data in addition to local normal-field data. Neumann compatibility and topological nullspaces must be enforced. The trivial toroidal vacuum field does not determine a unique plasma boundary and is not a good isolated test of an invertible free-boundary response.

For coupled matching, check

$$\mathbf n\cdot\mathbf B_{in}=\mathbf n\cdot\mathbf B_{out}=0,\qquad
p_{in}+B_{in}^2/(2\mu_0)=p_{out}+B_{out}^2/(2\mu_0),$$

with stated exterior pressure and any surface-current model. A tangential jump corresponds to

$$\mathbf K=\mathbf n\times(\mathbf B_{out}-\mathbf B_{in})/\mu_0.$$

Do not claim no surface current without checking this jump. Interior zero edge pressure alone does not construct an admissible exterior. Normal-field coil fitting alone is not a full stress/field match.

The coupled implicit derivative must include the plasma, vacuum unknowns, boundary motion, circulation constraints and coil/MGRID parameters actually varied. Verify the finite Newton anchor, strict failure statuses and true transpose residual on the same state used to evaluate the objective. Source contains anchoring functionality; stale older documentation is not the authority. Conversely, the existence of an anchor function does not certify all fallback paths. Exclude best-effort derivatives from accepted results.

Use coil-current amplitude and a smooth coil displacement as first parameters. Compare central differences of independently anchored coupled roots. Test a simultaneous rigid displacement of coils, domain and plasma as covariance; moving only the plasma while holding coils fixed is a physical perturbation, not a null test. Independently refine MGRID interpolation where used; direct coils and a gridded representation are different numerical paths.

**Exit:** one exact vacuum/operator fixture in each useful symmetry category, a bounded coupled-root benchmark with derivative certificates, and a separately labelled analytical-interior fitting outcome. A failure of exterior realization can be scientifically informative; never invent an exact free-boundary solution to fill a matrix cell.

## 12. R7: polishing with a correct derivative of its own problem

**Depends on:** R1 and the intended source/operator contract. **Maps to:** P7.

Compare pre/post polishing on the same continuous representation and held-out integration grid. A change of interpolation order or export mesh must be a separately controlled operation. Record physical field/current error, dimensional and normalized force, boundary/profile changes, coordinate regularity, residual norm and stationarity.

If polishing minimizes

$$\mathcal L(z,a)=\tfrac12 r(z,a)^T W r(z,a),$$

the implicit solution is a root of its stationarity equation, not generally r=0. For fixed W,

$$\nabla_z\mathcal L=J_r^TWr=0,\qquad
H=J_r^TWJ_r+\sum_i(Wr)_i\nabla_z^2r_i.$$

At nonzero residual, dropping the second term changes the response. State-dependent weights add further derivatives. Test the exact stationary derivative where implemented; label a Gauss-Newton response as approximate and quantify the error. Tightening stationarity does not by itself lower physical force below the representation floor.

Make a bounded attempt on one symmetric and one genuinely asymmetric or 3-D case. Use an independent field error to prevent a small training force norm from concealing movement away from the analytical equilibrium. Do not make polishing a way to erase an unaccepted original solve from the record.

**Exit:** a measured benefit, a demonstrated limitation, or a scoped failure with reproducible evidence and correct derivative semantics.

## 13. R8: optimization and extensions of the analytical work

**Depends on:** direct-reference optimization can start after reference/map validation; VMEX-mediated physical optimization requires R3/R4. **Maps to:** P8.

### 13.1 Coordinate optimization, with unchanged physical equilibrium

The first inexpensive optimization should seek a better regular chart, not a different equilibrium. Use a few regular gauge coefficients and minimize a declared spectral objective, for example a derivative-weighted high-mode tail of R/Z/lambda, normalized by a minor-size scale. Constrain map invertibility and independent held-out Eulerian B/J errors. Use a separate validation grid and show projection error, spectrum, solver effort and physical response before/after.

Hirshman-Breslau provides an explicit spectral-condensation reference [R06]. The action-based nested-coordinate work of Tecchiolli et al. provides a different initialization/regularity strategy for strongly shaped tori [R07]. The first application here should be a small adapter or initialization experiment, not importing an entire new solver architecture. A nicer spectrum alone is not a physical improvement.

### 13.2 Optimization within the exact physical families

Use direct analytical fields and differentiated quadratures to optimize the low-dimensional integer/stretch and sheared families. Repeated VMEX equilibrium solves are unnecessary for the reference optimization. VMEX should reproduce selected optimized configurations and responses independently.

Fix enough scales to avoid trivial improvements: zero edge pressure; stated mean field/flux and volume or size; limits on aspect ratio, elongation, current, Jacobian conditioning and the analytical-domain margin. Maximize beta only with these constraints and an explicit physical interpretation. Adding a constant pressure offset or unlimited elongation is not a useful optimization result.

Choose one primary tradeoff, such as beta versus current concentration and geometric distortion, or magnetic-well/field-variation metrics at fixed transform family. Use a smooth objective and independently evaluated final constraints. Do not assume these generic equilibria are quasisymmetric, omnigenous or stable.

Record all starts in the small parameter box, active constraints, the direct-reference derivative check, objective history, unsuccessful candidates and an independently refined final result. Repeat with a finer quadrature and nearby starts. A flat or unfavorable tradeoff is an acceptable resolved outcome.

### 13.3 Tangent versus transverse physical continuation

Write

$$P(a,\eta)=P_{exact}(a)+\Delta P(\eta),$$

where eta represents a few genuine boundary/profile changes, including controlled LASYM modes. Exact-family tangent directions provide the analytical reference; transverse directions require independently converged numerical evidence.

A useful physical diagnostic follows from

$$\mathbf J=\sigma\mathbf B+\frac{\mathbf B\times\nabla p}{B^2},\qquad
\mathbf B\cdot\nabla\sigma=-\nabla\cdot\left(\frac{\mathbf B\times\nabla p}{B^2}\right)=S.$$

For a closed field line, a necessary compatibility condition is

$$\oint S\,dl/|B|=0.$$

Measure this and its controlled parameter response on the rational-transform family. Compare with current convergence, scaled residual rank and physical sensitivity. Along a sheared family examine resolved rational surfaces and the corresponding Fourier convention; denominators involve m*iota-n*NFP when the phase is m*theta-n*NFP*phi. Avoid assuming that an observed small divisor proves a singular solution or that finitely many compatibility conditions prove existence.

Use pseudo-arclength continuation only if an actual branch fold requires it. Track the branch, current peaks, surface regularity and held-out force; do not let the optimizer jump among unlabelled roots. Start with a few modes and a trust region. Preserve exact-family tangent controls throughout.

### 13.4 General boundary and coil optimization last

Only after physical derivatives are certified should general boundary or coupled coil optimization be used to claim improvement. Compare final points at higher independent resolution, check gradients again and report error bars from representation/measurement refinement. An optimization that exploits field interpolation, an unresolved high-mode tail, a moving acceptance region, or a best-effort adjoint has not improved the physical problem.

For free-boundary source optimization, retain engineering regularization and the full matching budget from R6. Do not confuse the ability to minimize a normal-field residual with a proof that the desired free-boundary equilibrium has been realized.

**Exit:** one completed direct exact-family optimization, one controlled coordinate-optimization outcome, and a bounded tangent/transverse study. Broad physical optimization is either independently verified or explicitly left blocked by a named missing certificate.

## 14. Minimal code changes and script organization

Keep useful current scripts. Do not delete measured experiments simply because there are many filenames. Consolidate repeated operations only when it reduces ambiguity and preserves reproduce commands.

The highest-priority edits are to `plot_tcon_ladder.py`, `run_vmex.py`, `native_samples.py`, `score_samples.py` and `run_desc_coordinate.py`. Move import-time argument parsing out of reusable projection modules. Extend the existing tests or add one small harness test file. One shared adapter may expose native geometry/field samples, effective run metadata and verified parameter maps. It must not mirror all upstream APIs.

Suggested new drivers, only when the relevant phase starts:

| Driver or extension | Purpose |
|---|---|
| `verify_measurement.py` | Volume/shift/knot/inversion/scorer convergence on saved states |
| `probe_residual_contract.py` | Call-history independence, force-stage decomposition and tangent/transpose tests |
| `run_family_response.py` | Complete analytical input maps and nonzero/null/constraint responses |
| Extend `run_vmex.py` | Explicit source/config/run IDs, bounded continuation and two closures |
| Extend `run_desc_coordinate.py` | Independent labels, CPU/GPU selection, correct source metadata and current closure |
| `run_vacuum_checks.py` | Harmonic/operator tests before coupled free boundary |
| `optimize_family.py` | Low-dimensional direct-reference optimization and held-out verification |

These are proposed filenames, not claims that executable implementations already exist. Do not add empty stubs, fabricated records or a generic dispatch framework. Keep ordinary readable functions; line-count minimization must not remove validation, units, comments or clear exceptions. Avoid dense one-liners that conceal numerical choices.

For an upstream VMEX or adjacent-library defect, prepare one narrow branch and PR with a minimal reproducer, derivation where needed, tests and before/after measurements. Do not merge it or push to upstream main. Benchmark the historical baseline and proposed fix separately. Do not use the benchmark repository to accumulate an unreviewed fork of the solver.

## 15. Figures, performance and the README

Generate every figure from saved arrays and read-only summaries. Keep axes, units, physical region, representation, source pin and evidence class visible. Do not interpolate through an unrun cell or hide an iteration-capped point inside a line of accepted states.

Prioritize these figure groups:

| ID | Data and question |
|---|---|
| V2-F01 | Fixed-state norm versus quadrature size and shift, including legacy96: are the measurements converged? |
| V2-F02 | Native forward-chart versus inverse-chart field/current/force and coordinate backward error: where does evaluation error enter? |
| V2-F03 | Completed TCON radial/angular/tolerance ladders, projection and solved states separate: which trend survives measurement refinement? |
| V2-F04 | Force-stage, mode and radial decomposition, with pure-gauge and flux-surface displacement separated |
| V2-F05 | Scaled operator near-null directions classified by Eulerian physical response, plus call-history tests |
| V2-F06 | Nonzero and null exact-family derivatives, TCON sensitivity and finite-difference step intervals |
| V2-F07 | Axisymmetric/LASYM/sheared-A recovery and current-predicted iota; input and boundary-fit errors shown |
| V2-F08 | Matched finite-problem and exact-limit comparisons for VMEX/VMEC++/DESC and optional GVEC |
| V2-F09 | Diagnostics, vacuum target-distance tests, strict free-boundary responses and bounded polishing |
| V2-F10 | Direct exact-family tradeoff, chart optimization and tangent/transverse continuation |
| V2-F11 | End-to-end time to physical/derivative accuracy, including scoring, plus executed capability matrix |

Retain the old figures as historical evidence; supersede their claims only where new data warrant it. Avoid making the README a transcript of every diagnostic. Its lead should state the physical cases actually recovered, the most important limitation, the derivative status and the exact commands needed to reproduce the main result. Put detailed chronology in the logbook and link raw data.

Measure compilation, solve, root anchor, sampling, scoring, gradients, export and verification separately, plus the complete end-to-end cost. Synchronize device work. Distinguish cold process, warm same input, warm new parameters and cache state. Use repeated measurements with spread. Do not combine CPU/GPU, package changes and algorithm changes into a claimed speedup without controls. Peak host RSS is not GPU memory, and a process high-water mark is not automatically the memory used by its last stage.

The current scoring cost exceeds the solve cost on the small TCON cases. Reducing redundant inversions and compilation is therefore a more justified first performance task than changing the equilibrium algorithm to save a few iterations. All speed claims remain conditional on equal physical and derivative error.

## 16. Acceptance states and completion

Use a small explicit state vocabulary: `planned`, `implemented_not_run`, `diagnostic`, `passed`, `failed`, `blocked`, `unavailable`, `not_applicable`. Record the evidence class separately: exact reference, projection, analytical recovery, discrete consistency, independent numerical reference, integration, or exploratory optimization.

For an accepted analytical recovery require valid input/domain, resolved representation, successful intended nonlinear/root status, converged physical measurement, and the stated physical thresholds. For a derivative additionally require the correct parameter/gauge map, a certified linear solve, branch control and an independent comparison. A root is not accepted just because its score is small, and a small training score is not a certificate.

For the next handoff, success means R0/R1 are implemented, the missing R2 cell and residual-contract test have a recorded outcome, and at least one independently verified physical derivative/control path has progressed. The final all-capability program remains larger: meaningful F/T, both closures, 3-D and axisymmetry, vacuum/coupled free boundary, diagnostics, bounded polishing and optimization, plus scoped downstream/mirror tests.

A negative outcome can close a scientific experiment when its assumptions, convergence and failure mechanism are examined. It cannot label unsupported functionality as passed. Optional dependencies may be unavailable without blocking all core evidence. Do not silently convert previously required work into optional work because one implementation was inconvenient; record and justify any scope change.

## 17. Git identity, publication and privacy

Use the existing authenticated `rogeriojorge` account. Verify it before any write. Configure Git author and committer locally using the owner's approved email or the correctly derived account noreply address. Do not change global Git settings. All new commits, pushes, comments and PRs must use that account and must not contain automated-assistant authors or co-author trailers.

Preserve existing merge history. The reviewed merge commit has a GitHub web-flow committer; do not rewrite it to satisfy a prospective identity rule. If exact owner author/committer identity is required for future merges, use the authorized local merge/fast-forward workflow rather than assuming web UI merges have identical committer metadata. Do not merge upstream PRs as part of this assignment.

Inspect the staged diff before publication. Exclude credentials, private paths/data, environments, caches, unreviewed logs and large accidental artifacts. Preserve third-party licenses and scientific attribution. Store representative native states and large arrays as checksummed release assets when appropriate; do not omit the state needed to reproduce a reported diagnostic. CI should have read-only permissions and should not generate commits/comments.

## 18. Continuing logbook and exact resume state

After each meaningful block append an entry containing:

```text
Date/time and timezone:
Phase / task IDs:
Benchmark and dependency commits, dirty patches:
Question and expected discriminating observation:
Files changed and why:
Commands and effective configuration:
Run IDs, parent states, artifact paths and hashes:
Results, uncertainties and failed attempts:
Tests actually executed and tests skipped:
Interpretation supported; alternatives not excluded:
Branches, commits and PR state:
Exact next action and prerequisites:
```

Keep a compact current table above the append-only entries:

| Phase | State at this review | First unresolved action |
|---|---|---|
| R0 | Planned repairs; source defects identified | Preserve raw bytes and make report generation read-only |
| R1 | Legacy scorer implemented, not independently certified | Fixed-state grid/shift/volume/inversion checks |
| R2 | Nine historical ladder states; one NS129 cell missing | Finish projected zero-strength cell, then stateless residual tests |
| R3 | Explicit-reference derivatives only | Complete input map on an axisymmetric control; regular-gauge null |
| R4 | Partial recovery and initial DESC comparison | Sheared A and true LASYM volume path, matched controls |
| R5 | Broad integrations planned | Small exact-geometry consumers without waiting for all solves |
| R6 | Planned | Harmonic/vacuum identities, then a strict anchored coupled case |
| R7 | Planned | One same-representation bounded polishing experiment |
| R8 | Planned | Small direct-reference chart/family optimization, with held-out checks |

### Revision-2 review entry, 2026-09-24

The reviewed public benchmark commit is `575f13f67b346118c3d7f05cc60cc6e29131359a`; PR #1 is merged. Historical solver evidence remains pinned to VMEX `b5f5267...`. The review identified report mutation, sparse measurement, label/metadata and regularity issues in the harness, and a relevant external residual-policy precedent in VMEC++. These findings motivate the reordered phases above, not a declaration that the entire physical dataset or solver is incorrect.

Three standalone mathematical probes passed: coarse angular aliasing, one-period/full-torus volume normalization, and the leading radial order induced by the old m=3 gauge. No repository solver or benchmark test suite was rerun for this review. This review made no remote Git changes.

**First local action:** preserve the existing plan/logbook and raw TCON records; implement R0 with tests. Keep the original source and existing artifacts intact. Then run R1 on saved projection/solved controls before accepting new physical conclusions. The bounded missing NS129 experiment may run after R0, but it remains diagnostic until R1 is complete.
