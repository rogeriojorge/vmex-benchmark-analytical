> **Active plan.** Adopted 2026-09-24 from the review handoff in [`docs/handoff/review-05e9473/`](docs/handoff/review-05e9473/) (package checksums verified). Earlier plans and complete logbooks are preserved byte-for-byte: [through 575f13f](docs/history/plan_through_575f13f.md), [through d5484d1](docs/history/plan_through_d5484d1.md), [through 05e9473](docs/history/plan_through_05e9473.md). The previous agent prompt is archived at [docs/history/AGENT_PROMPT_through_05e9473.md](docs/history/AGENT_PROMPT_through_05e9473.md). The task table and continuing logbook are at the end of this file.

# VMEX analytical benchmarks: continuation after 05e9473

**Owner:** rogeriojorge  
**Repository:** https://github.com/rogeriojorge/vmex-benchmark-analytical  
**Reviewed benchmark:** `05e947370b9d0cebc0e508361e447b03523a331d`  
**Previous review:** `d5484d1e7a15c9cf10599c2b0f05bdf69e22e861`  
**Review date:** September 24, 2026, America/Chicago; GitHub receipts include September 25 UTC.

This is the next working contract, not a new project. Preserve the code, exact references, numerical states, amended historical records and the complete prior logbook. Archive the active plan byte-for-byte before adopting this one. Keep one active plan, with links to earlier plans and this review. Source labels S1-S12 and literature labels L1-L14 refer to [REFERENCES.md](docs/handoff/review-05e9473/REFERENCES.md). The concrete review findings are in [REVIEW_FINDINGS.md](docs/handoff/review-05e9473/REVIEW_FINDINGS.md).

The next work must distinguish three questions: what equation was actually solved, what physical field that state represents, and what map was actually differentiated. The new evidence makes this distinction more urgent, not less. Do not remove the coordinate constraint globally, increase failed iteration caps repeatedly, or defer every useful experiment until the worst existing state has an extremely precise force norm.

## 1. Scope and adoption

Compare the working tree and remote with the reviewed commit. Preserve later owner changes. Record a new baseline if main has advanced; do not overwrite the new work with this package. Work on a small continuation branch unless the owner explicitly chooses another branch policy. The last consolidation left the project on main; this does not authorize deleting branches or rewriting its history.

All new commits, pushes, comments and PRs must use the authenticated `rogeriojorge` account. Check the actual account with `gh api user --jq .login`. Configure author and committer locally, not globally. Do not add assistant authors or co-author trailers. Keep third-party licenses and scientific attribution. Necessary upstream corrections belong in narrow separate PRs; do not merge them or push to upstream default branches. Inspect existing publication scripts before invoking them: a script which pushes main is not appropriate for a new topic branch without a deliberate decision.

The review inspected repository changes, selected implementation bodies, numerical reports, the active logbook, CI, historical VMEX refinement, current upstream context, and the SOLVAX spectral Poisson implementation. It did not execute VMEX or the repository's 55-test suite locally, and is not a complete semantic audit of every VMEX or adjacent-code file. Keep these scope limits explicit. The ten delivered mathematical tests and the one-surface sheared-reference reconstruction did execute; neither is a numerical equilibrium recovery.

The scientific goals remain broad: exact fixed-boundary recovery; fields, currents and derivatives; meaningful symmetry coverage; current and transform closures; file and restart paths; diagnostics and downstream interfaces; exterior operators and coupled free-boundary responses; and verified optimization. A single difficult rational-transform example must not become the whole project.

## 2. Current evidence: retain what is finished

The six new commits contain substantial work. The clean reference import boundary and report-schema repairs are complete for the reported cases. Current reference CI at 05e9473 is successful, including pytest and both reference scripts. The recorded suite has 55 tests. Do not begin by fixing the old CI failure again. [S1, S2, S8]

The missing NS129 integer-3D projected-start TCON0=0 run has now been done. On the identical legacy96 points, its field/current/pressure-gradient/force errors are about `1.0795e-6 / 1.8027e-5 / 3.0036e-6 / 1.9413e-4`; the default-TCON state is much worse on that same grid. The zero-weight solve took 50 iterations rather than 420. This is a promising saved candidate, not yet a resolved volume certificate. Reuse it; do not list the cell as unrun. [S2]

The preconditioned single-grid residual value, JVP and VJP were history independent in the new test. Raw and other reached configurations still need coverage, but repeating the identical passed experiment adds little. Actual native-knot crossings were also located on one NS33 ray to about 3.55e-15 in the coordinate equation; the maximum displacement from reference knots was 0.00359. Bounded streaming/checkpoint mechanics were exercised. Neither result establishes complete volume-quadrature convergence. [S2, S6]

The axisymmetric work now contains genuine equilibrium-response calculations, not only input-map derivatives. A small dense active-system solve verifies the linear response against the structured solver. The nonzero c response is promising; the delta-null response and independently reconverged branch remain unresolved. A tested radial m=1 alignment follows the observed branch response but changes Cartesian B. It is not a physically null gauge. [S2-S4]

The sheared-A code now supplies a nonzero lambda seed. On unchanged geometry and the same 96 physical points it improves B/J/force errors by roughly 190/52/50 compared with lambda=0. From that seed, default TCON0 converges but degrades the analytical physical score. The matched zero-TCON run caps and drifts more. These opposite outcomes from the integer and sheared examples rule out a justified universal recommendation to set TCON0=0. They do not establish that the default is optimal. [S2, S5]

The latest NS65 zero-TCON axisymmetric base and plus/minus states have residuals `1.19e-7`, `8.42e-8`, and `1.21e-7`, rather than the required `1e-11`. Their field finite difference is not a certified equilibrium response. The status amendment is appropriate and must remain linked to the raw report. [S7]

### Working snapshot table

| Item | Snapshot or state |
|---|---|
| Benchmark main | `05e947370b9d0cebc0e508361e447b03523a331d` |
| Historical VMEX | `b5f5267efc0795c4a49a224e321e9b370975c14c` |
| Current VMEX, separate comparison | `926892ab7131a6bc0c5b61218d1f75e7b77bc401` |
| Analytical supplement | `4c0b690ddebdc71811c88223eb9f44a98ab64222` |
| Recorded DESC | `4f48720beac3d4169e9165923d730445118bc2de` |
| Inspected SOLVAX | `2e246a5d6093662f9b5f72c46f995cd7c4bbd479` |
| Optional VMEX PR448 | open draft, head `131580da529a5e7c9f9db31d6fc2361b225e9c24` |
| Reference CI | run `36086434493`, success at 05e9473 |

A new result must identify the imported file path, exact commit, tracked/untracked modifications relevant to execution, package versions, and full input/state/operator fingerprints. A version string alone is insufficient. Do not silently move a historical convergence curve onto current main.

## 3. First priority: observe one actual refinement

The current `_root_and_anchor` observer does not measure the transition its field names imply. It first calls `solve_implicit_with_aux`, whose callback already attempts refinement. It then measures a residual and calls the memoized `_refine_fixed_point` again at the same parameters/configuration. That second call can return the cached result. A zero reported shift therefore does not show that the original refinement took no step. A large measured residual still correctly disqualifies the state. [S3, S9]

Repair this observer before interpreting the proposed root-floor experiment. Start from a saved raw host state or obtain exactly one raw host solve. Run one explicitly identified, uncached refinement from it, with a fresh configuration or isolated process. Do not clear arbitrary global caches in a shared interactive session. Prefer a narrow upstream observation hook or a small version-pinned diagnostic adapter over copying the whole solver.

For each actual attempted correction save: input-state hash, frozen state and DOF-mask hashes, raw and preconditioned residual vectors or reduced block norms, linear right-hand side, original-operator linear residual, correction norm, step length, geometry validity, trial residuals, accepted/rejected decision, and output-state hash. Save the raw state before refinement and the final state before field scoring. Distinguish `host_returned`, `refinement_attempted`, `root_certified`, and `response_certified`.

Regression tests must show that the observation distinguishes a genuinely changed state, an unchanged failed refinement, and a second memo hit. Verify the memo's existing contract: it is keyed by configuration and parameter bytes, not by an arbitrary supplied state. Do not treat it as a general-purpose refiner of any state at the same parameters. Reusing it in that way is a benchmark error even if the production call sequence is valid.

At the benchmark boundary, refuse an accepted derivative when the relevant root certificate fails. The historical source explicitly allows refinement to be an optional improvement, and its status wrapper uses host-solve acceptance. A successful return is not proof that the intended frozen root was reached. Propose an upstream strict-certificate option only after a small reproducer and tests; keep legacy permissive semantics separate rather than silently changing all callers. [S9, L9, L10]

The old zero-TCON script's exact run-time bytes are missing. Preserve the amendment and original hashes; do not synthesize a supposed original script from the corrected version. Future expensive runs must capture script bytes or a committed tree plus the complete relevant patch before execution.

## 4. Resolve the branch-versus-tangent question algebraically

More coefficient fits will not settle the discrepancy. First write down one mathematical operator. Let

\[
 F_h(z,P;g)=0,
\]

where z is the independent state, P the physical input, and g the frozen auxiliary data: constrained combinations, gauge, runtime baselines and any genuinely frozen numerical choices. Preconditioning recomputed from the trial state belongs inside F, not in a falsely frozen list. Identify each item from code.

At the base root define A=F_z and b=F_P P_a. The fixed-operator response satisfies

\[
 A z_a=-b. \tag{1}
\]

For saved branch endpoints, form the actual signed differences

\[
 d_h=(z_+-z_-)/(2h),\qquad q_h=(P_+-P_-)/(2h).
\]

The decisive test is

\[
 e_h=A d_h+F_Pq_h. \tag{2}
\]

Evaluate both endpoints in the same base operator, as well as in each endpoint's own operator. Record the mapping into independent DOFs and the explicit parameter-dependent edge assembly. Do not apply a full-state difference to a reduced Jacobian without this conversion. Use the same parameter step for the tangent comparison or bound the measured input-tangent difference.

If own-operator roots are small but base-operator residuals are not, test whether the observed auxiliary drift explains the discrepancy through F_g g_a. If (2) is small but the state response differs substantially, examine conditioning and null directions in declared physical metrics. If neither is true, investigate implementation, parameter map, differentiation, or root accuracy. These are discriminating outcomes, not assumptions that an m=1 discrepancy is a gauge.

Run this first on the saved default-TCON NS65 delta branch. It is cheaper and more informative than another zero-TCON plus/minus campaign. Preserve pointwise signed data. The existing reconstruction JVP along the measured state FD is useful, but it does not establish (1).

### Units and sampling of responses

For dimensionless c and delta, dB/da has units of tesla, not tesla per metre. Use a declared parameter scale a_star and field scale B_star:

\[
 E_{B,a}=\frac{a_*}{B_*}
 \left[\frac{\sum_jw_j|B_{a,j}-B^{\rm exact}_{a,j}|^2}
 {\sum_jw_j}\right]^{1/2}. \tag{3}
\]

Choose c_star=c0 and delta_star=delta0 for fractional changes, or explicitly retain a fixed absolute convention. Preserve historical unnormalized norms as historical metrics. A norm over 96 entries and a norm over four points are not comparable merely because both are named `l2_over_fixed_scale`. Duplicating points and dividing their weights must leave (3) invariant. A four-point experiment can certify four pointwise observables; it is not a volume response norm. Keep a point-cloud label until the physical quadrature is independently resolved.

Evaluate the dimensionless reference at x/L_star. The current exact-tangent helper omits this conversion and the response scale uses B_star/L_star; both are hidden by current unit scales. Add a nonunit L_star/B_star test. Keep coordinate inversion, tensor ordering, parameter normalization, and quadrature weights identical across derivative methods.

## 5. Bounded root-floor diagnosis, not unlimited iteration

Use the saved NS65 zero-TCON base state, not another complete branch. Validate the actual raw state and perform the corrected single-pass observation. Start with a small active system and a dense QR/SVD reference when feasible; the existing NS17 dense test provides a useful pattern. This is an independent linear algebra control, not an independent magnetic model.

Scale independent state coordinates and residual rows with stated, fixed metrics. Inspect both raw and preconditioned equations. Condition numbers depend on these choices; an unscaled singular value is not a coordinate-invariant physical instability. Separate structurally absent DOFs from small singular directions.

For a candidate Newton step d and residual r, compute the true linear defect e=A d+r. The derivative of the squared residual merit is

\[
 r^T A d=-\|r\|^2+r^Te. \tag{4}
\]

Thus a sufficiently accurate step should be a descent direction for that same merit. If it is not, the original linear solve or operator consistency is wrong. If it is, try a bounded backtracking or trust-region control and record why every trial is rejected. Do not diagnose nonlinear failure from the norm of a preconditioned linear residual alone. The inexact-Newton forcing literature is directly applicable to this distinction. [L8, L9]

Check whether the residual has a significant component outside the numerical range of A, using a small SVD or a controlled matrix-free estimate. For a near-null direction, evaluate physical Cartesian changes in B, pressure, s and the boundary. A small A-vector product does not make the direction an admissible gauge. Do not discard inconvenient physical equations with a pseudoinverse tolerance chosen to obtain convergence.

A regular angular relabelling can be tested explicitly. With theta_old=theta_new-u,

\[
 \delta\mathbf x=-\mathbf x_\theta u,\qquad
 \delta\lambda=-(1+\lambda_\theta)u. \tag{5}
\]

A convenient regular family uses u=rho^m(1-s) times a sine/cosine mode, for m>=1 and s=rho^2. This preserves the outer boundary and has appropriate polar regularity. Apply the full transformation, not only one geometry coefficient. At fixed Cartesian points, B, pressure and normalized toroidal flux must remain unchanged up to the measured projection/evaluation error. The previously fitted radial-normalized-flux motion fails this requirement and must not be reused as a gauge fix.

A bounded diagnostic closes when it identifies an inaccurate linear solve, an operator-context mismatch, a geometry rejection, a residual range obstruction, a certified correction, or an unresolved mechanism with enough saved data for a narrow follow-up. Do not repeat more than one alternative globalization and one independently checked linear-solver control without new evidence. Broad solver redesign is a separate decision.

## 6. Certify useful saved states with complementary physical measurements

Prioritize the integer-3D NS129 projection, zero-TCON root and default-TCON root on matched domains. Retain legacy96 as a reproducible screening cloud. Use independent angular shifts/refinement, native-cell-aware radial quadrature, explicit near-axis/edge probes, and recorded coordinate inversion residuals. The bad NS33 state can remain a diagnostic rather than consuming the precision budget for a successful result.

There are two legitimate domain choices. Native-cell quadrature maps the actual numerical chart and tests the exact field at those points; its domain is the numerical plasma. Common/reference-domain quadrature compares the same physical domain across solvers, but must split at actual native-knot crossings along each integration curve. A one-ray crossing demonstration is not a full-volume construction. Report boundary mismatch separately and never silently drop invalid samples.

The final recovery gates remain relative B <=1e-5, relative J <=1e-3, and pressure-normalized force <=1e-3, supplemented by pressure/flux/geometry checks. Estimate quadrature and reference uncertainty independently; for acceptance aim for uncertainty below one tenth of the relevant gate and demonstrate a stable finest-grid comparison. Diagnostic screening far above the gates can use a looser stated uncertainty and still decisively classify failure. Do not force every coarse quadrature rule to satisfy the final threshold.

Keep reductions streamed and cache reusable geometric/field kernels. Commit completed grid records immediately, with hashes and all sufficient statistics. Retain a bounded audit sample, not every large tensor. A stopped child process has its own receipt; never leave a report appearing active after termination.

### A weak force-balance test that avoids numerical curl

Add an independent stress test beside pointwise J x B - grad p. Define

\[
 T=\frac{\mathbf B\mathbf B}{\mu_0}
 -\left(p+\frac{B^2}{2\mu_0}\right)I.
\]

Direct differentiation gives

\[
 \nabla\cdot T=\mathbf J\times\mathbf B-\nabla p
 +\mathbf B\,\nabla\cdot\mathbf B/\mu_0.
\]

For a smooth test vector v vanishing on the boundary,

\[
 \int_\Omega v\cdot F\,dV
 =-\int_\Omega T:\nabla v\,dV
 -\frac1{\mu_0}\int_\Omega(v\cdot B)\nabla\cdot B\,dV. \tag{6}
\]

For a certified divergence-free Clebsch representation, the last term vanishes. Otherwise retain it; do not declare solenoidality from a function name. If v is not boundary-zero, retain the surface term. On piecewise-smooth fields, account for any internal trace jumps; a continuous field stress has no internal surface-jump contribution.

Use compactly supported physical-space test functions or regular functions of an independently known reference chart. Increase their spatial range and mode content under independent quadrature. This tests B and p without building J from a second geometry derivative. Compare its result with the strong-force residual and the existing algebraic dJ/dB/pressure decomposition. A finite set of weak moments does not certify a pointwise force norm; it is a complementary diagnostic and a useful cross-code observable.

The delivered polynomial example tests (6) exactly to numerical precision and detects a pressure perturbation. It is not a toroidal solver benchmark. Port the operation into the shared scorer without adding a second equilibrium solver.

## 7. Recover lambda from the full magnetic field

The new sheared-A field-line solve is useful and should be retained as an independent method. Its equation fixes straightness of the field-line direction. It is not, by itself, an independent check of the magnetic amplitude, especially on rational surfaces where nonconstant homogeneous solutions can exist. Instead of spending the next block tuning that iterative solve, construct a direct full-flux reference. This is an application of the standard Clebsch/potential representation, not a claimed new equilibrium theorem. [S5, S12, L11]

Let x(s,theta,phi) be the known exact chart, with physical toroidal angle phi, and let

\[
 \mathcal J=x_s\cdot(x_\theta\times x_\phi),\quad
 U=\mathcal J B^\theta,\quad V=\mathcal J B^\phi.
\]

Keep the SIGNED Jacobian. A divergence-free field tangent to s surfaces satisfies

\[
 \partial_\theta U+\partial_\phi V=0.
\]

Let P and T be the signed toroidal and poloidal flux derivatives in this chart. Independent oriented flux integrals fix them; coordinate-period averages of V and U supply a consistency check, not an opportunity to fit away a wrong input flux. Then

\[
 U=T-P\lambda_\phi,\qquad
 V=P(1+\lambda_\theta),\qquad \iota=T/P. \tag{7}
\]

Consequently g_theta=V/P-1 and g_phi=(T-U)/P are the two derivatives of lambda. Their cross derivatives must agree and their period integrals must vanish. With phase m theta - n NFP phi, the zero-mean Fourier solution is

\[
 \widehat\lambda_{mn}
 =\frac{-i\left(m\widehat g_{\theta,mn}
              -nN_{\rm FP}\widehat g_{\phi,mn}\right)}
             {m^2+(nN_{\rm FP})^2},\qquad (m,n)\ne(0,0). \tag{8}
\]

There is no division by m iota - n NFP. This is the least-squares periodic gradient potential; it exactly reproduces compatible data, but projects inconsistent data. Measure the inconsistency and reject it above the declared input error budget. Do not silently remove a flux-period or solenoidal defect.

As a simple discriminator, with iota=1, NFP=2 and lambda=0.08 sin(2 theta-2 phi), both the nonzero lambda and lambda=0 satisfy the homogeneous straightness equation. Only the correct lambda reproduces both flux densities on a fixed geometry. The delivered test exercises this ambiguity. It is not a claim that the existing sheared-A solution necessarily suffers from it.

### Implementation and gates

Use an odd-grid NumPy FFT reference first, avoiding real even-grid Nyquist ambiguities. Verify its forward/transpose operation and compare it with the existing LSMR solution on the same surfaces. The inspected SOLVAX spectral Poisson routine can supply the JAX path with the correct minus-Laplacian sign. Check the mean before its nullspace projection. No new general linear-solver package is needed.

Construct x_s from the explicit reference with independently converged finite differences or the verified input-map JVP; angular derivatives of cylindrical R/Z are periodic, whereas Cartesian x/y are not periodic over a single field period. Obtain U and V directly from triple products with exact B. Check tangency, signed Jacobian, flux periods, cross-derivative compatibility, and 1+lambda_theta>0.

Reconstruct B from (7) and the coordinate tangents, then compare its vector at the same points with the exact Cartesian field. Check a finer held-out grid. Next project geometry and lambda into VMEX with its actual normalization and axis/edge closures. Measure the increase in B/J error due to that finite projection separately. Do not certify a projected field by the transport residual alone.

At the axis, a whole angular surface collapses. Do not use the arbitrary zero lambda row as proof of radial regularity. Fit the regular mode amplitudes or derive the limiting coordinate gauge on a small-radius ladder. Check smooth radial gauge choice between neighboring surfaces and isolate truncation from axis closure. Make NFP an argument, or explicitly reject any value other than the hard-coded current case.

The supplied `probes/sheared_flux_demo.py` reconstructs B on the s=0.5 sheared-A reference surface at grids 33,65,97 and three radial difference steps. The best sampled B errors are about 1.5e-10, with a minimum straight-angle derivative near 0.767. These are surface reference checks. They do not test current, radial lambda derivatives, axis regularity, or a VMEX solve. The executed reference was extracted from the initially supplied handoff; its file hashes are recorded. Rerun against the current repository reference before using the result as a current-code artifact.

After the direct seed is independently verified, run a short NS17/33 recovery ladder from the same seed family. Start with default TCON0; retain the existing NS17 zero-weight failure as a control. Do not expand into a large TCON sweep until projection, input and root errors are separated. Independently measure iota in current-prescribed runs rather than reading a prescribed transform as a successful output.

## 8. Exact-family response targets and full input changes

The existing references and derivations remain authoritative. Preserve them rather than implementing a second copy. The following relations identify the full parameter paths needed for the next tests.

For the diagonal-stretched integer family, define

\[
 u_a=-\frac{a^2-b^2}{4c^2},\quad
 H_a=\frac{a^2+b^2}{2}-\frac{(a^2-b^2)^2}{8c^2},
 \quad |u_a|+\sqrt\delta<\tfrac12.
\]

The boundary is the pressure surface centered at u_a, not an old boundary with only its height multiplied. In the dimensionless reference,

\[
 p=2c^2(\delta-\psi),\quad
 V=2\pi^2abc\delta,\quad
 \Phi_t=\pi abc\delta,\quad
 \langle B^2\rangle=H_a+c^2\delta,\quad
 \beta_V=\frac{2c^2\delta}{H_a+c^2\delta}. \tag{9}
\]

Here s=psi/delta. For a=b=1, the beta derivatives used in the new response code follow directly. At fixed physical position the field is independent of the outer label delta, so B_delta=0. Changing delta nevertheless changes the boundary, total toroidal flux, pressure profile and enclosed-current input. Omitting any of those produces a different experiment.

In the sheared family, psi=k^2/2 and s=Q(k)/Q(k_b), not k^2/k_b^2. With zero boundary pressure,

\[
 p=\frac{\delta-\psi}{\lambda_{\rm axial}^2},
\]

where the axial parameter is unrelated to the straight-field-line displacement. At fixed normalized flux, varying the axial parameter changes volume, flux and pressure, while the analytical transform is unchanged. A useful null test therefore uses current-prescribed VMEX with the entire current map updated. It is not a prescribed-iota check.

For stable differentiation of the flux inversion use u=k^2. If Q_tilde(u,a)=s Q_edge(a), then

\[
 u_a=\frac{s\,dQ_{edge}/da-\partial_a Q_{tilde}}
                {\partial_u Q_{tilde}}. \tag{10}
\]

Do not introduce a spurious k=0 singularity by differentiating an unsuitable inverse variable. Keep bisection branch choices fixed locally, check physical-angle graph validity and verify the derivative on both sides of quadrant conventions without differentiating a discontinuous angle-wrap operation.

For all families, dimensionful values are x=L_star xbar, B=B_star Bbar, p=B_star^2 pbar/mu0, J=B_star Jbar/(mu0 L_star), and Phi=B_star L_star^2 Phibar. VMEX AC in the selected power-series convention represents the shape of dI/ds, while CURTOR specifies I(1). Verify both independently. Hold profile and Fourier dimensions fixed across a local derivative experiment. Reject rather than truncate a fitted coefficient vector that exceeds an interface's supported length.

### Response acceptance

First certify a nonzero c response on the smallest inexpensive axisymmetric case, then the delta-null response through radial and angular refinement. A complete accepted response needs a small root residual in the declared operator, a certified original linear and transpose residual, a nontrivial finite-difference step window, and agreement with the analytical response within a stated scaled tolerance. Preserve the distinction between a linear path in (state,parameters), reconverged fixed-operator roots, and cold/warm public solves with their own auxiliary state.

A root residual gate is necessary but not always sufficient. For a scalar Q, solve A^T ell=Q_z. Locally the output correction due to the residual is approximately -ell^T r. The centered-FD contamination can be estimated from the two output corrections divided by 2h. Treat this as a linearized estimate, not a rigorous bound without remainder control. Use it to avoid choosing a finite-difference step so small that certified-looking root tolerances still dominate the derivative. [L12]

Use declared engineering tolerances initially: nonzero response relative error 1e-3 with an independently resolved analytical/input map; null response scaled RMS 1e-5 as a research target. These are proposed response gates, not previously achieved results. Tighten or amend them only with recorded physical reasoning before selecting a winning case. Report every refinement level; do not normalize a null by its own numerical value.

## 9. Broader capabilities and code comparisons

The mandatory physical matrix remains: axisymmetric integer/Solov'ev; genuine up-down-asymmetric Solov'ev; integer 3-D; mild sheared-A; selected stretched and rephased exact cases; both NCURR choices; LASYM false/true basis controls; and a separately labelled genuinely symmetry-broken numerical extension. Strongly shaped B/C cases are later stress tests, not prerequisites for the first resolved result.

For LASYM, first check symmetric physics in the full basis, then a genuinely asymmetric reference. Preserve all sine/cosine partners and axis regularity in field conversion and derivatives. A WOUT round trip does not certify native volume curl. Do not remove an unsupported-interface guard without implementing the missing harmonics. If a native pathway remains unavailable, report it and use an independently converged alternative representation under its own label.

Compare VMEX historical/current, DESC and VMEC2000/VMEC++ with the same physical problem and declared boundary fit, profile, flux, domain and metric. A finite fitted boundary differs from the exact boundary; match the same finite boundary for a solver comparison, or separately bound each reference-boundary error for an analytical convergence study. Do not infer solver rank from equal nominal mode counts or FTOL. Add GVEC only after its current source, input map and native evaluation are pinned and checked. [L4-L7]

Use each adjacent code for one justified question:

| Capability | Useful experiment | Required limit |
|---|---|---|
| SOLVAX | Dense/structured tangent and transpose parity; direct periodic lambda potential | True residuals and compatible flux periods; no assumption that every wrapper supports every AD transform |
| booz_xform_jax | Exact-field reconstruction, angle convention, current/transform consistency, LASYM and selected spectrum derivatives | Straight lambda is not yet Boozer; rational gauge and mode origins must be controlled |
| Bounce/NEO_JAX | Independent well endpoints and action quadrature; selected ripple comparison | Exact MHD is not an exact transport coefficient; topology changes may invalidate ordinary derivatives |
| ESSOS/tracing | Exact integer-family field-line trajectory/tangent and closure; later orbit invariants | Field-line time is not toroidal angle or charged-particle time; distinguish full-orbit and guiding-center models |
| GKX/DKX | Exact geometry, metric, curvature and normalization handoffs; one bounded matched kinetic calculation | No claim of exact growth rate, bootstrap current or ambipolar field from an exact MHD solution |
| Near-axis code | Axis jets and radius/order convergence on compatible models | Generic non-QS equilibrium must not be forced into a QS ansatz |
| Mirror lane | Separate vacuum polynomial/open-flux fixture and declared paraxial tests | Toroidal cases do not test open topology or unrestricted high-beta mirror physics |

The exact-flow control can run before nonlinear recovery. For the integer family and its commuting diagonal stretch, d x/dt=B gives the oscillator acceleration with Omega=diag(1,1,2):

\[
 x(t)=\cos(\Omega t)x_0+\Omega^{-1}\sin(\Omega t)B(x_0),
\]

\[
 D_{x_0}x(t)=\cos(\Omega t)+\Omega^{-1}\sin(\Omega t)\nabla B(x_0). \tag{11}
\]

Use the physical scaling of t and any rigid translation consistently. Check flow composition, closure, tangent-map finite differences and volume preservation. These give unusually strong tests of field/derivative interfaces without making up an exact kinetic prediction.

For spatial derivative orders 0-3, state the smoothness of the underlying radial interpolant and of the derived B field. Sampling a third derivative successfully with AD does not establish its global continuity. Test cell interiors, one-sided knot limits, near-axis behavior, and refined native high-order states separately. Do not average a knot singularity away and claim a uniform classical derivative.

## 10. Free boundary and polishing remain distinct experiments

Carry forward the three free-boundary evidence levels: exact vacuum/operator fixtures; independently converged coupled equilibria; and an exact interior with a separately fitted exterior. The last is an approximate coupled reference unless the entire matching problem is independently known. Zero edge pressure alone does not supply exterior coils or a unique boundary.

For exact operator fixtures, use source-free harmonic fields on the relevant vacuum domain, with toroidal circulation specified separately from a single-valued potential. Test MGRID interpolation, NESTOR normal-field cancellation, source-field signs, virtual-casing target-distance refinement and fixed-plan spatial/parameter derivatives. A pure toroidal vacuum field is useful for reconstruction but has boundary degeneracy; do not use it as evidence for an invertible free-boundary adjoint. [L13]

For a coupled root, use the actual simultaneous plasma/vacuum/edge residual, with boundary normal-field and total-pressure stress conditions. Separate optional surface-current assumptions from imposed tangential continuity. Sample the same anchored state for the objective and its linearization. Apply the single-pass root observation and original transpose residual gates developed above. Add both symmetry controls and one supported asymmetric case. A coil/source fit must report held-out exterior error and its effect on boundary response separately from the equilibrium error.

PR448 is optional research context, not the continuation's solver. Its R6 independent magnetic-normalized force target passes for one fixed-profile axisymmetric state, but projected stationarity remains about 2.22e-8 against a 1e-8 gate; derivatives are open. Its local/global step discrepancy is also unresolved. Do not merge it, call it production-ready, or plot its force `epsilon_B` as this project's field error E_B. [S11]

A bounded polishing test should compare before/after on identical native representation and quadrature, preserving boundary and profiles. Report field/current/force, feasibility, exact least-squares stationarity, and memory separately. If differentiating a nonzero-residual least-squares solution, include the residual-weighted second derivative in the stationary Hessian or explicitly label a Gauss-Newton approximation. Exact knot insertion and zero-padding are valuable continuation tools because they need not refit the physical state.

## 11. Optimization and the scientific extension

Do not wait for every module to launch a small exact-family optimization. Explicit equilibria can be optimized directly, independently of the VMEX solve, and then selected points can be verified through the numerical solver. This provides a useful result even if a particular nonlinear branch remains difficult.

First remove trivial scale freedoms. For example, fix physical volume and RMS field by choosing L_star and B_star from the exact dimensionless integrals. Keep boundary pressure zero. These normalization scales now depend on the shape parameters: differentiate that dependence. Do not apply the fixed-L_star, fixed-B_star delta-null identity to this rescaled optimization path. Choose a small shape parameter set, a domain margin, finite elongation/aspect-ratio bounds and nonzero symmetry-breaking bounds when a 3-D design is required. Optimizing beta by adding a pressure constant or collapsing to an axisymmetric limit is not the intended study.

A concrete first pair of objectives is current concentration and field-strength variation:

\[
 C_J=\mu_0\ell_*\langle J^2\rangle_V^{1/2}/B_{rms},\qquad
 C_B=\langle(B/B_{rms}-1)^2\rangle_V,
\]

with ell_star=V^(1/3), plus a prescribed beta range and geometric constraints. Compute a small Pareto set with a few starts and record all objective/constraint histories. Compare direct analytical gradients with finite differences. Hold out quadrature points and refine the final candidates. If the family offers no nontrivial tradeoff, document that outcome instead of adding many arbitrary metrics.

The next optimization is a coordinate-only study. Minimize resolved Fourier tail or a coordinate-condition measure over regular u in (5), with positive Jacobian and unchanged physical B, p, s and boundary. Treat this as representation optimization, not a plasma improvement. Compare cost-to-field/current accuracy before and after projection. Recover lambda consistently; never reduce spectral force by allowing the physical field to drift unnoticed.

Only then release a small number of physical boundary/profile parameters transverse to the exact family. Use verified VMEX derivatives, trust-region steps, explicit source/geometry constraints and reconvergence checks. The physical question is whether useful improvements remain under independent refinement, and whether tangent and transverse directions have different conditioning or resonance sensitivity. Preserve branch identity; a loss event, mode crossing or root switch is not an ordinary smooth derivative.

For the regularity study, current continuity with J=sigma B+(B x grad p)/B^2 gives a magnetic differential equation for sigma. On a closed line the integral of its source divided by |B| must vanish. Use this necessary compatibility condition as a diagnostic of resonant forcing, not a sufficient theorem of nearby smooth equilibria. Rational exact solutions are valuable controls; they are not automatically singular, uniquely invertible, omnigenous or reactor-like.

Coil optimization follows only after the exterior and coupled-response work. Separate boundary-only, fixed source-fitted, and truly coupled problems. Do not use the analytical interior as an external coil field. Optimize a held-out normal-field objective with length/curvature/current/clearance constraints, then verify coupled equilibrium response and cost.

## 12. Implementation sequence and completion gates

The new task IDs below do not erase P/R/T history. Their purpose is to prevent the next agent from repeating finished work or expanding an inconclusive run indefinitely.

| Task | Work | Exit evidence |
|---|---|---|
| C0 | Single-pass anchor observation; response units; run-source capture | Tests distinguish raw/refined/memo-hit states; nonunit/replicated-sample tests pass; failed roots cannot become accepted responses |
| C1 | Saved default-branch residual defect and saved zero-weight base root-floor test | Same-operator endpoint and tangent defects; real correction trace; scaled rank/range or globalization outcome |
| C2 | Independent scoring of the saved NS129 candidates and weak stress | Stable finest-grid norms or bounded failure; correct domain/knot semantics; complementary stress/current diagnosis |
| C3 | Full-flux lambda reference, projection and sheared-A refinement | Flux periods, compatibility, physical B and radial regularity checked; matched projected/solved states kept distinct |
| C4 | Accepted axisymmetric nonzero/null responses and selected 3-D response | Actual root/linear/FD/continuum certificates with invariant normalization |
| C5 | LASYM, current/iota, output/restart, spatial derivatives, exact flow and diagnostics | Executed capability entries with independent or explicitly limited oracles |
| C6 | Exterior operators, a coupled root and bounded optional polish | Matching/geometry/root/stationarity/derivative gates separately recorded |
| C7 | Direct-family and coordinate optimization; bounded transverse extension | Reproducible histories, gradient checks and independently refined outcomes |
| C8 | Cost, figures, README and final resumable handoff | Read-only plots, complete scope/limitations, accurate cost-to-error comparison and current logbook |

C0 comes first. C1 and C2 can then proceed independently. C3 and direct-family C7 can start after their reference-only tests, without waiting for the entire zero-TCON diagnosis. C4 can certify the nonzero axisymmetric case while the null case is investigated. C5 exact-flow/interface tests need not wait for nonlinear recovery. C6 coupled gradients and solver-driven C7 need the relevant root/response certificates.

The next five concrete actions are:

1. Adopt this plan without losing history; rerun the cheap existing reference suite and record its actual result. Do not change the now-working import boundary.
2. Correct `_root_and_anchor` observation and response normalization with focused tests. Capture runtime code before any new expensive solve.
3. Evaluate the saved default NS65 branch in one frozen residual, including (2), and observe one actual refinement of the saved zero-weight base. No new plus/minus solve is needed initially.
4. Run the full-flux prototype against the current `analytic.py`, then cross-check the existing sheared-A map and one projected state. In parallel, score the saved promising NS129 integer states on independently refined domains.
5. Choose the next nonlinear or derivative run from these discriminating results, rather than from another unconstrained TCON scan.

### Where to change code

Keep `analytic.py` as the shared field/reference implementation. Put only genuinely shared norm, stress and quadrature helpers in `measurement.py`. Extend the current `evidence.py` for exact execution-source capture and certificate semantics only where missing. Repair the existing response driver instead of adding one new script per fitted hypothesis.

A small flux-potential helper may be added if both projection and postprocessing call it; otherwise keep it near the sheared projection. Isolate version-specific private VMEX imports in the existing adapter/driver boundary. Do not copy the entire implicit solver or implement a new equilibrium solver. Replace hard-coded timestamp paths in active drivers with explicit arguments or a small selected-run record; historical reproduction scripts can remain untouched and clearly marked.

Tests should cover the scientific failure modes, not only function existence: cached anchor observations, incompatible flux periods, resonant straightness ambiguity, sign/NFP errors, invalid sample domains, response scaling, changed operator context, failed roots, and evidence promotion. Do not count a mock root test as physical recovery. Reuse fixtures and parameterization rather than creating a directory of nearly identical tests.

## 13. Figures and results to publish

Every figure is derived from saved immutable records. Plotters must not edit solver records, inject package versions, or promote acceptance. Keep input/record/generator hashes in a separate figure manifest. Show failed and capped runs with different markers. Do not connect an unrun point with a fitted curve.

| Figure | Scientific question and required evidence |
|---|---|
| Current evidence map | Which reference, projection, root, measurement and derivative levels actually pass? |
| Corrected anchor trace | What raw state entered refinement, which steps were tried, and why were they accepted/rejected? |
| Branch defect and response | Do saved endpoints solve the same operator, and does their signed FD satisfy its linearization? |
| Scaled conditioning | Are small directions coordinate-null or physically active, and what metric is used? |
| NS129 physical refinement | Does the zero-weight integer candidate remain accurate under resolved volume and boundary checks? |
| Strong/weak force comparison | Is the failure present in B/p stress moments, numerical curl, or both? |
| Sheared lambda reconstruction | Transport, full-flux reference and finite projection; compatibility, field error and cost |
| Sheared recovery ladder | Projection-to-root movement versus resolution, default and only justified controls |
| Derivative certificates | Nonzero c, null delta and later sheared axial-parameter responses; FD windows and root-error estimates |
| Breadth and integration | LASYM/current/iota/file/exact-flow results without combining unlike evidence classes |
| Exterior and polishing | Matching, force, stationarity and response errors, with the actual model declared |
| Optimization and cost | Verified objective/constraint tradeoffs and time/memory at equal physical accuracy |

The README should lead with a short current-results table and the major unresolved limits. Link the long logbook instead of repeating every historical run narrative. State that CI tests reference code, not every optional solver. Preserve failed measurements as linked records; do not let historical adoption text imply a presently failing CI or a presently missing NS129 run.

Measure compilation, raw nonlinear solve, refinement, field scoring, adjoint, input generation and I/O separately, plus end-to-end time. Synchronize accelerators. List devices actually used, not merely visible. Record host and device memory separately; a logical-array estimate is not a process RSS bound. Use repetitions when making performance claims. Independent analytical-reference cost may be reported separately, never hidden from an end-to-end claim.

## 14. Evidence contract and resource policy

Use separate fields for execution status, solver return/convergence, root certification, physical measurement resolution, recovery acceptance, linear response certification and continuum derivative acceptance. Do not overload one `passed` flag. A dense derivative at an unresolved state can be a valid linearization test, but is not an equilibrium sensitivity.

Store exact input/deck, state, parameter tangent, independent-DOF layout, frozen context, quadrature/domain, units, source and environment hashes. Save the raw state and every completed expensive child result before later processing. Reference functions and fitted profiles used to interpret an old state must be the recorded versions. An amended interpretation points to the unchanged raw file and gives its rule and author; it does not rewrite the old result.

Use finite process time and memory budgets established from a cheap pilot. A watchdog should record exit code, signal, actual wall time, sampled process-tree RSS, and expected-output existence. Reuse the useful pattern in PR448 only as a small utility; do not add a scheduler framework. On interruption, preserve completed outputs and write a terminal receipt. Repeating a failed experiment requires a new hypothesis, changed setting and predicted discriminating outcome in the logbook.

Do not silently turn nonfinite fields into zeros, discard failed inversion points, select a different root per finite-difference sign, or let arbitrary truncation in a pseudoinverse remove physical response. Unsupported hardware, interface or model assumptions remain explicit `unavailable`, `blocked` or `not_applicable` entries.

## 15. Completion, logbook and replacement-agent handoff

The core is complete only when the stated mild axisymmetric and non-axisymmetric cases have resolved physical recovery, meaningful LASYM and both closure coverage, at least one certified nonzero and one null physical response, independent diagnostic/interface checks, and both exterior-operator and strictly labelled coupled free-boundary evidence. A bounded optimization and polishing outcome must be recorded, but it may be a well-explained limit rather than a claimed improvement. Optional kinetic models do not redefine this core.

A negative result can close a specific question when its assumptions and numerical evidence are resolved. It cannot turn an unsupported capability into a pass. A model or source limitation should produce a minimal reproducer and a precise next step, not a growing set of unstructured long runs.

At adoption initialize C0-C8 as planned, carrying forward the completed T0 repairs and the existing partial evidence listed above. After every meaningful block append the following record to the active plan. Do not continually rewrite the historical entries.

```text
Date/time and timezone:
Question and task ID:
Benchmark/source commits and runtime source hash:
Input, state, frozen operator, tangent and domain hashes:
Command and effective configuration:
Raw/linear/physical/response certificates, with separate statuses:
Measured result and uncertainty; failed or interrupted attempts:
Artifacts and hashes:
Source/tests actually inspected or executed:
Branch, commit and PR state:
Decision and exact next action:
```

### Review entry: 05e9473 continuation

This review retires the old CI and missing-NS129 tasks, identifies that the root observer calls a previously attempted cached refinement, and replaces further coefficient fitting with the defect of the measured branch tangent in one declared operator. It also corrects dimensional/sample-count response comparisons, proposes a full-flux periodic lambda reference and weak magnetic-stress test, and keeps the failed zero-TCON examples separate from the promising integer candidate.

The supplied ten mathematical tests pass. The sheared surface prototype executed against the reference files supplied in the initial handoff, whose hashes are recorded. Its surface B reconstruction reached about 1.5e-10 relative error; no VMEX solve, current certificate, global radial certificate, full repository test rerun or current-main comparison was executed here. GitHub reference CI success was inspected independently. No remote repository write was made by this review.

**Exact next action:** verify the actual checkout, archive its active plan, repair the single-pass refinement observer and normalization, then use the saved default-branch endpoints and saved zero-weight base for the two algebraic diagnostics. Do not rerun the completed NS129 cell, call the radial m=1 candidate a gauge, or interpret the uncertified zero-weight branch FD as a physical derivative.

## 16. Task table (C0-C8)

| Task | Status | Evidence / next |
|---|---|---|
| T0 (carried) | done | Reference CI green at 05e9473; import/schema repair; NS129 zero-TCON run exists; preconditioned single-grid history test passed. Not reassigned. |
| C0 | done (2026-09-24) | `refinement_observer.py` + tests: raw host state, one uncached `_refined_state`, per-step true linear defects, memo-hit probe; eq.(3) response metric and x/L_star fix with nonunit/duplicate tests; `capture_execution_source` before runs; derivative gate on root certificate |
| C1 | done (2026-09-24) | Zero-weight base: certified root 6.2e-14 after a second pass, same operator; first pass stalls on a too-long accurate Newton step. Default branch: eq.(2) defect is O(h^2), branch FD is the base linear response; null response E_B,delta = 2.93e-4 (point cloud) is physical, not an operator defect |
| C2 | bounded failure recorded (2026-09-24) | NS129 zero and default TCON0 volume-scored: both fail at the axis cell; zero better everywhere; mid and edge meet gates for zero |
| C3 | in progress | Full-flux lambda verified (1.6e-10); projection ladder NS17-65 radial-limited; recovery solves next |
| C4 | pointwise certified (2026-09-25) | NS129 axisymmetric c and null delta pass root/linear/transpose/FD gates on 4 pointwise observables (delta JVP 1.57e-6 with matched input tangent); volume norms await the axis-row fix |
| C5 | partial | Exact flow done; LASYM basis control passes (round-off); asymmetric Solov'ev running; closures and interfaces remain |
| C6 | planned | Needs C0, C1 |
| C7 | direct stage closed | No substantive C_J/C_B tradeoff (bound-driven); solver-verified stage needs C4 |
| C8 | planned | Continuous |

## 17. Continuing logbook

### C0/C1 block, 2026-09-24 (America/Chicago; run IDs carry UTC 2026-09-25)

**Question and task ID:** C0: observe one raw host state and one genuinely uncached refinement; correct response units and weighted norms; capture execution source. C1: (a) observe one real refinement of the saved NS65 zero-TCON base; (b) test the saved default-TCON NS65 delta branch against one frozen operator, plan eq.(2).

**Benchmark/source commits and runtime source hash:** Benchmark base `05e947370b9d0cebc0e508361e447b03523a331d` on branch `c-05e9473` (uncommitted during runs; each run stores the exact bytes of its driver, `refinement_observer.py` (SHA-256 prefix `7ccb6b8e18a7`) and `evidence.py` in `execution_source/`, plus the tracked-diff hash). Historical VMEX `b5f5267efc0795c4a49a224e321e9b370975c14c`, clean isolated clone, verified by each run. SOLVAX imported from a local checkout at `2d96aa89f573dd9042148101588099f30fc9296f` (not the review's `2e246a5`; the refinement path imports its Krylov routines). JAX 0.11.0, CPU, float64.

**Adoption:** `plan.md` archived byte-for-byte as `docs/history/plan_through_05e9473.md`; previous prompt archived as `docs/history/AGENT_PROMPT_through_05e9473.md`; handoff package copied to `docs/handoff/review-05e9473/` with its `SHA256SUMS` verified. Reference suite rerun before changes: `pytest -q` 55 passed (15.8 s); `verify_reference.py` 14/14 pass.

**Source findings (pinned VMEX):** (1) `_refine_fixed_point` memo is keyed by config and parameter bytes; the historical `_root_and_anchor` therefore measured two post-refinement states. (2) `make_config` interns by content (`_canonical_config`): a second `make_config` with identical settings returns the *same object* in one process, so "fresh config" does not escape identity memos (measured: same `id`, `_LAST_SOLVE` hit in 25 us). The observer calls unmemoized `_refined_state` directly. (3) `_refined_state` freezes the operator at its input state; `frozen` only supplies non-evolved entries (fixed m=1 combinations, lambda axis row) plus the edge; preconditioning is recomputed at the trial point.

**C0 code:** `benchmarks/refinement_observer.py` (raw host state with `refine=False`; one `_refined_state(initial_correction=None)` with the three module seams wrapped; per step: input/output hashes, trial residual, reported and *true* linear defect `|A d + r|/|r|`, merit slope `r.A d/|r|^2` (eq. 4), acceptance by distance to the returned state; before/after residuals in the refinement operator and re-frozen operator, raw and preconditioned with blocks; statuses `host_returned`, `refinement_attempted`, `root_certified`, `response_certified`; `derivative_gate`). `_root_and_anchor` now uses it (historical keys kept). `measurement.scaled_response_error` implements eq.(3); `measurement.exact_field_parameter_tangents` evaluates the reference at x/L_star. `evidence.capture_execution_source`. Tests `tests/test_refinement_observer.py` (6, contract-level with a fake seam module, not physical roots): changed/certified, unchanged failed, memo hit (including a different state returning the memo), hash sensitivity, duplicate-point invariance and a_star/B_star scaling, nonunit L_star/B_star tangents.

**C1(a) command:** `PYTHONPATH=benchmarks:<pinned VMEX> python benchmarks/observe_root_refinement.py --deck results/audit/axisym_branch_tcon0/axisym-tcon0-branch-20260925T010824.995851Z/input_delta_ns65_base.indata --saved-state .../state_base.npz --ns 65 --label ns65-tcon0-base` (effective NS65, FTOL 1e-12, NITER 10000, refine_tol 1e-11, TCON0 0 from the deck; deck SHA-256 `7a75525e...`). Then `python benchmarks/diagnose_root_floor.py --observation <that run>` and `python benchmarks/score_saved_states.py ... --labels raw pass1 pass2 --points results/audit/axisym_radial_relabel/.../radial_relabel_field_ns65.npz`.

**C1(a) result:** Raw host solve: 1091 iterations, IER 11, preconditioned residual `5.518e-7` (raw formulation `8.86e-7`). The saved historical "base" is **not** the raw state: it equals the output of the first refinement (`1.188e-7`, state difference 0.0774). The historical record's zero shift and equal before/after residuals were therefore the memo artifact; the refinement had moved the state. One observed pass from the raw state: 4 block and 3 GCROT steps, all accurate Newton directions (true defect 2.8e-6..0.19 relative, merit slope -0.96..-1.000); the first full block step (norm 0.0772) raises the residual to 2.44e-4; the best trial (block step 4, `1.191e-7`) is returned; not certified. A second uncached pass from that state: three block steps, `1.19e-7 -> 5.1e-8 -> 7.0e-12 -> 6.22e-14`, quadratic. All three states have identical non-DOF entries (difference exactly 0), and the final state has `6.22e-14` preconditioned / `6.33e-12` raw in every one of the three operators: **root_certified for one equation** (tolerance 1e-11). Backtracking along the first Newton direction: residual grows as alpha^2 (2.44e-4, 6.1e-5, ..., 5.64e-7 at alpha=1/64); Armijo(1e-4) fails for alpha>=1/64, so the direction is a descent direction but the raw state lies ~1% (0.077 of state norm 8.01) from the root along a strongly curved near-null direction. Mechanism class: globalization/step-length, not an inaccurate linear solve, operator mismatch or range obstruction. Physical effect on the same 96 Cartesian points (point-cloud screening, uniform weights): B relative L2 `7.31e-6` raw, `1.667e-6` pass 1, `1.673e-6` certified root; max native flux-label error `2.61e-5`, `6.49e-6`, `6.50e-6`; raw-to-root B change `6.4e-6`. The near-null direction is physically active. First scoring run compared native rho with s (flux error 0.25); preserved with `amendment.json`, superseded by the rerun.

**C1(b) command:** `python benchmarks/branch_tangent_defect.py` (saved states only: base and h=1e-3,3e-4,1e-4 endpoints from `results/vmex/response_runs/axisym-root-response-20260924T232219.753772Z`, h=3e-5,1e-5 from `results/audit/axisym_branch_fine_steps/axisym-branch-fine-20260925T000429.848273Z`; parameters from the saved decks; operator frozen at the base; default TCON0=1).

**C1(b) result:** Base residual `5.70e-14`. `|A d_h + F_P q_h|/|F_P q_h|` = `3.02e-4, 2.71e-5, 3.01e-6, 2.71e-7, 3.01e-8` for h = `1e-3, 3e-4, 1e-4, 3e-5, 1e-5`: exactly O(h^2). Frozen-entry drift between endpoints: exactly 0. Every endpoint residual is 5.8e-14..8.1e-14 in both the base and its own operator. Hence the saved default branch is a smooth branch of roots of one operator and its centered FD satisfies the base linearization; there is no tangent/operator defect. The stored implicit tangent solves its own system to 1.2e-13, but it used the h=3e-4 input tangent; `q_h` changes by 2.2e-4 relative between h=3e-4 and h->0, which fully explains its 1.5e-4 plateau against the branch FD. The profile-fit input map is not converged at 1e-4 relative at h=3e-4; C4 must use matched q or bound this. Invariant eq.(3) null response of that branch (96-point cloud, a*=delta0=1/64, B*=1 T): `E_B,delta = 2.93e-4`, versus the 1e-5 research target. Since the linearization is consistent, this nonzero response belongs to the NS65 discrete equilibrium and input map, not to the derivative machinery.

**Status separation:** zero-weight base: host_returned, root_certified (after two passes), physical 96-point screening only, no response. Default branch: roots certified (~6e-14), linear response consistent, null response *not* accepted (fails 1e-5), measurement point-cloud only.

**Artifacts (SHA-256):** `results/audit/refinement_observation/ns65-tcon0-base-20260925T033309.807236Z/observation.json` `1951b699...`, arrays `c590b816...`; `results/audit/root_floor_diagnosis/ns65-tcon0-base-20260925T033309.807236Z-floor-20260925T033615Z/root_floor.json` `3a8194e6...`, certified-root states `2cb3f58a...`; an earlier diagnosis directory (`...-floor` without timestamp) holds only its source capture and a failure receipt (template runtime not primed); `results/audit/saved_state_scores/ns65-tcon0-root-floor-20260925T033801Z/scores.json` `e2c87918...` (superseded `...033714Z` kept with amendment); `results/audit/branch_tangent_defect/ns65-delta-default-20260925T033907Z/defect.json` `3d9c5097...`. Run times 65 s, 3 s, ~40 s, 5.6 s; peak host RSS about 1.9 GB, CPU only. An NS17 observer pilot (certified in one pass, 1.8e-7 -> 6.3e-15; memo hit confirmed) ran in scratch with a pre-fix trial matcher and is not a recorded result.

**Branch/PR state:** local branch `c-05e9473` from `05e9473`; to be pushed after validation. No PR, no upstream write, no solver change, TCON0 default unchanged.

**Decision and exact next action:** Do not rerun zero-TCON plus/minus branches yet. The root question is resolved for the base: a zero-weight root exists and is more accurate on the point cloud than the raw host state. Next: (C2) rescore the saved NS129 integer states with independent refined quadrature and add the weak-stress moment; (C3) run the full-flux lambda prototype against current `analytic.py`; (C4, later) measure `E_B,delta` versus NS at default TCON with matched input tangents, and re-derive the zero-TCON branch from certified roots (two-pass refinement) only if C2 supports it. Propose an upstream strict-certificate / damped-first-step refinement option only with a small reproducer.


### C3/C5/C7 reference-only block, 2026-09-24 (America/Chicago)

**Question and task IDs:** C3: does the full-flux lambda (eqs. 7-8) reproduce the sheared-A field on current `analytic.py`, agree with the existing transport lambda, and change the NS17 projection? C5: exact field-line flow control (eq. 11). C7: direct exact-family optimization of C_J and C_B.

**Source:** benchmark `ee69264` plus the uncommitted files named below (each run stores its script bytes under `execution_source/` or `results/*/source_captures/<run>/`); `analytic.py` SHA-256 `edea4490...` (the review prototype used `bcbe45a3...`); historical VMEX `b5f5267` only for the projection; JAX 0.11.0 float64 CPU.

**C3 surface reference.** Command: `python docs/handoff/review-05e9473/probes/sheared_flux_demo.py --reference-root . --output results/reference/full_flux_lambda/sheared_flux_demo_current_ee69264.json` (SHA-256 `acdab594...`). Best s=0.5 relative B error `1.58e-10` (handoff `1.54e-10`), closedness `7.8e-8`, min 1+lambda_theta `0.767`: reproduced on current code. `python benchmarks/compare_sheared_lambda.py` (run `compare-transport-20260925T034246Z`, SHA-256 `ffdb22c1...`): on s=0.25/0.5/0.75 and odd grids 33/65/97 the full-flux and LSMR transport lambdas agree to `6.6e-9..2.0e-7` relative L2; B rebuilt from the full-flux lambda has error `1.2e-10..1.9e-10` for n>=65, from the transport lambda `3.4e-9..6.4e-8` (limited by its iterative tolerance and slightly worse with n), and from lambda=0 `9-16%`. No resonant straightness ambiguity on these surfaces. The probe functions are now ported unchanged into `benchmarks/flux_lambda.py` with `exact_surface_lambda` (odd grid, signed Jacobian, one-sided second-order radial stencil at the edge); it refuses to run without float64 after a float32 call gave `6.5e-7` instead of `1.6e-10` (recurrent x64 trap).

**C3 projection.** `python benchmarks/project_sheared_straight_field.py --lambda-method full_flux` (new option; transport stays the default). Run `sheared-A-full-flux-20260925T040717.258975Z`, report SHA-256 `def912c8...`; source capture `results/projection/source_captures/sheared-a-full-flux-projection-c3-v2-20260925` (the first capture's attempt failed before reserving a directory; failure receipt kept). NS17 96-point scores: B/J/grad-p/force `7.0655e-4 / 5.1312e-3 / 3.5814e-4 / 0.024406`, identical to the transport projection to about 1e-12. Lambda accuracy is therefore no longer the NS17 projection floor; finite VMEX representation is. Projection-only status; no root, no volume measurement.

**C5 exact flow.** `tests/test_exact_flow.py` (3 tests): closed-form flow vs DOP853 ODE (`<1e-11`), tangent map vs integrated variational equation (`<1e-9`), det = 1 (`1e-12`), flux-label closure (`1e-13`), composition, and x = L xbar(B* t / L) scaling, for integer_axisymmetric and the stretched integer_3d. Reference-only interface control.

**C7 direct optimization.** `python benchmarks/optimize_integer_family.py` (C_J minimized under C_B caps 0.02-0.16) and `--minimize C_B --caps 2.24 2.245 3.0`; runs `direct-20260925T035905Z` (`9f63d1bc...`) and `direct-20260925T040050Z` (`3db804ca...`). c = 1 fixes the family's self-similarity (tested: B(3x; 3a,3b,3c) = 3B(x)), so C_J and C_B need no L*/B* derivative path. Quadrature reproduces V and beta_V closed forms to 1e-12 (beta 3.50877193% at the reference point). Constraints: margin 0.05, beta in [0.02, 0.05], a/b <= 3, a^2-b^2 >= 0.2. Every start converges to the same bound corner: beta = 0.02 and asymmetry at its floor. Pareto set (C_J, C_B): `(2.2317, 0.00233)`, `(2.2400, 0.00205)`, `(2.2450, 0.00203)`, `(2.2502, 0.00202)`; gradients vs FD `<=6.4e-9`; 2x held-out quadrature changes objectives by `<=4e-16`. Two SLSQP starts at the C_J minimum end with "positive directional derivative" at the same point as the successful start. Outcome: within this family and these bounds there is no substantial tradeoff; both objectives prefer minimum beta and minimum asymmetry (the axisymmetric/low-pressure direction the plan excludes). Documented as a limit, not an improvement; more metrics are not added.

**Branch/PR state:** uncommitted on `c-05e9473` (last push `ee69264`). No PR, no upstream write.

**Exact next action:** C3: projection ladder NS33/65 with the full-flux lambda (running) to measure the representation floor before any solve; then an NS17/33 recovery ladder from that seed at default TCON0. C7: a solver-verified point needs C4 certificates; the direct study is closed unless a new objective with a real tradeoff is justified.

**C3 addendum, projection ladder (full-flux lambda, MPOL 13, NTOR 12, 96-point cloud, projection only, no solve):** `python benchmarks/project_sheared_straight_field.py --lambda-method full_flux --ns {17,33,65}` (runs `sheared-A-full-flux-20260925T040717.258975Z`, `...T040857.962212Z`, `...T041109.088156Z`; source captures under `results/projection/source_captures/`). B/J/grad-p/force: NS17 `7.07e-4 / 5.13e-3 / 3.58e-4 / 2.44e-2`; NS33 `5.89e-5 / 2.18e-3 / 2.02e-5 / 1.04e-2`; NS65 `1.055e-5 / 6.79e-5 / 3.08e-6 / 3.34e-4`. Max flux-label error `2.7e-5, 1.95e-6, 1.92e-6`. Radial resolution dominated the NS17 floor. At NS65 J and force meet their gates and B is just above 1e-5; the flux-label error plateau from NS33 to NS65 points to the Fourier truncation next. **Next:** an MPOL/NTOR step at NS65, then a default-TCON0 recovery solve from the NS33 and NS65 seeds, before any TCON comparison.

### C4 block, 2026-09-24: axisymmetric c/delta responses versus NS (default TCON0)

**Commands:** `python benchmarks/null_response_ladder.py` (NS 17/33/65/129, h=1e-4), then `--ns 129 --step 3e-4` and `--step 3e-5`; then `python benchmarks/axisymmetric_root_response.py --ns 129 --fd-steps 3e-4 1e-4 3e-5 --frozen-steps 1e-4 1e-5` (repaired driver: single-pass observer, eq.(3) records, derivative gate, execution-source capture). Runs: results/audit/null_response_ladder/axisym-default-20260925T035733Z/ results/audit/null_response_ladder/axisym-default-20260925T040711Z/ results/audit/null_response_ladder/axisym-default-20260925T040810Z/ and `results/vmex/response_runs/axisym-root-response-20260925T040924.626718Z`. Historical VMEX b5f5267; each run holds its source bytes.

**Roots:** every base and endpoint at every NS was certified in one observed pass (3 block steps from about 6e-7): residuals ~8e-15 (NS17), ~2e-14 (NS33), ~5e-14 (NS65), 1.5-1.8e-13 (NS129), all below refine_tol 1e-11. The base-case B relative error on the 96-point cloud is 1.69e-4, 1.40e-4, 4.83e-6 and 8.83e-7. The per-endpoint values in the ladder are measured against the *base* exact field, so perturbed endpoints naturally show larger ones.

**96-point branch FD (eq.(3), a* = base value, B* = 1 T; point cloud):** c relative error 7.8e-3, 1.17e-2, 1.97e-3, 3.97e-5; E_B,delta 1.13e-3, 1.25e-3, 2.93e-4, 4.35e-6 (NS 17/33/65/129; NS65 reproduces the historical 2.93e-4). At NS129 the step window gives h=3e-4: 3.95e-5 / 4.10e-6 and h=3e-5: 3.98e-5 / 4.76e-6. FD truncation is small, so the values measure discretization. The null response goes below the 1e-5 research target at NS129.

**Repaired driver at NS129 (its own 4 fixed points):** root 1.75e-13; tangent linear residuals 2.0e-13 and 1.2e-11; transpose duality 2.9e-11 (c) and 4.7e-9 (delta); derivative gate open. For c, the JVP relative error to the exact response is 1.636e-5, and branch FD minus JVP is 2.7e-8, 5.4e-9 and 3.5e-9 at h = 3e-4, 1e-4, 3e-5. **Nonzero c response: certified as a 4-point pointwise response at NS129** (root, linear, transpose, stable FD window, 1e-3 gate); it is not a volume norm. For delta, branch FD gives E 1.01e-6 / 1.32e-6 / 1.52e-6, but the JVP gives 4.69e-5 with a constant FD-minus-JVP of 6.0e-3 across h. That matches the input-tangent error found in C1(b): the tangent uses the h=3e-4 profile-fit q, 2.2e-4 away from its limit, and a near-null direction of A amplifies it. **The null JVP is not accepted.**

**Next:** recompute the delta tangent with a converged input tangent (the existing `full_input_derivatives` map, or q from matched small h), then repeat the NS129 duality/FD check; extend both responses to a resolved volume quadrature once C2's near-axis issue is understood.

### C2 block, 2026-09-24: NS129 integer-3D volume scoring (zero vs default TCON0)

**States:** reproduced from the recorded input `inputs/input.integer_3d_iota` (SHA-256 `7a7b0cdb...`) and seed `results/projection/integer_3d_vmex_ns129/seed.npz` with `BENCH_FTOL=1e-10 BENCH_TCON0={0,default} python benchmarks/run_vmex.py inputs/input.integer_3d_iota 129 3000 <seed> --keep-terminal` (runs `integer3d-ns129-tcon0-{0,default}-reproduce-20260925`; source captures in `results/vmex/source_captures/`). The original run's WOUT was gitignored on another machine. The zero run reproduces the saved record (50 iterations, same FSQ, identical 96-point cloud, B observations within 4.9e-12); the default run reproduces 420 iterations and its recorded scores. Each measurement stores the scored `spectral_state.npz`.

**Command:** `python benchmarks/verify_measurement.py --input inputs/input.integer_3d_iota --wout <run>/wout.nc --state-id <id> --output-parent results/audit/measurement_c2 --profile full`. Zero run `integer3d-ns129-tcon0-0-full-c2-v3-20260925`: every grid checkpoint completed, then the final spread crashed on a key mismatch (route-difference short keys vs TARGETS names). Fixed with `ROUTE_KEY`; `interruption.json` lists the 12 completed grids, and its `measurement.json` still reads "running" as written, superseded by the receipt. Two earlier attempts (wrong `--input` location and wrong legacy-sample format) left `measurement_failed` records. Default run `integer3d-ns129-tcon0-default-full-c2-20260925`: completed, `measurement_diagnostic`, not resolved.

**Results (route A; route B agrees to about 1e-13), B / J / force-over-grad-p:**

| grid | zero TCON0 | default TCON0 |
|---|---|---|
| legacy96 | 1.08e-6 / 1.80e-5 / 1.94e-4 | 1.30e-4 / 2.92e-3 / 1.74e-2 |
| full 16x64x64 Gauss | 1.94e-5 / 2.82e-3 / 2.99e-2 | 2.26e-4 / 1.02e-1 / 1.13 |
| full midpoint | 1.52e-5 / 2.24e-3 / 2.40e-2 | 4.95e-4 / 7.69e-2 / 0.863 |
| near axis (s 1e-8..1e-2) | 3.06e-4 / 9.88e-3 / 1.46 | 9.53e-4 / 9.63e-2 / 14.6 |
| native knots and cells | 9.53e-5 / 9.46e-3 / 0.106 | 9.25e-4 / 1.08e-1 / 1.26 |
| near edge | 9.58e-7 / 2.51e-5 / 2.18e-4 | 8.86e-5 / 3.34e-2 / 0.268 |

Angular refinement and shifts change nothing at 9 digits; the radial Gauss/midpoint spread remains. For the zero state, the B error is constant at 3.5e-4 as s -> 0 and the J error peaks at 1-2e-2 in the first two or three native cells (s <= 0.012), while s = 0.5 and 0.99 give B ~9.4e-7 and J 7e-6..2.5e-5. **Outcome:** a bounded failure of volume recovery for both states, localized at the magnetic axis. Zero TCON0 is better than the default on every grid of this case (the opposite of the sheared-A NS17 outcome), so this is still not a global default recommendation. The 96-point legacy cloud missed the axis entirely.

**Next:** diagnose the axis defect in this state: axis position (R/Z m=0 at j=0) versus the exact axis, the lambda axis-row closure, and m=1 near-axis regularity. Compare the exact axis field with the native s->0 limit. Then a weak-stress moment near the axis (`measurement.weak_stress_moment`) to see whether the defect appears in B/p stresses or only in the numerical curl.

### C2 addendum, 2026-09-25: axis attribution and NS257 (office host)

**Axis position/field split (NS129, local):** the WOUT axis row of the zero-TCON0 root lies on the exact axis to 8.0e-7 in R and 6.4e-7 in Z; the default-TCON0 axis is displaced by 4.4e-4. The zero-state axis error is almost entirely in |B| (parallel 3.49e-4, perpendicular 1.6e-6), mostly toroidal, and varies with phi following the rotating ellipse. On 896 near-axis and knot points (`benchmarks/score_saved_states.py`, run `results/audit/axis_diagnosis/ns129-axis-projection-vs-root-20260925T044334Z`), the **exact projection with no solve** has the same error as the root (1.81e-4 overall; 3.50e-4 vs 3.49e-4 at s=1e-8). The axis defect precedes the solver.

**Projection ladder (no solve):** `python benchmarks/axis_projection_ladder.py` (run `projection-ladder-20260925T044531Z`). Axis-point B error 1.41e-3, 7.01e-4, 3.50e-4, 1.75e-4 for NS 33, 65, 129, 257, i.e. first order in the radial spacing; the s=0.5 error is 5.77e-9 at every NS. Near the axis the representation or its evaluation is O(Delta s).

**NS257 zero-TCON0 solve and full measurement (office; pinned VMEX b5f5267, JAX 0.11.0, CPU, 8 threads):** projection by `project_integer_vmex.py 257`; `BENCH_FTOL=1e-10 BENCH_TCON0=0 run_vmex.py inputs/input.integer_3d_iota 257 6000 <seed>` (run `integer3d-ns257-tcon0-0-office-20260925`): 39 iterations, IER 11, converged, 8.1 s. Full measurement `integer3d-ns257-tcon0-0-full-c2-office-20260925` (5276 s, `measurement_diagnostic`). B/J/force-over-grad-p: legacy96 2.64e-7 / 4.73e-6 / 5.0e-5; full Gauss 16x64x64 6.64e-6 / 1.65e-3 / 1.75e-2; full midpoint 5.83e-7 / 1.64e-4 / 1.79e-3; near axis 1.46e-4 / 9.56e-3 / 1.41; knots and cells 4.89e-5 / 6.88e-3 / 7.72e-2; near edge 2.51e-7 / 1.13e-5 / 9.88e-5. The axis B error halved from NS129, as predicted. The **near-axis current and force did not improve** (9.9e-3 to 9.6e-3; 1.46 to 1.41), so the axis-cell J error is limited by an NS-independent axis rule. The Gauss/midpoint volume spread is set by how close the radial nodes come to the axis.

**Interpretation and next:** away from the first cells the solved zero-TCON0 state converges toward the exact field; the gate failures are the axis cell. Next: find the NS-independent axis-cell J rule in the native field evaluator (how the first half-cell and lambda axis row are closed at the pin), with a minimal reproducer on the exact projection only; then decide between a documented evaluator limit and a narrow upstream issue. Report volume norms with the axis cell separated, and never drop those samples.

### C2/C5 block, 2026-09-25: axis-row root cause, exact-projection volume, LASYM basis control

**Exact projection, full measurement (local):** `verify_measurement.py --state-npz results/projection/integer_3d_vmex_ns129/seed.npz --profile full` (run `integer3d-ns129-projection-full-c2-20260925`, `measurement_diagnostic`). With no solve, B/J/force-over-grad-p: legacy96 2.42e-7 / 7.14e-6 / 8.15e-5; full Gauss 2.78e-5 / 1.66e-3 / 1.77e-2; full midpoint 3.22e-6 / 9.02e-4 / 9.61e-3; near axis 3.00e-4 / 1.23e-2 / 1.82; knots and cells 1.04e-4 / 1.14e-2 / 0.128; near edge 6.81e-7 / 1.97e-5 / 1.65e-4. The exact field projected onto the NS129 representation fails the volume J and force gates the same way the zero-TCON0 root does (the root is slightly better in volume). **The C2 volume failures are inherited from representation and evaluation, not produced by the solver.**

**Root cause (source-confirmed):** `vmex/core/extender.py::_radial_table` (pin b5f5267; unchanged on main 926892ab, line 492) regularizes coefficients as c/s^{|m|/2} and sets the m>0 axis row with `regular[0] = regular[1]`. This makes the axis value first order in Delta s and leaves the first cell with a near-zero s-slope, an O(1) error in R_s and Z_s, hence in the native Jacobian and current, independent of NS. Reproducer `benchmarks/axis_projection_ladder.py --axis-row {pinned_copy,linear_extrapolation}` (runs `projection-ladder-pinned_copy-20260925T064003Z`, `projection-ladder-linear_extrapolation-20260925T064032Z`; exact projection, no solve, monkeypatched in the benchmark only). Axis B 1.41e-3 / 7.01e-4 / 3.50e-4 / 1.75e-4 becomes 1.46e-5 / 3.63e-6 / 9.04e-7 / 2.26e-7 (NS 33/65/129/257, second order). J at s=1e-4 1.07e-2 / 1.02e-2 / 9.92e-3 / 9.65e-3 becomes 9.62e-5 / 4.60e-5 / 2.25e-5 / 1.09e-5 (now converging). The s=0.5 control is unchanged (B 5.77e-9, J 3.5e-7..2.8e-7; the curl index order "ij" is selected by that control). Two earlier attempts of this script failed (flattened derivative tensor; a reused name) and keep failure receipts.

**Upstream fix, prepared, not merged:** branch `fix/extender-axis-row-extrapolation` on uwplasma/vmex, commit `afd1392` on top of main 926892ab: `axis = 2 r[1] - r[2]` for ns > 2 (copy kept for ns = 2) plus `tests/test_mgrid.py::test_radial_parity_axis_row_is_extrapolated_not_copied` (exact s^{m/2}(a+bs) table). The test fails on main (value error 5.0e-5) and passes with the fix. `tests/test_native_interior_form.py` and `tests/test_mgrid.py` pass (23 passed, 5 skipped, 287 s); optimize, problem, near-surface and virtual-casing tests are running on office. The PR will be opened only after those pass and preflight/diff-cover are checked. It is a separate narrow change; the benchmark keeps using the historical pin.

**LASYM basis control (office):** `python benchmarks/lasym_basis_control.py` on `run_vmex.py` solves of `input.integer_axisymmetric_current` (LASYM=F) and `input.integer_axisymmetric_full_basis_current` (LASYM=T), FTOL 1e-12, NS 33 and 65 (runs `lasym-control-*-ns{33,65}-20260925`). Identical iteration counts (400 and 474). Symmetric blocks agree to max |diff| 2.9e-15 (rmnc), 4.7e-16 (zmns) and 1.1e-13 (lmns); asymmetric partners rmns/zmnc/lmnc stay at <=2.0e-14. **Pass: the full basis reproduces the symmetric solution to round-off.** The scores differ only because the evaluators differ (NS65 native Clebsch B/J/force 8.2e-6 / 2.48e-3 / 2.74e-2 vs LASYM fitted-state route 4.6e-6 / 5.6e-4 / 5.8e-3 on the same state). That independently corroborates the native axis-row defect and is not a LASYM physics difference. The genuinely asymmetric Solov'ev case is next.

### C4 addendum, 2026-09-25: matched input tangent (office)

`python benchmarks/axisymmetric_root_response.py --ns 129 --fd-steps 1e-4 3e-5 1e-5 --frozen-steps 1e-4 1e-5 --tangent-step 1e-5` (new `--tangent-step`; run `axisym-root-response-20260925T065107.516249Z`, office, pinned VMEX, CPU). The first attempt passed a single frozen step and was rejected by argument validation before creating a run directory; a second launch never started because an office `git pull` aborted on untracked copies of committed files (all 44 later verified byte-identical and moved to a backup). Base root 1.9e-13; tangent linear residuals 1.9e-13 and 1.2e-11; transpose duality 4.2e-11 (c) and 1.6e-9 (delta); derivative gate open. c: JVP relative error 1.636e-5, FD-minus-JVP 5.4e-9, 7.7e-9, 6.6e-9 at h = 1e-4, 3e-5, 1e-5. **delta: JVP E_B,delta = 1.57e-6** (previously 4.69e-5 with the h=3e-4 input tangent); branch FD 1.32e-6, 1.52e-6, 1.55e-6; FD-minus-JVP 7.7e-5, 1.0e-5, 6.8e-6 (raw metric), now converging with h. **Both the nonzero c and the null delta axisymmetric responses meet every stated gate at NS129 on the driver's 4 pointwise observables** (root, linear, transpose, FD window, 1e-3 and 1e-5 targets). Limits: pointwise observables, not a volume norm; volume response norms inherit the axis-row evaluator defect until it is fixed or the axis cell is reported separately.

### C3 addendum, 2026-09-25: why sheared-A solves leave the accurate seed

**NS65 default-TCON0 solve from the full-flux seed (local):** `BENCH_FTOL=1e-12 run_vmex.py inputs/input.sheared_A_current 65 20000 <seed_ns65>` (run `sheared-a-ns65-full-flux-seed-default-20260925`): capped at 20000 iterations, IER 2, FSQ ~1.8e-9, 673 s. B/J/grad-p/force 4.71e-4 / 1.08e-2 / 5.80e-3 / 5.04e-2, flux-label error 4.05e-3, against the seed's 1.06e-5 / 6.8e-5 / 3.1e-6 / 3.3e-4 and 1.9e-6. Not converged; drifts away from the exact field (same pattern as NS17).

**Newton from the exact seed (office):** `observe_root_refinement.py --no-host-solve --passes 4 --saved-state <seed_ns65>` (run `sheared-a-ns65-seed-newton-20260925T065342.030882Z`; the first attempt crashed on NumPy leaves in the pinned 3-D projector and keeps a failure receipt). Preconditioned residual 3.42e-2, mostly R_cos 2.8e-2 and Z_sin 2.0e-2 (lambda 9.4e-5). The first full block Newton step solves its linear system to 3e-11 (slope -1.000) but has norm 3.4 and raises the residual to 4.8e+8. The best trial is a GCROT step (2.08e-2); pass 2 finds no improvement. No certified root is reached near the projection.

**Residual ladder (`benchmarks/seed_residual_ladder.py`, no solve):** sheared-A seeds NS17/33/65 give preconditioned residuals 9.9e-3, 1.71e-2, 3.42e-2 (raw 4.7e-2, 8.2e-2, 1.65e-1). The integer-3D exact projections NS129/257 give 6.7e-2 and 1.97e-1 (raw 2.1, 6.3), yet the solver converges from those seeds in 39-50 iterations. The unnormalized residual norm grows with NS for both families, so it is not a distance to the root, and a seed-conversion defect (flat residual) is ruled out. **Next:** Newton from the integer NS129 projection as a control (running on office). If it certifies while sheared-A does not, the sheared difficulty is intrinsic to that equilibrium (for example rational surfaces or a near-singular Jacobian); then test a damped/continuation path, not more iterations.
