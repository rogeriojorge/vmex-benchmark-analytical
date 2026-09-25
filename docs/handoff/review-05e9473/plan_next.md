# VMEX analytical benchmarks: continuation after 05e9473

**Owner:** rogeriojorge  
**Repository:** https://github.com/rogeriojorge/vmex-benchmark-analytical  
**Reviewed benchmark:** `05e947370b9d0cebc0e508361e447b03523a331d`  
**Previous review:** `d5484d1e7a15c9cf10599c2b0f05bdf69e22e861`  
**Review date:** September 24, 2026, America/Chicago; GitHub receipts include September 25 UTC.

This is the next working contract, not a new project. Preserve the code, exact references, numerical states, amended historical records and the complete prior logbook. Archive the active plan byte-for-byte before adopting this one. Keep one active plan, with links to earlier plans and this review. Source labels S1-S12 and literature labels L1-L14 refer to [REFERENCES.md](REFERENCES.md). The concrete review findings are in [REVIEW_FINDINGS.md](REVIEW_FINDINGS.md).

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
