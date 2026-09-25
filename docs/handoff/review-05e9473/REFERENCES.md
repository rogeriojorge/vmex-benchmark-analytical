# Sources and methods for the 05e9473 continuation

Reviewed September 24, 2026, America/Chicago. Some GitHub records use September 25 UTC. Source claims below describe the stated snapshot, not every release or branch. The review used the connected GitHub reader; network cloning was unavailable in this environment. Large source files were inspected in selected ranges, not certified by a whole-tree semantic audit.

## Repository evidence

**S1. Benchmark snapshot and change set.**
https://github.com/rogeriojorge/vmex-benchmark-analytical/tree/05e947370b9d0cebc0e508361e447b03523a331d

https://github.com/rogeriojorge/vmex-benchmark-analytical/compare/d5484d1e7a15c9cf10599c2b0f05bdf69e22e861...05e947370b9d0cebc0e508361e447b03523a331d

Six commits advance the preceding review. The active plan's current task table and final work blocks supersede its historical adoption narrative. Read `plan.md`, `README.md`, and the numerical records together. A documentation-only consolidation is not another experiment.

**S2. Current plan and chronology.**
https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/05e947370b9d0cebc0e508361e447b03523a331d/plan.md

Inspected opening specification, current table around lines 460-545, and late work blocks around 690-830. These establish the completed NS129 zero-constraint experiment, partial measurement repair, actual response work, sheared lambda projection, failed radial-relabel interpretation, and uncertified zero-constraint roots. The earlier derivations and logbooks remain archived in the repository.

**S3. Root-response implementation.**
https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/05e947370b9d0cebc0e508361e447b03523a331d/benchmarks/axisymmetric_root_response.py

Inspected lines 1-525: physical input map, signed tangents, root helper, field evaluation, linear-path finite differences, transpose pairing, and reconverged branch tests. Two concrete concerns are the repeated cached anchor and dimensional/sample-count normalization of responses.

**S4. Linear and physical-response discriminators.**
https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/05e947370b9d0cebc0e508361e447b03523a331d/benchmarks/certify_dense_axisymmetric_response.py

https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/05e947370b9d0cebc0e508361e447b03523a331d/benchmarks/decompose_axisymmetric_branch_jvp.py

The dense solve checks another linear algebra implementation against the same residual derivative. It is not an independent continuum oracle. The component JVP decomposition correctly separates direct input changes from geometry/lambda changes; the next missing discriminator is the branch tangent's defect in one frozen residual.

**S5. Sheared straight-field projection.**
https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/05e947370b9d0cebc0e508361e447b03523a331d/benchmarks/project_sheared_straight_field.py

Inspected lines 1-225. The code solves the variable-coefficient field-line equation, using cylindrical R/Z derivatives and an augmented zero-mean gauge. This is a substantial improvement over lambda=0. Its transport operator hard-codes NFP=2, which is valid for the current experiment but not a general API. Whole-volume radial regularity, full field amplitude, and the axis row require independent checks.

**S6. Measurement implementation.**
https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/05e947370b9d0cebc0e508361e447b03523a331d/benchmarks/measurement.py

Inspected lines 1-200. Actual-knot crossing support and bounded sample retention now exist. `composite_spread` still recognizes orders 2/4 with the old absolute gate and a uniform-cell description; do not mistake this helper for a completed physically aligned volume certificate.

**S7. Root-status amendment.**
https://github.com/rogeriojorge/vmex-benchmark-analytical/blob/05e947370b9d0cebc0e508361e447b03523a331d/results/audit/axisym_branch_tcon0/axisym-tcon0-branch-20260925T010824.995851Z/root_status_amendment.json

The saved zero-TCON roots miss the 1e-11 residual tolerance. The separate amendment leaves raw observations intact. It also records that the checked-in probe does not have the hash of the run-time script; exact original script bytes are missing.

**S8. Current successful reference CI.**
https://github.com/rogeriojorge/vmex-benchmark-analytical/actions/runs/36086434493

At commit 05e9473 the reference job completed successfully, including requirements installation, pytest, reference identities, and reference derivatives. The repository reports 55 reference tests. This is no longer the failing CI from the preceding review; it is not evidence that solver-response tests ran in that job.

**S9. Historical VMEX implicit implementation.**
https://github.com/uwplasma/vmex/blob/b5f5267efc0795c4a49a224e321e9b370975c14c/vmex/core/implicit.py

Inspected lines 1360-1520 and 1600-1910, plus the auxiliary-solve entry point. `_host_solve_and_mask_impl` already calls `_refine_fixed_point`; the refinement memo is keyed by configuration and parameter bytes. `_refined_state` may return without meeting `refine_tol`. Status acceptance based on the original host solve is not a certificate of the anchored residual. These statements concern the observed implementation, not a claim of an untested failure in every production call.

**S10. Separate current VMEX snapshot.**
https://github.com/uwplasma/vmex/tree/926892ab7131a6bc0c5b61218d1f75e7b77bc401

Keep this separate from the historical baseline. Before rerunning, diff the reached residual, mask, field and refinement functions, plus their tests. No current-main solver run was performed in this review.

**S11. Force-recovery research PR448.**
https://github.com/uwplasma/vmex/pull/448

Reviewed branch head: `131580da529a5e7c9f9db31d6fc2361b225e9c24` on `rj/force-balance-recovery`. Open draft, unmerged. R6 describes an independently checked force target for one axisymmetric fixed-profile state, but projected stationarity remains above its unchanged gate. Its `epsilon_B` is a force normalization, not analytical magnetic-field error. Reuse isolated ideas such as point-oracle scheduling, exact knot insertion, true KKT residuals and checkpointing; do not import its success labels into this benchmark.

**S12. SOLVAX periodic elliptic implementation.**
https://github.com/uwplasma/SOLVAX/blob/2e246a5d6093662f9b5f72c46f995cd7c4bbd479/src/solvax/elliptic.py

Inspected lines 1-130. `periodic_poisson_eigenvalues` uses the continuous Fourier symbol, not a finite-difference Laplacian. `solve_periodic_poisson_spectral` solves minus-Laplacian and removes the incompatible mean. For the proposed lambda reconstruction, check compatibility before calling it; never let mean projection conceal an invalid field/flux pair.

## Physics and numerical methods

**L1. M. Landreman, Analytic toroidal 3D MHD equilibria and steady Euler flows with invariant surfaces (2026).**
https://arxiv.org/html/2609.26742v1

Supplement: https://github.com/landreman/analytic_3d_equilibria/tree/4c0b690ddebdc71811c88223eb9f44a98ab64222

Use explicit field, surface and flux formulas as independent references. The interior equilibrium does not by itself provide exterior vacuum matching, stability, kinetic transport, or a unique differentiable numerical branch. The oscillator flow gives exact field-line and tangent-map tests. The additional diagonal stretch remains the previously derived generalization, not a newly claimed priority result.

**L2. S. P. Hirshman and J. C. Whitson, Steepest-descent moment method for three-dimensional magnetohydrodynamic equilibria, Physics of Fluids 26, 3553 (1983).**
https://doi.org/10.1063/1.864116

**L3. Variational/spectral coordinate constructions.**
https://www.sciencedirect.com/science/article/pii/0010465584900468
https://www.sciencedirect.com/science/article/pii/001046558690127X

These are historical context for VMEC's coordinate choice and spectral condensation, not an argument for keeping or removing a term regardless of measured errors. Follow each term into the actual pinned residual, identify its varied coordinate, and distinguish the physical stationarity equations from coordinate constraints.

**L4. D. Panici et al., The DESC stellarator equilibrium solver. Part 1: High-order solutions through Newton-Krylov optimization.**
https://arxiv.org/abs/2203.17173

**L5. R. Conlin et al., The DESC Stellarator Code Suite Part II: Perturbation and continuation methods.**
https://arxiv.org/abs/2203.15927

Use native high-order evaluation as an independent representation and use predictor/corrector continuation for changing analytical parameters. Match boundary approximation, physical profiles, flux, region and error norm. A perturbation prediction needs a corrected root; second/third-order perturbation capability is not automatically a certified second derivative of the VMEX wrapper.

**L6. GVEC documentation.**
https://gvec.readthedocs.io/latest/

Arbitrary-degree radial B-splines and flexible mappings provide a useful additional representation comparison. Pin the runnable code before using it; do not turn an optional third solver into a blocker for repairing an existing measurement.

**L7. Tecchiolli and collaborators, toroidal coordinate construction using an action principle.**
https://arxiv.org/abs/2405.08173

Use as a bounded alternative initial coordinate map for difficult shapes. It constructs coordinates, not the MHD solution. Require a one-to-one physical-angle chart where the selected VMEX interface needs one.

**L8. S. C. Eisenstat and H. F. Walker, Choosing the Forcing Terms in an Inexact Newton Method, SIAM Journal on Scientific Computing 17, 16-32 (1996).**
https://doi.org/10.1137/0917003

Use a measured original-operator linear residual to determine whether a proposed Newton correction is accurate enough. Avoid solving linear systems far below the nonlinear/model floor. This literature motivates a bounded diagnostic, not an automatic replacement of the current algorithm.

**L9. PETSc SNES documentation.**
https://petsc.org/main/manual/snes/
https://petsc.gitlab.io/petsc/release/manualpages/SNES/SNESConvergedReason/

Borrow explicit convergence reasons, true function/step checks, line-search diagnostics and Jacobian verification. A step-tolerance exit or a returned state must not be relabelled as a small-function-norm root. PETSc is optional as a small independent control, not a new mandatory dependency.

**L10. JAX custom linear solve contract.**
https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html

The implicit derivative assumes the supplied solve satisfies its linear equation; JAX does not establish this invariant. Keep primal, transpose and original-equation residual certificates outside the AD wrapper. Probe transformations individually rather than assuming custom VJP implies forward-mode support.

**L11. booz_xform theory and implementation.**
https://hiddensymmetries.github.io/booz_xform/theory.html

Its theory distinguishes straight-field-line lambda from the further Boozer toroidal shift, and derives a potential from both covariant components. The proposed flux-density lambda construction is the analogous elementary periodic-potential calculation for contravariant flux densities; it is not itself a Boozer transform. Derive sign conventions directly rather than copying names or isolated equations, including apparent typographical component/index errors in rendered documentation.

**L12. R. Becker and R. Rannacher, An optimal control approach to a posteriori error estimation in finite element methods, Acta Numerica 10 (2001).**
https://doi.org/10.1017/S0962492901000010

Use dual-weighted residual ideas to estimate which residual components affect a physical observable, and calibrate predictions against actual refined results. The finite-element theory does not automatically prove an error bound for VMEX's nonstandard representation and constraints. Linearized estimates in the plan are labelled estimates.

**L13. S. P. Hirshman, W. I. van Rij and P. Merkel, Three-dimensional free boundary calculations using a spectral Green's function method, Computer Physics Communications 43, 143-155 (1986).**
https://doi.org/10.1016/0010-4655(86)90058-5

Read alongside NESTOR's actual source, source-field circulation constraints and boundary stress condition. Validate analytic vacuum operators before coupled equilibria. An exact interior is not proof of a realizable coil exterior.

**L14. R. Jorge, W. Sengupta and M. Landreman, Near-axis expansion of stellarator equilibrium at arbitrary order in the distance to the axis.**
https://arxiv.org/abs/1911.02659

Useful for unrestricted local field/axis-jet and asymptotic-order comparisons. Do not force a generic non-QS exact equilibrium into a restricted QS near-axis ansatz.

## Scope of the renewed literature review

The new implementation choices are primarily L8-L11: measure the actual nonlinear correction, test one well-defined residual, and recover the full magnetic coordinate potential rather than only field-line slopes. L1-L7 retain the established physical and cross-code context. Other kinetic and coil libraries remain scoped integrations from the prior plan. This review did not audit their entire current source trees or run their solvers.
