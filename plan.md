# Analytical benchmarks for VMEX

**Owner:** `rogeriojorge`  
**Target repository:** `https://github.com/rogeriojorge/vmex-benchmark-analytical` (public)  
**Plan revision:** 1, 2026-09-23  
**Primary solver snapshot:** `uwplasma/vmex@b5f5267efc0795c4a49a224e321e9b370975c14c`  
**Analytical supplement:** `landreman/analytic_3d_equilibria@4c0b690ddebdc71811c88223eb9f44a98ab64222`

This file is the implementation contract and the continuing logbook. Keep the scientific specification stable; append dated evidence and decisions to the logbook. A replacement agent must be able to resume from this file, the Git history, and the recorded artifacts without the original conversation.

## 1. Mandate and present state

Build a small, reproducible benchmark repository that answers four questions:

1. Does VMEX recover independently specified exact equilibria, rather than merely agree with VMEC's discrete residual or another implementation?
2. Do its physical fields, spatial derivatives, equilibrium sensitivities, diagnostics, and file interfaces converge to the correct quantities?
3. Which fixed-boundary, free-boundary, symmetry, and downstream capabilities have direct analytical evidence, and which have only numerical or consistency evidence?
4. Can a verified differentiable solver explore exact solution families and nearby equilibria without optimizing discretization error?

The starting references are Landreman's integer-transform and sheared-transform toroidal equilibria [R1, R2]. Add axisymmetric limits, an additional diagonal stretch, a genuinely up-down-asymmetric Solov'ev equilibrium, rigid-motion covariance tests, vacuum operator fixtures, and separate open-mirror fixtures. Carry the same fields through as much of VMEX as each model legitimately permits.

**Initial handoff contents:** executable analytical field, geometry, flux and current calculations; 14 reference configurations; 28 generated fixed-boundary input candidates; independent identity, quadrature, sign, symmetry and derivative tests; a physical-sample scorer; reference figures; and two VMEX smoke runners. Subsequent dated entries below record VMEX and DESC source pins, actual solver runs, failures, saved numerical artifacts and figures. Do not conflate the initial handoff with the later measured work.

**Do not call this a completed all-module source audit.** The preparation reviewed selected VMEX source paths, current changes and adjacent interfaces. `docs/SOURCE_REVIEW.md` states the scope. Phase P0 must inventory the local trees and complete the source-to-test review ledger. Reading a README, parsing an AST, seeing a green CI badge, and semantically reviewing the corresponding implementation are different actions.

The existing measured evidence is in `results/reference/`. Sampled agreement with a differential identity is not a proof over the continuum; the derivations below explain why the identities should hold. Likewise, an analytical geometry is not an analytical solution of every kinetic or stability model consuming it.

### State vocabulary

Every result must use one of these evidence classes:

| Class | Meaning |
|---|---|
| `analytic_reference_sampled` | Explicit reference equations checked on stated points/quadrature; no numerical equilibrium solve. |
| `analytic_projection` | Exact state projected into a numerical representation, without nonlinear recovery. |
| `analytic_recovery` | A solved equilibrium compared with an independent exact field. |
| `discrete_consistency` | Two evaluations of the same discrete problem, including adjoint versus discrete finite differences. |
| `independent_numerical_reference` | Native independently refined solver or integral implementation, not a closed-form answer. |
| `integration` | An interface or downstream calculation checked with explicitly limited physical scope. |
| `exploratory` | An optimization or continuation result requiring subsequent validation. |

Use statuses `planned`, `implemented_not_run`, `passed`, `failed`, `blocked`, `not_applicable`, and `unavailable`. A missing optional dependency is `unavailable`, not `passed`. A solver stopping successfully is not automatically `analytic_recovery`. Record unsuccessful attempts and their cost.

## 2. Operating rules

### Repository and identity

Create the public repository under the owner's account. `tools/publish.sh` stages an explicit allowlist and requires `PUBLISH=1` before it commits or pushes. Read its staged diff first. Confirm `gh api user --jq .login` is exactly `rogeriojorge`; derive the account's noreply email from its actual numeric ID, or use a verified owner-approved email. Configure both author and committer locally as `rogeriojorge`. All new project commits, pushes, comments, issues and pull requests must use that authenticated account. Do not add automated-assistant author names or co-author trailers.

This does not authorize rewriting upstream history or erasing someone else's copyright or scientific attribution. Keep third-party licenses and references. Do not import upstream Git histories into this new repository. Do not configure the user's global Git identity. Do not publish credentials, local paths revealing private information, personal files, caches, virtual environments or unreviewed terminal dumps. CI runs tests with read-only repository permissions and must not commit or comment automatically.

For VMEX or adjacent-library fixes, use separate worktrees and narrowly scoped branches. Open separate upstream PRs with reproducer, tests, measurements and pinned dependencies. Never merge those PRs or push directly to an upstream default branch as part of this task. Benchmark both the recorded baseline and the proposed fix before describing an improvement. An unrelated open PR is not an invitation to merge it.

### Implementation style

Use plain functions, a small amount of shared analytical code, JSON metadata, NumPy/SciPy/JAX, pytest, and Matplotlib. Reuse VMEX's public interfaces and existing operators. Do not build a plugin system, workflow framework, dashboard server, universal configuration language, or a second equilibrium solver.

Keep driver settings near the top of readable scripts. One script should correspond to one scientific experiment or a genuinely shared operation. Prefer one shared scorer over copies of force norms. Add a class only where it represents a real state or contract. Do not pursue a line-count target by compressing expressions, removing validation, or hiding physics in opaque helpers. A useful target is a small handful of analytical/adapter modules and approximately one driver per phase, not one file per test or every matrix cell.

Do not add all optional dependencies to the core environment. Pin the actual solver and reference versions in the result manifest. Keep DESC/reference-binary environments separate when their JAX or Python requirements conflict with VMEX. Transfer numerical data through documented file contracts rather than forcing incompatible environments together.

## 3. Source baseline and audit priorities

Before changing code, compare local VMEX with the pinned baseline. Keep the baseline immutable in a worktree. Obtain current heads, open and recently merged PRs, change logs, capability declarations, test manifests and benchmark records; record actual SHAs. Freeze a second comparison snapshot if main has advanced. Never silently replace the baseline midway through a convergence plot.

The source review identified the following concrete priorities [R3-R10].

* The current `freeboundary_implicit.py` implements a coupled Newton anchor, finite restart budgets and an anchor-failure status. An older paragraph in `docs/explanation/validation.md` still says the free-boundary state is unanchored. Test the implementation; do not reproduce the outdated limitation as a current fact. Conversely, the existence of an anchor routine does not prove that every return path enforces its residual certificate.
* VMEC's `FTOL` is a test on squared, normalized discrete force quantities. It is not a bound on Cartesian force error, state error, or derivative error. Its three convergence components are individually tested; a sum can be as large as roughly three times the per-component tolerance. Do not invent a conflicting sum-only convergence condition.
* The fixed-boundary implicit map contains constrained/frozen coordinate combinations. Frozen-path finite differences test that discrete map; independently reconverged physical observables test a different and necessary part of the argument. Both are required.
* Current field interpolation uses the native Clebsch representation where its required spectra are present. Older/fallback field reconstructions have different derivative accuracy. Record which path is actually evaluated. Earlier live-state versus WOUT errors motivate testing all interior surfaces, not just the last surface.
* `strong_force.py` uses a continuous spline/Fourier representation, distinct from the legacy half-mesh residual. Its coordinates use a field-period angle: physical cylindrical angle is that angle divided by NFP. Its lambda is the external straight-field-line displacement, not the internally rescaled solver unknown.
* Public polishing is an overdetermined physical least-squares problem. Small force, small least-squares stationarity residual, and accurate derivatives are separate gates. A Gauss-Newton matrix is not automatically the exact derivative of a nonzero-residual least-squares solution.
* The free-boundary reverse API is not evidence of a public forward-mode API. Probe transformations individually. `custom_vjp` entry points cannot simply be assumed to accept `jax.jvp`, `jacfwd`, or arbitrary higher derivatives.
* Current near-surface exterior evaluation uses graded quadrature; a removed continuation API still appears in some adjacent documentation. Freeze discrete quadrature plans when differentiating, and independently check whether the frozen plan remains adequate after geometry changes.
* `pyQSC_JAX`'s substantial implementation is in draft PR 2 at the pin in `sources.json`, not in its minimal main README. Treat it as an optional branch experiment, not a released prerequisite.

`tools/audit_sources.py /path/to/checkouts` creates a full tracked-source inventory without marking files reviewed. Resolve the null pins in `sources.json` first. For each VMEX module, and each imported adjacent-library implementation relevant to a scored capability, record: exact file hash; model and units; input/output contract; derivative semantics; existing tests and their oracle; uncovered branch; benchmark fixture; reviewer notes; and completion status. Review associated tests, not only their names. Include error/status code paths and serializers. Nonreachable adjacent modules may be explicitly out of scope with a reason; do not claim to benchmark the entire uwplasma organization.

## 4. Case matrix: geometry, symmetry, closure and boundary conditions

### Fixed-boundary matrix

| Case group | Physical class | LASYM | Independent target | Purpose |
|---|---|---|---|---|
| Integer family, a=b=c=1 | Axisymmetric | F and T | Exact field, iota, current, pressure, volume, energies | Basic recovery and same-physics basis control. |
| Sheared family, epsilon=0 | Axisymmetric with shear | F, then T | Exact field plus flux quadrature | Profile inversion and current-prescribed transform. |
| Solov'ev, chi=0 | Axisymmetric, up-down symmetric | F and T | Separate Grad-Shafranov formula | Independent analytical construction. |
| Solov'ev, chi=0.3 | Axisymmetric, up-down asymmetric | T | Exact asymmetric pressure/field | Genuine LASYM physics, not just moving a symmetry plane. |
| Integer family, a!=b | Three dimensional | F and T | Exact Cartesian field and nested tori | 3-D nonlinear recovery and rational-transform response. |
| Sheared A | Three dimensional, sheared iota | F and T | Exact field, independent flux/current quadratures | Primary 3-D sensitivity and diagnostics case. |
| Sheared B and C | Stronger shaping | F; selected T repeats | Same references | Angular conditioning and representation stress. |
| Stretched integer family, c!=1 | Three dimensional | F and T | Derived exact family | Extra shape and pressure responses. |
| Rigidly rephased/shifted exact cases | Same physical equilibrium, hidden symmetry | T | Euclidean covariance | Nonzero sine/cosine partners and coordinate-origin invariance. |
| Generic transverse perturbations | Genuinely symmetry-broken 3-D equilibria | T | Independently converged numerical solutions | Extension beyond exact families; never label these exact. |

For each mandatory mild case, solve with both `NCURR=0` and `NCURR=1` after converting the same physical profiles correctly. Running a symmetric geometry with LASYM=T must recover the same physical state; this is distinct from testing genuinely asymmetric geometry. Do not count the duplicated basis control as an additional physical configuration.

The initial JSON includes 14 reference cases. Expand the complete study through shared configuration records, not duplicated scripts. Initial M=12, N=12 boundary candidates for sheared B and C are unresolved; the forward smoke runner refuses their current manifests. Refine angular resolution and quadrature before launching them. The initial mild inputs are smoke inputs, not final accuracy specifications.

### What can and cannot be called exact for free boundary

Both axisymmetric and 3-D free-boundary calculations, with LASYM=F and T, belong in the program. However, the paper gives an interior MHD equilibrium, not a complete coil-vacuum-plasma free-boundary solution. Setting edge pressure to zero or feeding its boundary into NESTOR does not supply that missing solution.

Use three distinct free-boundary tiers:

1. **Exact operator fixtures:** analytic vacuum fields and harmonic potentials for NESTOR, MGRID interpolation, coil kernels, source separation and their derivatives on prescribed surfaces.
2. **Independent numerical coupled equilibria:** established axisymmetric and 3-D coil/MGRID cases, with full-basis controls and truly asymmetric geometry/coils where supported; compare independently reconverged roots and native fields.
3. **Analytical-interior target with fitted exterior:** fit an admissible external source to an exact interior target, solve free boundary, and separately converge exterior representation and plasma response. This is an approximate coupled benchmark unless the entire matching problem has an independently established exact solution.

Never populate an imaginary exact-reference entry for every Cartesian product of flags. A pure toroidal vacuum field is useful for field reconstruction, but does not select a unique nested plasma boundary. Its boundary degeneracy makes it a poor test of an invertible free-boundary adjoint.

## 5. Equations and independent references

### 5.1 Units, signs and spatial comparison

Analytical code uses dimensionless coordinates and mu0=1. For dimensional length L and field B0,

$$
\mathbf r=L\bar{\mathbf r},\quad \mathbf B=B_0\bar{\mathbf B},\quad
p=\frac{B_0^2}{\mu_0}\bar p,\quad
\mathbf J=\frac{B_0}{\mu_0L}\bar\nabla\times\bar{\mathbf B},\quad
\Phi_t=B_0L^2\bar\Phi_t.
$$

Record the numerical value of mu0 rather than mixing constants packages. The provided input generator uses `4*pi*1e-7`. Pressure is in Pa, current in A, flux in Wb, and positions in m in VMEX inputs. The dimensionless current obtained from an Ampere integral scales as B0 L / mu0.

The provided surface parameter increases counterclockwise in an R-Z section. The exact fields in these test conventions have negative signed transform. The paper/supplement uses a clockwise convention giving positive transform. The existing DESC-to-VMEX interface also reverses poloidal orientation [R2, R3]. Validate signs by field components and oriented flux integrals, not by comparing absolute iota alone.

On an R-Z loop increasing counterclockwise, the oriented normal is -e_phi. Thus the positive toroidal current is minus the loop integral of B dot dl divided by mu0. Toroidal flux is the positive-e_phi flux through that section. These conventions are explicitly tested in the reference code.

Always evaluate reference and numerical vectors at identical Cartesian points. Do not compare mode coefficients until radial labels, angle gauges, NFP factors and normalization have been matched. Record whether a tensor stores component or derivative direction first. VMEX `gradB` has entries dB_i/dx_j, whereas its SIMSOPT-compatible `dB_by_dX` transposes those axes [R7].

### 5.2 Integer-transform family and diagonal-stretch extension

Let a,b,c>0 and define

$$
q=(x/a)^2+(y/b)^2,\qquad f=\sqrt{2q-q^2-4(z/c)^2},
$$

$$
\mathbf B=\left(
\frac{2zx/c-(a/b)fy}{q},
\frac{2zy/c+(b/a)fx}{q},
c(1-q)\right).
$$

Define

$$
u_a=-\frac{a^2-b^2}{4c^2},\qquad
H_a=\frac{a^2+b^2}{2}-\frac{(a^2-b^2)^2}{8c^2},
$$

$$
H=\frac{x^2+y^2+4z^2+B^2}{2},\qquad
\psi=\frac{H-H_a}{2c^2},\qquad p=2c^2(\delta-\psi).
$$

Here `u_a` is the field-line-label center, not a radial variable. Require

$$ |u_a|+\sqrt{\delta}<\frac12. $$

The paper's displayed family has a=sqrt(1+epsilon), b=sqrt(1-epsilon), c=1. The pressure-boundary extension releases c. It is not obtained by stretching an old pressure boundary while holding its label center fixed.

A global chart uses field-line labels u,v and parameter t:

$$
\ell=\sqrt{\frac{1+\sqrt{1-4(u^2+v^2)}}2},
$$
$$
\mathbf r(u,v,t)=\left(
 a\left[\ell\cos t+\frac{u\cos t+v\sin t}{\ell}\right],
 b\left[\ell\sin t+\frac{v\cos t-u\sin t}{\ell}\right],
 c[v\cos2t-u\sin2t]\right).
$$

Its Jacobian is -abc, B=partial_t r, and psi=(u-u_a)^2+v^2. Use the chart to cross-check the Cartesian label, vector field, closed-line winding, surface regularity and volume. The physical cylindrical angle is not generally t. `surface()` performs the exact integer-family conversion to physical phi.

The extension follows from the axisymmetric seed's tension identity

$$ (\mathbf B_0\cdot\nabla)\mathbf B_0=-D\mathbf r,\quad D=\mathrm{diag}(1,1,4). $$

For constant A=diag(a,b,c), set B_A(r)=A B_0(A^{-1}r). Divergence is preserved and ADA^{-1}=D. The vector identity

$$ (\nabla\times\mathbf B)\times\mathbf B=(\mathbf B\cdot\nabla)\mathbf B-\nabla(B^2/2) $$

gives the pressure above. More generally, a constant linear deformation supplies a scalar tension potential only if ADA^{-1} is symmetric, equivalently [A^T A,D]=0. Arbitrary shear is not a legitimate MHD-solution generator. Orthogonal transformations preserve equilibrium but do not necessarily produce new intrinsic symmetry classes.

Exact scalar targets are

$$
V=2\pi^2abc\delta,\quad \Phi_t=\pi abc\delta,\quad
\langle B^2\rangle_V=H_a+c^2\delta,\quad
\langle p\rangle_V=c^2\delta,\quad
\beta_V=\frac{2c^2\delta}{H_a+c^2\delta}.
$$

Since s=Phi_t(psi)/Phi_t(delta)=psi/delta, the VMEX pressure is linear in normalized toroidal flux. The physical field is independent of the chosen outer label delta at fixed Cartesian position. This gives a nontrivial null test of the complete boundary/profile/flux sensitivity chain.

For the original family write D0=1-epsilon^2/2+delta. Then

$$
\partial_\epsilon\beta_V=\frac{2\delta\epsilon}{D_0^2},\qquad
\partial_\delta\beta_V=\frac{2(1-\epsilon^2/2)}{D_0^2}.
$$

At epsilon=0.5, delta=1/64, beta=2/57. Treat the additional stretch as a derivation to verify and compare with the literature, not an asserted priority claim.

### 5.3 Sheared-transform family

Use epsilon>=0, lambda>0, 0<k_b<1, delta=k_b^2/2, and S>asin(k_b). Set w=x+i y and

$$
K=\bar w\sqrt{1+\epsilon/\bar w^2},\quad
\Xi=wK+\frac\pi2-S,
$$
$$
B_x+iB_y=\frac{i e^{-i\lambda z}\sin\Xi}{2K},\qquad
B_z=\frac{\mathrm{Re}(e^{-i\lambda z}\cos\Xi)}{\lambda},
$$
$$
\psi=\frac{\sin^2(\lambda z)+(\lambda B_z)^2}{2},\qquad
p=\frac{\delta-\psi}{\lambda^2}.
$$

Keep the prescribed square-root branch. Replacing K with a superficially equivalent principal square root of a different complex expression can change the field. Real/imaginary extraction makes ordinary complex-step differentiation inappropriate here. The supplied tests compare real central differences with JAX real-coordinate derivatives.

The explicit surface chart in `analytic.py` follows [R1, R2]. Its auxiliary t is not cylindrical phi. Bisection in the supplied NumPy surface sampler is suitable for generating inputs, but is **not** the differentiable input map for P4. If g(t,a)=phi(t,a)-phi_target=0, differentiate using

$$ \partial_a t=-\frac{\partial_a g}{\partial_t g}, $$

with a certified nonzero denominator and unique branch. Implement a small custom derivative or use an existing implicit scalar-root primitive. Do not differentiate Boolean bisection decisions and call the resulting near-zero derivative correct. Test the reconstructed position and its derivative against independent physical-phi samples. Reject geometries that lose a single-valued cylindrical chart or monotone angle map.

Let Q(k) be enclosed toroidal flux, A(k) the consistently oriented poloidal flux. Reference quadratures give iota=A'(k)/Q'(k) in the paper's orientation. `shear_flux_rates` provides Q'(k)/k and A'(k)/k, regular at k=0. Compute

$$ s=Q(k)/Q(k_b),\qquad \psi(s)=k(s)^2/2. $$

Never replace this with s=k^2/k_b^2. Fit pressure and current in s after inversion. For differentiable inversion use u=k^2 and mathcal Q(u,a)=Q(sqrt(u),a), so

$$
\left.\frac{\partial u}{\partial a}\right|_s=
\frac{s\,dQ(k_b,a)/da-\partial_a\mathcal Q(u,a)}{\partial_u\mathcal Q(u,a)}.
$$

The total derivative of edge flux includes motion of k_b. This formulation avoids a spurious axis 0/0 from Q'(0)=0. A profile fit also has parameter dependence; use fixed nodes and degree within a derivative experiment, and differentiate the fit or its converged linear coefficients. Changing the degree is a discrete outer decision.

Primary 3-D case A: epsilon=1.08, S=3, lambda=3.5, k_b=0.70. B and C are angular-resolution stress cases, not the first performance targets.

At fixed epsilon,S,delta, the quadratures imply Q proportional to lambda^{-1}, so normalized toroidal flux and iota(s) are independent of lambda. At the same time V scales as lambda^{-1}, and p(s) as lambda^{-2}. Test

$$
\partial_\lambda\iota(s)=0,\quad
\partial_\lambda V=-V/\lambda,\quad
\partial_\lambda\Phi_t=-\Phi_t/\lambda,\quad
\partial_\lambda p(s)=-2p(s)/\lambda.
$$

The iota test is scientifically meaningful in a **current-prescribed** solve with the analytically changing enclosed-current profile. Prescribing iota and recovering it is largely an input check.

### 5.4 Exact up-down-asymmetric Solov'ev family

Let U=R^2-R0^2, b,g>0 and |chi|<1. Define

$$
\psi=bU^2+gZ^2+2\chi\sqrt{bg}\,UZ,
$$
$$
F^2(\psi)=F_0^2-4g\psi,\quad
p=8b(\psi_a-\psi),\quad
(B_R,B_\phi,B_Z)=\frac1R(\psi_Z,F,-\psi_R).
$$

All quantities here use mu0=1. The axisymmetric Grad-Shafranov operator satisfies

$$
\Delta^*\psi=\psi_{RR}-\psi_R/R+\psi_{ZZ}=8bR^2+2g
=-R^2p'(\psi)-FF'(\psi).
$$

The UZ term is homogeneous under this operator. This is an elementary member of the Solov'ev solution space, not a claim of a new general Grad-Shafranov method [R11]. A symbolic test verifies the identity in the bundle. Nonzero chi produces genuine up-down asymmetry about the magnetic axis at Z=0; a vertical translation cannot eliminate the cross term.

Explicit surfaces are

$$
U=\sqrt{\frac\psi b}\frac{\cos\theta}{\sqrt{1-\chi^2}},\qquad
Z=\sqrt{\frac\psi g}\left(\sin\theta-
\frac{\chi\cos\theta}{\sqrt{1-\chi^2}}\right),\quad
R=\sqrt{R_0^2+U}.
$$

Require R0^2>sqrt(psi_a/[b(1-chi^2)]) and F0^2>4g psi_a. These ensure positive R and a regular nonzero toroidal field on the chosen domain.

The independent scalar targets include

$$
V=\frac{\pi^2\psi_a}{\sqrt{bg}\sqrt{1-\chi^2}},
$$
$$
\frac{d\Phi_t}{d\psi}=
\frac{\pi\sqrt{F_0^2-4g\psi}}
 {2\sqrt{bg}\sqrt{1-\chi^2}\sqrt{R_0^4-\psi/[b(1-\chi^2)]}},\qquad
|\iota|=\frac{2\pi}{d\Phi_t/d\psi}.
$$

Again, generally s is not psi/psi_a. The bundle checks flux using a separate cross-section integral and volume using a boundary integral. For R0=1,b=1/4,g=1,F0=1,chi=0, the field agrees pointwise with the integer family's axisymmetric member. The chi=0.3 counterpart exercises nonzero LASYM coefficients without relying on that same field formula.

### 5.5 Rigid-motion and scaling invariance

For proper orthogonal Q and displacement d,

$$ B'(r)=Q B(Q^T(r-d)),\qquad p'(r)=p(Q^T(r-d)). $$

A toroidal rephasing and vertical translation preserve the simple cylindrical chart and are provided. Later add a small generic tilt/translation only after validating the toroidal graph, moving to NFP=1 as needed. These configurations retain hidden physical symmetries; label them accordingly. Do not claim generic intrinsically symmetry-broken 3-D exact equilibria merely because LASYM coefficients become nonzero.

Field scaling B->a B, p->a^2 p, J->a J leaves geometry, beta and iota invariant if flux and current scale consistently. Length scaling changes volume, flux and current according to section 5.1. These supply additional value and sensitivity checks and detect normalization errors.

### 5.6 Vacuum and mirror fixtures

Use vacuum fixtures only in current-free regions that exclude their singular sources. Examples include G/R e_phi away from R=0, a displaced point dipole away from its source, analytic circular-coil on-axis fields, and gradients of harmonic scalar potentials. Derive all fields and derivative tensors independently of the implementation being tested.

For an open axisymmetric mirror, an exact polynomial vacuum field is

$$
\Phi_m=B_0 z+\alpha\left(z^3/3-z(x^2+y^2)/2\right),\qquad B=\nabla\Phi_m,
$$
$$
B_x=-\alpha xz,\quad B_y=-\alpha yz,\quad
B_z=B_0+\alpha[z^2-(x^2+y^2)/2].
$$

Both curl and divergence vanish. The flux label is

$$ \psi_m=\frac12(B_0+\alpha z^2)r^2-\frac\alpha8 r^4. $$

Restrict the domain so B_z remains positive and the selected flux surfaces are regular. Add harmonic quadrupole terms only as additional exact **field** fixtures unless a matching nested open-surface construction is independently established. Finite-beta long-thin mirror formulas test an asymptotic model; separate truncation error in a/L from discretization error. Do not map toroidal Landreman solutions into the mirror lane by changing a geometry flag.

## 6. Input construction, representation and error budget

### Profile and boundary adapters

The input adapter must supply the complete physical problem: boundary, pressure profile, total toroidal flux, and either iota(s) or enclosed toroidal current. The current `PCURR_TYPE='power_series'` interpretation uses `AC` as the shape of I'(s), with `CURTOR` setting total current [R3]. The generated inputs differentiate an independently fitted enclosed-current polynomial. Never copy an enclosed-current polynomial directly into derivative-profile coefficients.

Test power-series and cubic-spline representations of the same resolved physical profiles. Confirm axis and edge values, integrated current, zero edge pressure, and profile derivative accuracy. Keep GAMMA=0 for prescribed-pressure benchmarks. Explicitly inspect SPRES_PED, BLOAT, pressure scale, flux orientation and profile normalization rather than inheriting unrelated settings from a demonstration deck. For nonuniform APHI tests, map every physical profile through the changed radial coordinate; merely reusing old coefficient arrays changes the problem.

The input deck's Fourier mode truncation, angle-conversion error and pressure/current fit error must be converged independently of the equilibrium solve. A small geometry error alone does not bound magnetic-field error. Refine the input representation until its effect on every scored output is below one tenth of the selected output tolerance, then vary solver resolution.

Use the exact boundary at physical phi. Preserve the appropriate LASYM partners and NFP. Check the orientation/Jacobian and fit on held-out angular nodes. Serialize and re-read inputs before solving, then verify that VMEX interpreted the intended physical profiles. The bundle's candidate inputs are not proof of the internal parser's sign conventions.

### Projection before recovery

Construct exact states in two ways where practical: (a) an independent geometry/field projection, and (b) flux-coordinate geometry plus the required lambda/flux functions. Check the resulting physical B, not only R and Z. Landreman's integer-family gauge can simplify lambda in an appropriate straight-field-line angle, but a generic interpolated geometric theta cannot be assigned lambda=0 without proof. The sheared family needs a nontrivial straight-field-line map or an equivalent field-consistent reconstruction.

First score the exact projected state without iterating. This measures representation and diagnostic error. Then perturb its admissible state or start from a conventional initializer and solve the nonlinear problem. Report projection, warm/perturbed recovery and cold recovery separately. A fitted exact state that is never evolved is not a solver benchmark.

Near the axis, represent regular Fourier amplitudes with their required powers of rho. Do not obtain rho^m behavior by dividing tiny noisy coefficients by rho^m at the axis. Test lambda's internal/external scaling, constrained m=1 combinations, full versus half radial meshes, and limiting axis values. Evaluate derivative convergence away from interpolation knots and at knots separately; a C2 interpolant does not provide an everywhere classically smooth third derivative.

### Scoring

For volume quadrature weights w_i>0, define RMS_v(q)=sqrt(sum w_i |q_i|^2 / sum w_i). Score at the same physical positions:

$$
E_B=\frac{\|B_h-B_e\|_{L^2}}{\|B_e\|_{L^2}},\quad
E_J=\frac{\|J_h-J_e\|_{L^2}}{\|J_e\|_{L^2}},\quad
F_h=J_h\times B_h-\nabla p_h.
$$

Report dimensional RMS_v(F_h), RMS_v(F_h)/(B0^2/(mu0 L)), and, for finite-pressure cases, RMS_v(F_h)/RMS_v(grad p_e). Also report maximum sampled error, pressure-gradient error, analytic pressure-surface-label error, axis displacement, energy, volume, iota and total current. Maxima over samples are sampled maxima, not rigorous sup-norm bounds.

Use radial bins near-axis, bulk and edge, with the exact bin definitions in metadata. Do not hide a bad axis by silently excluding it. Use separate axis-limit checks when ordinary flux coordinates are singular. Samples used to validate a fit or polish must not be its training grid.

The bounded pointwise diagnostic 2|F|/(|JxB|+|grad p|+floor) saturates near vacuum and cannot rank such cases. Do not quote it alone. In exact zero-current or zero-derivative tests, report absolute error against a declared global scale, not relative error divided by zero.

### Initial acceptance targets, not claimed outcomes

The first mild-case recovery targets are E_B<=1e-5, E_J<=1e-3, pressure-normalized force RMS<=1e-3, and resolved nonzero scalar quantities within 1e-5 where their conditioning permits. These are starting goals for the local study, not a guarantee that the current solver achieves them. Require at least three useful resolution levels and a demonstrated separation of boundary/profile/diagnostic error from solve error. Strongly shaped cases may initially fail these targets; record the failure instead of silently loosening the metric or removing the case.

For reference identities on well-separated admissible points, use the existing 1e-11-scale dimensionless gates, independent real finite differences, symbolic identities and quadrature refinement. Roundoff-level sampled identity residuals do not imply roundoff-level solver accuracy.

For nonzero equilibrium gradients, aim first for 1e-4 relative agreement on mild cases, with a finite-difference step-size interval and derivative convergence under spatial refinement. For null responses use absolute nondimensional error and its refinement trend. The final gradient target should be justified by a posteriori linear/nonlinear residual and conditioning estimates; a universal tolerance unrelated to the smallest response singular value is not meaningful.

## 7. Phases and executable deliverables

Follow dependencies. Do not begin a large optimization campaign before the relevant recovery and gradient gates pass.

### P0. Environment, repository, source audit and inventory

**Inputs:** this bundle; local terminal; owner GitHub authentication; available hardware and solver checkouts.

1. Inspect the bundle, initialize the repository, verify identity and publish the plan with the explicit script. Record the repository URL and initial commit in the logbook. Do not confuse publishing the scaffold with finishing the study.
2. Clone or locate baseline VMEX, Landreman supplement, DESC, SOLVAX and booz_xform_jax outside the benchmark repository. Resolve all used pins. Record clean/dirty status and any patch hashes. Install a separate benchmark environment and write exact dependency, Python, compiler, BLAS, OS, CPU/GPU, thread and device metadata.
3. Run the supplied reference tests, scripts and figures. Compare measured output with the handoff records. Variations at roundoff are acceptable; do not use literal platform-string or timing equality as a physics test.
4. Execute the inventory tool. Complete the semantic source/test ledger across all VMEX modules and the relevant adjacent import paths. Review current main changes and PRs only to identify separately pinned comparisons. Do not accept source comments as evidence without the corresponding test/result.
5. Probe actual public APIs and their transformation support. Fix the smoke drivers only as necessary, with a small tested compatibility change, not speculative wrappers for every old version.
6. Add `results/status.json` from the phase state table and enumerate real supported, limited and unavailable capabilities. Keep unsupported physics such as anisotropic/ANIMEC outside the scalar-pressure claim.

**Exit:** reproducible reference suite; pinned baseline; authenticated public repo; complete scope-qualified review ledger; no dependency ambiguity. If the runtime cannot run a particular external solver or device, record that limitation and continue the independent phases.

### P1. Analytical oracles, domains and input adapters

**Already provided:** `analytic.py`, `verify_reference.py`, `reference_derivatives.py`, `build_inputs.py`, and the reference tests.

Complete the following before interpreting VMEX results:

1. Independently verify both original fields and the stretch and Solov'ev additions. Include pressure-surface tangency, divergence, force, Jacobian, nonzero-field/domain margins, oriented flux and current, volume and averages. Use independent algebra/integrals rather than making the tested implementation its own reference.
2. Verify the sheared physical-phi inversion globally on the sampled domain: residual, branch uniqueness, positive angular derivative and graph validity. Explicitly inspect strong shaping near branch/domain boundaries. Add bad-domain tests.
3. Test the complete signed mapping into VMEX's boundary/profile representation. Compare parser-evaluated profiles with independent arrays. Validate both NCURR closures. The input generator's real-coordinate sign choice is only accepted after this test.
4. Resolve sheared B/C Fourier truncation; record the smallest useful mode ladders and input-fit floors. Do not run a high-radial-resolution equilibrium with an unresolved low-angular-resolution boundary and call the result a radial accuracy limit.
5. Add dense independent reference arrays only when needed for cross-process interfaces. Keep their generators and hashes. The analytical evaluator, not an imported DESC HDF5 file, remains the principal truth.

**Exit:** oracles and complete input problem certified to tighter accuracy than the first solver targets. Scalar means and flux inversion agree with independent quadratures. Correct sign is established using physical vectors, not inferred from a plot.

### P2. Representation, fields and diagnostic operators without nonlinear solving

**Implement:** `benchmarks/projection.py` and the smallest shared VMEX adapter needed by subsequent drivers.

1. Project the exact state into VMEX's spectral and high-order spline representations. Score native field, Cartesian current and force without nonlinear recovery. Use `HighOrderEquilibriumState`, its field evaluator and force certification where appropriate, while maintaining a separate reference implementation.
2. Exercise all six R/Z/lambda sine/cosine families through LASYM controls, exact reframing and the asymmetric Solov'ev state. Test NFP=1 versus a physically equivalent NFP representation where legal. Include mode-table, phase and axis normalization checks.
3. Test `VmecInteriorField`: B, |B|, gradB, Hessian B and third spatial derivative on representative interior points. Compare native and WOUT-created objects, seeded and unseeded coordinate inversion, legitimate out-of-domain behavior, axis limits and radial-knot behavior. Record tensor orientation and dimensions. Do not suppress warning/fallback paths.
4. Verify force/current calculations from independent Cartesian derivatives and from native curvilinear operators. Check divB, J dot grad psi, B dot grad psi, and pressure gradients. Scalar energy/volume integrals must use consistent physical Jacobians.
5. Test radial/angle reconstruction, Fourier cutoffs, spline degree and radial knot count independently. Never use an underdetermined high-order fit to invent smooth curvature. Establish the oracle's own error floor before using it to judge a solver.
6. Round-trip INDATA/JSON, native state, WOUT and Boozer files. Verify metadata, dimensions, asymmetry flags, NFP, full/half meshes, profile units and field reconstruction throughout the volume. A boundary-only round trip misses interior scaling errors.

**Exit:** quantified projection and field-derivative error curves; no ambiguity about which interpolation path was scored. Higher spatial derivatives may have a narrower certified region/resolution than B itself.

### P3. Nonlinear fixed-boundary recovery

**Starting driver:** `benchmarks/run_vmex.py`; **implement:** a modest recovery/convergence driver around the shared adapter and `score_samples.py`.

1. Begin with the symmetric axisymmetric case, then asymmetric Solov'ev, integer 3-D, and sheared A. Use both prescribed current and prescribed transform after P1.
2. Run cold initialization, exact-projection initialization and small admissible perturbations of the projected state. Distinguish failure to recover an equilibrium from a field/coordinate error. An exact equilibrium need not attract the chosen energy-relaxation algorithm; track instability or branch problems rather than declaring the analytic field invalid.
3. Run radial ladders such as NS=17,33,65,129,257 while holding boundary/Fourier/diagnostic errors small; then angular ladders at converged radius. Select counts from measured costs and accuracy, not a universal requirement to run every large 3-D combination.
4. Test single-grid versus multigrid, hot restart at equal and changed resolution, repeated deterministic runs, and perturbations of pressure/current/flux. Ensure interpolated states remain admissible. A restart trajectory may differ while the converged physical state agrees.
5. Compare CLI and Python routes, cold/warm executable reuse, and native versus WOUT analysis. Existing VMEC2000 and VMEC++ runs are implementation baselines, not the exact reference.
6. Use the provided DESC scripts only in a separate environment and with independently resolved native DESC states. Compare fields at common physical points, matched inputs, identical norms and error regions. Do not rank native solvers from a table of differently interpolated WOUT files [R12].
7. Add B/C stress cases, smaller nonzero pressure gradients, more extreme admissible stretch, and a reduced safe domain-margin case. Do not approach analytic singularities inadvertently.

**Exit:** every mandatory mild fixed-boundary matrix cell has a resolved recovery record or a specific reproducible failure. At least one axisymmetric LASYM=T and one 3-D LASYM=T recovery are genuine tests, not only zero extra coefficients. The README shows physical errors and resolution, not just FSQ and iteration counts.

#### P3 coordinate-parameterization comparison (S03a–S03c)

The analytical chart is not required to minimize VMEC's spectral width. Its poloidal angle can be changed in the continuum if geometry and the Clebsch stream function are transformed together. This does not guarantee that a finite Fourier/radial projection preserves the same field: mode truncation, radial representation, lambda scaling, boundary parameterization and Jacobian admissibility all need measurement. The single-mode NS=33 trial below reduced the constraint residual only modestly and worsened sampled physical errors. It does not establish that reparameterization is impossible.

1. Complete a native DESC source/test review at the pin in `sources.json`, including its default `Equilibrium.solve` objective and constraints, boundary fit/angle convention, `lambda`, field/current evaluation and initial-guess paths. Use Landreman's DESC scripts as starting cases, not as an analytical oracle. Keep DESC in its own environment and record package/JAX versions. Pin and review VMEC2000, VMEC++ and GVEC before attributing a result to a particular implementation.
2. On the axisymmetric control and integer 3-D case, construct matched physical boundaries and profiles in the original analytical chart and at least one nontrivial smooth compensated poloidal remap. Preserve the physical boundary, toroidal flux, signed transform/current and pressure; update lambda consistently. Extend to sheared A only after its input gate. First certify continuum chart invariance against the independent exact Cartesian field and a nonvanishing, orientation-consistent Jacobian. Then measure the finite native projection's geometry, B, J and force errors, boundary-fit error and spectral width separately at increasing angular and radial resolution. A continuum identity is not a finite-state certificate.
3. For each representation, run cold and certified projected/warm starts where supported. For VMEX compare the default constraint with a bounded `TCON0` ladder including zero; do not infer a new root from a same-state force switch. Run native DESC, VMEC2000 and VMEC++ comparisons, and GVEC if its pinned build and input mapping are verified. Use the same physical problem and held-out Cartesian points. Record solver convergence, raw and code-specific residual definitions, coordinate drift, Fourier width/high-mode energy, Jacobian, B/J/physical-force errors, wall time and peak memory. Keep projected, solved and independent numerical evidence distinct. A code without VMEC's explicit constraint can still have coordinate sensitivity through its basis, optimizer or initial guess.
4. Test whether smaller or zero VMEX constraint strength improves accepted *physical* solutions under resolution refinement without gauge drift, ill-conditioning, broad spectra or loss of root/derivative certificates. If zero strength exposes an angle null space, test a small explicit gauge condition or reparameterized seed before considering a DESC-like solver design. Compare a matching discrete weak-form energy directional derivative to the physical force when constraint-free residuals disagree. Change no default or upstream code merely to lower FSQ on one projected state.

**Exit for S03a–S03c:** a reproducible matched-input and held-out-scoring protocol, continuum and finite-projection remap errors, independently converged or specifically failed native solves, angular/radial convergence, and a stated decision on constraint strength or alternative gauge treatment. A claim that coordinate mismatch is harmless requires converged physical observables and stable derivatives despite the mismatch; a claim that it is harmful requires a resolved physical difference, not only a large code-specific residual. If a solver is unavailable, label that comparison unavailable and retain the design question open.

### P4. Equilibrium derivatives and optimization infrastructure

**Starting driver:** `run_gradient_smoke.py` checks one discrete scalar response. It is not the final analytical-family derivative test. **Implement:** `benchmarks/equilibrium_derivatives.py` with shared input-map functions.

Let a parameterize the analytical family and P(a) include boundary, pressure/current profiles and total flux. For the actual constrained discrete residual,

$$ F_h(x_h,P(a))=0,\qquad F_x x_a=-F_P P_a. $$

For a scalar objective Q,

$$ F_x^T\lambda=Q_x^T,\qquad dQ/da=Q_a-\lambda^TF_P P_a. $$

Audit the exact DOF mask, gauge, anchored root and preconditioning. At a root F=M f with nonsingular M gives the same response as f=0; off root, the extra (dM)f term is controlled by the actual raw residual, not by casually identifying FTOL with its norm.

Required experiments:

1. **Complete family tangent.** Differentiate boundary, flux, pressure/current, surface-label inversion, physical-angle inversion and profile-fit coefficients together. Hold static basis, quadrature plans, masks and branch conventions fixed during a local derivative check. Compare integrated beta/energy/volume and pointwise physical fields with independent analytical derivatives.
2. **Eulerian null.** For the integer family, partial_delta B at fixed physical position is zero. In a moving computational chart subtract the convective term `(grad B) r_delta` from the material derivative. Do not compare the two derivatives without this correction.
3. **Current-prescribed shear null.** Verify partial_lambda iota(s)=0 for sheared A with NCURR=1 and the changing analytic current profile, boundary, pressure and flux. Also check the nonzero pressure, volume and flux scalings.
4. **Nonzero responses.** Include epsilon/stretch derivatives of beta, a pressure/current response, one actual physical-field component, and an asymmetric boundary degree of freedom. Volume determined solely by a fixed input boundary is useful but not sufficient to validate the equilibrium response.
5. **Discrete tangent and transpose.** Compare small assembled residual Jacobians with JVP/VJP products; test `<u,F_x v>=<F_x^T u,v>`. Solve the tangent equation separately where a public JVP of the nonlinear wrapper does not exist. Compare direct small dense/SciPy solves with SOLVAX block and Krylov paths, including multiple right-hand sides and factor reuse. Certify the unpreconditioned operator residual and actual conditioning.
6. **Three-way gradient comparison.** Compare implicit response, frozen-path finite differences, and independently anchored/reconverged physical perturbations. Use multiple h and multiple spatial resolutions. A plateau in h can be solver noise, and a gap between cold and frozen paths can reveal gauge dependence. Do not substitute a frozen consistency test for the continuum target.
7. **Taylor tests.** Plot |Q(a+h v)-Q(a)-h g dot v| versus h. Seek a second-order interval before numerical floors. For centered finite differences seek the corresponding accuracy interval. Near-zero derivatives require absolute scaled errors. Keep the same physical objective and sampling convention for all three evaluations.
8. **Higher derivatives.** Probe HVP/Hessian support rather than assuming it. Compare analytic-family Hessians and finite differences of certified first derivatives where feasible. Distinguish a Hessian of an explicit reference, a derivative of a discrete solve, and a Gauss-Newton approximation. Unsupported transformations are reported, not emulated silently by a different algorithm.
9. **Failure paths and caches.** Test anchor rejection, insufficient nonlinear convergence, insufficient adjoint convergence, singular/near-singular charts, finite/nonfinite penalties, updated parameters with reused executable/factors, and cold/warm result invariance. Benchmark certification uses strict error policies, not best-effort adjoints.

**Exit:** at least one nontrivial continuum sensitivity and one nontrivial null response pass in each mandatory geometry class, with the full chain and correct physical location. Report source, nonlinear and adjoint residuals, conditioning evidence and derivative-order limitations alongside errors.

### P5. Boozer, bounce, geometry and equilibrium diagnostics

**Implement:** `benchmarks/diagnostics.py`, reusing the reference evaluator and common samples. Avoid treating a downstream computation as analytically solved just because B is exact.

* **Boozer transform:** reconstruct B and geometry in physical space; compare fluxes, I/G conventions, Jacobian, field-line straightness, surface averages and spectra at matched gauges. The axisymmetric aligned case has no genuine n!=0 |B| modes. Generic Landreman cases are not assumed quasisymmetric or quasi-isodynamic. Compare booz_xform_jax with the independently installed reference transform. Test file and live-state routes, LASYM coefficients, resolution changes and parameter derivatives. For rational iota, handle gauge/nullspace choices explicitly; do not infer nonexistence of Boozer coordinates merely from rational transform.
* **Field lines:** integrate the explicit B and the numerical B with independent ODE control. Compare pressure-label conservation and winding. Integer-family lines close; closure on one line is not proof that the field is correct. In the sheared family compare iota from flux derivatives, line tracing and current-prescribed output. Keep toroidal winding unwrapped and distinguish t from physical phi.
* **Bounce action:** VMEX's kernel uses `J=2 integral sqrt(1-pitch B) dl`, not the dimensional mechanical action without its mass/speed factor [R8]. Compare with independently located bounce points and adaptive quadrature on the exact field. Test multiple wells, finite traces, well masks, overflow, turning-point approach and gradients only inside a fixed smooth well topology. Distinguish exact `kernel_floor=None` results from a smoothed optimization surrogate, and refine the floor separately. A constant transform does not make a field omnigenous.
* **Geometry and stability:** compare metric factors, curvature, grad-B drift, magnetic shear, magnetic-well measures, j dot B, surface averages, and pressure/current profile derivatives against exact geometry. For Mercier, resistive interchange and ballooning quantities, use an independent implementation and a matched model. Near zero shear or a vanishing denominator, test limits/status explicitly. Force balance alone proves neither local nor global stability. A coefficient-space residual Jacobian's smallest singular value is not an MHD stability eigenvalue.
* **Bootstrap and transport-facing data:** validate the geometry and normalization, then compare a separately prescribed kinetic problem with an independent solver where appropriate. The MHD total-current profile is not generally equal to bootstrap current. Specify density, temperature, species, collisions and electric field separately. Do not impose unsupported QS/large-aspect-ratio bootstrap formulas on a generic exact 3-D field and label the mismatch an equilibrium error.
* **NEO_JAX:** compare epsilon_eff with the reference implementation at declared nonrational/sheared surfaces and refine angular, pitch, line-length and well resolution. Record whether outputs are epsilon_eff or epsilon_eff^(3/2). A globally rational integer-transform family is a stress test of closure/sampling assumptions, not an automatic production transport benchmark based on surface ergodicity.

**Exit:** each advertised diagnostic has either an independent target, a clearly scoped consistency/integration test, a reproducible limitation, or an explicit not-applicable status. No all-green summary obtained from skipped optional packages.

### P6. Free boundary, exterior fields and coil sensitivities

**Implement:** `benchmarks/free_boundary.py` and `benchmarks/exterior.py` only after their common fixtures and scorer exist. Keep three evidence tiers separate as defined in section 4.

The vacuum region must satisfy

$$ \nabla\times B_v=0,\qquad \nabla\cdot B_v=0. $$

At a stationary flux boundary, require B dot n=0 on each side and total-pressure balance,

$$ p_p+\frac{B_p^2}{2\mu_0}=p_v+\frac{B_v^2}{2\mu_0}. $$

A tangential-field jump corresponds to a sheet current

$$ K_s=\frac{n\times(B_v-B_p)}{\mu_0}. $$

Specify whether sheet currents are allowed, represented and measured. Matching normal field alone does not enforce pressure balance or remove a tangential jump. The exact interior formula generally has nonzero curl outside its chosen boundary and is not a valid vacuum extension. Virtual casing separates fields by **source-current location**, not simply by whether a target point is inside or outside the surface [R9].

**P6a: operator and source tests.**

1. Test NESTOR's vacuum boundary integral with analytic harmonic/dipole fields on aligned and asymmetric surfaces. Include nontrivial toroidal circulation and all normalization factors. Gauge and harmonic-flux constraints must be imposed explicitly; a Neumann problem is not fixed by a scalar potential constant alone in a multiply connected domain.
2. Test coil Biot-Savart values and derivatives against a circular-loop on-axis formula and an independent off-axis quadrature. Include current linearity and shape perturbations, not only reused samples from the same field class.
3. Compare analytic field callbacks, direct-coil callbacks and MGRID interpolation. Independently refine the MGRID R/phi/Z sampling, include periodic wrapping and boundaries, and verify named current-group scaling. Do not double-multiply currents when importing an already scaled table.
4. Test virtual-casing normal fields and exterior B, gradB, Hessian and third derivative against independent volume/surface integrals or an independently converged reference implementation. Refine source quadrature and target distance separately. Sample distances as fractions of minor radius and source-panel size. Compare direct and graded quadrature; no near-surface accuracy claim without a target-distance study.
5. Verify interior/exterior limits, jump relations and signs. Freeze the numerical precision/patch plan for a derivative, then check that recomputing an adequate plan gives the same converged physical result. A changing adaptive plan is not a differentiable physical degree of freedom.

**P6b: coupled root recovery.**

1. Reproduce one axisymmetric finite-beta case and one 3-D finite-beta case from VMEX's verified input/coil assets. Preserve hashes and exact current groups; do not redistribute large or unlicensed assets unnecessarily.
2. Run symmetric cases with LASYM=F and T. Add a genuinely asymmetric axisymmetric boundary/source case where applicable, and a truly asymmetric 3-D coil/current perturbation with NFP reduced as needed. Use independent numerical reference roots for those coupled cases.
3. Test both MGRID and direct coils, multigrid and hot restart, and pressure/current/flux changes. Report B.n, stress balance, field/current error, displaced boundary and nonlinear coupled residual, not only plasma FSQ.
4. Compare the ordinary forward state with the Newton-anchored coupled state. Verify that the objective and derivative read the same state. Exercise anchor status 3, finite restart budgets and deterministic cold retries. Use `adjoint_fail='error'` and finite `refine_tol` for accepted gradients. Best-effort and skipped-anchor runs may be recorded only as diagnostics.
5. Compare `coupled_gcrot`, `boundary_schur` and `edge_response` using actual available APIs. Test transpose duality, exact coupled residual certificates and gradients against independently reconverged current/coil-shape perturbations over a step-size interval. A fixed-boundary virtual-casing objective derivative is not this coupled derivative.
6. Measure peak memory and compile time before attempting larger matrices or GPU placement. Public forward-mode derivatives remain a capability probe; a separate tangent linear solve is acceptable when correctly identified.

**P6c: exact-interior target and exterior design.**

1. Supply the exact surface and boundary field to an independent source-separation calculation. Fit an external coil/current-potential representation to the required exterior contribution, with explicit circulation/current constraints.
2. At fixed target surface, report full boundary field/stress errors as well as B.n. Use constraints on source distance, current magnitude and surface regularity so a nearly singular source is not mistaken for an improvement.
3. Independently increase coil/current-potential resolution, vacuum quadrature and plasma resolution. Re-solve free boundary, allowing the surface and axis to move. Compare the recovered interior with the exact target at common physical points only in their common admissible domain; report any boundary mismatch separately.
4. Separate exterior fit error, NESTOR/MGRID error, nonlinear error and plasma discretization error. Without an exact matching exterior this is not an exact complete free-boundary equilibrium benchmark.
5. Extend sensitivities through source fitting only after its own optimality conditions are certified. Holding a fitted coil set fixed and differentiating a geometry-dependent coil fit are different questions and must have separate experiment labels.

**Exit:** exact vacuum/operator tests, independently validated coupled axisymmetric and 3-D cases, strict reverse derivative evidence, and a documented outcome for the exact-interior matching experiment. Failure to realize a chosen exact interior with the chosen coil class is a physical/design result, not grounds to fabricate a passing exact free-boundary cell.

### P7. High-order polishing and stationary-response checks

**Implement:** `benchmarks/polishing.py`; reuse the native projection and scorer.

1. Compare unpolished and polished states in the same continuous representation, on the same held-out quadrature and with the same boundary/profiles/flux. Separately measure WOUT export/reconstruction effects at a fixed export mesh.
2. Begin with symmetric and asymmetric Solov'ev cases. Then attempt the mild integer 3-D and sheared A cases within measured time/memory budgets. Respect the driver's automatic decline policy; do not turn on an hours-long solve accidentally or call a declined case a success.
3. Record physical force, exact field/current error, norm of the residual, stationarity norm, Newton/Krylov certificates, rank/conditioning and geometry/domain margins. Better force does not automatically mean better field or more accurate sensitivity.
4. If the polish minimizes 1/2 ||r(x,a)||^2, its defining stationarity equation is G=J^T r=0. Its exact x derivative is

$$ G_x=J^TJ+\sum_i r_i\nabla_x^2 r_i. $$

Derivatives of the least-squares stationary state must use this operator, or explicitly bound the error made by neglecting the second term. Gauss-Newton is a useful optimization approximation, not an exact stationary derivative at a nonzero residual by definition.

5. Repeat the P4 gradient tests for accepted polished states. Treat gauge choices, overdetermined residuals and branch selection explicitly. Compare the earlier square-root/homotopy route only as a separately labelled diagnostic.

**Exit:** at least an accurate axisymmetric improvement or a documented counterexample; evidence for any 3-D improvement must include full field and derivative checks. Do not promote a universal 3-D polish claim from one favorable plot.

### P8. Optimization and extension of the analytical work

**Implement:** `benchmarks/optimize_family.py`, followed by `benchmarks/continue_transverse.py`. Add a coil optimization driver only after P6. These are studies with testable questions, not promises of a better reactor configuration.

#### Study A: exact-family parameter optimization

Optimize explicit fields/geometry/quadratures directly. Re-solving MHD at every step is unnecessary within an exact family. Compare direct gradients with the verified VMEX chain at selected points, and use VMEX for diagnostics or departures from the family.

Use the integer/stretch parameters or epsilon,S,lambda,delta of the sheared family. Remove pure field and length rescalings from the search by fixing volume and a field scale, or enforce equivalent normalization constraints. Set edge pressure to zero. Fix or bound aspect ratio, elongation, current and distance to the analytical singular domain. Impose a nonaxisymmetry floor for a genuinely 3-D search so an optimizer cannot win by returning to the axisymmetric limit. Parameterize domain constraints smoothly inside one branch; an absolute-value cusp at the axisymmetric point is not a useful gradient test.

Start with a transparent smooth objective such as a declared combination of field-strength variation, normalized current concentration, and target beta/shape constraints. For example,

$$
Q=w_B\frac{\langle(B-\langle B\rangle)^2\rangle}{\langle B\rangle^2}
 +w_J\frac{L^2\langle J^2\rangle}{\langle B^2\rangle}
 +\sum_k w_k c_k(a)^2,
$$

in mu0=1 units, with every average, weight and constraint stated. This objective is not asserted to equal confinement quality. Add validated magnetic-well, bounce-action, ripple or stability measures one at a time. Report the tradeoff curve rather than hiding opposing trends in a single weighted sum.

Use the sheared family's lambda direction to ask how stability/confinement diagnostics change while the transform profile is unchanged. Include both moderate and high-beta members only where representation and model assumptions remain valid. A higher beta reached by changing the additive pressure constant is not a meaningful optimization result; it is excluded by the zero-edge-pressure condition.

Run a modest set of physically distinct initial parameters, certify any inner solves, compare AD with finite differences at representative iterates, and validate the final parameters on denser independent quadratures. Report all starts and constrained failures. Never claim global optimality from a local gradient run.

#### Study B: exact-family tangent versus transverse response

Write P(a,eta)=P_exact(a)+Delta P(eta), where eta releases a small set of boundary/profile degrees of freedom not generated by the exact family. Identify tangent directions in a consistent, scaled, gauge-controlled representation. Orthogonalize the chosen transverse directions against that span for diagnostic clarity; this is not a coordinate-invariant mathematical definition of the full solution manifold.

Current continuity, with

$$ J=\sigma B+\frac{B\times\nabla p}{B^2}, $$

implies

$$ B\cdot\nabla\sigma=-\nabla\cdot\left(\frac{B\times\nabla p}{B^2}\right)=S. $$

On a closed field line, a necessary compatibility condition is

$$ \oint \frac{S}{|B|}\,dl=0. $$

For a near-rational Fourier mode, resonant denominators involve m*iota-n*NFP in a straight-field-line convention. Exact-family tangent changes preserve the exact compatibility identities. Generic perturbations need not preserve the same cancellations.

Measure current concentration, continuous force error, resolution dependence, gauge-constrained residual singular values, tangent norm and resonant source integrals as small eta is continued in both directions. Compare continuation at decreasing eta and increasing spatial resolution. Do not interpret an isolated small singular value as a physical MHD instability, nor a smooth finite-resolution nested-surface solution as proof of a smooth continuum equilibrium. The closed-line compatibility condition is necessary, not sufficient.

The desired result is an evidence-based description of which directions are well conditioned and which are sensitive. A regular nearby family or a reproducible loss of regularity can both be informative outcomes. Do not prescribe the conclusion before computing it.

#### Study C: physical boundary optimization outside the exact family

Starting from a verified sheared reference, release a small, documented set of boundary coefficients with fixed pressure/current/flux choices. Compare an implicit-gradient optimizer with a central-difference baseline at the same tolerances, objective, parameter scaling and evaluation budget. Keep geometric admissibility, normalization and nontrivial physical constraints. Use actual held-out force/field checks as acceptance criteria, not only a decreasing optimizer loss.

Distinguish training-resolution progress from final-resolution improvement. Re-solve at higher resolution, restart from more than one admissible state and compare independent physical diagnostics. Reject improvements that disappear under refinement. Intrinsically asymmetric 3-D numerical configurations belong here and in Study B, not in the exact-reference table.

#### Study D: exterior/source optimization

After P6, optimize currents or a modest coil shape set with source-distance and current constraints. Certify any inner field fit and its differentiability. First test the fixed-boundary source objective, then the coupled free-boundary objective. The two are not interchangeable. Track boundary displacement, normal field, stress balance and interior accuracy after re-solving.

**Exit for P8:** reproducible objective/constraint definitions, gradients checked at multiple iterates, complete histories, independent final verification, and a documented scientific outcome. A flat tradeoff or failed coil realization is acceptable evidence if the failure is resolved and explained; an unsupported success claim is not.

### P9. Adjacent-code and open-mirror coverage

Use these optional integrations to measure how equilibrium error propagates, not to turn the benchmark into a second kinetic-code development project.

| Integration | Required useful check | What it does not establish |
|---|---|---|
| SOLVAX | Residual/JVP/VJP factor tests; dense independent solves; factor reuse; true residual and response accuracy | Every solver or truncation mode in SOLVAX is validated by VMEX. |
| booz_xform_jax | Physical reconstruction, independent transform, LASYM, NFP, differentiable spectra | Generic exact field is QS/QI or all rational gauges are unique. |
| virtual_casing_jax | Independent source decomposition, target-distance convergence, fixed-plan derivatives | A free-boundary coil solution exists for every exact interior. |
| ESSOS | Coil kernels, field handoff, tracing, orbit invariants and integration refinement | Guiding-center magnetic moment is an exact invariant of full particle orbits. |
| NEO_JAX | Geometry and ripple quadrature on suitable surfaces; independent reference and derivative checks | Exact equilibrium gives an exact transport coefficient without solving transport. |
| GKX | Metric, curvature and drift data from exact geometry; optional certified isolated linear eigenvalue response | Exact MHD implies an analytically known gyrokinetic growth rate or heat flux. |
| DKX | Geometry/normalization handoff, a prescribed kinetic case, conservation and matched SFINCS comparison | MHD total current equals bootstrap current or fixes an ambipolar electric field. |
| pyQSC_JAX PR 2 | Geometric axis/field-jet checks and appropriate asymptotic limits on compatible cases | A QS near-axis ansatz can represent arbitrary non-QS Landreman equilibria exactly. |
| VMEX mirror lane | Exact vacuum polynomial field, open-surface flux tests, separate paraxial finite-beta checks, mirror adjoint | Toroidal exact equilibria verify the open topology or high-beta mirror closure. |

For trajectories, conserve energy in a static magnetic field and use the correct invariant for the chosen full-orbit or guiding-center model. In an axisymmetric field, compare canonical toroidal momentum with a consistent vector potential/gauge. Separate orbit-step error, guiding-center ordering error and equilibrium interpolation error. Near loss boundaries or bounce topology changes, an indicator-valued loss fraction is not a smooth objective; use a stated surrogate and check convergence back to the physical diagnostic.

For GKX eigenvalue derivatives, require an isolated converged eigenpair, an original-operator residual, correctly normalized left/right eigenvectors and branch tracking. At a crossing, do not claim a unique ordinary derivative. For DKX ambipolar roots, a near-zero derivative of the ambipolarity function can make the root response ill conditioned; distinguish branch continuation from picking a different root at each trial.

Near-axis comparisons must separate expansion truncation from numerical error by a radius ladder at fixed expansion order. Do not expand a force-balanced finite-radius field into a restricted symmetry ansatz without verifying compatibility. The draft branch is optional and must not block the primary toroidal benchmark.

**Exit:** one small, well-defined test per available relevant integration, with imported package versions and actual executed counts. Record missing hardware/packages as unavailable. Unrelated PIC, materials, relativistic or transport-evolution repositories are outside this equilibrium benchmark unless a concrete shared operator supplies a justified test.

### P10. Performance, figures, README and final handoff

Measure performance only after accuracy is established. Include input preparation, nonlinear solve, root anchor, field reconstruction, Boozer transform, scalar gradient/Jacobian, export and final verification as separate stages. Also report complete end-to-end time. GPU timings must synchronize results; warm timing must not omit later phases. Cold compilation uses an isolated benchmark cache, never deletion of the user's global cache.

For representative workloads, record cold process time, warm same-input time, warm new-parameter time, peak host resident memory and device memory if measurable. Use repetitions and show medians plus spread. The first measurement may include compilation; record this rather than averaging it invisibly with warm runs. Measure crossover with resolution and parameter/output count; never promise a GPU win or order-of-magnitude speedup in advance.

For matched solver comparisons, compare time to a stated physical error or gradient error, not just equal NS/MPOL/NTOR or nominal FTOL. Include failed or out-of-budget runs. Source generation, external quadrature and analytical-reference cost may be reported separately, but not silently removed from a claimed end-to-end workflow.

Produce figures from stored numerical arrays and a manifest tying each image to inputs, commit, metric and generator. Re-run from a clean environment and inspect each rendered image. Labels must state dimensions, normalization, physical region and whether the data are analytical, projected, solved, or numerical-reference values. Use equal physical scales for geometric plots. Avoid interpolated curves that disguise sparse data or unmeasured resolutions.

Finish the README with the main verified outcomes and exact run commands. Keep complete derivations and the full test matrix here or in a short model note. Link failed/limited cases rather than suppressing them. Do not write adjectives such as "fully validated" unless the completed, scope-specific matrix supports them. Do not present a projected exact field as a solved VMEX result.

## 8. Figure plan and provenance contract

Each row is a scientific figure, not a compulsory multi-panel dashboard. Separate plots are preferable when scales or evidence classes differ.

| ID | Figure | Required data | Principal question |
|---|---|---|---|
| F01 | Exact surfaces and cross sections, including asymmetric Solov'ev and reframed 3-D case | Reference parameters and Cartesian geometry | Which physical cases and symmetry classes are actually tested? |
| F02 | Boundary/profile representation error under independent refinement | Held-out boundary samples; pressure/current/flux errors | Is input conversion limiting the benchmark? |
| F03 | B recovery versus radial and angular resolution | Native common-point field errors | Does the nonlinear solution approach the exact field? |
| F04 | Current and force errors, including radial bins | J, grad p, volume weights, dimensional force | Does small discrete force correspond to physical force accuracy? |
| F05 | Spatial derivative order 0-3 and inversion error | Tensor errors, interpolation path, knot/axis masks | How much accuracy survives differentiation and field lookup? |
| F06 | Current-prescribed iota and profile recovery | Signed iota(s), enclosed I(s), exact flux mapping | Does VMEX predict, rather than merely impose, the transform? |
| F07 | Gradient/Taylor and finite-difference step studies | Objective, complete parameter direction, root/adjoint certificates | Is there a resolved derivative interval and continuum convergence? |
| F08 | Null-response convergence | Eulerian delta field response; current-prescribed lambda iota response | Are required cancellations recovered? |
| F09 | Boozer and bounce reconstruction errors | Physical reconstruction, spectra/gauge, independent well quadratures | Do postprocessing and derivatives preserve the exact physics? |
| F10 | Vacuum/MGRID/virtual-casing error versus grid and target distance | Source plans, target distance, derivatives and independent integrals | Are exterior numerical and near-surface errors controlled? |
| F11 | Coupled free-boundary boundary/stress and gradient errors | Anchored root, displaced surface, full field/stress, FD intervals | Are free-boundary values and derivatives of the same physical problem? |
| F12 | Polishing before/after on identical representation and quadrature | Force, field/current, stationarity and derivatives | Does polishing improve what matters, and where? |
| F13 | Exact-family optimization tradeoffs and histories | All starts, constraints, direct and VMEX checks | What changes are available within an exact solution space? |
| F14 | Tangent versus transverse response and resonant forcing | Resolution/amplitude ladders, source integrals, scaled Jacobian diagnostics | Which directions remain regular and well conditioned? |
| F15 | Boundary/coil optimization validation | Training and held-out metrics, costs and physical constraints | Does optimization improve the resolved problem? |
| F16 | Time and memory to physical accuracy | Per-stage cold/warm measurements and errors | What is the computational cost at comparable fidelity? |
| F17 | Capability/evidence matrix | Completed statuses, oracles and unavailable cases | What has and has not been tested? |
| F18 | Coordinate gauge sensitivity across native solvers | Same physical cases and held-out samples; projected and solved states, constraint ladder, spectral width and physical errors | Does angle mismatch change physical accuracy or only a code-specific residual? |

The existing figures are reference-only illustrations and input-fit measurements. They are not placeholders labelled as completed F03-F18. Generate those only after the corresponding experiments exist.

## 9. Result schema, budgets and acceptance

For each run store a small JSON record with:

* unique run and case IDs; parent run for continuation/restart; evidence class; status; exact command and inputs;
* Git SHAs and dirty/patch hashes for benchmark and all imported source packages; package versions and interpreter/hardware metadata;
* units, mu0, L and B0; coordinate/orientation conventions; boundary/profiles/current/flux hash; symmetry/NFP and actual degrees of freedom;
* solver, vacuum, boundary-fit, profile-fit and scoring resolutions separately; tolerances, quadrature plans and derivative method;
* nonlinear status and residuals; anchor status/residual; linear status and unpreconditioned residual; gauges/constraints and branch evidence;
* physical errors, scalar outputs, error regions and normalization; timings and memory; all failure/skip reasons;
* paths and SHA256 hashes of numerical arrays needed to regenerate reported figures.

Large WOUTs, dense arrays, traces and movies can be checksummed release assets rather than Git history. Keep small summary JSON/CSV and the scripts in Git. A GitHub release asset upload must also use the owner's account, not a bot. Preserve enough representative state data to reproduce source-error decompositions without rerunning every expensive solve.

Before a large run, measure one small rung and estimate peak memory from actual arrays and compilation behavior. Set and record a local time/memory budget. Stop safely on budget exhaustion, save available diagnostics, and mark the case blocked or failed. Do not let a solver silently continue indefinitely. An agent restart must not repeat already completed expensive measurements without a reason recorded in the logbook.

Use strict failures for invalid domains, nonfinite samples, wrong signs, unsupported flags, unresolved reference calculations and uncertified derivatives. Do not use `nan_to_num`, clip invalid analytic radicands, or substitute a prior successful state to make an accuracy test pass. The small nonnegative roundoff guard in the endpoint quadrature has a specific removable-endpoint purpose; it is not permission to hide a field-domain error.

## 10. Repository layout and development order

The bundle intentionally contains only the reference implementation and small first runners. Future filenames below are planned deliverables, not a claim that missing scripts already work.

```text
README.md                      measured overview and run commands
plan.md                        contract, decisions and logbook
AGENT_PROMPT.md                 resumable task prompt
cases.json                     exact reference configurations
sources.json                   source pins and dependency roles
docs/SOURCE_REVIEW.md           inspected source scope and follow-up audit
benchmarks/analytic.py          shared exact fields, surfaces and quadratures
benchmarks/verify_reference.py  reference identities and checks
benchmarks/reference_derivatives.py
benchmarks/build_inputs.py      boundary/profile input generation
benchmarks/score_samples.py     common physical scorer
benchmarks/run_vmex.py          initial fixed-boundary forward smoke
benchmarks/run_gradient_smoke.py
benchmarks/plot_results.py      existing reference figures
benchmarks/<phase driver>.py    add only when a phase needs it
tests/test_reference.py         small independent reference suite
tools/audit_sources.py          local source inventory, not semantic review
tools/publish.sh                explicit owner-authenticated publication
inputs/                        generated candidate decks and fit manifest
results/reference/             actual handoff measurements
results/<experiment>/          later measured records, not invented results
figures/                       generated plots with evidence descriptions
```

Add a small `vmex_adapter.py` only when it prevents actual duplication between P2-P7. It should expose physical samples and verified parameter maps, not mirror every upstream API. Keep changes in VMEX itself separate and minimal. If a production capability cannot be tested without a broad solver rewrite, first publish the minimal failing reproducer and evidence, then discuss the required scope in the logbook.

The mandatory core is P0-P6 plus a bounded polishing attempt and the first exact-family optimization. P8 transverse/free-boundary optimization and P9 integrations deepen the study. They remain required tracked work items, but unsupported packages or unresolved physical assumptions must be reported honestly rather than blocking all core evidence or manufacturing a passing status.

## 11. Final completion conditions

The final repository must contain the actual executed capability matrix, not only this plan. Completion requires:

- resolved exact-reference recovery for the stated mild axisymmetric and 3-D fixed-boundary cases, including meaningful LASYM coverage and both closure choices;
- field/current/force and derivative convergence evidence, with projection and reconstruction separated;
- both exact vacuum/operator tests and properly labelled coupled free-boundary tests; no claim that an interior field alone supplies an exact exterior;
- strict root and adjoint certificates and at least the nonzero and null physical sensitivity experiments;
- a bounded, recorded outcome for polishing, exact-family optimization and transverse continuation; independent verification of any claimed improvement;
- meaningful tests or explicit unavailable/not-applicable statuses for the relevant downstream and mirror capabilities;
- a complete source-scope ledger, reproducible environment, figure/data provenance, concise README, and a logbook that records failures and next actions.

A negative scientific result can close an experiment if its mathematical assumptions, numerical convergence and failure mechanism have been examined. It cannot turn an unsupported capability into a pass. If a blocker remains, preserve the state and leave the exact next experiment; do not announce that an unrun phase is completed.

## 12. Logbook and replacement-agent handoff

### Current phase table

| Phase | Status on delivery | Evidence / next action |
|---|---|---|
| P0 | Partial | Public repo and required source pins exist; DESC is pinned at `4f48720`. The ledger has 512 entries, 25 partial and no claim of complete semantic review. Eight entries record targeted DESC source/test paths and findings; full VMEX and adjacent-library semantic review remains. |
| P1 | Partial | 29 tests and 14 references reproduced; all 28 decks passed parser/setup checks. B/C fits clear the smoke gate; sampled sheared charts and local implicit-root derivatives pass. A sampled guard now rejects measured near-domain folds. Final output error budget remains. |
| P2 | Partial | Native integer 3-D projected volume B/J/force passes first targets at NS=129 without solving; continuum gauge identity, the new five-surface mapped B/curl-B check, and finite NS33 surface projection are separately measured. The new sampled remap check has B error <=2.59e-10, curl-B error <=1.16e-4 and positive map derivative >=0.95. Symmetric Solov'ev projection and LASYM WOUT round trip also run. Higher derivatives, global Jacobian proof, LASYM volume and other file routes remain. |
| P3 | Partial | DESC base/remapped comparisons remain at angular `M=N=6/8` (radial `L=8/10`); the remapped M8 run remains above optimizer gtol. Added the NS33 VMEX `TCON0={1,0.1,0}` cold/projected ladder, plus projected-start default/zero runs at NS65 and default at NS129. All nine measured states reached discrete `FTOL=1e-10`, but none met all physical B/J/force targets. Zero strength improved projected-start scores through NS65; NS129 zero remains unrun. Multi-resolution coordinate stability, additional gauges/remaps, derivatives, and VMEC2000/VMEC++ comparisons remain before any design recommendation. |
| P4 | Reference derivatives run; solver work planned | Smoke runner exists; complete family input-map differentiation is not implemented. |
| P5 | Planned | Review/execute Boozer, bounce and diagnostics tests with independent references. |
| P6 | Planned | Vacuum operators first, then anchored coupled roots, then exterior fitting. |
| P7 | Planned | Same-representation polishing and stationary response. |
| P8 | Planned | Direct exact-family optimization, then transverse and source studies. |
| P9 | Planned | Small scoped adjacent/mirror integrations, with unavailable statuses where needed. |
| P10 | Partial | Reference, axisymmetric, integer 3-D projection/loose-root, short warm trajectory, force-localization, constraint-switch, gauge-scan, gauge-surface, VMEX/DESC coordinate-comparison, NS33 TCON0 ladder, and partial radial-resolution figures generated from saved data and visually inspected; broader matched performance and resolution work remains. |

The current P3 coordinate comparison has nine measured VMEX TCON0 states: six NS33 cold/projected cells, projected default/zero at NS65, and projected default at NS129. All nine reached the discrete `FTOL=1e-10` stop; none passed all native physical B/J/force gates. The NS129 projected zero-strength cell is intentionally recorded as unrun. See the newest logbook entry and the partial summary under `results/vmex/tcon_resolution_ladder/` for the immediate resume action.

### Entry 2026-09-23: handoff preparation

**Changed:** added explicit integer/stretch, sheared and asymmetric Solov'ev references; physical-angle surface samplers; toroidal-flux inversion; independent Ampere-current quadrature; 14-case matrix; both closure input candidates; reference checks; physical scorer; first solver runners; source inventory and owner-only publish helper.

**Measured:** `python -m pytest -q` passed 29 tests. `verify_reference.py` passed all 14 sampled reference cases. Sampled force RMS divided by pressure-gradient RMS was of order 1e-15. Real finite differences independently checked the field Jacobians. Symbolic asymmetric Grad-Shafranov identity, independent volume/flux integrals, frame covariance, profile inversion and null sensitivities passed. Exact values and package versions are recorded in the JSON, not inferred from prose.

**Important non-result:** no VMEX or other equilibrium/kinetic solver ran. Candidate input fit error is measured, but parser/sign closure and equilibrium recovery are not established. Sheared B and C are boundary-underresolved at the supplied first Fourier truncation. The forward smoke refuses them until their manifests are regenerated with adequate fits.

**Source finding:** current free-boundary source and change log include Newton anchoring, while an older validation paragraph says anchoring is absent. Resolve the documentation/code/test discrepancy locally; do not assume either universal success or an unfixed absence.

**Validation packaging:** a combined multi-command validation call reached the execution wrapper limit. Completed stages were checkpointed and the remaining commands succeeded when run separately. The final execution record includes the successful test, reference, input-generation, plotting and syntax-check commands. No partial solver output was counted as evidence.

**Next actions, in order:** publish the reviewed scaffold with owner identity; pin and install the baseline; inventory and review source; rerun reference checks; verify generated input conventions in VMEX; implement native physical sampling; solve `integer_axisymmetric` in both closures; then `solovev_asymmetric`, `integer_3d` and `sheared_A`.

### Template for every subsequent entry

### Entry 2026-09-23: local reference rerun and publication preparation

**Phase / run IDs:** P0 and P1; R01 and R02 repeated. The owner authentication returned `rogeriojorge`, numeric account ID `6816712`; local author and committer are `rogeriojorge <6816712+rogeriojorge@users.noreply.github.com>`. The public repository did not exist at this check. Initial Git staging has not been committed or pushed yet.

**Commands and results:** `python3 -m pytest -q` passed 29 tests in 11.57 s. `python3 benchmarks/verify_reference.py` passed all 14 sampled cases; the largest printed force ratio was 1.65e-15. `python3 benchmarks/reference_derivatives.py`, `python3 benchmarks/build_inputs.py`, and `python3 benchmarks/plot_results.py` completed. The largest regenerated candidate boundary errors remain 2.80e-3 m for sheared B and 1.33e-2 m for sheared C. These are reference and candidate-input measurements, not VMEX results. The scripts regenerated `results/reference/`, `inputs/manifest.json`, and `figures/`.

**Source and environment:** existing local VMEX and adjacent checkouts were found, but their current heads differ from the pinned VMEX baseline; the analytical supplement was not yet located locally. The workstation is macOS arm64 with an Apple M3 Max, Python 3.11.14. No VMEX solve, semantic source review, GPU run or parser certification was performed in this block.

**Publication review:** inspected `tools/publish.sh` before running its nonpublishing stage mode. Its local identity settings match the authenticated owner. The staged content requires a final privacy/diff review before publication.

**Next exact actions:** finish staged diff/privacy inspection; invoke `PUBLISH=1 sh tools/publish.sh` only if clean; record the resulting URL and commit; create a clean pinned VMEX baseline worktree and clone the supplement; then inventory and review the parser/profile/field paths before checking input interpretation.

**Working tree / branch / PR state:** new local `main` repository with staged scaffold; no public repository or PR at this entry.

### Entry 2026-09-23: pinned input conversion and first native recovery

**Phase / run IDs:** P0-P3, P10; I01, R03, S01a, S01b. The public `rogeriojorge/vmex-benchmark-analytical` repository was created under the verified owner account at initial commit `8171799`. No upstream branch or PR was created.

**Source commits and environment:** clean detached VMEX baseline `b5f5267efc0795c4a49a224e321e9b370975c14c`, analytical supplement `4c0b690ddebdc71811c88223eb9f44a98ab64222`, SOLVAX `2e246a5d6093662f9b5f72c46f995cd7c4bbd479`, and booz_xform_jax `cd25084422de10b620bd86ede0bbd51ba06d7fa6`. The VMEX runtime used Python 3.11.14 and VMEX 0.8.1 on a CPU. Exact private checkout/environment locations are in a Git-local handoff note, excluded from commits. `tools/audit_sources.py` inventoried 657 tracked text/source files; `results/audit/review_ledger.json` has 504 code/config entries, 3 marked partial and none marked complete. This is not a complete semantic audit. Optional integrations remain unpinned and unrun.

**Changes and commands:** added `benchmarks/verify_inputs_vmex.py`, native physical sampling, a recovery plot generator and run records. From the repository root: `python3 benchmarks/build_inputs.py`; `PYTHONPATH="$VMEX_BASELINE" "$BENCHMARK_PYTHON" benchmarks/verify_inputs_vmex.py`; `PYTHONPATH="$VMEX_BASELINE" "$BENCHMARK_PYTHON" benchmarks/run_vmex.py inputs/input.integer_axisymmetric_iota 33` (repeat with 65 and 129); `PYTHONPATH="$VMEX_BASELINE" "$BENCHMARK_PYTHON" benchmarks/run_vmex.py inputs/input.integer_axisymmetric_current`; `python3 benchmarks/plot_vmex_recovery.py`. The two environment variables are defined privately in the Git-local handoff note. The reference suite was rerun and still passed 29 tests.

**Input results and failed attempts:** a Fourier ladder showed the first B/C boundaries were underresolved. For B, `(max_m,max_n)=(16,96)` gave held-out maximum 8.59e-9 m. For C, `(24,100)` gave 3.34e-7 m; a trial with `n=112` reached 9.80e-8 m but VMEX rejected its RBC subscript because its declared boundary limit is `|n|<=101`. The supported `n=100` candidate clears the 1e-6 m smoke gate. All 28 regenerated decks passed parser/setup checks of boundary coefficients, pressure, signed flux, iota or normalized current, with no unexpected theta flip. Artifact: `results/inputs/vmex_parser.json`. This does not certify the boundary contribution to every final solver output.

**Recovery measurements:** axisymmetric integer prescribed-iota cold NS=33/65/129 converged with final FSQR components below 1e-14. Native-sample `(E_B,E_J,F/RMS(grad p_exact))` were `(5.76e-5,2.58e-2,2.88e-1)`, `(7.90e-6,2.49e-3,2.76e-2)`, `(6.51e-7,8.24e-5,8.18e-4)`. At NS=129, `FSQR=9.66e-15`, `FSQZ=3.53e-15`, `FSQL=1.71e-16`; native sample flux-label error was 3.17e-6. The final NS=129 sample meets the initial single-run targets. Current-prescribed multigrid NS=65 converged with `(7.97e-6,2.49e-3,2.75e-2)`, failing the current/force physical targets at that resolution. The near-axis Gauss point dominated the NS=65 current/force error. The first uniform-radial sampler gave an incomplete-volume weight sum, so it was replaced by Gauss radial quadrature and both closure runs were repeated. Solver portions took 2.42-6.07 s per process; native sampling/scoring took about 43-44 s and dominated wall time. Peak memory was not captured. Records and small sampled arrays are in `results/vmex/`; WOUT files are local and excluded from Git.

**Independent checks and figure:** the scorer evaluates the analytical field at the same Cartesian points and obtains J and grad p independently. `figures/vmex_axisymmetric_recovery.png` was generated from saved run records, visually inspected, and linked to source hashes in `results/vmex/figure_manifest.json`. The volume quadrature sums to 0.308425146 m3 for the axisymmetric case, matching its independent exact volume at the shown precision. These measurements are nonlinear recovery evidence for one case, not a projection or broad capability certificate.

**Decision and next exact action:** keep P0-P3 partial. Complete semantic review of the native field, input, setup, profile and solver paths with tests; implement an exact-state projection to separate representation from recovery; then run axisymmetric current at NS=129 and start asymmetric Solov'ev with both closure modes. Before any wider 3-D or gradient study, verify source/field error floors and final input-fit effects on scored observables. The present benchmark working tree has unpublished changes on `main`; no upstream PR exists.

### Entry 2026-09-23: symmetric projection and second closure recovery

**Phase / run IDs:** P0, P2, P3; G01a and S01c. Benchmark base commit `d04e42e`; VMEX baseline remains clean at `b5f5267efc0795c4a49a224e321e9b370975c14c`. No upstream branch or PR was created.

**Changed and commands:** added `benchmarks/projection.py`, which builds the exact symmetric Solov'ev regular series in VMEX's continuous basis, then calls its existing field evaluator and strong-force certificate. The radial and straight-field-line derivation is attributed to the pinned VMEX projection test in `NOTICE.md`. Commands: `PYTHONPATH="$VMEX_BASELINE" "$BENCHMARK_PYTHON" benchmarks/projection.py`; `PYTHONPATH="$VMEX_BASELINE" "$BENCHMARK_PYTHON" benchmarks/run_vmex.py inputs/input.integer_axisymmetric_current 129`. Local baseline tests used `"$BENCHMARK_PYTHON" -m pytest -q tests/test_strong_force_solovev.py` and two selected native interior tests. Private environment variable values remain in the Git-local handoff note.

**Projection results:** on 96 held-out Cartesian points, degree-5, mmax=12, spans 2/4/8 gave B relative L2 `1.589e-12`, `1.065e-12`, `1.065e-12` without nonlinear solving. Strong-force absolute RMS was `5.17e-4`, `1.15e-4`, `1.13e-4` N/m3 and its bounded normalized L2 was `1.42e-9`, `2.96e-10`, `2.87e-10`; the final change is a floor, not a demonstrated asymptotic order. Radial quadrature differences were `3.56e-5`, `5.69e-5`, `1.65e-3`. Artifact: `results/projection/solovev_symmetric.json`. This continuous basis is distinct from the solved state's radial/field interpolation; these results cannot be substituted for the latter's error.

**Source-test results and limitations:** the pinned upstream exact Solov'ev projection module passed 7 tests in 75.86 s. Two selected native interior tests gave 1 pass, 1 skip, 2 fallback accuracy warnings in 79.84 s; the skipped high-derivative test is marked `full` and was not executed. `results/projection/upstream_solovev.json` records the counts. The review ledger now has 5 partial entries, none complete; most VMEX and adjacent source paths remain unreviewed semantically.

**Current-closure recovery:** cold current-prescribed NS=129 converged with `(FSQR,FSQZ,FSQL)=(9.89e-15,3.73e-15,1.73e-16)`. The same native sampler gave `(E_B,E_J,F/RMS(grad p_exact))=(6.89e-7,8.25e-5,8.18e-4)` and maximum VMEX inverted-s difference `3.14e-6`; the initial single-run targets pass. Record and sampled array: `results/vmex/integer_axisymmetric_current_ns129/`. This does not close the full P3 matrix or P2 field-derivative gate. Peak memory remains unmeasured.

**Decision / exact next action:** extend the exact projection to the genuinely asymmetric and 3-D representations or record their projection floors; run `PYTHONPATH="$VMEX_BASELINE" "$BENCHMARK_PYTHON" benchmarks/run_vmex.py inputs/input.solovev_asymmetric_iota 33` as a bounded first recovery rung, then score and decide whether to refine to NS=65/129. Continue the semantic module/test ledger before relying on derivative or free-boundary paths. Current benchmark changes are uncommitted on `main`; no PR is open.

### Entry 2026-09-23: asymmetric Solov'ev route separation

**Phase / run IDs:** P2-P3; G02a/G02b and S01d. Benchmark base commit `d04e42e`; pinned VMEX unchanged. The first `NS=33` asymmetric prescribed-iota solve converged at `(FSQR,FSQZ,FSQL)=(9.81e-15,5.37e-15,1.11e-15)`, but native Cartesian postprocessing raised `NotImplementedError: live-state field evaluation supports lasym = False only`. This is an API limitation, not a failed nonlinear solve.

**Independent branch and failure analysis:** the existing `surface_field_data_from_wout` supports LASYM and evaluates B from full-mesh geometry plus adjacent half-mesh spectra. `benchmarks/score_wout_surfaces.py` scored five surfaces per run against the independent exact Cartesian field. The worst area-weighted surface B relative L2 errors for `NS=33,65,129` were `1.2615e-4`, `3.1471e-5`, `7.8566e-6`; `NS=129` passes the initial B target on these sampled surfaces. The file route supplies no continuous-volume J or force certificate. Saved outputs: `results/vmex/solovev_asymmetric_iota_ns*/wout_surface_scores.json`. `figures/vmex_lasym_surface_B.png` was generated from those JSON files, inspected, and checksummed by `results/vmex/lasym_figure_manifest.json`.

**Failed fitted-state attempts:** using `lift_high_order_state` with a fixed eight-span cap gave B relative L2 `3.61e-2` at NS=33 and `7.81e-1` at NS=65; J/force were worse. Those arrays and records are preserved with `span8_` names. Allowing 14 and 30 spans improved the fitted B errors to `1.66e-4` and `1.82e-5`, but J relative errors stayed `4.79e-3` and `3.65e-3`, with force ratios `5.69e-2` and `4.27e-2`; these fail the first physical targets. A fixed 32-span cap at NS=129 worsened B error to `9.92e-4` and was preserved as `span32_` artifacts. Removing the cap hit the lift's rank guard (`rank 65/66`); no NS=129 continuous score was accepted. A WOUT-based span sweep gave B error `1.60e-5` at 32 spans for NS=65 and `1.40e-4` at 40 spans for NS=129, while larger NS=129 bases became underdetermined. These are failed representation attempts, not evidence that the original discrete equilibrium worsens with refinement.

**Commands, status and next action:** run `PYTHONPATH="$VMEX_BASELINE" "$BENCHMARK_PYTHON" benchmarks/score_wout_surfaces.py results/vmex/solovev_asymmetric_iota_ns129/wout.nc solovev_asymmetric` to reproduce the final surface score; `python3 benchmarks/plot_lasym_surfaces.py` regenerates its figure. The source ledger now has 7 partial code/config entries and no complete semantic module reviews. Next, compare the direct live half-mesh field/geometry at interior surfaces against WOUT, then design a narrow LASYM Cartesian field implementation or an independently certified continuous interpolation before scoring volume J/force. Complete the source/test review for that path and open a narrow upstream PR only with a failing reproducer and a verified fix. Other P3 cases remain unrun; P4 and later phases have not started. Current changes are uncommitted on benchmark `main`; no upstream branch or PR exists.

### Entry 2026-09-23: sheared chart domain and inverse check

**Phase / run ID:** P1, R04. Benchmark base commit `3d374be`; no source pin changed. Added `benchmarks/verify_sheared_chart.py` and ran `python3 benchmarks/verify_sheared_chart.py` from the repository root. It tested every sheared reference variant on 513 toroidal nodes, 64 poloidal labels and five radial fractions, then 1024 random physical-angle inversions per case. Artifact: `results/reference/sheared_chart.json`.

**Results:** all six cases passed sampled monotonicity, full 2π angle coverage, positive radius, physical-angle inversion and analytical label checks. The smallest finite-difference angular slope normalized by the t-step was 0.436 in sheared C. Its smallest sampled cylindrical radius was 1.165 in reference length units, and its largest label error divided by edge label was 2.34e-14. This is sampled evidence, not a proof for every point in the domain or a derivative of the bisection input map.

**Failed attempt and correction:** the first script version omitted JAX float64 configuration and reported 1e-6-scale label discrepancies. Enabling float64 before importing the oracle reduced them to roundoff-scale values; the initial numbers were precision artifacts and were not accepted. Two initial script errors (a missing parenthesis and unbroadcast arrays) were corrected before the measured run. No solver or GPU was used in this block.

**Next exact action and state:** add the implicit derivative of the sheared physical-angle scalar root, with a nonzero angular derivative certificate and fixed branch, then compare its boundary response against real central differences at more than one step. Continue the LASYM live-field diagnosis and source/test ledger before treating the broader P2/P3 gates as complete. The benchmark changes are uncommitted on `main`; no upstream PR exists.

### Entry 2026-09-23: sheared implicit-root directional derivative

**Phase / run ID:** P1, R05. Benchmark base commit `3d374be`; source pins unchanged. Added `benchmarks/sheared_map.py` with a scalar implicit-root custom JVP and `benchmarks/verify_sheared_derivative.py`. The primal uses fixed-quadrant bisection; the tangent solves the angle-crossing equation with its local nonzero angular derivative. Ran `python3 benchmarks/verify_sheared_derivative.py` from the repository root. Artifact: `results/reference/sheared_derivative.json`.

**Results:** the six sampled A/B/C points passed position agreement with the independently implemented surface inversion, nonzero angular derivative and central-difference checks at `h=1e-3,1e-4,1e-5,1e-6`. The smallest directional-position errors for each point ranged from `7.38e-13` to `1.95e-12` in reference length units; all central-difference series showed roundoff growth at the smallest step. The check perturbed radial label coordinate, poloidal angle, physical toroidal angle and three sheared parameters together. It establishes local sample derivatives, not a global branch-boundary derivative, shape derivative at fixed physical point or VMEX equilibrium response. No solver or GPU was used; runtime was 2.1 s, memory unmeasured.

**Failed attempts, branch state and next exact action:** no failed derivative candidate was retained. Commit and publish the reviewed benchmark block with owner identity; then add explicit bad-domain and branch-boundary cases and resume the LASYM live-state field diagnosis/source ledger before additional recovery claims. No upstream branch or PR exists.

### Entry 2026-09-23: sheared near-domain graph failure and input guard

**Phase / run ID:** P1, R06. Benchmark base commit `aacde00`; source pins unchanged. Added `benchmarks/verify_sheared_domain_edges.py` and a sampled graph guard in `benchmarks/analytic.py`, invoked by `benchmarks/build_inputs.py`. Ran `python3 benchmarks/verify_sheared_domain_edges.py`, `python3 -m pytest -q` (29 passed in 10.78 s), and `python3 benchmarks/build_inputs.py` (all 14 cases regenerated for two closures). Artifact: `results/reference/sheared_domain_edges.json`; no inputs changed in Git.

**Measured finding and failed attempt:** the first near-domain check expected all analytically admissible margins to remain physical-angle graphs and failed. The C-family margins `S-asin(sqrt(2 edge)) = 0.001, 0.01, 0.1` instead had sampled minimum normalized angular slopes `-3.224, -2.222, -0.209`. Their selected target angles each had three chart crossings, although the chart radius stayed positive and the existing smooth-domain validator accepted the parameters. Margins `0.2, 0.5, 1.0, 3.073` had sampled minimum slopes `0.0137, 0.0685, 0.1565, 0.4355`. The added guard rejects the three folded samples before producing VMEX input decks; five invalid-parameter cases were independently rejected by the analytical validator. Fifteen samples at toroidal quadrant boundaries and offsets agreed with the independent bisection surface to roundoff at one safe margin. The guard is a finite-grid check, not a global proof or a certified threshold. Runtime of the final probe was 2.0 s; memory unmeasured.

**Branch/PR state, blockers and next exact action:** benchmark changes are uncommitted on `main`; no upstream branch or PR exists. Review staged diff and publish this block as owner. Then compare direct live LASYM state surface geometry/field against its WOUT result and complete the reachable VMEX source/test ledger before a volume current/force implementation. Do not treat the failed fitted-state lift as a nonlinear recovery failure.

### Entry 2026-09-23: LASYM restart surface consistency

**Phase / run ID:** P0/P2, G03a. Benchmark base commit `dca6df4`; source pins unchanged. Added `benchmarks/check_lasym_roundtrip.py`, read the WOUT/restart/surface and interior-field source paths stated in `results/audit/review_ledger.json`, and updated only the inspected ranges in that ledger. Ran the checker on the asymmetric Solov'ev prescribed-iota WOUTs at NS=33,65,129 with the pinned VMEX Python environment. Each run took 3.0-3.2 s; memory unmeasured. Three selected pinned upstream restart tests passed in 2.88 s. Artifacts: `results/vmex/solovev_asymmetric_iota_ns{33,65,129}/lasym_roundtrip.json`.

**Results and limits:** WOUT to native restart state to WOUT preserved sampled surface geometry to at most `1.39e-17` m and surface B to at most `3.42e-14` relative L2, across five surfaces at each NS. At NS=129 the original WOUT surface field errors against the independent analytical B ranged from `5.20e-7` to `7.93e-6` in an unweighted surface L2 measure. This is a value-only serialization/field consistency result; it does not exercise a live field from an in-memory solve or supply continuous-volume J, pressure gradient or force. Source inspection found that the current live spectra constructor rejects LASYM and the interior geometry/native field evaluator uses only the symmetric R/Z/lambda harmonics. Merely removing the guard would be invalid. The source ledger has 9 partial code/config/test entries and no complete semantic module review.

**Failed attempts, branch/PR state, blockers and next exact action:** no failed candidate was retained in this block. Benchmark changes are uncommitted on `main`; no upstream branch or PR exists. Review staged diff and publish as owner, then build a minimal direct live LASYM surface parity check against WOUT before implementing an asymmetric continuous Cartesian field. Keep the fitted-state volume failure separate from the successful WOUT surface result.

### Entry 2026-09-23: cold integer 3-D solve failure

**Phase / run ID:** P3, S02a. Benchmark base commit `0845dc5`; source pins unchanged. Ran `benchmarks/run_vmex.py inputs/input.integer_3d_iota 33` with the pinned VMEX Python environment and its original 30,000-iteration limit. VMEX reported an initial Jacobian sign change, improved the magnetic-axis guess, repeatedly reduced its step after force components near `1e-12`, and ended with `VmecConvergenceError: MORE ITERATIONS REQUIRED`. The final terminal FSQR/FSQZ/FSQL were `1.15e-10 / 8.18e-11 / 4.54e-11`. Runtime was approximately 91 s from command waits, not measured precisely; memory unmeasured. No WOUT or physical scorer artifact was emitted because the solve did not return a converged state.

**Change and reproducible failure:** `benchmarks/run_vmex.py` now accepts an explicit iteration cap and saves a structured `forward.json` when VMEX raises `VmecConvergenceError`. A repeated cold NS=33 run capped at 3,000 iterations took 12.44 s including first compilation, ended with the same exception, and last printed components `3.62e-9 / 1.87e-9 / 8.87e-10`. Artifacts: `results/vmex/integer_3d_iota_ns33_niter3000/forward.json` and `attempt_history.json`. These are failed solver attempts, not numerical-reference or exact-equilibrium failures. No Cartesian field, current or force was scored.

**Branch/PR state, blockers and next exact action:** benchmark changes are uncommitted on `main`; no upstream branch or PR exists. Review staged diff and publish as owner. Next, construct and verify a field-consistent exact-state projection for integer 3-D, then use it as a warm initialization at NS=33; inspect the solver's admissible axis and Jacobian before any longer cold run. Resume the LASYM direct live-surface diagnosis independently.

### Entry 2026-09-24: integer 3-D field-consistent projected state

**Phase / run IDs:** P0/P2, G04. Benchmark base commit `be9c840`; VMEX source pin unchanged. Added `benchmarks/integer_surface_projection.py`, `benchmarks/project_integer_vmex.py` and `benchmarks/plot_integer_projection.py`. The exact geometric chart's Cartesian tangents and exact B give toroidal/poloidal flux derivatives and both angular derivatives of lambda. Three surfaces at `s=0.2,0.5,0.8` gave `chip/phip=-2` within `1e-10`; the residual lambda gradient was `8.81e-11` to `9.56e-11` of the flux scale with `h=1e-5`. The first projection check divided by the almost-zero lambda gradient itself and falsely failed at relative 0.61-0.73; the corrected declared flux scale gave the meaningful result. A broadcasting error in its first field reconstruction call was fixed before the measured run. Artifact: `results/projection/integer_3d_surface.json`.

**State conversion and measurements:** sampled exact full-mesh R/Z Fourier coefficients were mapped through VMEX's mode normalization and m=1 constraint with lambda zero. At NS=33 the resulting half-mesh Jacobian had one sign (`tau` from `-0.01101` to `-0.00639`), unlike the cold initializer. Generated state arrays and independent native volume scores are in `results/projection/integer_3d_vmex_ns{33,65,129}/`. Native 96-point volume B relative errors were `2.59e-5, 2.98e-7, 2.42e-7`; J relative errors were `3.55e-3, 9.72e-5, 7.14e-6`; pressure-normalized force ratios were `3.78e-2, 1.03e-3, 8.15e-5`. The NS=129 projected state meets the initial B/J/force targets, without any nonlinear solve. Separate WOUT surface B maxima were `1.42e-4, 3.55e-5, 8.86e-6`; the first two fail the surface B target even though native volume B at NS=65 passes. Memory unmeasured. The source ledger now has 14 partial entries, no complete semantic review.

**Figure and next action:** `python3 benchmarks/plot_integer_projection.py` generated `figures/vmex_integer_3d_projection.png` from saved JSON, visually inspected it and recorded its SHA in `results/projection/integer_3d_figure_manifest.json`. Continue with strict warm recovery from the projected state, keeping the no-solve scores separate.

### Entry 2026-09-24: strict warm failure and loose-root physical gap

**Phase / run IDs:** P3, S02b/S02c. Extended `benchmarks/run_vmex.py` to accept a projected state seed and an explicitly named `BENCH_FTOL` override, recording seed SHA, tolerance, initialization and failure status. At NS=33 with the strict `1e-14` target, the projected warm solve avoided the cold Jacobian sign error but ended `MORE ITERATIONS REQUIRED` at 3,000 iterations. Its printed FSQR/FSQZ/FSQL reached about `9.00e-13 / 9.50e-13 / 7.12e-13` at iteration 1,000 and drifted to `1.39e-10 / 9.23e-11 / 6.17e-11` by 3,000. Artifact: `results/vmex/integer_3d_iota_ns33_niter3000_projected/attempt_summary.json`; no strict solved field was scored.

**Exploratory loose roots:** with explicit `BENCH_FTOL=1e-10`, the warm solve returned at iterations 311, 335 and 420 for NS=33,65,129. Their native 96-point volume B relative errors were `8.71e-4, 7.06e-4, 1.30e-4`; J relative errors `0.439, 0.149, 2.92e-3`; force ratios `4.93, 1.70, 1.74e-2`. All fail the initial physical targets and are not accepted recoveries. The reference-to-VMEX normalized surface-label discrepancies reached `0.00424, 0.00470, 0.00534`, while projected seed discrepancies were at most `1.45e-6` on the same volume points. Saved WOUTs and native samples remain in `results/vmex/integer_3d_iota_ns{33,65,129}_niter3000_projected_ftol1e-10/`; the solver times were 3.34, 4.33 and 6.87 s, and native sampling took 45.73, 46.38 and 47.79 s. Memory unmeasured. The result may involve a weakly controlled trajectory, gauge or branch; no cause is established from these scores alone.

**Branch/PR state, blockers and next exact action:** benchmark changes are uncommitted on `main`; no upstream branch or PR exists. Run the repository tests and JSON checks; inspect staged diff and `tools/publish.sh`, then publish as the verified owner. Next, compare the raw unpreconditioned VMEX force/residual of the projected seed and loose roots under the identical grid and gauge, and probe short warm trajectories before changing the convergence rule. Do not relax acceptance from the loose-root FSQ alone. Continue the direct live LASYM surface parity check separately.

### Entry 2026-09-24: invariant-force comparison on projected and WOUT-restarted states

**Phase / run ID:** P3, S02d. Benchmark base commit `10e1b02`; pinned VMEX commit `b5f5267efc0795c4a49a224e321e9b370975c14c`. Added `benchmarks/compare_integer_residuals.py` and `results/vmex/integer_3d_raw_residual_comparison.json`. Ran the comparison in the pinned VMEX Python environment with the same prescribed-iota deck, profile, radial grid, and fixed-boundary setup at each NS. `vmex.core.solver.evaluate_forces` measured invariant unpreconditioned FSQR/FSQZ/FSQL for the saved projected seed and a state rebuilt from each loose-root WOUT. Compact restarted spectral states are saved beside each loose-root forward record; `BENCH_USE_SAVED_STATE=1` reproduced the comparison from those public arrays without requiring the ignored WOUT files. This is a diagnostic of VMEX's discrete force; it is distinct from the Cartesian physical scorer and from the 3-D exact field.

**Results and checks:** projected `(FSQR,FSQZ,FSQL)` were `(0.253876,0.919231,2.88e-8)` at NS=33, `(0.509236,1.843785,3.50e-9)` at NS=65, and `(1.473595,5.335401,4.34e-10)` at NS=129. WOUT-restarted loose roots gave all three components near `1e-10`; recomputed minus saved solver components were at most `6.41e-15`. No Jacobian sign change was reported for either state at any NS. The projected NS=129 Cartesian B/J/force scores remain small while its discrete R/Z force is large, so a low physical sampling error does not imply a small VMEX discrete residual. Conversely, the loose root's low VMEX residual did not imply accepted physical recovery. The state shifts in all six spectral blocks and energy scalars are saved. No root cause is assigned yet. Memory unmeasured; no new nonlinear solve was run in this block.

**Branch/PR state, blockers and next exact action:** the preceding block was published to benchmark `main` as `10e1b02`; this diagnostic is pending publication on `main`. No upstream branch or PR exists. Run the repository tests and JSON validation, inspect staged diff and `tools/publish.sh`, then publish with the verified owner. Next, evaluate the projected state's force blocks by radial location and mode, and record a short warm trajectory using a bounded public solver path before changing any convergence rule. Test whether the observed residual is concentrated at the axis, edge, a flux convention, or a coordinate/gauge mode. Keep all physical acceptance gates unchanged. Continue direct live LASYM surface parity independently.

### Entry 2026-09-24: warm-start baseline correction and bounded trajectory

**Phase / run ID:** P3/P10, S02d/S02e. Benchmark base commit `2ca5de5`; VMEX pin unchanged. The preceding invariant-force diagnostic was published before noticing that `solve(initial_state=...)` applies `hot_restart_state` and `runtime_with_baselines` before its first force evaluation. Corrected `benchmarks/compare_integer_residuals.py` to make those same calls separately for each state. Recomputed all three grids. FSQ values moved only at roundoff: at NS=33 projected `(0.253876,0.919231,2.878e-8)` and at NS=129 `(1.473595,5.335401,4.342e-10)` remain. The transferred boundary's R/Z coefficient change was `2.50e-15` at each grid. WOUT-restarted solver residual agreement remained within `6.41e-15`. The original conclusion holds for this correction; the earlier method was incomplete and is superseded by the revised artifact.

**Bounded run and independent score:** `benchmarks/probe_integer_warm.py` ran the pinned single-grid solver from the NS=33 projected seed at explicit `FTOL=1e-4`, 500-iteration limit, returning after 65 iterations with `(FSQR,FSQZ,FSQL)=(4.62e-5,9.11e-5,2.58e-6)`. Its saved 65-row force history, state, native 96-point sample arrays and score are in `results/vmex/integer_3d_short_warm_ns33_ftol1e-4/`. Native B relative error `1.51e-3`, J relative error `0.326`, and force over exact pressure-gradient RMS `3.72` worsened from the projected seed's `2.59e-5`, `3.55e-3`, and `3.78e-2`. The solve took 2.43 s including first compile; native sampling and scoring took 45.92 s; peak memory unmeasured. The short run is not an accepted equilibrium. `benchmarks/plot_integer_trajectory.py` generated `figures/vmex_integer_3d_trajectory.png` from saved history and scores; it was visually inspected and its SHA recorded in the figure manifest.

**Checks, state and next exact action:** the repository's 29 tests passed after the force-comparison correction; the saved-state path reproduced the WOUT-derived residuals. The benchmark repository has this block unpublished on `main`; no upstream branch or PR exists. Validate JSON and staged diff, inspect `tools/publish.sh`, and publish as the verified owner. Then localize the NS=33 projected force by radial row and Fourier mode using VMEX's force-pipeline outputs under the corrected warm-start runtime. Compare axis, interior and edge contributions before choosing an upstream fix or any solver modification. Keep physical acceptance gates unchanged. Continue direct live LASYM surface parity as a separate P2 experiment.

### Entry 2026-09-24: spectral force localization

**Phase / run ID:** P3/P10, S02f. Benchmark base commit `e2f8330`; VMEX pin unchanged. Added `benchmarks/localize_integer_force.py` to evaluate the pinned solver's scaled spectral force blocks at NS=33 on the projected seed, iteration-65 state and loose root. This diagnostic uses private VMEX `_field_chain_lane`, `_geometry` and `_force_pipeline` after public `hot_restart_state`, `runtime_with_baselines` and `evaluate_forces`; it is pinned-source dependent and does not change VMEX. Its first run failed because absent symmetry blocks are `None`; filtering them gave the measured run. For all three states and R/Z/lambda, the summed squares of saved block marginals reproduce `evaluate_forces`'s unpreconditioned `gcr2/gcz2/gcl2` within the declared numerical check. Full radial, poloidal and toroidal fractions and top cells are saved in `results/vmex/integer_3d_force_localization_ns33.json`.

**Result and interpretation:** for the projected seed, radial rows 16–31 of 0–32 carry 86.08% of R and 85.72% of Z force sums; rows 0–3 carry 0.0098% and 0.0107%. Poloidal modes 3–6 carry 85.11% and 86.95%; toroidal modes 0–4 carry 99.87% and 99.99%. The single largest R/Z cell is at radial row 22, m=3, |n|=0. Lambda's much smaller force is concentrated near the axis, a different distribution. This rules out an R/Z residual confined to the axis or fixed edge; it does not distinguish flux/profile convention, finite radial discretization, gauge or force-kernel error. `benchmarks/plot_integer_force_localization.py` generated `figures/vmex_integer_force_localization.png` from saved JSON; its first layout clipped a long y label, which was shortened and visually rechecked. The final figure SHA is in `results/vmex/integer_3d_force_localization_figure_manifest.json`. No new nonlinear solve or physical scoring occurred in this block; elapsed time and peak memory were not measured.

**Checks, branch/PR state and next exact action:** the benchmark changes are pending publication on `main`; no upstream branch or PR exists. Run tests and JSON checks, inspect staged diff and `tools/publish.sh`, then publish as the verified owner. Next, compare VMEX real-space force kernels against the independent exact `J×B−∇p` on matching radial/angular quadrature at NS=33 and 65, checking flux and pressure normalization explicitly. Separate weak-form aliasing and radial-discretization effects before changing a solver or opening an upstream PR. Keep native physical acceptance gates unchanged and continue live LASYM surface parity separately.

### Entry 2026-09-24: spectral-condensation constraint switch

**Phase / run ID:** P3/P10, S02g. Benchmark base commit `f5cc050`; VMEX pin unchanged. Before attempting the preceding entry's proposed direct Cartesian force comparison, source review showed that `vmex/core/forces.py` returns radially staggered flux-form variational kernels with spectral-condensation terms, not pointwise `J×B−∇p` vectors. A direct componentwise comparison would be invalid; that proposed experiment is superseded. The valid next comparisons are a matching discrete weak-form/energy variation or a physical-field-preserving coordinate remap.

**Diagnostic and results:** added `benchmarks/probe_integer_constraint.py`, which reevaluates the saved projected, loose-root and NS33 iteration-65 states with identical input, warm-start boundary transfer and rebound baselines, changing only `tcon0` from the deck default to zero. The projected `(FSQR,FSQZ)` fell from `(0.253876,0.919231)` to `(8.51e-7,9.99e-7)` at NS33, from `(0.509236,1.843785)` to `(2.09e-7,2.48e-7)` at NS65, and from `(1.473595,5.335401)` to `(5.17e-8,6.18e-8)` at NS129. The lambda force was unchanged. On the loose roots, zeroing the constraint instead raised R/Z residuals by factors of roughly 5–150, so this is not a universal residual reduction. These counterfactual force evaluations did not modify or rescore the states; the full values and applied tcon ranges are in `results/vmex/integer_3d_constraint_switch.json`. They isolate the large projected discrete R/Z residual to the spectral-condensation constraint, without proving that a physically equivalent VMEX gauge or accepted root exists. No new solver run occurred; elapsed time and memory were unmeasured.

**Figure, checks and next exact action:** `benchmarks/plot_integer_constraint.py` generated `figures/vmex_integer_constraint_switch.png` from the saved JSON. The first figure had a redundant second legend, which was removed; the final figure was visually inspected and hashed in its manifest. The source ledger now has 16 partial code/config reviews and no complete semantic review. Current benchmark changes are unpublished on `main`; no upstream branch or PR exists. Run tests and validate JSON, inspect staged diff and `tools/publish.sh`, and publish under the verified owner. Next, test a physical-field-preserving poloidal-coordinate/lambda remap of the exact projected state against VMEX's constraint at NS33; compare native B/J/force before and after remapping, then evaluate default-tcon and zero-tcon residuals. If that fails, compare the VMEX force to an independent *weak-form* energy directional derivative on the same discrete grid. Do not relax P3 physical acceptance. Continue live LASYM surface parity separately.

### Entry 2026-09-24: bounded poloidal gauge projection trial

**Phase / run ID:** P3/P10, S02h. Benchmark base commit `dfb04c3`; VMEX pin unchanged. Added `benchmarks/probe_integer_gauge.py` to define `theta_new = theta_old + u`, with `u=a*s*(1-s)*sin(m*theta_new-n*nfp*phi)`. At the continuous Clebsch level, the compensating physical lambda is `-phipf(s)*u`, keeping the same B when geometry is resampled at `theta_old=theta_new-u`; the script converts lambda using VMEX's `lamscale` and projects R/Z/lambda to the pinned finite Fourier state. This continuum identity is a construction, not a certificate for the finite projected state. Scanned `(m,n)=(2,0),(3,0),(2,1),(3,1)` at amplitudes `-0.2,-0.1,0.1,0.2` on NS33 with the default coordinate constraint, comparing VMEX FSQR/FSQZ/FSQL without a solve.

**Results and failed attempt:** among 16 scanned points, `(m,n,a)=(2,0,-0.1)` minimized FSQR+FSQZ at `0.79057`, down from the unshifted seed's `1.17311` but nowhere near solver acceptance. Its native 96-point `(B,J,force)` scores against the exact field were `(3.12e-5,4.67e-3,5.14e-2)`, slightly worse than the unshifted projected state's `(2.59e-5,3.55e-3,3.78e-2)`. On identical held-out points, selected-versus-unshifted weighted relative differences were B `2.70e-5`, J `3.46e-3`, pressure gradient `1.15e-5`. Thus the finite projection did not preserve sampled fields to roundoff. No scanned state changed Jacobian sign. The selected state, native samples and full scan are in `results/projection/integer_3d_gauge_scan_ns33/` and `results/projection/integer_3d_gauge_scan_ns33.json`. Total scan plus scoring took 46.17 s, of which native sampling took 43.78 s; peak memory unmeasured. This is a failed narrow gauge trial, not an exact recovery or evidence that no suitable gauge exists.

**Figure, checks and next exact action:** `benchmarks/plot_integer_gauge_scan.py` generated `figures/vmex_integer_gauge_scan.png` from saved data and the unshifted residual; its first line plot interpolated across unsampled zero amplitude, so it was replaced with a scatter plot and visually rechecked. The figure SHA is in its manifest. Benchmark changes are unpublished on `main`; no upstream branch or PR exists. Run tests and JSON checks, inspect staged diff and `tools/publish.sh`, then publish as the verified owner. Next, audit the finite lambda normalization and geometry/field preservation on one shifted surface against the exact Clebsch transformation, then test a small multi-mode constrained gauge fit only if that certificate passes. Otherwise use the independent weak-form energy directional derivative proposed above. Keep P3 physical acceptance unchanged and continue direct live LASYM surface parity separately.

### Entry 2026-09-24: continuum gauge identity and finite VMEX surface

**Phase / run IDs:** P2/P10, G05a/G05b. Benchmark base commit `262e86a`; VMEX pin unchanged. Added `benchmarks/verify_integer_gauge_surface.py` for the selected `(m,n,a)=(2,0,-0.1)` shift at `s=0.5`. It differentiates the transformed exact chart in `s`, poloidal angle and toroidal angle, uses the independent exact flux derivatives and `lambda=-phipf*u`, and reconstructs Cartesian B. Analytical evidence is saved separately in `results/projection/integer_3d_gauge_surface_continuum.json`; projected VMEX state evidence is in `results/projection/integer_3d_gauge_surface_vmex.json`. The latter uses `surface_field_data_from_state` on the saved finite NS33 state and compares geometry at the same gauge-transformed angular points plus B against the exact field at the VMEX points. There was no nonlinear solve.

**Measured convergence and representation gap:** with central-difference steps `2e-4,1e-4,5e-5,2e-5,1e-5`, continuum reconstructed B relative errors were `1.82e-8,4.54e-9,1.12e-9,1.80e-10,1.05e-10`; this agrees with second-order convergence until roundoff. The exact chart's minimum oriented Jacobian was about `-0.00934`, with no observed sign change. VMEX's parsed `phipf=-0.006765823467065927` differs from the independent exact `-0.00676582346718246` by `1.17e-13`, and its `lamscale=0.006765823467065927`. On a 16×32 surface grid, the finite state's geometry maximum absolute difference from the transformed exact surface was `2.06e-9`; its B relative error was `2.06e-5`, compared with `1.57e-5` for the unshifted VMEX state at the same surface. Thus the continuum transformation is certified on this one surface, but the finite VMEX surface field does not meet the initial `1e-5` target. This does not certify the whole volume or a suitable gauge.

**Figure, checks and next exact action:** `benchmarks/plot_integer_gauge_surface.py` generated `figures/vmex_integer_gauge_surface.png` from the two separate records; it was visually inspected and hashed in its manifest. The source ledger now has 17 partial code/config reviews and no complete semantic review. The benchmark changes are unpublished on `main`; no upstream branch or PR exists. Run tests and JSON checks, inspect staged diff and `tools/publish.sh`, then publish as the verified owner. Next, derive the VMEX spectral-condensation target from its `rcon/zcon` source and solve a bounded multi-mode gauge fit at NS33 while checking both exact-gauge continuum invariance and finite-state native B/J/force after each accepted trial. If representation error grows, stop and use the independent weak-form energy directional derivative. Keep P3/P2 gates unchanged, and continue live LASYM surface parity separately.

### Entry 2026-09-24: pin DESC and plan matched coordinate tests

**Phase / run IDs:** P0/P3, S03a–S03c planned. Benchmark base commit `b78350d`; primary VMEX pin unchanged. The upstream DESC default branch is `master`. A separate clean DESC checkout was fast-forwarded with `git pull --ff-only origin master` from `ad105c5` to `4f48720beac3d4169e9165923d730445118bc2de`; its resulting worktree was clean. Another existing DESC checkout had uncommitted changes and was left untouched. `sources.json` now pins the clean revision. A targeted read covered the current `Equilibrium.solve` default, `ForceBalance` setup, optional `FixThetaSFL`, and boundary-fitting angle input; no full DESC source/test review is claimed.

**Decision and changes:** added the S03a–S03c protocol to P3 and `benchmark_matrix.json`, a planned F18 figure, and the limited DESC review scope to `docs/SOURCE_REVIEW.md`. The protocol separates exact continuum reparameterization, finite native projection and actual nonlinear recovery, and compares VMEX constraint strengths with native DESC/VMEC2000/VMEC++ and, when pinned and runnable, GVEC. A single-mode remap's failure does not establish that the analytical input cannot be reparameterized. A same-state `TCON0=0` residual drop does not establish that removing the constraint improves a solved physical equilibrium. A DESC-like redesign or VMEX default change remains a decision gated by accepted roots, physical accuracy, resolution and coordinate stability.

**Commands, results, artifacts and next action:** `git pull --ff-only origin master` completed in 2.7 s; `python -m json.tool sources.json`, `python -m json.tool benchmark_matrix.json`, and `git diff --check` passed. No DESC import, reference calculation, projection, solve, GPU run, timing study or memory measurement was performed. No numerical artifact or figure was generated. Benchmark branch was `main`; no upstream branch or PR was created. Next, complete the native DESC input/field/source-test review and run S03a's continuum-versus-finite projection control at the same held-out Cartesian points, beginning with the axisymmetric case and integer 3-D at NS=33. Record the physical field and Jacobian before any cross-code solve or constraint-strength recommendation.

### Entry 2026-09-24: initial native DESC coordinate comparison

**Phase / run IDs:** P0/P3/P10, S03a and S03b diagnostic subset. Benchmark base commit `12e75529`; VMEX remains pinned at `b5f5267efc0795c4a49a224e321e9b370975c14c`; DESC pinned at `4f48720beac3d4169e9165923d730445118bc2de`. The DESC driver uses x64, JAX `0.6.2`, SciPy `1.15.3`, NumPy `2.2.6`, DESC `0.17.3+27.g4f48720be`, and reported `cuda:0`. Fixed iota profile is `+2` in DESC's orientation. Physical scoring uses the same 96 held-out Cartesian points and exact reference arrays as the corresponding VMEX cases.

**Changes:** added `benchmarks/run_desc_coordinate.py` for the original and smooth compensated remapped charts; `benchmarks/score_desc_boundary.py` for a dense native LCFS fit check; and `benchmarks/plot_coordinate_comparison.py`. Saved each DESC projected score separately from solved/terminal output under `results/desc/coordinate/`, including compressed physical samples and the small (120–132 KiB each) HDF5 states. The HDF5 files were force-added because the repository-wide `*.h5` ignore rule would otherwise omit these reproducibility artifacts. Added source/test review hashes, matrix statuses and two publication-quality README figures. The remap is `theta_new=theta_old+0.1*s*(1-s)*sin(2*theta_new)` and lambda is adjusted by `-u`; the analytical boundary is unchanged. The initial remap implementation sampled geometry at the new rather than old angle and produced grossly incorrect projection errors. Source inspection caught that parameterization bug; those measurements were discarded, the mapping corrected, and all remapped cases rerun. A first DESC custom-grid scoring attempt also failed because integration-dependent quantities were requested with `override_grid=False`; the driver now uses DESC's documented quadrature-grid override and transfers dependencies to the held-out nodes. An initial `JAX_PLATFORMS=cuda` launch failed at platform initialization; subsequent reported runs selected the GPU through DESC before importing equilibrium components.

**Results:** the DESC run tags L6/L8 denote angular `M=N=6/8`, with radial `L=8/10`. Integer-3D base/remapped M6 projections gave B/J relative L2 `4.837e-4/1.101e-2` and `5.398e-4/3.546e-2`; the corresponding converged force-balance outputs were `1.3013e-4/1.4431e-3` and `1.3009e-4/1.4418e-3`. Thus this chart change worsens the finite projection but the M6 solver outputs agree within about `0.1%` on these sampled physical errors. At M8, base projection/solver errors were `4.694e-5/9.647e-4` and `1.052e-5/1.482e-4`; the remapped projection was `5.853e-5/4.098e-3`. Its terminal state at 120 iterations scored `1.599e-5/2.246e-4`, but optimization reported `success=false`, optimality `5.25e-8` for `gtol=1e-8`; it is not a completed solve. DESC dense LCFS maximum errors were `5.95e-5 m` at M6/L8 and `4.15e-6 m` at M8/L10, with base/remapped values matching to reported precision. Minimum Jacobian over scored 3-D points was positive, between `3.95e-3` and `4.06e-3`; this is not a global orientation certificate. DESC axisymmetric M6/L8 projection and solved B/J were `1.118e-5/2.150e-4` and `1.991e-6/8.717e-6`. For comparison, VMEX integer-3D NS129 projection and loose-root B/J were `2.416e-7/7.138e-6` and `1.299e-4/2.921e-3`; these different bases and resolutions do not establish solver ranking. Prescribed profiles set iota exactly, so the DESC report does not call that an independently measured transform.

**Timings and memory:** DESC projection-plus-solve wall time ranged from `69.45 s` (axisymmetric M6/L8) to `121.45 s` (remapped M8/L10, 120-iteration cap); reported process peak RSS ranged from `3919` to `4041 MiB`. Dense boundary checks took `3.39–4.58 s` with process peak RSS `1739–1740 MiB`; no GPU-memory counter was collected. Figures `figures/coordinate_solver_comparison.png` and `figures/coordinate_axisymmetric_control.png` were generated from the saved JSON scores, visually inspected, and hashed in `results/desc/coordinate/figure_manifest.json`.

**Source/tests and failures:** updated `results/audit/review_ledger.json` with eight partial DESC source/test entries. Read the pinned solve/default constraint setup, force objective, surface angle fit, initial-guess interface, custom-grid dependency handling, B/J/grad-p definitions, plus the surface-fit and solve/load test paths. `tests/test_surfaces.py::TestFourierRZToroidalSurface::test_surface_from_values` passed (1 test, 9.53 s). The targeted solve/load test could not be collected because `tests/test_equilibrium.py` imports `Qic`, absent from the installed PyPI `qic` package; its test module remains unavailable in this environment. This review is explicitly partial. An early 60-iteration remapped L8 attempt is saved separately; it also hit the cap and is superseded by the 120-iteration terminal attempt, not relabeled as convergence.

**Decision, branch/PR state and next exact action:** DESC can form and optimize the same physical boundary from both charts without VMEX's explicit spectral-condensation constraint. The finite projection responds to the remap, especially in J, while at M6 the converged DESC physical scores return to nearly the same values. That is evidence the coordinate effect is not unique to VMEX, but it neither proves mismatch harmless nor justifies removing/downweighting VMEX's constraint: only one remap, fixed-iota closure, incomplete derivative and resolution studies, and one capped DESC run are represented. S03a–S03b remain diagnostic-only; S03c is unstarted. The results/code/README commit was published to benchmark `main` as `8dcb5f5` under the verified owner identity; no PR or upstream changes exist. This status was written in the follow-up logbook commit. Next, extend the independent continuum remap checker to the exact DESC remap and certify B/J invariance and `d(theta_old)/d(theta_new)>0` over the sampled volume; if it passes, run the VMEX cold/certified-start `TCON0` ladder at NS33 with the same scorer, then expand the DESC/current-closure and additional-code matrix.

### Entry 2026-09-24: sampled volume certificate for the compensated remap

**Phase / run IDs:** P2/P3/P10, G05c and S03a. Benchmark base commit `eb16374`; VMEX pin remains `b5f5267efc0795c4a49a224e321e9b370975c14c`, DESC pin remains `4f48720beac3d4169e9165923d730445118bc2de`. Added `benchmarks/verify_integer_gauge_volume.py`, which reconstructs the integer-family field from the exact surface geometry, the flux-scaled Clebsch lambda and its mapped covariant basis, then compares curl B computed from covariant derivatives with the independent JAX Cartesian curl. The tested map is the DESC-style `(m,n)=(2,0)`, amplitude `0.1` remap over normalized surfaces `s=0.1,0.3,0.5,0.7,0.9`, with 32 poloidal and 16 toroidal points per surface. Geometry derivatives use `h=2e-5`; radial derivative steps were `4e-4,2e-4,1e-4`.

**Results and failed attempts:** the successful run saved `results/projection/integer_3d_gauge_volume_continuum.json` with `status=passed`, maximum remapped B relative L2 `2.583e-10`, maximum curl-B relative L2 `1.156e-4`, minimum `d(theta_old)/d(theta_new)=0.95`, and preserved Jacobian sign on every sampled node. The largest curl error is at `s=0.9`; it remained within `1.2e-9` across the radial step changes. The first checker run failed its curl gate because the FFT derivative axes included the vector-component axis; this was fixed and rerun. A first RSS conversion reported an impossible `387968 MiB` because macOS returns `ru_maxrss` in bytes; platform-aware conversion gives `384.72 MiB`. Runtime was `3.42 s`. These results certify one smooth continuum remap on sampled interior surfaces, not every volume point, every gauge mode, DESC's profile convention, or equilibrium-response derivatives; no solver was run.

**Decision, branch/PR state and next exact action:** the sampled continuum B/J/orientation gate passed, so the remap is eligible for the planned finite-state and solver tests. It does not recommend a weaker VMEX constraint by itself. Benchmark branch `main` is at `eb16374` with the certificate script/result and documentation updates uncommitted; no upstream branch or PR exists. Next, make the bounded VMEX `TCON0` matrix runner record equivalent projected and solved states at NS33, beginning with the default and zero settings plus one intermediate setting, using cold and certified projected starts and the existing 96-point scorer. Expected observables are solver exit status, native B/J/force errors, coordinate drift, raw residuals and elapsed/RSS; do not infer a recovered root from the same-state force switch.

### Entry 2026-09-24: NS33 VMEX constraint-strength recovery ladder

**Phase / run IDs:** P3/P10, S03b and S02b. Benchmark base commit `fe23c17`; pinned VMEX source is `b5f5267efc0795c4a49a224e321e9b370975c14c`. `solve_multigrid` at the pinned source accepts explicit stage arrays and a `tcon0` override; `TCON0` is part of each stage runtime. Targeted source review covered `vmex/core/multigrid.py` and the multigrid ladder tests, with both ledger entries kept partial. No VMEX source was changed.

**Runner and controls:** extended `benchmarks/run_vmex.py` to pass `NS`, `FTOL`, `NITER` and `TCON0` explicitly, preserve terminal states for bounded TCON experiments, record the iteration count, actual controls and peak process RSS, and score native Cartesian fields in batches of 32. Added `benchmarks/plot_tcon_ladder.py`. The six exact reproducible command templates are in `results/vmex/tcon_ladder_ns33/summary.json`; each used `BENCH_FTOL=1e-10`, `NITER=3000`, the integer-3D prescribed-iota deck, NS33, and `TCON0=1` (input default), `0.1`, or `0`, from a cold or saved certified projected initialization.

**Results:** all six solver runs met the discrete componentwise stop at `FTOL=1e-10`. Iterations for projected starts at TCON0 `{1,0.1,0}` were `{311,309,95}`; for cold starts they were `{366,362,498}`. On the same 96 held-out Cartesian points, projected B/J/pressure-normalized force errors were `{8.713e-4,4.391e-1,4.929}`, `{3.061e-4,2.891e-1,3.295}`, and `{4.476e-5,1.727e-2,1.819e-1}`. Cold-start errors were `{9.465e-4,4.654e-1,5.241}`, `{5.361e-4,2.672e-1,3.045}`, and `{3.074e-4,4.957e-2,4.811e-1}`. The corresponding maximum absolute VMEX normalized-flux-label drift was `{4.243e-3,3.599e-3,9.205e-5}` from projected starts and `{4.435e-3,5.379e-3,1.299e-2}` from cold starts. Thus the best run, projected TCON0=0, still misses the initial B/J/force targets (`1e-5/1e-3/1e-3`); all six are `accepted=false`. Zero strength strongly improved the projected-start state but caused much larger label drift from cold initialization. This shows resolution-level and initialization sensitivity, not that the constraint is universally harmful or safe.

**Timing, memory, checks and failure:** solve time ranged `13.86–17.13 s`; native sampling/scoring took `89.38–94.20 s`; process peak RSS ranged `3358–3513 MiB`. Two NVIDIA RTX A4000 devices were visible; per-device GPU memory use was not measured. Repeating the projected default score with sample batch sizes 8 and 32 gave identical points/weights/labels; maximum absolute changes were `3.22e-15 T` in B, `4.10e-7 A/m^2` in J and `4.01e-8 Pa/m` in grad p, with physical score differences at roundoff. The plot `figures/vmex_tcon_ladder_ns33.png` was generated from saved scores, visually inspected and hashed as `4744efc2d0790b9541cdfc8c812c416d70323997c8139e9122beb443551fa974`. The repository suite passed 29 tests before the final reporting edits; final validation remains below. One setup attempt omitted `BENCH_FTOL`, so the runner used its strict `1e-14` default and the projected start hit 3,000 iterations (`FSQR/FSQZ/FSQL=1.39e-10/9.23e-11/6.17e-11`). It produced no physical score and is retained separately as `results/vmex/tcon_ladder_ns33/invocation_failure.json`, not counted among the six runs.

**Decision, branch/PR state and exact next action:** keep S03b diagnostic-only and leave the constraint-design decision open. The six outputs and compressed 96-point native samples are in the six run directories named by the summary; run environment and hashes are recorded there. Update the README with the measured plot/table and qualify the single-resolution result. Benchmark branch `main` is based on `fe23c17`; these changes are uncommitted, and there is no upstream branch or PR. Next, run the certified projected integer-3D seeds at NS65 and NS129 with TCON0 default and zero, the same `FTOL=1e-10` and held-out scorer; record convergence, native errors, flux-label drift, timing and process RSS. Do not recommend a constraint change unless physical gates, resolution trends and gauge/derivative stability support it.

### Entry 2026-09-24: projected TCON0 radial comparison and stop handoff

**Phase / run IDs:** P0/P3/P10, S03b and S03c diagnostic subset. Benchmark base is `fe23c17`; VMEX source is pinned at `b5f5267efc0795c4a49a224e321e9b370975c14c`, and DESC at `4f48720beac3d4169e9165923d730445118bc2de`. The runner and plotting generator are `benchmarks/run_vmex.py` and `benchmarks/plot_tcon_ladder.py`. The existing source review entries for VMEX's multigrid solve and ladder tests are targeted/partial, not a complete semantic source audit.

**Completed run cells:** in addition to the six NS33 runs documented above, three certified projected-start runs were scored on the same 96 held-out Cartesian points at `FTOL=1e-10`, `NITER=3000`: NS65 with default `TCON0=1`, NS65 with `TCON0=0`, and NS129 with default `TCON0=1`. Their componentwise final discrete residuals `(FSQR,FSQZ,FSQL)` were respectively `(9.73e-11,8.12e-11,8.48e-11)`, `(9.45e-11,9.07e-11,1.18e-11)`, and `(9.05e-11,9.77e-11,9.96e-11)`. Their native `(E_B,E_J,force/RMS(grad p_exact))` values were `(7.0633e-4,1.4922e-1,1.6986)`, `(8.2595e-6,1.5122e-3,1.6702e-2)`, and `(1.2991e-4,2.9206e-3,1.7449e-2)`. Maximum normalized-flux-label drifts were `4.6952e-3`, `1.7850e-5`, and `5.3377e-3`. The NS65 zero-strength state passes the B gate only; none passes all three physical gates. These are single runs at each cell, not statistical or resolution-converged uncertainty estimates.

The NS65 default/zero runs took `17.04/13.36 s` to solve and `92.69/92.96 s` to sample; the NS129 default run took `22.84 s` to solve and `92.61 s` to sample. Process peak RSS was `3366.1`, `3370.9`, and `3400.0 MiB`; GPU-memory peak was not measured. Each forward record, compressed native sample array, SHA-256, runtime controls and actual residuals are linked from `results/vmex/tcon_resolution_ladder/summary.json`. `figures/vmex_tcon_resolution_ladder.png` and the NS33 figure are regenerated from saved scores by `python3 benchmarks/plot_tcon_ladder.py`; their data/figure hashes are in the corresponding figure manifests. Both were visually inspected. The README now displays the measured VMEX and DESC comparisons and labels the capped DESC remapped M8 state and incomplete VMEX radial matrix honestly.

**Interpretation and limits:** zero `TCON0` improved all three projected-start scores at NS33 and NS65 relative to the default at those same resolutions, and improved the NS65 B score below `1e-5`; however, its NS65 current and force scores still miss their gates. Default-strength scores improve from NS33 to NS129 in J and force but are not monotone in B and do not pass the physical gates. The NS33 cold-start ladder shows initialization-dependent drift, including larger drift at zero strength. Therefore these data support a coordinate/initialization sensitivity concern and justify continuing the controlled comparison. They do not establish that the coordinate constraint should be removed, that its weight should be reduced, or that a compact unconstrained VMEX representation will be stable.

**Interrupted cell and exact next action:** the user requested that work stop and reconvene after the handoff; no NS129 zero-strength solver was started. It is explicitly `not_run` in the partial summary. Resume with the existing environment and projected seed using `BENCH_FTOL=1e-10 BENCH_TCON0=0 python benchmarks/run_vmex.py inputs/input.integer_3d_iota 129 3000 results/projection/integer_3d_vmex_ns129/seed.npz`; score on the same saved 96 points, regenerate the summary/figure, and compare it with NS129 default before making any interpretation. Then proceed with the plan's Jacobian regularity, Fourier high-mode energy, gauge/derivative stability, additional remaps, and matched DESC/VMEC2000/VMEC++ (GVEC only if pinned and runnable). Preserve separate projection, analytical, solved, consistency and integration evidence. Do not change the recovery gates or solver default to make a cell pass.

**Validation and publication handoff:** final checks passed: `python3 -m py_compile benchmarks/run_vmex.py benchmarks/plot_tcon_ladder.py`, `python3 benchmarks/plot_tcon_ladder.py`, a parse of every repository JSON file, `pytest -q` (29 passed in 10.13 s), and `git diff --cached --check`. The two TCON figures were regenerated and visually inspected; hashes are recorded in their manifests. `tools/publish.sh` was inspected and run only in its default staging mode. It stages an explicit allowlist; `PUBLISH=1` commits and pushes directly to `main`, so the staged result is instead being published on a review branch. Authentication was verified as GitHub login `rogeriojorge` (account ID `6816712`); local Git author and committer are `rogeriojorge <6816712+rogeriojorge@users.noreply.github.com>`. The staged-diff privacy scan found no local paths, home paths, host alias, or private environment names. At entry time the work is on a new review branch and no PR exists yet; leave the PR unmerged for review.

```text
Date/time and benchmark commit:
Phase / run IDs:
Question or hypothesis:
Source commits and patches:
Changed files and reason:
Commands and environment:
Results, residuals, resolutions and artifact paths:
Independent checks performed:
Failures / unavailable dependencies / budget used:
Decision (including any target change and justification):
Next exact command or implementation step:
Working tree / branch / PR state:
```

### End-of-session checklist

Commit small coherent changes with the owner's identity. Update the phase table and append an entry. Record running processes, saved outputs, input hashes, local worktree locations and uncommitted changes without publishing private paths unnecessarily. Stop or explicitly hand over any long-running local jobs. Do not claim background work is continuing when no process exists. Leave the next command and expected observable result, not a vague instruction to "continue benchmarking".

## References

[R1] M. Landreman, *Analytic toroidal 3D MHD equilibria and steady Euler flows with invariant surfaces*, arXiv:2609.26742v1 (2026). https://arxiv.org/abs/2609.26742 . Sections 2-3 give the two exact fields and geometry; distinguish paper formula numbers across HTML/PDF renderings.

[R2] Analytical supplement at `4c0b690ddebdc71811c88223eb9f44a98ab64222`: https://github.com/landreman/analytic_3d_equilibria/tree/4c0b690ddebdc71811c88223eb9f44a98ab64222 . In particular the integer and sheared DESC scripts give independent numerical solves, oriented flux/current mappings and quadratures. They are not VMEX results.

[R3] VMEX baseline: https://github.com/uwplasma/vmex/tree/b5f5267efc0795c4a49a224e321e9b370975c14c . See `docs/all-of-vmex.md`, `docs/howto/profiles.md`, `docs/explanation/validation.md`, `CHANGELOG.md`, the input/profile/wout implementations and their tests.

[R4] VMEX fixed-boundary derivative implementation: https://github.com/uwplasma/vmex/blob/b5f5267efc0795c4a49a224e321e9b370975c14c/vmex/core/implicit.py . Inspect actual masks, raw/preconditioned residuals, root refinement and frozen-path checks.

[R5] Coupled free-boundary implementation: https://github.com/uwplasma/vmex/blob/b5f5267efc0795c4a49a224e321e9b370975c14c/vmex/core/freeboundary_implicit.py . Source is newer than parts of the accompanying validation narrative.

[R6] Continuous force and polishing: `vmex/core/strong_force.py`, `polish.py`, `polish_driver.py` and associated tests at [R3]. The exact Solov'ev projection test is not evidence of nonlinear recovery.

[R7] Interior/exterior fields: https://github.com/uwplasma/vmex/blob/b5f5267efc0795c4a49a224e321e9b370975c14c/vmex/core/extender.py . Tensor orientation, inversion, native versus fallback representations and near-surface modes matter.

[R8] Bounce kernels: https://github.com/uwplasma/vmex/blob/b5f5267efc0795c4a49a224e321e9b370975c14c/vmex/core/bounce.py . Note the normalized action, topology masks and optional floor.

[R9] Virtual casing: https://github.com/uwplasma/virtual_casing_jax and independent reference https://github.com/hiddenSymmetries/virtual-casing . Pin both implementations before comparison. Preserve source versus target terminology and derivative-plan semantics.

[R10] Adjacent implementations: https://github.com/uwplasma/SOLVAX , https://github.com/uwplasma/booz_xform_jax , https://github.com/uwplasma/ESSOS , https://github.com/uwplasma/NEO_JAX , https://github.com/uwplasma/GKX , https://github.com/uwplasma/DKX . Optional near-axis work uses https://github.com/uwplasma/pyQSC_JAX/pull/2 at the recorded draft head. Source pins not yet specified in `sources.json` are a P0 task, not an implicit latest-version dependency.

[R11] A. J. Cerfon and J. P. Freidberg, *One size fits all analytic solutions to the Grad-Shafranov equation*, Physics of Plasmas 17, 032502 (2010), doi:10.1063/1.3328818. Use as context for analytical axisymmetric solution spaces; the specific polynomial identities in this plan are derived and tested directly.

[R12] D. Panici et al., *The DESC stellarator equilibrium solver. Part 1. High-order solutions through Newton-Krylov optimization*, Journal of Plasma Physics (2023), doi:10.1017/S0022377823000272, arXiv:2203.17173. Consult alongside the actual pinned DESC source and Landreman's supplied scripts; compare native representations with common physical norms.

[R13] S. P. Hirshman, W. I. van Rij and P. Merkel, *Three-dimensional free boundary calculations using a spectral Green's function method*, Computer Physics Communications 43, 143-155 (1986). Inspect its vacuum/circulation and free-boundary assumptions together with the current NESTOR implementation rather than inferring the model from a function name.
