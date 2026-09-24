# Source review and remaining audit

Review date: 2026-09-23. The VMEX snapshot is `b5f5267efc0795c4a49a224e321e9b370975c14c`. The Landreman supplement is `4c0b690ddebdc71811c88223eb9f44a98ab64222`. A source file's blob SHA is not a repository commit SHA; do not interchange them when pinning a run.

## Scope

Preparation used the connected GitHub reader to inspect current VMEX changes, documentation, selected implementation sections and adjacent-library interfaces. The working runtime could execute the analytical reference code but could not clone or install the solver repositories. No whole-tree semantic review, VMEX build, complete regression suite or GPU test is claimed here.

The inspected source is sufficient to identify the major benchmark interfaces and several important traps. It is not sufficient to certify every module or exceptional path. Phase P0 and `tools/audit_sources.py` make that remaining audit explicit. The inventory tool reads and hashes all tracked source/text files in local checkouts; it never marks them semantically reviewed.

## VMEX paths inspected during preparation

Paths below refer to the pinned tree. Ranges refer to source file lines, not the line numbers of a JSON wrapper returned by a repository connector.

| Path / range | Finding used in the plan | Required local follow-through |
|---|---|---|
| `docs/all-of-vmex.md`, complete returned file | Public solve, output, optimization and mirror entry points; separation of fixed-boundary source objectives and coupled free boundary | Verify exports/signatures against installed pin and trace actual implementations. |
| `docs/explanation/validation.md`, introductory material from previous review and current lines 215-460 | Existing evidence classes, native versus WOUT distinction, optional dependency skips, limited model scope | Reconcile with new source and rerun tests behind relevant statements. |
| `CHANGELOG.md`, lines 1-155 | New free-boundary anchoring, edge response, native interior interpolation and near-surface changes | Match release/PR changes to the actual baseline tree and tests. |
| `vmex/core/implicit.py`, lines 1-155 | Raw/preconditioned derivative relation, root refinement, DOF/gauge constraints, custom VJP and frozen-path semantics | Read implementation bodies and masks; test complete root and failure-return paths. |
| `vmex/core/freeboundary_implicit.py`, lines 1-230 and `_anchor_root` search | Coupled plasma-vacuum derivative, anchor, restart budget, status 3, strict versus best-effort policies, three transpose solvers | Read complete coupled operator, anchor stopping conditions, low-rank edge assembly, status wrappers and tests. |
| `vmex/core/strong_force.py`, lines 1-165 | Independent continuous representation, field-period angle, external lambda, regular radial amplitudes, bounded force normalization | Review all reconstruction/field/certification code and independent exact-state tests. |
| `vmex/core/extender.py`, lines 1-185 | Interior/exterior interfaces, source-field separation, tensor ordering, parameter factoring, near-surface modes | Read native and fallback fields, coordinate inversion, bounds checks, high spatial derivatives and quadrature selection. |
| `vmex/core/polish_driver.py`, lines 1-135 | Public collocation least squares, separate diagnostic root route, budget policy and stationarity stopping | Audit exact stationary derivative, residual scaling, chart rank and certificate. |
| `vmex/core/bounce.py`, lines 1-125 | Action normalization; masks for marginal/merged/truncated/overflow wells; floor versus exact kernel | Audit remaining crossing/interval code and root-motion derivatives with independent quadrature. |
| `examples/take_fixed_boundary_gradients.py`, complete returned file | `implicit.run`, scalar outputs, parameter replacement and reverse-mode API | Run against generated input; do not mistake its FD example for a continuum derivative proof. |
| `tests/test_strong_force_solovev.py`, lines 1-190 in preceding review | Exact projection and representation convergence, not nonlinear recovery | Compare complete test with the new asymmetric Solov'ev field and solve it from independent initial states. |
| `docs/howto/profiles.md`, complete returned file in preceding review | AM/AI powers in s, AC current derivative shape, CURTOR normalization, splines and profile pitfalls | Confirm actual `profiles.py`, `setup.py` and parser semantics at the pinned baseline before claiming closure recovery. |

The source directory listing and current open-PR metadata were also inspected. A directory listing or PR description is not a review of every file it names. The open release PR and winding-sensitivity work are context, not automatically incorporated dependencies.

## Findings that change the benchmark design

### Free-boundary root anchoring: source and narrative disagree

The current `freeboundary_implicit.py` docstring and controls explicitly include Newton anchoring. Its configuration accepts finite `refine_tol`, and an unsuccessful anchor is status 3. The current change log describes the same correction. An older section of `validation.md` still describes unanchored free-boundary objective values. The plan therefore tests the actual anchored implementation and records any stale documentation separately.

Do not infer that every solve is anchored merely because the routine exists. In particular, inspect status/fallback paths, explicit infinite refinement tolerance, best-effort adjoints, nonlinear stopping and the actual state used to compute the objective.

### Discrete versus physical gradients

Fixed-boundary masks freeze certain coordinate combinations. The source documents differences between frozen-path and independently reconverged perturbations. Both comparisons are useful, but only the exact-family physical-space tests can determine whether the intended continuum response is approached under refinement.

The preconditioner derivative term is proportional to the raw residual. Since the reported solver tolerances involve squares and normalization, residual-based error budgets should use measured norms rather than treating FTOL itself as a raw force bound.

### Field interfaces are important benchmark targets

The change log describes previous mistakes in odd-mode radial scaling and interior inversion. These motivate pointwise comparisons throughout the volume, not only matching the last surface or WOUT scalars. Native Clebsch reconstruction, fitted fallback and output interpolation need separate measurements. High derivatives are especially sensitive to interpolation regularity and knots.

### Polishing is not one certificate

The current driver minimizes an overdetermined physical residual. The force error, exact-field error and least-squares stationarity residual answer different questions. At nonzero residual, the true derivative of stationarity includes residual-weighted second derivatives. A Gauss-Newton approximation cannot be silently relabelled as an exact implicit derivative.

### Rational transform is useful but special

The integer family gives exact closure and strong null tests, but does not establish ergodicity. Treat Boozer gauge choices, zero shear, trapped-well topology and transport sampling assumptions explicitly. A numerical routine's failure near a rational denominator must be compared with its mathematical assumptions, not automatically interpreted as an incorrect analytical equilibrium.

## Adjacent repositories: inspected evidence and scope

| Repository | Evidence inspected | Benchmark role and remaining review |
|---|---|---|
| `landreman/analytic_3d_equilibria` | README, repository tree, integer/sheared DESC script sections in preceding review; current commit rechecked | Independent geometry, normalization, flux and current quadratures. Read complete scripts and notebooks locally; run DESC separately. |
| `uwplasma/SOLVAX` | README through line 145 | Structured factors, transpose solves, implicit versus raw iterative algorithms. Review actual imported functions, rank/conditioning, reuse and derivative contracts. |
| `uwplasma/booz_xform_jax` | README through line 165 | Functional transform and file interface, LASYM and differentiability. Review `jax_api` and transformation kernels before a response claim. |
| `uwplasma/virtual_casing_jax` | README through line 190 | Source separation, singular quadrature, fixed precision plans. Correct repository uses underscores. Some older near-surface documentation differs from current VMEX APIs; inspect version-specific kernels. |
| `uwplasma/ESSOS` | `essos/dynamics.py` through line 130 | Real orbit/particle interface and field imports. Review coils, fields, guiding-center/full-orbit conventions and stepping. |
| `uwplasma/NEO_JAX` | README through line 155 | Ripple/current workflow and rational-surface work controls. Review actual geometry and integration paths; distinguish epsilon_eff powers and model assumptions. |
| `uwplasma/GKX` | README through line 115 | Geometry handoff and certified eigenvalue interface. Read geometry/normalization and optional eigenvalue response code for the chosen integration test. |
| `uwplasma/DKX` | README through line 110 | Neoclassical geometry, LASYM, closure and solver interfaces. Exact MHD does not imply exact kinetic transport. |
| `uwplasma/pyQSC_JAX` | Minimal main README and detailed PR 2 metadata | Main and draft refactor are different products. Optional draft head is pinned in `sources.json`; review branch source before use, and respect the restricted near-axis symmetry ansatz. |

An additional concrete ESSOS review lead: the viewed particle constructor accepts an initial full-orbit phase parameter but assigns the stored phase to zero, and the conversion routine constructs a perpendicular basis using the z direction. Check phase handling and fields parallel to that direction with a minimal reproducer before calling either a defect. These are source-review leads, not reproduced failures in this bundle.

## How to complete the ledger

Clone or locate the pinned sources outside this repository, resolve null pins, then run:

```sh
python tools/audit_sources.py /absolute/path/to/checkouts
```

Create `results/audit/review_ledger.json` entries with `repository`, `commit`, `path`, `sha256`, `review_scope`, `model`, `units`, `derivative_semantics`, `tests_read`, `oracle`, `benchmark_cell`, `finding`, and `status`. Keep every module initially unreviewed. Mark it reviewed only after inspecting implementation and relevant tests. For very large files, record reviewed ranges and remaining ranges; do not label a prefix review complete.

Include all VMEX Python modules, relevant configuration and serializers, and the reachable source paths of adjacent libraries. For an unrelated adjacent module, state why it is not in the benchmark's dependency/physics scope. Inspect tests that call external codes or use `importorskip`; a skipped test supplies no local evidence. Record which import path actually ran, including installed wheels versus local editable checkouts.

The final scope statement must distinguish full local source inventory, semantic module review, executed tests and measured physical validation. These are four different columns, not one percentage.
