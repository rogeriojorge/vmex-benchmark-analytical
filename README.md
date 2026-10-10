# Testing MHD equilibrium codes against exact solutions

This repository compares four ideal-MHD equilibrium codes against equilibria whose answer is known exactly: VMEX, VMEC2000, VMEC++ and DESC. The comparison found two defects in VMEX, both fixed upstream, and shows where each code is accurate and where it is not.

## What is compared

**Codes.** All versions are pinned; each run records its exact commit or binary hash.

| Code | Version | Notes |
|---|---|---|
| [VMEX](https://github.com/uwplasma/vmex) | main `f16c300` | JAX port of VMEC with a continuous field evaluator and implicit derivatives |
| [VMEC2000](https://github.com/PrincetonUniversity/STELLOPT) | STELLOPT `3e1439d` | Reference Fortran code |
| [VMEC++](https://github.com/proximafusion/vmecpp) | 0.5.2 | C++ reimplementation of VMEC |
| [DESC](https://github.com/PlasmaControl/DESC) | `4f48720` | Pseudo-spectral code with a Zernike basis in the poloidal cross-section |

**Exact solutions.** Most tests of 3-D equilibrium codes compare one code with another or with a finer run of itself. Neither check can catch an error that is the same at every resolution. [Landreman (2026)](https://arxiv.org/abs/2609.26742) found closed-form 3-D equilibria with finite pressure and nested flux surfaces. For these, B, J, the pressure and the flux surfaces are known at every point, including the magnetic axis. This benchmark uses two of his families, an integer-transform family and a sheared family, plus two Solov'ev tokamak equilibria. One of those is an up-down asymmetric variant derived here. [Issan et al. (2026)](https://arxiv.org/abs/2610.07304) extended both families with one parameter, `tau`, that breaks stellarator symmetry; `tau` and `-tau` are mirror images. Those cases test the full Fourier basis (`LASYM = T`) of every code, and the Boozer, effective-ripple and virtual-casing tools downstream of it (section 7).

![Flux surfaces of the exact solutions](figures/exact_cases.png)

*Exact flux surfaces. Blue: toroidal angle 0; orange: a quarter field period later. The asymmetric Solov'ev case is axisymmetric.*

**Scoring.** VMEX, VMEC2000 and VMEC++ read the same generated input file. Each output is scored against the exact solution by the same script, using three measures:

- **Flux-surface error:** how far each computed flux surface lies from the exact one. This is pure geometry.
- **B and J errors,** relative to the exact field.
- **Axis position.**

DESC is run at its own spectral resolution and scored by evaluating its fields at chosen points.

## Main results

- VMEX, VMEC2000 and VMEC++ give the same equilibrium from the same input. VMEX's improvements are elsewhere: in how fields are read out, and in its derivatives.
- The exact solutions exposed two VMEX defects that resolution studies had missed. Fixes: [uwplasma/vmex#452](https://github.com/uwplasma/vmex/pull/452) (field at the axis) and [#453](https://github.com/uwplasma/vmex/pull/453) (derivative refinement), both merged.
- VMEX's continuous field gives J 10–70× more accurately than the standard WOUT output file, except within a few surfaces of the axis.
- DESC is the most accurate code in this comparison, including at the axis, and it recovers the cases where the VMEC-type codes fail from a cold start.
- Without stellarator symmetry, VMEX's full-basis path adds no error (the `tau = 0` equilibrium in the full basis is identical to the symmetric one), mirror pairs agree to 1e-9, and the native field again gives J ~10-30x more accurately than WOUT. Booz_xform_jax, NEO_JAX and virtual_casing_jax were made to work without symmetry and checked against the exact fields ([uwplasma/vmex#573](https://github.com/uwplasma/vmex/pull/573), [#525](https://github.com/uwplasma/vmex/pull/525), [NEO_JAX#7](https://github.com/uwplasma/NEO_JAX/pull/7), [virtual_casing_jax#21](https://github.com/uwplasma/virtual_casing_jax/pull/21)).
- With `tau != 0` the `iota = 2` family is degenerate: VMEX converges (1e-11) to a neighbouring equilibrium whose surfaces differ from the exact ones only in harmonics resonant with `iota = 2`, at 8e-4 independent of resolution.

## Results

### 1. Same input, same equilibrium

The three VMEC-type codes solve the same discrete equations. On every case where they converge, their flux surfaces agree to the plotted precision, and VMEX and VMEC2000 agree on every score to all printed digits. They also fail together. From a cold start, none of them reaches the integer 3-D or sheared-A equilibria; their flux-label errors are 0.1 and 0.5. VMEC++ writes no output in those cases.

![Flux-surface error for VMEX, VMEC2000 and VMEC++](figures/code_parity.png)

*Left: converged cases; the three codes overlap. Right: hard cases from a cold start.*

### 2. VMEX fix: the field at the axis (#452, merged)

VMEX evaluates B and J between its radial surfaces by interpolating Fourier coefficients. Each coefficient is scaled near the axis so that it stays smooth. The axis value of that scaled coefficient was copied from the first surface instead of extrapolated. As a result, the axis field was only first-order accurate, and the current next to the axis stayed wrong at the 1% level no matter how many surfaces were used. Only an exact solution shows an error like this: the axis field is known exactly, and the error does not go away under refinement. With linear extrapolation both errors converge.

![Axis field and near-axis current before and after #452](figures/fix_axis_row.png)

*The exact solution projected onto VMEX's representation, with no solve, so only the evaluator is tested. Integer 3-D case.*

### 3. VMEX fix: refinement before derivatives (#453, merged)

Before computing derivatives, VMEX refines the equilibrium with Newton steps until the force residual is below `refine_tol`. When the first step overshot, the refinement stopped early and derivatives were taken at an unconverged state. It now restarts from its best point, at most twice. On the case below one call goes from 1e-7 to 6e-14.

![Refinement residual before and after #453](figures/fix_refinement.png)

*Integer axisymmetric case, NS65, starting from the same solver state.*

### 4. Reading B and J from an equilibrium

VMEC2000 and VMEC++ provide their result as a WOUT file. The file stores the field on a staggered radial grid and the current as a derived quantity. VMEX also provides a continuous field, `VmecInteriorField`, that can be evaluated anywhere. The plot evaluates both at the same points of one equilibrium, VMEC2000's own output. Between s = 0.25 and 0.75 the native field gives J 10–70× more accurately, and B is also better. Within the first few surfaces of the axis the WOUT current is better.

The WOUT format fixes what is stored, and VMEX writes it the same way as VMEC2000 so that existing tools keep working. The more accurate route is therefore the native field, not a different WOUT file. VMEX users get it directly in Python; users of VMEC2000 and VMEC++ can convert a WOUT with `vmex.state_from_wout` and use the same evaluator.

![WOUT output versus VMEX native field](figures/field_readout.png)

*Integer axisymmetric case, NS129, one VMEC2000 equilibrium read two ways.*

### 5. Integer 3-D: DESC and VMEX

For the integer 3-D case, DESC converges spectrally: the volume current error falls by about ten for each increase of the resolution by two. VMEX solves this case only when started from the exact state. It then reaches J errors of about 1e-4 at NS257, limited by the first cells next to the axis (see *Open problems*).

![Volume current error, DESC and VMEX](figures/integer_3d_accuracy.png)

*Volume-averaged J error. The two panels use different resolution parameters, so compare the levels, not the slopes.*

### 6. Sheared A and the asymmetric Solov'ev case

Sheared A has strong magnetic shear and a large pressure. The VMEC-type codes do not converge on it, while DESC reaches B errors near 1e-5 at M = 10. DESC did not meet its own stopping tolerance within 150 iterations there, so the figure marks it. For the asymmetric Solov'ev case VMEX and VMEC2000 converge. DESC is much more accurate there, partly because this solution is a low-order polynomial that its basis represents almost exactly. VMEC++ does not support up-down asymmetric (LASYM) equilibria.

![Sheared A and asymmetric Solov'ev across codes](figures/hard_cases.png)

*B error at mid radius (s near 0.5).*

### 7. Without stellarator symmetry

The Issan families are Landreman's with one extra parameter. Family A keeps `iota = 2`; family B keeps the sheared transform, which `tau1` lowers from 2.80 to 2.55 at `|tau1| = 0.5`. The closed forms in `benchmarks/analytic.py` reproduce the paper's volume, beta and transform (its Figs. 3, 6 and 7) and pass the same force-balance, divergence and quadrature checks as every other case.

![Issan families](figures/issan_cases.png)

*Boundaries at `tau = 0` (grey) and `+-tau` (blue, orange); beta and iota against `tau`, with the paper's quoted value. The [animation](figures/issan_tau.gif) sweeps `tau` through both families.*

**Equilibria.** From a cold start no VMEC-type code converges on these cases, as for their symmetric parents. Seeded with the exact geometry (`project_integer_vmex.py --solve`), VMEX converges family A to 1e-11 at NS 33, 65 and 129 and stalls on family B at 1e-6 to 1e-7. Three results on family A:

- The `tau = 0` equilibrium in the full basis gives the symmetric result to every digit, and `tau` and `-tau` agree to 1e-9 in every score. The asymmetric path adds nothing.
- The native field gives J to 1e-4 at mid radius against 3e-3 from WOUT; B is right to 2.5e-5.
- The flux surfaces miss the exact ones by up to 8e-4 at every resolution tried (NS 65 and 129, 13 or 17 poloidal modes, a recentred copy). The miss has zero mean and lives in the harmonics (1,1), (2,2), (3,3): resonant with `iota = 2` at two field periods, where a deformation costs no force. Stellarator symmetry forbids it at `tau = 0`; breaking the symmetry admits it. Both states are converged equilibria with the prescribed boundary, pressure and transform.

![VMEX on the asymmetric family A](figures/asymmetric_vmex.png)

*Left: exact-label error of VMEX surfaces. Middle: current error from WOUT and from the native field. Right: largest harmonics of the label error at mid radius.*

**Downstream tools.** Each was checked against the exact field rather than another code:

- *Boozer transform* (booz_xform_jax): points mapped from Boozer angles land on the exact surface and reproduce |B| to 1e-7 on the asymmetric Solov'ev equilibrium (converged at mboz 16), and to 4e-5 on family A.
- *Effective ripple*: a real-space reference from the exact field (Nemov et al. eq. 29, `benchmarks/ripple_reference.py`) gives `epsilon_eff^{3/2}` for family B with `tau1 = +-0.4` equal to 1e-4. NEO_JAX traces the mirrored line backwards and agrees to its line-length error (3e-3); on the asymmetric Solov'ev equilibrium, where the exact ripple vanishes, it floors at 1e-5 with or without the sine tables. The NEO_JAX check against family B awaits a converged family-B equilibrium.
- *Virtual casing* (virtual_casing_jax): against the Biot-Savart integral of the exact current, the plasma field converges at the symmetric rate, 3.5e-9 off the surface at a 24x48 grid and 2e-5 to 6e-4 on it; mirror pairs agree to 5.5e-16 off the surface.
- *Coils and free boundary* ([coils-for-new-landreman#1](https://github.com/rogeriojorge/coils-for-new-landreman/pull/1)): ESSOS coils reach B.n/|B| of 1e-4 (family A) and 4e-4 (family B) mean. VMEX free boundary with them does not converge on either case, for the reasons above.

![Boozer, ripple and virtual casing without symmetry](figures/asymmetric_chain.png)

*Left: Boozer |B| against the exact field. Middle: exact effective ripple of family B. Right: virtual-casing B.n on the boundary against Biot-Savart of the exact current.*

## Open problems

- **Near-axis current in solved VMEX states.** After #452 the exact projection is accurate at the axis, but solved states still carry a current error of about 1e-2 in the first two or three cells. The error is the same at NS129 and NS257. VMEC2000 gives the identical solution, and smoothing the first surfaces does not remove it. It is therefore a property of the VMEC discretization near the axis, not of the evaluator, and a fix would change that discretization.
- **Family B (Issan) in VMEC-type codes.** VMEX stalls at 1e-7 even from the exact geometry, so the NEO_JAX and free-boundary checks on it are open.
- **Sheared A in VMEC-type codes.** Cold starts end on a wrong state, and a solve started from the exact state drifts away from it. VMEC2000 and VMEC++ behave the same way; DESC does not.

## Reproduce

```sh
python -m pip install -r requirements.txt
python -m pytest -q                        # exact-solution checks (no solver needed)
python benchmarks/verify_reference.py      # force balance of every exact case
```

With VMEX, VMEC2000 (`xvmec2000`) and VMEC++ installed:

```sh
python benchmarks/cross_code_run.py --deck input.integer_axisymmetric_iota --ns 129 --code vmec2000
python benchmarks/score_wout_exact.py --run-dir results/cross_code/runs/integer_axisymmetric_iota-ns129-vmec2000 --case integer_axisymmetric
python benchmarks/cross_code_summary.py
python benchmarks/plot_figures.py
```

The non-symmetric results: `benchmarks/build_inputs.py <case>` writes a deck, `benchmarks/project_integer_vmex.py 65 --case integer_3d_tau --solve --ftol 1e-11` solves it from the exact geometry, `benchmarks/asymmetric.py` scores everything into `results/asymmetric/summary.json`, and `benchmarks/ripple_reference.py reference` computes the exact effective ripple.

DESC cases run with `benchmarks/desc_general_case.py --case integer_3d --M 8 --output-dir <dir>`. The two fix reproductions are `benchmarks/axis_projection_ladder.py` and `benchmarks/observe_root_refinement.py`.

## Layout

- `benchmarks/analytic.py`: the exact solutions. `build_inputs.py` writes the input decks in `inputs/`.
- `benchmarks/cross_code_run.py`, `score_wout_exact.py`, `cross_code_summary.py`: the VMEX, VMEC2000 and VMEC++ comparison.
- `benchmarks/desc_general_case.py`: DESC runs.
- `benchmarks/boozer_exact.py`, `ripple_reference.py`, `asymmetric.py`: Boozer, effective-ripple and non-symmetric records.
- `benchmarks/plot_figures.py`: every figure, drawn from `results/` only.
- `results/`: per-run receipts (input hash, code version, command) and scores.

The full development history, including earlier diagnostics and plans, is kept at the git tag `archive-2026-09-25`.

## References

- M. Landreman (2026), [arXiv:2609.26742](https://arxiv.org/abs/2609.26742); supplement [landreman/analytic_3d_equilibria](https://github.com/landreman/analytic_3d_equilibria).
- W. Issan, J. Citrin, M. Landreman, B. Tracey (2026), [arXiv:2610.07304](https://arxiv.org/abs/2610.07304).
- V. V. Nemov, S. V. Kasilov, W. Kernbichler and M. F. Heyn, *Evaluation of 1/nu neoclassical transport in stellarators*, Phys. Plasmas 6, 4622 (1999).
- S. P. Hirshman and J. C. Whitson, *Steepest-descent moment method for three-dimensional magnetohydrodynamic equilibria*, Phys. Fluids 26, 3553 (1983). (VMEC)
- D. W. Dudt and E. Kolemen, *DESC: A stellarator equilibrium solver*, Phys. Plasmas 27, 102513 (2020).
