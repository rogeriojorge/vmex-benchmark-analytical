# VMEX analytical benchmarks

Exact toroidal equilibria, physical-field checks and a staged benchmark of VMEX's derivatives, diagnostics and free-boundary interfaces.

The primary references are [Landreman's analytical equilibria](https://arxiv.org/abs/2609.26742), their [supplementary implementation](https://github.com/landreman/analytic_3d_equilibria), and an explicitly derived asymmetric Solov'ev case. The implementation plan and continuing logbook are in [plan.md](plan.md). Start a local implementation session with [AGENT_PROMPT.md](AGENT_PROMPT.md).

## Results currently included

These are **analytical reference results, not VMEX solver results**. The reference suite passed 29 tests, and sampled force identities passed on 14 configurations. The largest sampled force RMS divided by pressure-gradient RMS was 1.66e-15. Tests also cover an independent Grad-Shafranov identity, volume and toroidal-flux integrals, coordinate covariance, current integration, flux inversion, field Jacobians and sensitivity cancellations.

| Reference | Volume-averaged beta | Axis iota, counterclockwise R-Z convention |
|---|---:|---:|
| Integer 3-D, epsilon=0.5, delta=1/64 | 3.50877193% | -2.00000000 |
| Sheared A | 19.66235650% | -5.68886661 |
| Sheared B | 7.46725404% | -4.36231483 |
| Sheared C | 14.09668185% | -3.88989400 |

The paper uses the opposite poloidal orientation. The signs above are deliberate; the local VMEX parser/field conversion still needs its own physical-vector check. The exact field's derivative with respect to the integer family's outer label is zero at a fixed interior Cartesian point. The sheared transform's lambda derivative also vanishes; measured explicit-reference values and Taylor curves are in [results/reference/derivatives.json](results/reference/derivatives.json).

![Exact boundaries](figures/reference_geometry.png)

![Asymmetric Solovev reference](figures/solovev_asymmetry.png)

## Run the reference checks

Run from the repository root in a dedicated environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python benchmarks/verify_reference.py
python benchmarks/reference_derivatives.py
python benchmarks/build_inputs.py
python benchmarks/plot_results.py
```

The scripts use float64. Recorded package versions and machine information are in [results/reference/identities.json](results/reference/identities.json); test and command evidence is in [results/reference/execution.json](results/reference/execution.json). Exact local versions, rather than an assumed latest release, must accompany later solver results. Python 3.12 or newer is a practical starting point for the separate current-VMEX environment; resolve its actual dependency constraints during phase P0.

![Reference Taylor test](figures/reference_derivatives.png)

## Candidate VMEX inputs

`inputs/` contains 28 generated INDATA candidates: 14 configurations, each with prescribed iota and prescribed current. The current profile is generated from an independent Ampere integral and converted to VMEX's derivative-profile convention. Pressure, flux and geometric scales are recorded in [inputs/manifest.json](inputs/manifest.json).

The first Fourier truncation resolves the mild examples for a smoke run, but **not sheared B and C**. Their sampled boundary-fit errors are about 2.8e-3 m and 1.3e-2 m at a 1 m length scale. Refining radial resolution cannot repair that input error. The first forward runner refuses those manifests until they are refined. No candidate has yet been certified by VMEX in this bundle.

![Boundary input fit error](figures/boundary_fit.png)

After installing the pinned VMEX and its dependencies in a separate local environment:

```sh
python benchmarks/run_vmex.py inputs/input.integer_axisymmetric_iota
python benchmarks/run_vmex.py inputs/input.integer_axisymmetric_current
python benchmarks/run_gradient_smoke.py inputs/input.integer_axisymmetric_iota
```

Those scripts were syntax-checked but not run in the handoff environment, where VMEX was absent. The forward runner records scalar and solver output; it does not certify physical accuracy. The gradient smoke varies only PHIEDGE at fixed boundary and profiles. It does not differentiate the complete analytical family. Complete those adapters and the native-field sampler in phases P1-P4 before reporting recovery or continuum derivatives.

The shared scorer accepts an NPZ file of **native physical samples**, with its contract in the script and plan:

```sh
python benchmarks/score_samples.py samples.npz scores.json
```

It compares Cartesian B, J and grad p against the reference at the same points. Its `accepted` field remains unset until the experiment's resolution and error gates have been applied.

## Study coverage

The [execution matrix](benchmark_matrix.json) records implemented, planned and blocked experiments. The plan covers axisymmetric and non-axisymmetric equilibria, LASYM=F/T, genuine asymmetric Solov'ev physics, exact coordinate changes, both current/transform closures, field derivatives, implicit responses, file/restart paths, Boozer and bounce diagnostics, polishing, free-boundary operators and coupled roots, source fitting, optimization and scoped adjacent-code integrations.

An exact interior field is not automatically an exact free-boundary solution. The free-boundary plan distinguishes exact vacuum/operator tests, independently converged numerical coupled equilibria, and approximate exterior fits to an exact interior target. Genuinely symmetry-broken 3-D extensions outside the exact families also require numerical references. These distinctions are part of the benchmark, not missing labels to be filled with assumed answers.

No VMEX, DESC, free-boundary, GPU or kinetic benchmark has been run yet. No remote repository was created by this handoff preparation. The local agent creates it using the owner's authentication. [The source review](docs/SOURCE_REVIEW.md) identifies inspected paths and the full local audit still needed; a whole-source semantic review is not claimed from an interface inspection.

## Local source inventory and publication

Resolve the used pins in `sources.json`, locate checkouts outside this repository, then run:

```sh
python tools/audit_sources.py /absolute/path/to/checkouts
sh tools/publish.sh
# Read the staged diff before authorizing publication:
PUBLISH=1 sh tools/publish.sh
```

The inventory tool does not mark files reviewed. The publication helper verifies the authenticated owner and configures only local Git identity; it does not force-push or rewrite history. Preserve third-party notices. Update the phase table and append a logbook entry after each meaningful experiment.

## License and references

Original code: MIT. See [NOTICE.md](NOTICE.md) for attribution and [plan.md](plan.md#references) for the scientific and implementation references. Large future state files should be checksummed release assets, while code, small numerical summaries and figure generators remain in Git.
