# Continuation handoff after 05e9473

Use this package with the existing public benchmark repository, not as a replacement checkout.

- [Plan and continuing-logbook contract](plan_next.md)
- [Source findings and completed work](REVIEW_FINDINGS.md)
- [References and concrete applications](REFERENCES.md)
- [Local-agent prompt](AGENT_PROMPT.md)
- [Snapshot and review scope](review_snapshot.json)
- [Task index](tasks.json)

The main priorities are single-pass refinement observation, a same-operator branch-tangent check, independent scoring of useful saved states, and full-flux lambda reconstruction. Previous CI/import repairs and the missing NS129 run are now completed and are not reassigned.

## Executed mathematical probes

From this directory:

```sh
python -m pytest -q probes/test_review_probes.py
python probes/sheared_flux_demo.py \
  --reference-root /path/to/vmex-benchmark-analytical \
  --output /path/to/fresh/sheared_flux_demo_results.json
```

The ten tests require NumPy and pytest. The surface demonstration additionally uses the repository's analytical reference and its JAX/SciPy dependencies, but never imports VMEX. The destination must not already exist.

`probes/flux_potential.py` is a small independent NumPy implementation, not production integration. It uses both signed contravariant flux densities to reconstruct a periodic zero-mean lambda, with explicit incompatibility rejection. The tests cover multiple NFP values, a resonant straightness ambiguity, incompatible data, weak stress, response scaling, operator-context changes and a true Newton linear-defect identity.

The included surface-result JSON was generated from the analytical files in the originally supplied `vmex-benchmark-analytical-handoff.zip`, not a fresh checkout of current main. It records their exact hashes. It reconstructs the sheared-A magnetic vector on one surface; it does not verify radial current, a VMEX projection, a nonlinear solution or equilibrium derivatives. Best sampled relative B error was approximately 1.5e-10. Rerun it against current source before incorporating it into the live repository's results.

## Evidence limits

The review inspected the successful GitHub reference CI, source changes and saved measurements. It did not rerun the full reference suite or any VMEX/DESC/free-boundary/kinetic solver here. No remote repository changes were made. `probe_execution.json` records the tests actually executed in this review. Historical source and current/experimental VMEX snapshots stay separate.

Before adoption, verify `SHA256SUMS`, compare the real checkout to the reviewed commit, and preserve all later owner work. The source archive and long historical logbook already in the repository must remain available.
