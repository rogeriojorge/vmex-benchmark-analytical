# Prompt for the continuing local agent

Continue the existing public repository `rogeriojorge/vmex-benchmark-analytical`. Do not recreate it. Read this package's `plan_revision_2.md`, `REVIEW_FINDINGS.md`, `REFERENCES.md`, the repository's current plan/logbook, `sources.json` and saved result manifests before changing anything.

The reviewed benchmark head is `575f13f67b346118c3d7f05cc60cc6e29131359a`; PR #1 is merged. Preserve later changes. Archive the previous plan and its full logbook before adopting the revision as the active plan. Keep the historical VMEX baseline `b5f5267efc0795c4a49a224e321e9b370975c14c` immutable and pin a separate current comparison. Never pool different solver/environment generations silently.

Start with R0. The current TCON plotting script overwrites primary reports and inserts hard-coded metadata. Make plotting read-only, preserve original bytes, and store historical corrections with their evidence. Repair effective run controls, immutable run IDs, failed-run memory reporting, source provenance and the DESC sampler's inherited VMEX flux label. Add narrow regression tests. Do not reinterpret old arrays as fabricated or discard useful results.

Then implement R1: retain `legacy96` for continuity but certify scores on independent shifted/refined grids; verify full-torus volume weights, near-axis/edge behavior, inverse-coordinate backward error and direct forward-chart derivatives. A small error on 96 regular points is not a converged volume norm.

Complete the historical missing NS129 projected-start TCON0=0 cell with explicit FTOL=1e-10 and the saved exact-state seed. Record a bounded outcome and compare with the saved default case on both old and new grids. Do not infer the unrun result or change the global TCON0 default from this test.

Test whether the actual reusable VMEX residual is independent of prior evaluations, including both directions across m=1 policy thresholds. VMEC++ issue #624/PR #626 is a precedent, not proof of a VMEX defect. Separate coordinate-constraint force, physical force, finite representation, inversion and normalized-flux-surface displacement.

Run an axisymmetric complete-family derivative test and sheared-A recovery in parallel with the integer-3D diagnosis. Use regular gauge maps: the old m=3 `s*(1-s)` envelope is not analytic at the axis. Check nonzero physical responses, pure-gauge Eulerian nulls, the integer family's delta-field null and the sheared current-prescribed lambda-iota null. Test TCON sensitivity and gauge-defined rank before claiming an exact adjoint.

Continue the broader LASYM, current/iota, spatial derivative, Boozer/bounce, vacuum, anchored free-boundary, bounded polishing, exact-family optimization and scoped adjacent-library phases according to their dependencies. An exact interior does not supply an exact vacuum exterior. Do not assume custom-VJP APIs support public JVPs or Hessians. Keep failed, projected, solved and independently certified states distinct.

Keep code small, readable and deliberate. Reuse existing scripts/operators. Do not create a workflow framework, rewrite VMEX coordinates, train a new equilibrium model or start a broad kinetic campaign before the discriminating measurements are resolved. Necessary upstream fixes belong in separate narrow PRs with reproducer, tests and matched before/after data. Do not merge them or push upstream main.

Verify GitHub authentication is `rogeriojorge`. Configure new Git author and committer locally as that owner using an approved email. No automated-assistant authors or co-author trailers. Preserve existing history, third-party licenses and scientific attribution. Inspect every staged diff before publishing; exclude credentials, private files and caches. Do not rewrite an old GitHub web-flow merge commit.

After each meaningful block update the current phase table and append a logbook entry containing the actual command/configuration, source pins, run IDs, results, uncertainties, failures, hashes, branch/PR state and exact next action. Generate and inspect figures from saved read-only data. README claims must be supported by executed experiments. Leave an honest, resumable handoff even when a capability remains blocked.
