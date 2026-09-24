# Revised VMEX analytical-benchmark handoff

This package updates the existing `rogeriojorge/vmex-benchmark-analytical` project after review of commit `575f13f67b346118c3d7f05cc60cc6e29131359a`. It is not a replacement repository and does not duplicate its numerical states or original reference code.

Read [the revised plan](plan_revision_2.md), [the source-review findings](REVIEW_FINDINGS.md), and [the references and application notes](REFERENCES.md). Give [AGENT_PROMPT.md](AGENT_PROMPT.md) to the local agent. Preserve the old plan and append-only logbook before installing the revision as the active plan. Keep later user changes intact.

The priority is measurement and operator correctness: immutable reporting, independent scoring, the missing NS129 zero-constraint run, deterministic residuals and physically meaningful derivatives. The full axisymmetric/3-D, symmetry, closure, free-boundary, diagnostics and optimization program remains in scope.

## Included checks

`audit_probes.py` is a small NumPy-only demonstration of angular aliasing, field-period volume normalization, and the nonanalytic axis behavior of the previous m=3 gauge envelope. Run it with:

```sh
python audit_probes.py
```

Its actual output is in [audit_probe_results.json](audit_probe_results.json). These are mathematical checks, not new VMEX or DESC benchmark results. No solver suite was rerun for this review, and no remote repository was changed.

`review_tasks.json` is a compact task index, not an execution engine. `SHA256SUMS` covers the delivered files. The detailed plan contains the operating rules, equations, phases, gates, proposed scripts, figure plan, optimization scope and continuing logbook template.
