# References and attribution

The original benchmark implementation is provided under the accompanying MIT license. Analytical formulas and reference quadratures follow M. Landreman, arXiv:2609.26742v1, and the associated `landreman/analytic_3d_equilibria` supplement. The diagonal-stretch and asymmetric Solov'ev derivations are written out in `plan.md`; they are not assertions of priority over the literature.

No VMEX, DESC, SOLVAX, ESSOS, Boozer, virtual-casing, NEO, GKX, DKX, or pyQSC source tree is vendored in this bundle. Their code, assets and licenses remain with their repositories. Preserve all applicable notices if code or numerical reference assets are later imported. Scientific references and source provenance must not be removed to enforce the owner's requested new-project Git authorship.

Figures and summaries carry separate analytical-reference, projected-state and solved-VMEX evidence labels. The initial reference figures are not VMEX results. The analytical archive was checked during preparation; its fields and numerical values were reimplemented and rechecked here, rather than treating a past summary as independent evidence. The symmetric Solov'ev projection formulas in `benchmarks/projection.py` follow the derivation in pinned VMEX `tests/test_strong_force_solovev.py`; its source remains separately licensed and attributed to VMEX's authors.
