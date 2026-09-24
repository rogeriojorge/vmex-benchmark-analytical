# References and attribution

The original benchmark implementation is provided under the accompanying MIT license. Analytical formulas and reference quadratures follow M. Landreman, arXiv:2609.26742v1, and the associated `landreman/analytic_3d_equilibria` supplement. The diagonal-stretch and asymmetric Solov'ev derivations are written out in `plan.md`; they are not assertions of priority over the literature.

No VMEX, DESC, SOLVAX, ESSOS, Boozer, virtual-casing, NEO, GKX, DKX, or pyQSC source tree is vendored in this bundle. Their code, assets and licenses remain with their repositories. Preserve all applicable notices if code or numerical reference assets are later imported. Scientific references and source provenance must not be removed to enforce the owner's requested new-project Git authorship.

The figures and numerical summaries currently included were produced by the supplied analytical reference scripts. They are not measurements of a VMEX equilibrium solve. The previously supplied reference archive was checked during preparation; its fields and numerical values were reimplemented and rechecked here, rather than treating a past summary as independent evidence.
