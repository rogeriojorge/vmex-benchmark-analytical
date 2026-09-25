# Review findings after 05e9473

This review concerns benchmark commit `05e947370b9d0cebc0e508361e447b03523a331d`, six commits beyond the preceding `d5484d1` review. It uses source inspection, saved reports and current CI metadata. It does not claim a full local VMEX rerun. Reference labels resolve in [REFERENCES.md](REFERENCES.md).

## 1. Completed work must be retired from the next to-do list

**Status: verified progress, not a new defect.** [S1, S2, S8]

The previous solver-independent test import failure is repaired and the reference CI at the current commit succeeds. The repository records 55 tests. Report schema normalization, immutable attempts and bounded measurement checkpointing have been exercised. Native-knot crossings were solved accurately on one NS33 ray. The previously missing projected-start NS129 TCON0=0 run is complete, and the preconditioned single-grid residual value/JVP/VJP passed a history-independence test.

**Next action:** keep regression tests and rescore useful saved states. Do not repeat these completed experiments as though no progress occurred. Do not expand the one-ray or one-operator checks into claims about all quadratures or all residual formulations.

## 2. The before/after anchor observer starts after the first anchor attempt

**Status: source-confirmed instrumentation mismatch; highest priority.** [S3, S9]

`axisymmetric_root_response.py::_root_and_anchor` calls `solve_implicit_with_aux`, then measures `before`, then calls `_refine_fixed_point`. The historical VMEX callback has already called `_refine_fixed_point`. Its memo uses the configuration and parameter bytes. The observer's second call can therefore be a cache hit.

The consequence is narrow but important: a zero `state_anchor_shift_l2` cannot establish that the actual refinement took no step. Both reported residuals can be post-attempt measurements. Their smallness or largeness is still useful; the large zero-TCON residuals remain disqualifying. The observation is not evidence that every VMEX refinement is broken.

**Repair:** observe one raw host state and one uncached actual refinement, with fixed operator and full input/output hashes. Record attempted corrections, true linear defects, trial residuals and decisions. Test changed-state, unchanged-failure and memo-hit paths. Keep memoized public workflow and arbitrary-state refinement contracts distinct.

## 3. A returned state is not a certified equilibrium root

**Status: source-confirmed contract limitation, reproduced in saved benchmark evidence.** [S7, S9]

The historical refiner may retain an unchanged or merely improved state without meeting `refine_tol`. The status wrapper bases acceptance on the original host solve rather than the final root certificate. This is consistent with the source's optional-improvement policy, but not with interpreting every successful call as a certified differentiable equilibrium.

The current separate amendment correctly reclassifies the three NS65 zero-weight states: their residuals are approximately 1.19e-7, 8.42e-8 and 1.21e-7 versus 1e-11. Their recorded field FD must remain diagnostic. Do not infer that their physical branch is better or worse from this unqualified comparison.

**Repair:** gate accepted derivatives at the benchmark boundary using the exact operator certificate. A narrow upstream strict option can be proposed separately. Preserve diagnostic sampling of failed states, but make the status impossible to confuse with a root.

## 4. The branch tangent has not yet been tested against one frozen operator

**Status: missing discriminating experiment, not an established cause.** [S3, S4]

The new dense check validates the linear algebra for the selected residual; the reconstruction JVP along a measured branch FD shows that the field evaluator follows that state variation. Neither proves that the endpoint states solve the same frozen equation differentiated at the base.

**Next action:** compute the reduced-state FD, its defect `A d + F_P q`, and both endpoint residuals in the base operator. Compare with their own-context residuals. Track frozen-auxiliary changes rather than assuming they are harmless gauges. Match the parameter tangent step and independent-DOF assembly. Only use rank/conditioning explanations after this algebraic check.

The delivered scalar probe illustrates how exact roots in changing contexts can disagree with a frozen derivative. It does not diagnose which context, if any, differs in VMEX.

## 5. Response norms currently mix conventions hidden by unit scales

**Status: source-confirmed generalization defect; current dimensionless unit values hide it.** [S3]

The c and delta parameters are dimensionless, so their field derivatives scale with B_star/a_star, not B_star/L_star. `_exact_field_tangents` also evaluates its reference at the passed physical points without dividing by L_star. Existing L_star=B_star=1 runs do not expose either issue.

The code reports raw Euclidean vector norms. Increasing the number of samples changes those norms even for identical response fields. The four-point and 96-point values therefore cannot be read as comparable RMS quantities. The reported value near 0.184 is a derivative-vector norm, not a finite magnetic-field displacement in tesla.

**Repair:** retain signed arrays, define parameter scales, and use fixed-domain weighted RMS. Test nonunit length/field scales and duplicate-point invariance. Keep old values under their original metric names and add a derived normalized record instead of rewriting raw evidence.

## 6. The tested radial m=1 alignment is not a gauge

**Status: important negative result already established by the work.** [S2]

The candidate follows the independently reconverged branch response but changes Eulerian B. Normalized toroidal flux cannot be arbitrarily relabelled while retaining its physical definition. Coefficient correlation or a low geometry fit residual does not establish a gauge transformation.

**Next action:** keep this hypothesis closed unless a different, fully specified transformation is tested. Use a regular poloidal map with compensating lambda and fixed-point B/p/s checks for any true gauge test. Do not fit away the physical response error.

## 7. The sheared lambda solve fixes direction; verify both flux densities

**Status: useful implementation with an additional independent oracle proposed.** [S5, S12, L11]

The existing transport equation and nonzero-lambda projection greatly improve the sheared-A seed. Its surface tangents correctly differentiate periodic R/Z and include the cylindrical basis derivatives. Keep this work.

A zero-mean transport solution alone is not always unique at rational transform and does not independently establish the full magnetic amplitude. The full-flux potential in the continuation plan recovers lambda from both `J B^theta` and `J B^phi`, checks flux periods and solenoidality, and avoids resonant field-line denominators. It can be implemented with the already available spectral Poisson operator.

Additional source limitations are a hard-coded NFP=2 in the transport operator, even-grid Nyquist choices, and an axis lambda row assigned without a complete radial regularity certificate. These are limitations to test, not proof that the current one-surface field is wrong.

**Executed review pilot:** the supplied NumPy reconstruction, using the original supplied analytical reference, reproduced B on a sheared-A s=0.5 surface with relative errors down to about 1.5e-10. It did not project into VMEX or evaluate current. Current-code and full-volume replication are next tasks, not completed claims.

## 8. Constraint removal has different outcomes in the two tested families

**Status: measured result; no universal default change supported.** [S2]

The integer NS129 zero-TCON candidate is much more accurate on legacy96 than the matched default case. Conversely, default TCON0 converged from the corrected sheared-A NS17 seed, whereas zero weight oscillated to the cap with a large flux-label drift. Both sheared outputs remain unaccepted and worse than the analytical projection.

**Next action:** converge physical scoring and study projection-to-root movement. A low discrete residual can move away from the exact field at finite resolution. Do not treat every such movement as a constraint defect before checking full input, representation, operator and field evaluation.

## 9. Measurement repair is partial, so use complementary tests and useful candidates

**Status: demonstrated mechanics, unresolved volume certification.** [S2, S6]

Exact crossings on one ray and streamed smoke measurements do not establish an aligned full-volume quadrature. `composite_spread` retains the old order-2/order-4 structure and naming. Historical raw reports remain useful, but acceptance needs the actual domain, knot set, inversion error and stable fine-grid comparisons.

**Next action:** use saved promising NS129 states rather than demanding high absolute precision from the worst NS33 state. Add weak stress moments which depend on B and p, not a numerical curl. This complements the pointwise current/force test and cannot replace it. Account for nonzero divergence, boundary terms and possible internal trace jumps explicitly.

## 10. The original probe bytes are missing; do not repair history by invention

**Status: documented provenance gap.** [S7]

The root-status amendment records different hashes for the original runtime script and checked-in corrected probe. The exact former bytes are not included. This does not erase the observed residual failure, but prevents byte-exact reproduction of that old implementation from the checkpoint alone.

**Repair:** capture a committed tree or complete execution patch and script bytes before future runs. Reproduce with corrected source under a new run ID after the base-root diagnosis. Keep the original hash, record and amendment unchanged.

## 11. The current upstream research branch has progressed, but is not the benchmark's answer

**Status: refreshed upstream context.** [S10, S11]

VMEX main is now 926892ab. PR448 remains open draft at 131580da. Its R6 description provides independent force certification for a restricted axisymmetric fixed-profile state, while projected stationarity still misses its gate and derivative qualification is open. The local/global step comparison also remains provisional.

**Next action:** keep historical/current/experimental snapshots separate. Borrow small tested ingredients, not acceptance claims. In particular, PR448's magnetic force normalization is not analytical B error. No merge or blanket adoption is authorized.

## 12. The study needs one active response path, not more hard-coded diagnostic variants

**Status: maintainability recommendation based on the reached source.** [S3-S6]

Several new scripts reproduce input-map, configuration, state-load and physical-JVP setup with timestamp-specific paths. Those experiments have scientific value, but repeated implementations make it harder to keep the operator and source identities aligned.

**Next action:** consolidate only the real shared operations into existing helpers. Add explicit paths/configuration arguments to active drivers. Preserve historical scripts for reproduction; do not rewrite them all at once or build a workflow framework. Give each new block a question, a bounded decision test and a stop condition.

## Review execution limits

The GitHub reference CI was inspected, not rerun locally. No new VMEX, DESC, coupled free-boundary, kinetic or production optimization run was performed. The ten delivered mathematical tests and the sheared-reference surface demo did run. The source ranges and actual pins are listed in the references and snapshot file. A whole-tree inventory, semantic audit, executed unit tests and physical validation remain different forms of evidence.
