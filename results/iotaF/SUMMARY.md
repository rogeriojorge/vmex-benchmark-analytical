# Lopez D: is the exact iota = 1 the obstruction? (track iotaF)

VMEX 3b73d6f (CPU, x64) and VMEC2000 on office, one job at a time. Run times were measured under heavy shared load
(office load 40-68 on 36 cores during the solves; 190-650 s per 3000-6000 iteration NS=17 run), so wall times are not benchmarks.
WOUTs are not committed: `wout_manifest_iotaF.json` lists office path + sha256. Per-run receipts, scorer records and solver logs are in `solves/`;
`collect_final.json` aggregates them, `energy_all.json/txt` is the energy test over all 47 scored states (28 earlier + 19 new).

## Finding that changes the premise

Every earlier "exact seed" run was a cold start. The seed was built in the flux-uniform angle (lambda = 0), which violates VMEC's
m = 1 polar constraint (R_1^s = Z_1^c at every phi). `hot_restart_state` then shears the deck-boundary edge (rotated by VMEC's theta shift, 4e-3 = 20 % of the minor
radius) against the interior, the first Jacobian changes sign, and `reguess_initial_axis` rebuilds the cold interior guess
(all old seed logs print `INITIAL JACOBIAN CHANGED SIGN! TRYING TO IMPROVE INITIAL MAGNETIC AXIS GUESS`; iteration-1 fsq is 0.6, a cold interior guess's, not the 6e2 of the exact state).
`lopez_polar.py` puts the same exact surfaces in the polar gauge, theta' = theta - omega(s, phi), omega = -arg z+, lambda = omega (m = 0, winding 0, range 4.3 rad),
so the exact state is a legitimate VMEC state: edge mismatch 6e-14, no Jacobian flag, B error 3e-4 (oracle) / 1.3e-3 (VMEX wout at NS 17).

## Experiments (prediction under H / outcome)

| # | experiment | H predicts | outcome |
|---|---|---|---|
| 1 | signed energy, 47 states | settled W below exact only if deck/scoring bug | scorer integral never below: min signed dW = +3.0e-6; flux-corrected dW/e_B^2 = 0.90-2.48 (Dirichlet identity predicts 1). Solver's own wb (theory K = 2(2 pi)^2) is below exact in 41/47 (down to -4.2e-3), tracking the scorer excess: discrete-functional error, not a different problem. No deck/scoring defect, H not tested by this. |
| 2 | iteration-0 residual of the exact state | resonant (d = n - c m = 0) concentration | d = 0 holds 0.3-1.4 % of the residual (uniform mode share 0.7 %), smooth in d. Physical (tcon0 = 0) residual fsqr/fsqz = 4.2e-4/7.1e-5, 1.1e-4/1.8e-5, 3.0e-5/5.3e-6 at NS 17/33/65 (NS^-2, 85-99 % in the first shells: axis discretization). With the constraint force on it is 6e-2/3e-1, 1e-1/5e-1, 2e-1/1.0 (grows with NS) and 3e2 in the unshifted eta gauge: the exact state is not in VMEC's spectral-condensation gauge. |
| 3 | iota scan, cold start NS 17, 6000 it | away from 1 converges to <= 1e-11 | refuted: iota = 0.8, 0.95, 0.99, 1.0, 1.01, 1.05, 1.2, 1.618, sheared 0.9-1.1 and 1.55-1.65 all stall at fsq_min 2.6e-6 - 3.9e-6, DELT falls to 0.12-0.14 (repeated growth restarts), none converged. Irrational/avoiding profiles behave like iota = 1. Polar-seeded iota = 1.05 / 0.95 (one run each): fsq_min 7.1e-8, spikes to 9e-5, none reaches 1e-11. |
| 4 | gauge shift theta' = theta - k phi (c = 0) | converges if the resonance were representational | the resonant set n = c m is gauge invariant, and the m = 1 constraint pins the gauge, so this cannot remove a resonance. Cold NS 17: iota closure fsq_min 3.3e-6, B error 7e-2; current closure fsq_min 2.8e-6, predicted iota' = 0.016 (same 1.5 % as c = 1). Fails identically. |
| 5 | Solov'ev asymmetric (LASYM) control, same driver | driver defect would fail | VMEX and VMEC2000, NS 33 and 65: fsqr/fsqz/fsql < 1e-14, ier 0. Driver is sound. |
| 6 | new: alternative foliation of the same field (shifted contours, `lopez_foliation.py`), polar gauge | a continuum of exact states at iota = 1 | exists: same boundary, flux and B, different interior (axis moved 0.0016 = 0.1 label semi-axis); residual of the same order as the standard foliation (tcon0 = 0: 5.5e-4/5.6e-5), solver started there stays (B error 2.4e-3, signed dW +3.3e-6). The degeneracy is real but is not what stalls the cold solve. |

## Valid-seed solves (NS 17 unless noted, 6000 iterations, ftol 1e-12; none reaches ftol)

| seed | closure | fsq_min | B vol | axis/minor | signed dW | note |
|---|---|---|---|---|---|---|
| polar eta | iota | 1.5e-7 | 1.35e-3 | 0.006 | +3.7e-6 | old (invalid seed): B 2.0e-2, axis 0.51, dW +9.7e-4 |
| polar eta | current | 1.7e-7 | 5.0e-3 | 0.011 | +3.9e-5 | predicted iota error 1.5e-3 |
| polar eta, tcon0 = 0 | iota | 1.1e-7 | 4.2e-3 | | +3.9e-5 | |
| polar shift 0.1 | iota | 6.0e-8 | 2.4e-3 | | +3.3e-6 | other foliation |
| polar eta NS 33 | iota | 7.8e-7 | 5.2e-3 | | +3.8e-5 | |

The solver now stays on the exact state (0.1-0.6 % of the minor radius off axis instead of 50 %, B error at the projection floor), while fsq keeps spiking
(1e-8 to 1e-4, DELT 0.5 -> 0.17): a time-step/restart dynamics problem at fsq <~ 1e-6, not a displaced or re-foliated attractor.

## Verdict on H

Refuted as the cause of the recovery failure: cold-start stalls are iota-independent (incl. irrational iota), the residual is not resonant, the symmetric control converges, and
the "attractor with displaced axis" was the cold start reached by a mis-built seed. H's structural half (non-unique foliation at iota = 1) is true (exp. 6) but harmless here.
Open: why fsq limit-cycles at 1e-8..1e-4 even from a valid seed (restart/time-step control, constraint force O(1) in the polar gauge), and the cold-start basin.
