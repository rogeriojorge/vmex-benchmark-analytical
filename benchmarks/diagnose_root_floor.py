"""C1: bounded root-floor diagnosis from a saved single-pass observation.

Uses the arrays of one ``observe_root_refinement.py`` run.  Questions:

1. Do the raw, first-pass and second-pass states share the frozen (non-DOF)
   entries, i.e. does every pass solve the same equation F(z, P; g) = 0?
2. What is the residual of the final state in the raw-state operator?
3. Along the first block Newton direction from the raw state, does a bounded
   backtracking (alpha = 2^-k, k <= 6) lower the merit, and does one observed
   refinement from the best damped point certify?  (One globalization control.)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from analytic import ROOT
from evidence import capture_execution_source, reserve_run_directory, sha256_file, write_json
import refinement_observer as observer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation", type=Path, required=True,
                        help="directory of an observe_root_refinement.py run")
    parser.add_argument("--output-parent", type=Path,
                        default=ROOT/"results/audit/root_floor_diagnosis")
    args = parser.parse_args(argv)
    import vmex
    from vmex.core import implicit
    parent = json.loads((args.observation/"observation.json").read_text())
    run_id, out = reserve_run_directory(args.output_parent, 
                                        f"{parent['run_id']}-floor-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    here = Path(__file__).resolve().parent
    capture_execution_source(out, [Path(__file__), here/"refinement_observer.py", here/"evidence.py"],
                             packages={"uwplasma/vmex": vmex.__file__})
    started = perf_counter()
    arrays = np.load(args.observation/"observation_arrays.npz")
    if sha256_file(args.observation/"observation_arrays.npz") != parent["arrays"]["sha256"]:
        raise SystemExit("observation arrays do not match their record")
    deck = Path(parent["deck"]["path"])
    if not deck.is_absolute():
        deck = ROOT/deck
    if sha256_file(deck) != parent["deck"]["sha256"]:
        raise SystemExit("deck hash changed")
    eff = parent["effective"]
    inp = vmex.VmecInput.from_file(deck)
    cfg = implicit.make_config(inp, ns=eff["ns"], ftol=eff["ftol"], max_iterations=eff["niter"],
                               refine_tol=eff["refine_tol"], adjoint_tol=1e-10)
    params = implicit.params_from_input(inp)
    # Prime concretely, as the host callback does, before any traced residual.
    implicit._template_runtime(cfg)
    implicit._boundary_pack_tables(cfg)
    from vmex.core.solver import SpectralState

    def load(prefix):
        return SpectralState(**{n: np.asarray(arrays[f"{prefix}_{n}"]) for n in observer.FIELDS})

    raw, mask = load("raw"), load("mask")
    pass1 = load("refined_from_raw_host")
    pass2 = load("refined_from_saved") if "refined_from_saved_R_cos" in arrays else None
    states = {"raw": raw, "pass1": pass1, **({"pass2": pass2} if pass2 is not None else {})}
    project = implicit._dof_projector(cfg, mask)

    def frozen_part(state):
        return observer._map(lambda a, b: np.asarray(a) - np.asarray(b), state, project(state))

    record = {"schema": 1, "run_id": run_id, "parent_run_id": parent["run_id"],
              "parent_observation_sha256": sha256_file(args.observation/"observation.json"),
              "question": "C1 root floor: same operator? damped first step?",
              "state_sha256": {k: observer.tree_hash(v) for k, v in states.items()}}
    record["non_dof_difference_l2"] = {
        k: observer.tree_norm(observer._map(lambda a, b: np.asarray(a)-np.asarray(b),
                                            frozen_part(v), frozen_part(raw)))
        for k, v in states.items()}
    record["cross_operator_residuals"] = {
        f"{state}_in_{frozen}_operator": observer.residual_record(
            implicit, cfg, params, states[frozen], mask, states[state])[0]
        for state in states for frozen in states}

    # Bounded backtracking along the first block Newton direction.
    z0 = project(raw)
    factors = implicit._refine_block_factors(cfg, params, raw, mask, z0)
    z1, _, norm1, linear = implicit._refine_block_step(cfg, params, raw, mask, z0, factors)
    direction = observer._map(lambda a, b: np.asarray(b)-np.asarray(a), z0, z1)
    F = implicit.residual_fn(cfg, raw, mask)
    base = observer.tree_norm(F(z0, params))
    rows = []
    for k in range(7):
        alpha = 2.0**-k
        trial = observer._map(lambda a, d: np.asarray(a)+alpha*np.asarray(d), z0, direction)
        rows.append({"alpha": alpha, "preconditioned_residual": observer.tree_norm(F(trial, params)),
                     "armijo_1e-4": observer.tree_norm(F(trial, params)) <= (1-1e-4*alpha)*base})
    record["backtracking"] = {"base_residual": base, "direction_norm": observer.tree_norm(direction),
                              "full_step_linear_relative_residual": float(linear), "trials": rows}
    accepted = [r for r in rows if r["armijo_1e-4"]]
    if accepted:
        alpha = accepted[0]["alpha"]
        # Full state with only the evolved entries moved: raw - P(raw) + (z0 + alpha d).
        damped =observer._map(lambda s, pz, zt: np.asarray(s) - np.asarray(pz) + np.asarray(zt),
                               raw, z0, observer._map(lambda a, d: np.asarray(a)+alpha*np.asarray(d), z0, direction))
        refined, obs, _ = observer.observe_refinement(implicit, cfg, params, damped, mask)
        record["damped_then_refined"] = {"alpha": alpha, "observation": obs}
        states["damped_refined"] = refined
        record["damped_vs_pass2_l2"] = (observer.tree_norm(observer._map(
            lambda a, b: np.asarray(a)-np.asarray(b), refined, pass2)) if pass2 is not None else None)
    npz = out/"root_floor_states.npz"
    with npz.open("xb") as handle:
        np.savez_compressed(handle, **{f"{k}_{n}": np.asarray(getattr(v, n))
                                       for k, v in states.items() for n in observer.FIELDS})
    record["arrays"] = {"path": npz.name, "sha256": sha256_file(npz)}
    record["elapsed_seconds"] = perf_counter() - started
    write_json(out/"root_floor.json", record, exclusive=True)
    print(run_id)


if __name__ == "__main__":
    main()
