"""C1: does a saved centered branch difference satisfy the base linearization?

Plan equation (2): with A = F_z and F_P at the base root (operator frozen at the
base state), d_h = P(x+ - x-)/(2h) and q_h = (P+ - P-)/(2h),

    e_h = A d_h + F_P q_h.

A smooth branch of roots of ONE operator gives |e_h| = O(h^2).  Each endpoint is
also evaluated in the base operator and in its own operator, and the frozen
(non-DOF, non-edge) entries are compared, so auxiliary drift F_g g_a is visible.
Saved states only; no new solve.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import ROOT
from evidence import capture_execution_source, reserve_run_directory, sha256_file, write_json
import refinement_observer as observer

RESPONSE = ROOT/"results/vmex/response_runs/axisym-root-response-20260924T232219.753772Z"
FINE = ROOT/"results/audit/axisym_branch_fine_steps/axisym-branch-fine-20260925T000429.848273Z"


def _sources(step):
    if step >= 1e-4:
        return RESPONSE/"response_ns65.npz", f"state_delta_h{step:.0e}", RESPONSE
    return FINE/"fine_branch_states_ns65.npz", f"state_{step:.0e}", FINE


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=float, nargs="+", default=(1e-3, 3e-4, 1e-4, 3e-5, 1e-5))
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/audit/branch_tangent_defect")
    args = parser.parse_args(argv)
    import vmex
    from vmex.core import implicit
    from vmex.core.solver import SpectralState
    run_id, out = reserve_run_directory(
        args.output_parent, f"ns65-delta-default-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    here = Path(__file__).resolve().parent
    capture_execution_source(out, [Path(__file__), here/"refinement_observer.py", here/"evidence.py"],
                             packages={"uwplasma/vmex": vmex.__file__})
    started = perf_counter()
    base_deck = RESPONSE/"base_input_ns65.indata"
    inp = vmex.VmecInput.from_file(base_deck)
    cfg = implicit.make_config(inp, ns=65, ftol=1e-12, max_iterations=10000,
                               refine_tol=1e-11, adjoint_tol=1e-10)
    implicit._template_runtime(cfg)
    implicit._boundary_pack_tables(cfg)
    mask = jax.tree.map(np.asarray, implicit._fixed_boundary_dof_mask(cfg))
    project = implicit._dof_projector(cfg, mask)
    edge = implicit._edge_mask(cfg)
    base_params = implicit.params_from_input(inp)
    response = np.load(RESPONSE/"response_ns65.npz")

    def load(data, prefix):
        return SpectralState(**{n: jnp.asarray(data[f"{prefix}_{n}"]) for n in observer.FIELDS})

    def sub(a, b):
        return observer._map(lambda x, y: np.asarray(x)-np.asarray(y), a, b)

    def frozen_part(state):
        # Entries neither evolved nor assembled from the boundary: the g data.
        return observer._map(lambda x, pz, e: (np.asarray(x)-np.asarray(pz))*(1-np.asarray(e)),
                             state, project(state), edge)

    base = load(response, "state_base")
    F_base = implicit.residual_fn(cfg, base, mask)
    z_base = project(base)
    tangent_z = project(load(response, "state_tangent_delta"))
    record = {"schema": 1, "run_id": run_id, "question": "C1 eq.(2) branch-tangent defect, default TCON NS65 delta",
              "base_deck_sha256": sha256_file(base_deck),
              "response_arrays_sha256": sha256_file(RESPONSE/"response_ns65.npz"),
              "fine_arrays_sha256": sha256_file(FINE/"fine_branch_states_ns65.npz"),
              "base_state_sha256": observer.tree_hash(base), "dof_mask_sha256": observer.tree_hash(mask),
              "tcon0": float(inp.tcon0),
              "base_residual_in_own_operator": observer.residual_record(
                  implicit, cfg, base_params, base, mask, base)[0]}
    rows = []
    for step in args.steps:
        path, prefix, deck_dir = _sources(step)
        data = np.load(path)
        ends, params = {}, {}
        for sign in ("plus", "minus"):
            ends[sign] = load(data, f"{prefix}_{sign}")
            deck = deck_dir/f"input_delta_ns65_h{step:.0e}_{sign}.indata"
            params[sign] = implicit.params_from_input(vmex.VmecInput.from_file(deck))
        d = observer._map(lambda a, b: np.asarray(a)/(2*step),
                          project(sub(ends["plus"], ends["minus"])), z_base)
        q = jax.tree.map(lambda a, b: (a-b)/(2*step), params["plus"], params["minus"])
        (_, Ad) = jax.jvp(lambda z: F_base(z, base_params), (z_base,), (d,))
        (_, Fq) = jax.jvp(lambda p: F_base(z_base, p), (base_params,), (q,))
        e = observer._map(lambda a, b: np.asarray(a)+np.asarray(b), Ad, Fq)
        (_, At) = jax.jvp(lambda z: F_base(z, base_params), (z_base,), (tangent_z,))
        row = {"step": step,
               "Ad_norm": observer.tree_norm(Ad), "FPq_norm": observer.tree_norm(Fq),
               "defect_norm": observer.tree_norm(e),
               "defect_over_FPq": observer.tree_norm(e)/observer.tree_norm(Fq),
               "defect_blocks": observer.block_norms(e),
               "d_norm": observer.tree_norm(d),
               "d_minus_implicit_tangent_over_tangent": observer.tree_norm(sub(d, tangent_z))/observer.tree_norm(tangent_z),
               "implicit_tangent_Az_plus_FPq_over_FPq": observer.tree_norm(
                   observer._map(lambda a, b: np.asarray(a)+np.asarray(b), At, Fq))/observer.tree_norm(Fq),
               "frozen_entry_drift_over_2h": observer.tree_norm(
                   sub(frozen_part(ends["plus"]), frozen_part(ends["minus"])))/(2*step)}
        for sign in ("plus", "minus"):
            own = implicit.residual_fn(cfg, ends[sign], mask)
            in_base = observer.tree_norm(F_base(project(ends[sign]), params[sign]))
            in_own = observer.tree_norm(own(project(ends[sign]), params[sign]))
            row[f"{sign}_residual_base_operator"] = in_base
            row[f"{sign}_residual_own_operator"] = in_own
            row[f"{sign}_state_sha256"] = observer.tree_hash(ends[sign])
        rows.append(row)
        print(step, row["defect_over_FPq"], row["d_minus_implicit_tangent_over_tangent"],
              row["frozen_entry_drift_over_2h"], row["plus_residual_base_operator"], row["plus_residual_own_operator"])
    record["rows"] = rows
    record["elapsed_seconds"] = perf_counter()-started
    write_json(out/"defect.json", record, exclusive=True)
    print(run_id)


if __name__ == "__main__":
    main()
