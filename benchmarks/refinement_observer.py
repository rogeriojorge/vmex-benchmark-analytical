"""Observe one raw VMEX host state and one uncached fixed-point refinement.

The public ``solve_implicit_with_aux`` callback already calls the memoized
``_refine_fixed_point`` (keyed by ``cfg`` identity and parameter bytes, not by
the supplied state).  Calling that memo again after the public solve can return
the cached result, so a zero shift measured that way says nothing about the
original refinement.  This adapter instead:

* obtains the raw host state with ``refine=False`` (no refinement attempted);
* calls the unmemoized ``_refined_state`` exactly once, with no warm
  correction, while recording every staged step through the module-level seams
  ``_refine_block_factors``, ``_refine_block_step`` and ``_refine_step``;
* measures each step's true linear defect ``e = A d + r`` in the formulation
  the step linearized, and the final state in the refinement's own operator
  (frozen at the input state) and in the re-frozen operator.

It is tied to the private API of the historical VMEX pin; ``implicit`` is passed
in so the contract can be tested without VMEX.  Certificates are kept separate:
``host_returned``, ``refinement_attempted``, ``root_certified`` and
``response_certified`` (never set here).
"""
from __future__ import annotations

import contextlib
import hashlib

import numpy as np

OBSERVER_VMEX_PIN = "b5f5267efc0795c4a49a224e321e9b370975c14c"
FIELDS = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")


def tree_hash(tree) -> str:
    """SHA-256 of named float64 leaves in fixed field order."""
    digest = hashlib.sha256()
    for name in FIELDS:
        value = getattr(tree, name, None)
        if value is None:
            continue
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _leaves(tree):
    return [np.asarray(getattr(tree, name), dtype=np.float64)
            for name in FIELDS if getattr(tree, name, None) is not None]


def tree_norm(tree) -> float:
    return float(np.sqrt(sum(float(np.vdot(v, v)) for v in _leaves(tree))))


def tree_dot(left, right) -> float:
    return float(sum(float(np.vdot(a, b)) for a, b in zip(_leaves(left), _leaves(right))))


def block_norms(tree) -> dict[str, float]:
    return {name: float(np.linalg.norm(np.asarray(getattr(tree, name))))
            for name in FIELDS if getattr(tree, name, None) is not None}


def _map(function, *trees):
    first = trees[0]
    return type(first)(**{name: function(*(getattr(t, name) for t in trees))
                          for name in FIELDS})


def _as_jax(tree):
    """The pinned projector updates leaves with ``.at``; NumPy leaves fail on 3-D decks."""
    import jax.numpy as jnp
    return type(tree)(**{name: jnp.asarray(getattr(tree, name)) for name in FIELDS})


def residual_record(implicit, cfg, params, frozen, mask, state) -> dict:
    """Raw and preconditioned residual norms of ``state`` in operator ``frozen``."""
    frozen, mask, state = _as_jax(frozen), _as_jax(mask), _as_jax(state)
    project = implicit._dof_projector(cfg, mask)
    z = project(state)
    record = {"frozen_state_sha256": tree_hash(frozen),
              "dof_mask_sha256": tree_hash(mask),
              "state_sha256": tree_hash(state)}
    vectors = {}
    for formulation in ("preconditioned", "raw"):
        value = implicit.residual_fn(cfg, frozen, mask, formulation=formulation)(z, params)
        record[formulation] = {"norm": tree_norm(value), "blocks": block_norms(value)}
        vectors[formulation] = value
    return record, vectors


def _linear_defect(implicit, cfg, params, frozen, mask, z, z_new, formulation):
    """Return ``|A d + r|/|r|`` and ``r.A d/|r|^2`` for the step ``d = z_new - z``."""
    import jax
    residual = implicit.residual_fn(cfg, frozen, mask, formulation=formulation)
    d = _map(lambda a, b: np.asarray(b) - np.asarray(a), z, z_new)
    r, Ad = jax.jvp(lambda t: residual(t, params), (z,), (d,))
    r_norm = tree_norm(r)
    defect = _map(lambda a, b: np.asarray(a) + np.asarray(b), Ad, r)
    scale = max(r_norm, 1e-300)
    return {
        "linearized_formulation": formulation,
        "input_residual_norm": r_norm,
        "correction_norm": tree_norm(d),
        "true_linear_defect_over_residual": tree_norm(defect)/scale,
        # Equation (4): r.A d = -|r|^2 + r.e; -1 means an exact Newton step.
        "merit_directional_derivative_over_residual_sq": tree_dot(r, Ad)/scale**2,
    }


@contextlib.contextmanager
def _recording(implicit, trace, measure_defects=True):
    originals = {name: getattr(implicit, name) for name in
                 ("_refine_block_factors", "_refine_block_step", "_refine_step")}

    def block_factors(cfg, params, frozen, mask, z):
        trace.append({"kind": "block_factorization", "at_state_sha256": tree_hash(z)})
        return originals["_refine_block_factors"](cfg, params, frozen, mask, z)

    def block_step(cfg, params, frozen, mask, z, factors):
        z_new, fz, norm, linear = originals["_refine_block_step"](
            cfg, params, frozen, mask, z, factors)
        row = {"kind": "block_newton_step", "input_sha256": tree_hash(z),
               "output_sha256": tree_hash(z_new),
               "trial_preconditioned_residual": float(norm),
               "reported_linear_relative_residual": float(linear),
               "finite": bool(np.isfinite(float(norm))), "_z": z_new}
        if measure_defects:
            row["defect"] = _linear_defect(implicit, cfg, params, frozen, mask,
                                           z, z_new, "raw")
        trace.append(row)
        return z_new, fz, norm, linear

    def krylov_step(cfg, params, frozen, mask, z, fz):
        z_new, fz_new, norm, linear = originals["_refine_step"](
            cfg, params, frozen, mask, z, fz)
        row = {"kind": "gcrot_newton_step", "input_sha256": tree_hash(z),
               "output_sha256": tree_hash(z_new),
               "trial_preconditioned_residual": float(norm),
               "reported_linear_relative_residual": float(linear),
               "finite": bool(np.isfinite(float(norm))), "_z": z_new}
        if measure_defects:
            row["defect"] = _linear_defect(implicit, cfg, params, frozen, mask,
                                           z, z_new, "preconditioned")
        trace.append(row)
        return z_new, fz_new, norm, linear

    implicit._refine_block_factors = block_factors
    implicit._refine_block_step = block_step
    implicit._refine_step = krylov_step
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(implicit, name, value)


def observe_refinement(implicit, cfg, params, state, mask, *, measure_defects=True):
    """Run exactly one uncached ``_refined_state`` from ``state`` and record it.

    Returns ``(refined_state, record)``.  The memo ``_LAST_REFINED`` is neither
    read nor written; the record says so by comparing its entry before/after.
    """
    state, mask = _as_jax(state), _as_jax(mask)
    memo_before = implicit._LAST_REFINED.get(cfg)
    before, before_vectors = residual_record(implicit, cfg, params, state, mask, state)
    trace = []
    with _recording(implicit, trace, measure_defects):
        refined = implicit._refined_state(cfg, params, state, mask,
                                          initial_correction=None)
    memo_after = implicit._LAST_REFINED.get(cfg)
    project = implicit._dof_projector(cfg, mask)
    in_hash, out_hash = tree_hash(state), tree_hash(refined)
    projected_out = project(refined)
    scale = max(tree_norm(projected_out), 1e-300)
    for row in trace:
        if "_z" in row:
            row["distance_to_returned_over_state"] = tree_norm(_map(
                lambda a, b: np.asarray(a) - np.asarray(b), row.pop("_z"), projected_out))/scale
    # The returned state is state + P(best - z0): match trials to roundoff.
    accepted = [i for i, row in enumerate(trace)
                if row.get("distance_to_returned_over_state", np.inf) <= 1e-13]
    after_same, after_vectors = residual_record(implicit, cfg, params, state, mask, refined)
    after_refrozen, _ = residual_record(implicit, cfg, params, refined, mask, refined)
    tol = float(cfg.refine_tol)
    steps = [row for row in trace if row["kind"] != "block_factorization"]
    for index, row in enumerate(trace):
        if row["kind"] != "block_factorization":
            row["accepted_as_returned_state"] = index in accepted
    final = after_same["preconditioned"]["norm"]
    record = {
        "observer": "single_pass_uncached_refined_state",
        "observer_vmex_pin": OBSERVER_VMEX_PIN,
        "memo_untouched": memo_before is memo_after,
        "refine_tol": tol,
        "input_state_sha256": in_hash,
        "output_state_sha256": out_hash,
        "state_changed": in_hash != out_hash,
        "correction_norm": tree_norm(_map(lambda a, b: np.asarray(b) - np.asarray(a),
                                          state, refined)),
        "attempted_steps": len(steps),
        "trace": trace,
        "returned_trial_indices": accepted,
        "residual_before_refinement_operator": before,
        "residual_after_refinement_operator": after_same,
        "residual_after_refrozen_operator": after_refrozen,
        "status": {
            "host_returned": True,
            "refinement_attempted": bool(steps),
            "root_certified": bool(np.isfinite(final) and final <= tol),
            "response_certified": False,
        },
    }
    return refined, record, {"before": before_vectors, "after": after_vectors}


def memo_probe(implicit, cfg, params, state, mask):
    """Two public memo calls: the second must be a hit with no new refinement."""
    def refinements():
        return implicit._SOLVE_STATS.get(cfg, {}).get("refinements", 0)
    counts = [refinements()]
    first = implicit._refine_fixed_point(cfg, params, state, mask)
    counts.append(refinements())
    second = implicit._refine_fixed_point(cfg, params, state, mask)
    counts.append(refinements())
    return {"config_is_content_interned_note": "make_config interns by content; "
            "identical settings share this memo within one process",
            "first_call_refined": counts[1] > counts[0],
            "second_call_memo_hit": counts[2] == counts[1],
            "same_object": first is second,
            "output_sha256": tree_hash(first)}


def raw_host_state(implicit, cfg, params):
    """One raw host solve with refinement disabled; returns state, mask, record."""
    import jax
    params_np = jax.tree.map(lambda a: np.asarray(a, dtype=np.float64), params)
    state, mask = implicit._host_solve_and_mask(cfg, params_np, refine=False)
    state = type(state)(**{n: np.asarray(getattr(state, n)) for n in FIELDS})
    mask = type(mask)(**{n: np.asarray(getattr(mask, n)) for n in FIELDS})
    result = implicit._LAST_SOLVE[cfg][1]
    return state, mask, {
        "host_returned": True,
        "converged": bool(result.converged),
        "iterations": int(result.iterations),
        "ier_flag": int(result.ier_flag),
        "fsqr_fsqz_fsql": [float(result.fsqr), float(result.fsqz), float(result.fsql)],
        "raw_state_sha256": tree_hash(state),
        "dof_mask_sha256": tree_hash(mask),
    }


def derivative_gate(root_record: dict) -> dict:
    """Refuse an accepted derivative unless the root certificate passed."""
    certified = bool(root_record.get("status", {}).get("root_certified"))
    return {"root_certified": certified,
            "derivative_may_be_accepted": certified,
            "reason": None if certified else "root certificate failed; diagnostic only"}
