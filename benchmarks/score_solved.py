"""Score solved VMEC-format states with the projection certificate's norm set.

``certify_projection`` scores a projected state's continuous interior field on exact-chart volume
points (B, J = curl B / mu0, and the force J x B - grad p against the exact grad p; relative L2 over
the whole volume and over axis / bulk / edge shells, quadrature uncertainty from a ladder of rules).
This module applies the same rules, the same gates and the same code path to a SOLVED state, so a
projected state and a solver output are measured by one set of norms:

* the state comes from any VMEC-format WOUT (VMEX, VMEC2000, VMEC++) through
  ``vmex.core.restart.state_from_wout`` and VMEX's continuous interior field;
* points where the native field cannot be evaluated (outside the solved last surface, or a failed
  inversion in an unconverged state) are excluded from the sums but counted and reported with their
  volume fraction, never dropped silently;
* the code-neutral surface metrics of ``score_wout_exact`` (schema 2) are kept alongside.

``score_state`` is importable; the command line scores run directories written by ``matrix_solve``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import certify_projection as cp
from analytic import cases

SCHEMA = 1
DEFAULT_LEVELS = cp.MESH_LEVELS  # radial panels aligned with the solved state's mesh


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_evaluator(field):
    """One compiled (B, grad B) evaluator per field, reused across quadrature rules."""
    return jax.jit(lambda p: (field.B(p), field.gradB(p)))


def native_BJ_safe(evaluate, xyz, chunk=16384):
    """Native B, J = curl B (mu0 = 1 units) and a validity mask, never raising on a failed inversion.

    The jitted call returns NaN for a point the field rejects (outside the last surface, or an
    inversion that did not converge); the mask marks those points.
    """
    n = len(xyz)
    B = np.empty((n, 3))
    g = np.empty((n, 3, 3))
    for i in range(0, n, chunk):
        block = xyz[i:i+chunk]
        pad = chunk-len(block)
        if pad:
            block = np.concatenate([block, np.repeat(block[-1:], pad, 0)])
        b, gb = evaluate(jnp.asarray(block))
        B[i:i+chunk] = np.asarray(b)[:chunk-pad]
        g[i:i+chunk] = np.asarray(gb)[:chunk-pad]
    valid = np.isfinite(B).all(-1) & np.isfinite(g).all((-1, -2))
    J = np.stack((g[:, 2, 1]-g[:, 1, 2], g[:, 0, 2]-g[:, 2, 0], g[:, 1, 0]-g[:, 0, 1]), -1)
    return B, J, valid


def field_from_wout(wout, deck):
    """VMEX's continuous interior field of the state stored in ``wout`` (target deck ``deck``)."""
    import vmex
    from vmex.core.restart import state_from_wout
    inp = vmex.VmecInput.from_file(deck)
    return vmex.VmecInteriorField.from_state(inp, state_from_wout(wout, inp=inp))


def score_state(field, rules):
    """Metrics of ``field`` on every quadrature rule, plus the ladder uncertainty when the rules allow."""
    scores, seconds = {}, {}
    evaluate = make_evaluator(field)
    for name, rule in rules.items():
        t = time.perf_counter()
        B, J, valid = native_BJ_safe(evaluate, rule["xyz"])
        scores[name] = cp.metrics(rule, B, J, valid)
        seconds[name] = time.perf_counter()-t
    record = {"scores": scores, "scoring_seconds": seconds}
    if cp.ladder_of(scores):
        record["uncertainty"] = cp.uncertainty(scores)
    return record


def headline(record):
    """Whole-volume B, J, force at the finest rule, with their quadrature uncertainty and invalid fraction."""
    ladder = cp.ladder_of(record["scores"])
    level = ladder[0] if ladder else list(record["scores"])[-1]
    a = record["scores"][level]["all"]
    out = {"level": level, "B": a["B"], "J": a["J"], "force": a["force"],
           "invalid_volume_fraction": a["outside_boundary_volume_fraction"]}
    if "uncertainty" in record:
        out["uncertainty"] = {k: record["uncertainty"]["all"][k]["uncertainty"] for k in ("B", "J", "force")}
        out["relative_uncertainty"] = {k: record["uncertainty"]["all"][k].get("relative_uncertainty") for k in ("B", "J", "force")}
    return out


def score_run(run_dir, rules, case_name, surface=True):
    """Score one ``matrix_solve`` run directory; the WOUT is read, never modified."""
    import vmex
    run_dir = Path(run_dir)
    receipt = json.loads((run_dir/"receipt.json").read_text())
    if not receipt.get("wout"):
        raise SystemExit(f"{run_dir.name}: no WOUT")
    wout_path = run_dir/receipt["wout"]["path"]
    deck = run_dir/f"input.{receipt['run_id']}"
    wout = vmex.read_wout(wout_path)
    t = time.perf_counter()
    record = {"schema": SCHEMA, "evidence": "solved_state_certificate_norms", "run_id": receipt["run_id"],
              "case": case_name, "code": receipt["code"], "protocol": receipt["protocol"], "ns": receipt["ns"],
              "mpol": int(wout.mpol), "ntor": int(wout.ntor), "lasym": bool(wout.lasym),
              "wout_sha256": sha256(wout_path), "gates": cp.GATES, "levels": {k: list(cp.LEVELS[k]) for k in rules},
              "fsq": [float(wout.fsqr), float(wout.fsqz), float(wout.fsql)], "ier_flag": int(getattr(wout, "ier_flag", -1))}
    try:
        record.update(score_state(field_from_wout(wout, deck), rules))
        record["headline"] = headline(record)
    except Exception as exc:  # recorded as a failure of the native route, not hidden
        record["native_error"] = f"{type(exc).__name__}: {exc}"
    if surface:
        from score_wout_exact import score
        try:
            record["surface"] = score(wout_path, case_name, deck=deck)
        except Exception as exc:
            record["surface_error"] = f"{type(exc).__name__}: {exc}"
    record["score_seconds_total"] = time.perf_counter()-t
    record["scorer_sha256"] = {"score_solved": sha256(__file__), "certify_projection": sha256(Path(cp.__file__))}
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--levels", nargs="*", default=list(DEFAULT_LEVELS), choices=list(cp.LEVELS))
    parser.add_argument("--no-surface", action="store_true", help="skip the schema-2 surface score")
    parser.add_argument("--force", action="store_true", help="overwrite an existing score")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    case = cases()[args.case]
    cache = {}

    def rules_for(ns):
        """Fixed-level rules are built once; mesh levels once per NS (a few at a time are kept)."""
        if len(cache) > 8:
            cache.clear()
        for name in args.levels:
            key = (name, ns if name in cp.MESH_LEVELS else None)
            if key not in cache:
                t = time.perf_counter()
                cache[key] = cp.volume_rule(case, name, ns if name in cp.MESH_LEVELS else None)
                print("rule %s (ns %s) built in %.1f s" % (name, key[1], time.perf_counter()-t), flush=True)
        return {name: cache[(name, ns if name in cp.MESH_LEVELS else None)] for name in args.levels}

    def run_ns(d):
        return json.loads((d/"receipt.json").read_text())["ns"]
    for run_dir in sorted(args.run_dirs, key=run_ns):
        target = run_dir/f"volume_score_v{SCHEMA}.json"
        if target.exists() and not args.force:
            print(f"{target} exists; skipping", flush=True)
            continue
        receipt = json.loads((run_dir/"receipt.json").read_text())
        if not receipt.get("wout"):
            print(f"{run_dir.name}: no WOUT; skipped", flush=True)
            continue
        record = score_run(run_dir, rules_for(receipt['ns']), args.case, surface=not args.no_surface)
        target.write_text(json.dumps(record, indent=1, allow_nan=False)+"\n")
        h = record.get("headline")
        print(run_dir.name, "B %.2e J %.2e force %.2e invalid %.1e" % (h["B"], h["J"], h["force"], h["invalid_volume_fraction"])
              if h else record.get("native_error"), flush=True)


if __name__ == "__main__":
    main()
