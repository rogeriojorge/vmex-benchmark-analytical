"""C0/C1: raw host state plus one observed, uncached refinement for a saved deck.

Example (historical VMEX pin on PYTHONPATH)::

    python benchmarks/observe_root_refinement.py \
        --deck results/audit/axisym_branch_tcon0/<run>/input_delta_ns65_base.indata \
        --saved-state results/audit/axisym_branch_tcon0/<run>/state_base.npz \
        --ns 65 --label ns65-tcon0-base
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import platform
import resource
import sys
from pathlib import Path
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from analytic import ROOT
from evidence import capture_execution_source, reserve_run_directory, sha256_file, write_json
import refinement_observer as observer


def _rss_mib():
    divisor = 1024**2 if platform.system() == "Darwin" else 1024
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/divisor)


def _arrays(prefix, tree):
    return {f"{prefix}_{name}": np.asarray(getattr(tree, name)) for name in observer.FIELDS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--saved-state", type=Path)
    parser.add_argument("--ns", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--ftol", type=float, default=1e-12)
    parser.add_argument("--niter", type=int, default=10000)
    parser.add_argument("--refine-tol", type=float, default=1e-11)
    parser.add_argument("--skip-memo-probe", action="store_true")
    parser.add_argument("--output-parent", type=Path, default=ROOT/"results/audit/refinement_observation")
    args = parser.parse_args(argv)

    import vmex
    from vmex.core import implicit
    run_id = datetime.now(timezone.utc).strftime(f"{args.label}-%Y%m%dT%H%M%S.%fZ")
    run_id, out = reserve_run_directory(args.output_parent, run_id)
    here = Path(__file__).resolve().parent
    source = capture_execution_source(
        out, [Path(__file__), here/"refinement_observer.py", here/"evidence.py"],
        packages={"uwplasma/vmex": vmex.__file__})
    if source["packages"]["uwplasma/vmex"].get("commit") != observer.OBSERVER_VMEX_PIN:
        raise SystemExit("imported VMEX differs from the observer pin")
    started = perf_counter()
    record = {"schema": 1, "run_id": run_id, "question": "C0/C1 single-pass refinement observation",
              "command": " ".join(["python", "benchmarks/observe_root_refinement.py", *sys.argv[1:]]),
              "execution_source_sha256": sha256_file(out/"execution_source.json"),
              "deck": {"path": str(args.deck), "sha256": sha256_file(args.deck)},
              "jax": jax.__version__, "vmex_distribution": importlib.metadata.version("vmex"),
              "devices_used": [str(jax.devices()[0])]}
    inp = vmex.VmecInput.from_file(args.deck)
    record["effective"] = {"ns": args.ns, "ftol": args.ftol, "niter": args.niter,
                           "refine_tol": args.refine_tol, "tcon0": float(inp.tcon0),
                           "ncurr": int(inp.ncurr)}
    arrays = {}

    def fresh_config():
        return implicit.make_config(inp, ns=args.ns, ftol=args.ftol,
                                    max_iterations=args.niter, refine_tol=args.refine_tol,
                                    adjoint_tol=1e-10)

    cfg = fresh_config()
    params = implicit.params_from_input(inp)
    record["params_sha256"] = observer.hashlib.sha256(implicit._params_key(params)).hexdigest()

    t = perf_counter()
    raw, mask, host = observer.raw_host_state(implicit, cfg, params)
    host["seconds"] = perf_counter() - t
    record["raw_host"] = host
    arrays.update(_arrays("raw", raw))
    arrays.update(_arrays("mask", mask))

    starts = [("raw_host", raw)]
    if args.saved_state:
        saved_npz = np.load(args.saved_state)
        saved = type(raw)(**{n: np.asarray(saved_npz[n]) for n in observer.FIELDS})
        record["saved_state"] = {
            "path": str(args.saved_state), "file_sha256": sha256_file(args.saved_state),
            "state_sha256": observer.tree_hash(saved),
            "identical_to_raw_host": observer.tree_hash(saved) == host["raw_state_sha256"],
            "difference_from_raw_host_l2": observer.tree_norm(
                observer._map(lambda a, b: np.asarray(a)-np.asarray(b), saved, raw)),
        }
        if not record["saved_state"]["identical_to_raw_host"]:
            starts.append(("saved", saved))
    write_json(out/"observation.partial.json", record)

    record["observations"] = {}
    for name, start in starts:
        t = perf_counter()
        refined, obs, vectors = observer.observe_refinement(implicit, cfg, params, start, mask)
        obs["seconds"] = perf_counter() - t
        obs["derivative_gate"] = observer.derivative_gate(obs)
        record["observations"][name] = obs
        arrays.update(_arrays(f"refined_from_{name}", refined))
        for stage, formulations in vectors.items():
            for formulation, tree in formulations.items():
                arrays.update(_arrays(f"residual_{name}_{stage}_{formulation}", tree))
        write_json(out/"observation.partial.json", record)

    if not args.skip_memo_probe:
        t = perf_counter()
        probe_cfg = fresh_config()
        memo_config_interned = probe_cfg is cfg
        observer.raw_host_state(implicit, probe_cfg, params)
        memo = observer.memo_probe(implicit, probe_cfg, params, raw, mask)
        memo["seconds"] = perf_counter() - t
        memo["make_config_returned_same_object"] = memo_config_interned
        memo["matches_observer_output"] = (
            memo["output_sha256"] == record["observations"]["raw_host"]["output_state_sha256"])
        record["public_memo_probe"] = memo

    npz = out/"observation_arrays.npz"
    with npz.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    record["arrays"] = {"path": npz.name, "sha256": sha256_file(npz)}
    record["elapsed_seconds"] = perf_counter() - started
    record["peak_host_rss_mib"] = _rss_mib()
    write_json(out/"observation.json", record, exclusive=True)
    (out/"observation.partial.json").unlink()
    raw_obs = record["observations"]["raw_host"]
    print(run_id, "changed", raw_obs["state_changed"], "steps", raw_obs["attempted_steps"],
          "before", raw_obs["residual_before_refinement_operator"]["preconditioned"]["norm"],
          "after", raw_obs["residual_after_refinement_operator"]["preconditioned"]["norm"],
          "certified", raw_obs["status"]["root_certified"])


if __name__ == "__main__":
    main()
