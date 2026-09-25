"""Check that the implicit residual and its derivatives ignore call history."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import vmex
from vmex.core import implicit
from vmex.core.input import VmecInput
from vmex.core.solver import SpectralState, evaluate_forces

from evidence import sha256_file, source_metadata, write_json


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--state-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-vmex", required=True)
    parser.add_argument("--ns", type=int, required=True)
    parser.add_argument("--perturbation", type=float, default=1e-7)
    return parser


def _flat(tree):
    return np.concatenate([np.asarray(leaf, dtype=float).ravel()
                           for leaf in jax.tree.leaves(tree)])


def _difference(reference, actual):
    delta = _flat(actual)-_flat(reference)
    base = _flat(reference)
    return {
        "max_abs": float(np.max(np.abs(delta), initial=0.0)),
        "l2": float(np.linalg.norm(delta)),
        "relative_l2": float(np.linalg.norm(delta)/max(float(np.linalg.norm(base)),
                                                       np.finfo(float).tiny)),
    }


def _ready(tree):
    return jax.block_until_ready(tree)


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.ns < 4 or not np.isfinite(args.perturbation) or args.perturbation <= 0:
        raise SystemExit("NS must be at least four and perturbation must be finite and positive")
    inp = VmecInput.from_file(str(args.input))
    with np.load(args.state_npz, allow_pickle=False) as arrays:
        state = SpectralState(**{name: arrays[name] for name in
            ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})
    if int(state.R_cos.shape[0]) != args.ns:
        raise SystemExit("state radial mesh does not match requested NS")
    source = source_metadata(vmex.__file__, "uwplasma/vmex", importlib.metadata.version("vmex"))
    if source.get("commit") != args.expected_vmex:
        raise SystemExit("imported VMEX source does not match the requested source pin")
    cfg = implicit.make_config(inp, ns=args.ns, ftol=1e-10, max_iterations=1)
    params = implicit.params_from_input(inp)
    mask = implicit._fixed_boundary_dof_mask(cfg)
    project = implicit._dof_projector(cfg, mask)
    frozen = jax.tree.map(jax.lax.stop_gradient, state)
    z_a = project(state)
    rng = np.random.default_rng(20260924)

    def perturb(seed):
        local = np.random.default_rng(seed)
        noise = jax.tree.map(lambda a: jnp.asarray(local.standard_normal(np.shape(a))), state)
        scaled = jax.tree.map(
            lambda a, m, n: a + args.perturbation*m*n, state, mask, noise)
        return project(scaled)

    z_b, z_c = perturb(17), perturb(29)
    tangent = perturb(41)
    tangent = jax.tree.map(lambda a, b: a-b, tangent, z_a)
    tangent = project(tangent)
    cotangent = jax.tree.map(
        lambda a, m: m*jnp.asarray(rng.standard_normal(np.shape(a))), state, mask)
    # Materialize the parameter-shaped runtime before tracing F. This is the
    # public implicit API's setup boundary and keeps its nested setup kernels
    # outside the reusable residual executable.
    runtime = implicit.runtime_from_params(params, cfg)
    jax.block_until_ready(runtime.setup.s_full)
    residual = implicit.residual_fn(cfg, frozen, mask, formulation="preconditioned")

    start = time.perf_counter()
    value_before = _ready(residual(z_a, params))
    _, jvp_before = jax.jvp(lambda z: residual(z, params), (z_a,), (tangent,))
    jvp_before = _ready(jvp_before)
    _, pullback_before = jax.vjp(lambda z: residual(z, params), z_a)
    vjp_before = _ready(pullback_before(cotangent)[0])
    value_b = _ready(residual(z_b, params))
    value_c = _ready(residual(z_c, params))

    # Exercise a fresh evaluator cache and then a returned cache before the
    # reusable residual is called again at exactly z_a.
    x_b = implicit._assemble(z_b, runtime, frozen, project, implicit._edge_mask(cfg))
    _, _, diag_b = evaluate_forces(x_b, runtime)
    x_c = implicit._assemble(z_c, runtime, frozen, project, implicit._edge_mask(cfg))
    evaluate_forces(x_c, runtime, cache=diag_b.cache, iteration=2, iter_last_reset=1)

    value_after = _ready(residual(z_a, params))
    _, jvp_after = jax.jvp(lambda z: residual(z, params), (z_a,), (tangent,))
    jvp_after = _ready(jvp_after)
    _, pullback_after = jax.vjp(lambda z: residual(z, params), z_a)
    vjp_after = _ready(pullback_after(cotangent)[0])
    elapsed = time.perf_counter()-start
    record = {
        "schema": 1,
        "status": "diagnostic_residual_history_test",
        "case": "integer_3d",
        "ns": args.ns,
        "residual_formulation": "preconditioned",
        "gauge_and_dof_mask": "fixed_boundary_analytic_mask; same mask and frozen state throughout",
        "operator_path": "direct single-grid evaluate_forces; multigrid not exercised",
        "source": source,
        "input_sha256": sha256_file(args.input),
        "state_sha256": sha256_file(args.state_npz),
        "perturbation_scale": args.perturbation,
        "fresh_then_reused_cache_interleave": True,
        "values": {
            "F_a_after_b_c_cache_interleave": _difference(value_before, value_after),
            "F_a_norm": float(np.linalg.norm(_flat(value_before))),
            "F_b_norm": float(np.linalg.norm(_flat(value_b))),
            "F_c_norm": float(np.linalg.norm(_flat(value_c))),
        },
        "jvp": {"Fzv_a_after_interleave": _difference(jvp_before, jvp_after)},
        "vjp": {"FzT_w_a_after_interleave": _difference(vjp_before, vjp_after)},
        "elapsed_seconds": elapsed,
        "host_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/
                                    (1024**2 if platform.system() == "Darwin" else 1024)),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output, record, exclusive=True)
    print(json.dumps({
        "status": record["status"],
        "value_history_difference": record["values"]["F_a_after_b_c_cache_interleave"],
        "jvp_history_difference": record["jvp"]["Fzv_a_after_interleave"],
        "vjp_history_difference": record["vjp"]["FzT_w_a_after_interleave"],
        "elapsed_seconds": elapsed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
