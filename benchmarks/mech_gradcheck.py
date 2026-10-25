"""Stage 6: is the discrete force the gradient of the discrete energy?  (asymmetric Z_00 channel and others)

For a state x, ``F_i(x)`` is VMEX's scaled force on dof i and ``W(x) = wb - wp`` (gamma = 0 energy, the functional whose
variation the force implements; ``wb``, ``wp`` from the same funct3d pass).  A consistent pair has
``dW/dx_i = -c_i F_i`` with ``c_i`` independent of the state.  We take central differences of W along single dofs
(``Z_cos`` m = 0 at shell js, and a few others), at the exact state and at randomly perturbed states, and report

* the finite-difference gradient dW/dx_i and the analytic force F_i at each state,
* the ratio c_i = -(dW/dx_i)/F_i per dof and its spread across states (consistency of the kernel with the energy),
* the gradient at the exact state itself (is the exact state a stationary point of the discrete energy?).

Every number is with ``tcon0 = 0`` (the constraint force has no energy).  FD step is a ladder (1e-5, 1e-6, 1e-7 times the
state scale) to show the plateau.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech_core as M  # noqa: E402

ROOT = M.ROOT


@jax.jit
def _energy(state, cache, rt):
    import vmex.core.solver as S
    r = S._evaluate(state, cache, jnp.asarray(101), jnp.asarray(100), jnp.asarray(1.0), rt)
    return r.wb - r.wp


def perturbed(seed, scale, rng, channels=None):
    from dataclasses import replace
    ns, mn = np.asarray(seed.R_cos).shape
    kw = {}
    channels = channels or (("Z_cos", "R_sin", "R_cos", "Z_sin") if np.any(np.asarray(seed.Z_cos)) else ("R_cos", "Z_sin"))
    for c in channels:
        a = np.asarray(getattr(seed, c)).copy()
        noise = rng.standard_normal(a.shape)*scale
        noise[0] = 0.0
        noise[-1] = 0.0
        kw[c] = jnp.asarray(a+noise)
    return replace(seed, **kw)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="solovev_asymmetric")
    p.add_argument("--ns", type=int, default=17)
    p.add_argument("--shells", type=int, nargs="+", default=[1, 2, 3, 5, 9, 13])
    p.add_argument("--states", type=int, default=3)
    p.add_argument("--perturb", type=float, default=1e-3)
    p.add_argument("--out", type=Path, default=ROOT/"results/mechanism/gradcheck.json")
    p.add_argument("--scratch", type=Path, default=Path("/tmp/mech_decks"))
    a = p.parse_args(argv)
    from dataclasses import replace
    inp, rt, seed, _ = M.exact_seed(a.case, a.ns, 0.0)
    rt = M.with_baseline(rt, seed, "vmec")
    rng = np.random.default_rng(7)
    m = np.asarray(rt.modes.m)
    n = np.asarray(rt.modes.n)
    if bool(inp.lasym):
        dofs = [("Z_cos", 0, 0), ("R_cos", 0, 0), ("R_sin", 1, 0), ("Z_cos", 1, 0), ("R_cos", 1, 0), ("Z_sin", 1, 0), ("L_cos", 1, 0)]
    else:
        dofs = [("R_cos", 0, 0), ("R_cos", 1, 0), ("Z_sin", 1, 0), ("R_cos", 2, 0), ("Z_sin", 2, 0), ("R_cos", 3, 0), ("L_sin", 1, 0), ("L_sin", 2, 0)]
    states = [("exact", seed)] + [(f"perturbed{i}", perturbed(seed, a.perturb, rng)) for i in range(a.states-1)]
    rows = []
    for name, st in states:
        cache, _ = M.fresh_cache(st, rt)
        force = M.scaled_force(st, rt, 1.0, cache=cache)[0]
        for ch, mm, nn in dofs:
            col = int(np.where((m == mm) & (n == nn))[0][0])
            for js in a.shells:
                grads = []
                for h in (1e-5, 1e-6, 1e-7):
                    def W(eps):
                        arr = np.asarray(getattr(st, ch)).copy()
                        arr[js, col] += eps
                        return float(_energy(replace(st, **{ch: jnp.asarray(arr)}), cache, rt))
                    grads.append((W(h)-W(-h))/(2*h))
                f = float(np.asarray(getattr(force, ch))[js, col])
                rows.append(dict(state=name, channel=ch, m=mm, n=nn, js=js, fd_grad=grads, force=f,
                                 ratio=[-g/f if f != 0 else None for g in grads]))
        print(name, "done", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(case=a.case, ns=a.ns, rows=rows), indent=1, allow_nan=False)+"\n")
    for r in rows:
        print(f"{r['state']:11s} {r['channel']:6s} m{r['m']} js{r['js']:2d} F {r['force']:+.3e} dW {r['fd_grad'][1]:+.3e} (h-ladder {r['fd_grad'][0]:+.3e} {r['fd_grad'][2]:+.3e}) ratio "
              + ("%.4g" % r['ratio'][1] if r['ratio'][1] is not None else "-"))


if __name__ == "__main__":
    main()
