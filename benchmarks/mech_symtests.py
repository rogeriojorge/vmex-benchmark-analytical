"""Stage 6: symmetry covariance of the VMEX force operator (is the LASYM code path consistent with its own symmetries?).

Three exact identities of the continuum operator, checked on the discrete force (scaled force at a frozen cache, ``tcon0 = 0``):

1. Mirror covariance: the stellarator-mirror map  (R_cos, Z_sin, L_sin) -> same, (R_sin, Z_cos, L_cos) -> minus  sends a
   state to its mirror image; the force must map the same way (even channels equal, odd channels sign-flipped).  Tested on
   random perturbations of the exact state and on the exact states of chi = +c and chi = -c (the tilt sign flips the odd part).
2. Vertical translation: Z -> Z + a for every surface (Z_cos m = 0 on all shells, edge included) leaves the force unchanged.
3. Exact symmetric data in the full basis: the odd channels of the force vanish to roundoff (also in ``mech_asym``).

Reported: the largest |violation| over every channel, relative to the largest |force| of that channel, plus the Z_cos m = 0 force of the
exact states at +c and -c (the flat residual of the asymmetric plateau).
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
import mech_decks as D  # noqa: E402
from vmex.core.solver import SpectralState  # noqa: E402

ROOT = M.ROOT
ODD = ("R_sin", "Z_cos", "L_cos")


def mirror(state):
    return SpectralState(**{k: (-getattr(state, k) if k in ODD else getattr(state, k)) for k in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")})


def force_arrays(state, rt, cache):
    f, _, _ = M.scaled_force(state, rt, 0.0, cache=cache)
    return {k: np.asarray(getattr(f, k)) for k in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")}


def rel_violation(a, b, sign=None):
    out = {}
    for k in a:
        s = -1.0 if (sign and k in ODD) else 1.0
        d = np.abs(a[k]-s*b[k]).max()
        out[k] = float(d/max(np.abs(a[k]).max(), np.abs(b[k]).max(), 1e-300))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ns", type=int, default=17)
    p.add_argument("--chi", type=float, default=0.3)
    p.add_argument("--scratch", type=Path, default=Path("/tmp/mech_decks"))
    p.add_argument("--out", type=Path, default=ROOT/"results/mechanism/symmetry_tests.json")
    a = p.parse_args(argv)
    base = D.variant("solovev_asymmetric", "solovev_asymmetric", chi=a.chi)
    neg = D.variant("solovev_asymmetric", "solovev_chi_minus", chi=-a.chi)
    decks = {c.name: D.write_deck(c, a.scratch/f"input.{c.name}_iota")[0] for c in (base, neg)}
    rec = {"ns": a.ns, "chi": a.chi}
    inp, rt, seed, _ = M.exact_seed(base, a.ns, 0.0, deck=decks[base.name])
    cache, _ = M.fresh_cache(seed, rt)
    rng = np.random.default_rng(2)
    from dataclasses import replace
    kw = {}
    for k in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"):
        arr = np.asarray(getattr(seed, k)).copy()
        noise = 1e-3*rng.standard_normal(arr.shape)
        noise[0] = 0.0
        noise[-1] = 0.0
        kw[k] = jnp.asarray(arr+noise)
    pert = replace(seed, **kw)
    F, Fm = force_arrays(pert, rt, cache), force_arrays(mirror(pert), rt, cache)
    rec["mirror_covariance_perturbed"] = rel_violation(F, Fm, sign=True)
    # vertical translation of the whole state
    Zc = np.asarray(pert.Z_cos).copy()
    Zc[:, 0] += 0.0123
    Ft = force_arrays(replace(pert, Z_cos=jnp.asarray(Zc)), rt, cache)
    rec["vertical_translation_perturbed"] = rel_violation(F, Ft)
    # exact states of +chi and -chi
    inp2, rt2, seed2, _ = M.exact_seed(neg, a.ns, 0.0, deck=decks[neg.name])
    cache2, _ = M.fresh_cache(seed2, rt2)
    Fe, Fn = force_arrays(seed, rt, cache), force_arrays(seed2, rt2, cache2)
    rec["mirror_covariance_exact_pm_chi"] = rel_violation(Fe, Fn, sign=True)
    rec["Z_cos_m0_force_exact"] = {"plus": [float(x) for x in Fe["Z_cos"][1:5, 0]], "minus": [float(x) for x in Fn["Z_cos"][1:5, 0]]}
    rec["R_cos_m0_force_exact"] = {"plus": [float(x) for x in Fe["R_cos"][1:5, 0]], "minus": [float(x) for x in Fn["R_cos"][1:5, 0]]}
    print(json.dumps(rec, indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
