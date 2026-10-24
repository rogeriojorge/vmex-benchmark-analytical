"""Added current at iota = 1: a sheared field through the resonance, relaxed without nested surfaces (MRX).

The start field is the Lopez D vacuum 2-form B0 plus eps * curl(a(rho) d zeta), a toroidal-current-like
poloidal field that shears the (exactly closed) lines: the rotation per turn becomes 1 + d_iota(rho) with
d_iota crossing zero at rho_star (an iota = 1 surface inside the plasma). A resonant seed
(``mrx.relaxation.seeding.parallel_seed``, m, n = 1, 1 at rho_star) lets an island open. The field is then
relaxed by MRX's ideal energy descent (Newton direction): field-line topology is frozen, the force residual
falls, the Poincare map and island width are measured before and after.

    python lopez_mrx_relax.py --eps 0.02 --seed 1e-3 --ns 8 12 24 --steps 40 --out results/lopez_mrx/relax/x
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lopez_mrx as lm  # noqa: E402


def shear_field(seq, a_profile):
    """Dirichlet 2-form curl(a(rho) d zeta) (no normal component on the wall)."""
    import jax.numpy as jnp
    free = seq.free
    A = free.interpolate(lambda x: jnp.array([0.0, 0.0, a_profile(x[0])]), 1, frame="logical")
    return seq.restrict(free.G[1] @ A, 2)


def rotation_profile(seq, dof, r_grid, turns=30, steps=48, theta0=0.13):
    seeds = np.concatenate([[[0.01, 0.0]], np.stack([r_grid, np.full_like(r_grid, theta0)], -1)])
    ys, ok = lm.trace_lines(seq, dof, seeds, turns, steps)
    return lm.rotation_per_turn(ys[1:], steps, lm.axis_track(ys[0], steps))


def measure(seq, dof, label, lines=80, periods=200):
    from mrx.diagnostics.islands import islands
    from mrx.diagnostics.poincare import poincare
    res = poincare(seq, dof, lines=lines, periods=periods, planes=2)
    chains = islands(seq, dof, res, m_max=1, periods=periods)
    shown = res["shown"]
    return dict(label=label, chaotic=int((res["keep"] & res["chaotic"]).sum()), lost=int((~res["keep"]).sum()),
                iota_min=float(res["iota"][shown].min()), iota_max=float(res["iota"][shown].max()),
                islands=[dict(r_chain=c["r_chain"], width=c["width"], residue=c["residue"], n_O=len(c["O"]),
                              n_X=len(c["X"])) for c in chains]), res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.02, help="shear amplitude (rotation change ~ eps * 0.5)")
    ap.add_argument("--rho-star", type=float, default=0.6)
    ap.add_argument("--seed", type=float, default=0.0, help="resonant seed amplitude dBr/B^zeta (0: none)")
    ap.add_argument("--ns", type=int, nargs=3, default=(8, 12, 24))
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=5)
    ap.add_argument("--newton", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from mrx.relaxation.loop import TimeStepper, initial_state, relax
    from mrx.relaxation.seeding import parallel_seed
    t0 = time.time()
    seq = lm.build_sequence(a.ns, a.p, lm.LabelMap(twist=-1))
    B0, _ = lm.vacuum_field(seq)
    B0 = B0/seq.l2_norm(B0, 2)
    rs = a.rho_star
    shear = shear_field(seq, lambda r: a.eps*(r**4/4 - rs**2*r**2/2))
    B = B0 + shear
    rec = dict(args=vars(a), built_s=time.time() - t0)
    r_grid = np.linspace(0.1, 0.9, 9)
    rec["rotation_vacuum_dev"] = (rotation_profile(seq, B0, r_grid) - 1).tolist()
    rec["rotation_sheared_dev"] = (rotation_profile(seq, B, r_grid) - 1).tolist()
    rec["r_grid"] = r_grid.tolist()
    if a.seed:
        rr = np.linspace(0, 1, 401)
        prof = np.exp(-((rr - rs)/0.08)**2)
        dB = parallel_seed(seq, B, 1, 1, prof, rr)
        B = B + a.seed*dB/seq.l2_norm(dB, 2)*seq.l2_norm(B, 2)
    stages = []
    m0, _ = measure(seq, B, "start")
    stages.append(m0)
    ts = TimeStepper(seq, newton=a.newton)
    state = initial_state(B, ts)
    res = relax(state, ts, a.steps, chunk=a.chunk, verbose=True)
    Bf = res.state.best.B
    rec["relax"] = dict(steps=int(res.steps), resid=[float(x) for x in res.trace["resid"]][::max(1, len(res.trace["resid"])//40)],
                        E=[float(x) for x in res.qoi["E"]], beta_vol=[float(x) for x in res.qoi["beta_vol"]],
                        helicity=[float(x) for x in res.qoi["helicity"]], wall=float(res.wall))
    m1, pres = measure(seq, Bf, "relaxed")
    stages.append(m1)
    rec["stages"] = stages
    rec["rotation_relaxed_dev"] = (rotation_profile(seq, Bf, r_grid) - 1).tolist()
    rec["total_s"] = time.time() - t0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(rec, indent=1, default=float) + "\n")
    print(json.dumps(stages, default=float))


if __name__ == "__main__":
    main()
