"""Stage 6: the half-mesh |B|^2 that the force kernel uses, against the exact |B|^2 on the same surfaces.

The force of the exact state involves P = p + |B|^2/2 on the half mesh, built by ``fields.magnetic_fields`` from the
projected R, Z, lambda (B^u = (chi' + dlam/dzeta)/sqrt g, B^v = (phi' + dlam/dtheta)/sqrt g).  The continuous interior field
used by the scorer (``VmecInteriorField``) reproduces B to the projection floor (1e-7), but that is a different code path
(and for lambda a different parity route).  Here |B|^2 of the force path is compared with the exact |B|^2 at the exact
surface of the half-mesh label (surface(label(s_half), theta, phi) on the same theta grid): the theta-dependent part
(B^2 / <B^2>_theta - 1, the part that enters  d/ds oint P R R_theta d theta)  as a function of NS, for symmetric and
asymmetric Solov'ev.  A resolution-independent mismatch that exists only with the asymmetric (cos) lambda would point at the lambda
(cos, m = 1) route of the force path.

Half-mesh index j (>= 1) is the cell between surfaces j-1 and j at s = (j - 1/2)/(NS-1); the exact surface at that label is
not the average of its neighbours, so a smooth O(h^2) difference is expected and is what the NS scan separates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mech_core as M  # noqa: E402
from analytic import cases, field, label_at_s, surface  # noqa: E402

ROOT = M.ROOT


def half_mesh_B2(seed, rt):
    import vmex.core.solver as S
    s = rt.setup.s_full
    setup = rt.setup
    _, geometry = S._geometry(seed, rt)
    jacobian = S.half_mesh_jacobian(geometry, s=s)
    metrics = S.metric_elements(geometry, s=s)
    fields = S.magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig, s=s, phips=setup.phips, phipf=setup.phipf,
        chips=setup.chips, signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass, ncurr=setup.ncurr, enclosed_current=setup.icurv)
    b2 = np.asarray(fields.bsupu*fields.bsubu + fields.bsupv*fields.bsubv)    # |B|^2 on the half mesh (T^2 in these units)
    return b2, np.asarray(jacobian.sqrt_g)


def run(case_name, ns, out=None, polar=False):
    case = cases()[case_name]
    inp, rt, seed, info = M.exact_seed(case_name, ns, 0.0, polar=polar)
    shift = float(info.get('polar_delta', [0.0])[-1])
    b2, sg = half_mesh_B2(seed, rt)
    ntheta3, nzeta = b2.shape[1], b2.shape[2]
    ntheta = int(rt.resolution.ntheta)
    theta = 2*np.pi*np.arange(ntheta3)/ntheta
    phi = 2*np.pi*np.arange(nzeta)/(nzeta*case.nfp)
    hs = 1.0/(ns-1)
    rows = []
    for j in range(2, ns):          # skip the first half cell (axis), the integrals are dominated by the bulk
        sh = (j-0.5)*hs
        xyz = np.asarray(surface(case, float(label_at_s(case, sh)), (theta+shift)[:, None], phi[None, :]))
        Bex = np.asarray(field(case, xyz.reshape(-1, 3))[0]).reshape(ntheta3, nzeta, 3)
        b2ex = np.sum(Bex**2, -1)
        ratio = b2[j]/b2ex
        # theta-dependent part: (B^2/<B^2>_theta) of each, compared
        a = b2[j]/b2[j].mean(0, keepdims=True)
        b = b2ex/b2ex.mean(0, keepdims=True)
        rows.append(dict(js=j, s=sh, mean_ratio=float(ratio.mean()), ratio_spread=float(ratio.max()-ratio.min()),
                         theta_part_max_diff=float(np.abs(a-b).max()), theta_part_rms_exact=float(np.sqrt(np.mean((b-1)**2)))))
    rec = dict(case=case_name, ns=ns, polar=polar, rows=rows,
               max_theta_part_diff=float(max(r["theta_part_max_diff"] for r in rows)),
               max_ratio_spread=float(max(r["ratio_spread"] for r in rows)),
               mean_ratio_range=[float(min(r["mean_ratio"] for r in rows)), float(max(r["mean_ratio"] for r in rows))])
    print(case_name, ns, "max theta-part diff %.2e (exact theta-part rms %.2e) ratio spread %.2e mean ratio %.6f..%.6f" % (
        rec["max_theta_part_diff"], rows[len(rows)//2]["theta_part_rms_exact"], rec["max_ratio_spread"], *rec["mean_ratio_range"]), flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1, allow_nan=False)+"\n")
    return rec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", nargs="+", default=["solovev_symmetric", "solovev_asymmetric"])
    p.add_argument("--ns", type=int, nargs="+", default=[17, 33, 65])
    p.add_argument("--out", type=Path, default=ROOT/"results/mechanism/fieldcheck")
    p.add_argument("--polar", action="store_true", help="seed re-parametrised into the m = 1 polar gauge; the exact surface is sampled at theta' + delta")
    a = p.parse_args(argv)
    for c in a.case:
        for ns in a.ns:
            run(c, ns, a.out/f"{c}-ns{ns}{'-polar' if a.polar else ''}.json", polar=a.polar)


if __name__ == "__main__":
    main()
