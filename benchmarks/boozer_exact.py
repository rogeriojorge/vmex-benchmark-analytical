"""Score booz_xform_jax output on an exact equilibrium, with and without stellarator symmetry.

No Boozer reference spectrum is needed: at a grid of Boozer angles, R, Z and
phi = zeta_B - nu from the transformed tables must lie on the exact flux
surface, and |B| from the |B| table must equal the exact field there.
"""
import argparse
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import cases, field, flux, iota, label_at_s


def booz_series(b, name, j, angle):
    """Cosine and (when present) sine Boozer tables of one surface at the given angles."""
    cos_name, sin_name = (name + "mnc_b", name + "mns_b")
    total = 0.
    for tab, trig in ((cos_name, np.cos), (sin_name, np.sin)):
        table = getattr(b, tab, None)
        if table is not None:
            total = total + trig(angle) @ np.asarray(table)[:, j]
    return total


def score(wout, case_name, mboz, nboz, s_values=(0.25, 0.5, 0.75), n=24):
    from booz_xform_jax import Booz_xform
    import vmex
    case = cases()[case_name]
    w = vmex.read_wout(wout)
    s_half = (np.arange(int(w.ns)-1)+0.5)/(int(w.ns)-1)
    surfs = sorted({int(np.argmin(abs(s_half-s))) for s in s_values})
    b = Booz_xform(verbose=0, mboz=mboz, nboz=nboz)
    b.read_wout(str(wout), flux=False)
    b.compute_surfs = surfs
    b.run()
    th, ze = np.meshgrid(2*np.pi*np.arange(n)/n, 2*np.pi*np.arange(n)/(n*int(w.nfp)), indexing="ij")
    angle = th.ravel()[:, None]*np.asarray(b.xm_b) - ze.ravel()[:, None]*np.asarray(b.xn_b)
    edge_flux = float(flux(case, case.edge))
    rows = []
    for j, js in enumerate(surfs):
        R, Z = booz_series(b, "r", j, angle), booz_series(b, "z", j, angle)
        phi = ze.ravel() - booz_series(b, "nu", j, angle)
        xyz = jnp.asarray(np.stack((R*np.cos(phi), R*np.sin(phi), Z), -1))
        B, label = jax.vmap(lambda x: field(case, x))(xyz)
        s_exact = np.asarray([float(flux(case, max(float(x), 0.))) for x in label])/edge_flux
        modB, exact = booz_series(b, "b", j, angle), np.linalg.norm(np.asarray(B), axis=-1)
        rows.append({"s": float(s_half[js]), "flux_label_error_max": float(np.max(abs(s_exact-s_half[js]))),
                     "modB_relative_max": float(np.max(abs(modB-exact)/exact)),
                     "iota_error": float(abs(np.asarray(b.iota)[js]
                                             - iota(case, float(label_at_s(case, s_half[js])))))})
    return {"case": case_name, "wout": Path(wout).name, "lasym": bool(w.lasym), "mboz": mboz,
            "nboz": nboz, "surfaces": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wout")
    parser.add_argument("case")
    parser.add_argument("--mboz", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = [score(args.wout, args.case, m, m) for m in args.mboz]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    for r in result:
        print(r["mboz"], [(round(x["s"], 3), f'{x["flux_label_error_max"]:.1e}', f'{x["modB_relative_max"]:.1e}')
                          for x in r["surfaces"]])
