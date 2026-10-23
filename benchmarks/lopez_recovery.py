"""Lopez D recovery protocols: the exact state projected onto the VMEC representation.

``oracle_wout`` builds, in NumPy only, the WOUT-like state of the exact solution at
(ns, mpol, ntor): full-mesh R, Z harmonics of the exact flux surfaces, lambda = 0 and
the exact transform, contravariant field B^zeta = Phi_edge/(2 pi |sqrt g|),
B^theta = iota B^zeta with sqrt g from the half-mesh geometry exactly as the scorer (and VMEC)
form it. Scoring it with ``score_lopez`` measures the resolved-projection floor: what a
VMEC representation of that size can hold, before any solver is involved. ``kind='alpha'``
is the direction-only negative control: the same surfaces and the same dtheta/dzeta = iota but
the ordinary ellipse angle, which is not flux-uniform, so the assumed uniform field density
is wrong.

``vmex_projection`` puts the same surfaces into a VMEX SpectralState, writes it through
``wout_from_state`` (no solve) and returns it for the same scorer; the exact-seed solve
starts from it. Cold-start solves run through ``cross_code_run.py``.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lopez_surfaces as ls  # noqa: E402
import score_lopez as sl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TWO_PI = 2*np.pi


def gauge_record():
    rec = next(r for r in json.loads((ROOT/"inputs/manifest.json").read_text())["records"]
               if r["case"] == "lopez_D")["lopez"]
    return (np.array(rec["frame_rotation"]), np.array(rec["frame_origin"])), int(rec["sigma"]), int(rec["iota"])


def surface_harmonics(ns, mpol, ntor, kind="eta", grid=(64, 256), frame_gauge=None):
    """Full-mesh (ns) R/Z cos and sin harmonics of the exact surfaces at s = j/(ns-1)."""
    frame, sigma, c = frame_gauge or gauge_record()
    th = TWO_PI*np.arange(grid[0])/grid[0]
    ph = TWO_PI*np.arange(grid[1])/grid[1]
    xm, xn = sl.mode_table(mpol, ntor)
    R = np.empty((ns,) + tuple(grid))
    Z = np.empty_like(R)
    for j in range(ns):
        R[j], Z[j] = ls.boundary_rz(th[:, None], ph[None, :], frame, sigma, c, s=j/(ns - 1), kind=kind)
    rc, rs = sl.analyze(R, xm, xn, 1)
    zc, zs = sl.analyze(Z, xm, xn, 1)
    return xm, xn, rc, rs, zc, zs


def oracle_wout(ns, mpol, ntor, kind="eta", grid=(64, 256), nyq=None):
    """Exact state projected onto the (ns, mpol, ntor) VMEC representation (no solver)."""
    frame_gauge = gauge_record()
    c = frame_gauge[2]
    xm, xn, rc, rs, zc, zs = surface_harmonics(ns, mpol, ntor, kind, grid, frame_gauge)
    nyq = nyq or (2*mpol, 2*ntor)
    xmn, xnn = sl.nyquist_table(*nyq)
    zero = np.zeros((ns, len(xmn)))
    w = SimpleNamespace(nfp=1, ns=ns, mpol=mpol, ntor=ntor, lasym=True, xm=xm, xn=xn, xm_nyq=xmn, xn_nyq=xnn,
                        rmnc=rc, rmns=rs, zmnc=zc, zmns=zs, bsupumnc=zero, bsupvmnc=zero, bsupumns=zero,
                        bsupvmns=zero, iotaf=np.full(ns, float(c)), phi=ls.flux_edge()*np.linspace(0, 1, ns),
                        fsqr=0., fsqz=0., fsql=0., ier_flag=0, signgs=-1)
    w = sl.as_wout(w)
    egrid = sl.grid_for(w)
    half = sl.evaluate_surfaces(w, "half", egrid)
    g = np.abs(sl.sqrtg(half))
    bv = float(ls.flux_edge())/(TWO_PI*g)
    bu = c*bv
    for name, f in (("bsupu", bu), ("bsupv", bv)):
        cos, sin = sl.analyze(f, xmn, xnn, 1)
        for suffix, a in (("mnc", cos), ("mns", sin)):
            full = np.zeros((ns, len(xmn)))
            full[1:] = a
            setattr(w, name + suffix, full)
    return w


def deck_for(mpol, ntor, ns, kind="eta", closure=1, directory=None):
    """VmecInput of the vacuum deck at (mpol, ntor) with a single radial stage ns.

    kind 'alpha' writes the boundary in the ordinary ellipse angle (the direction-only gauge)."""
    import tempfile
    import vmex
    frame, sigma, c = gauge_record()
    if kind == "eta":
        rows, _ = ls.boundary_rows(frame, sigma, c, mpol, ntor)
    else:
        xm, xn, rc, rs, zc, zs = surface_harmonics(2, mpol, ntor, kind, (96, max(256, 6*ntor)), (frame, sigma, c))
        rows = np.array([[m, n, a, b, d, e] for m, n, a, b, d, e in zip(xm, xn, rc[-1], rs[-1], zc[-1], zs[-1])])
    path = Path(directory or tempfile.mkdtemp())/f"input.lopez_{kind}_{mpol}_{ntor}"
    path.write_text(ls.deck_text(rows, closure, sigma, c))
    inp = vmex.VmecInput.from_file(path)
    return replace(inp, ns_array=np.asarray([ns]), ftol_array=np.asarray([1e-14]), niter_array=np.asarray([3000]))


def vmex_state(inp, ns, kind="eta", grid=(64, 256)):
    """Exact surfaces put into a VMEX SpectralState (lambda = 0); no solve."""
    from vmex.core.fourier import mode_table, trig_tables
    from vmex.core.residuals import m1_physical_to_constrained
    from vmex.core.solver import SpectralState, resolution_from_input
    from vmex.core.transforms import physical_to_internal_scale
    res = resolution_from_input(inp, ns=ns)
    modes = mode_table(res.mpol, res.ntor)
    scale = physical_to_internal_scale(modes, trig_tables(res))
    th = TWO_PI*np.arange(grid[0])/grid[0]
    ph = TWO_PI*np.arange(grid[1])/grid[1]
    frame, sigma, c = gauge_record()
    coef = np.zeros((4, ns, modes.mnmax))
    for j in range(ns):
        R, Z = ls.boundary_rz(th[:, None], ph[None, :], frame, sigma, c, s=j/(ns - 1), kind=kind)
        cr, sr = sl.analyze(R, np.asarray(modes.m, float), np.asarray(modes.n, float), 1)
        cz, sz = sl.analyze(Z, np.asarray(modes.m, float), np.asarray(modes.n, float), 1)
        coef[:, j] = (cr, sz, sr, cz)   # R cos, Z sin, R sin, Z cos
    R_cos, Z_sin, R_sin, Z_cos = (coef[i]*scale[None, :] for i in range(4))
    R_cos, Z_sin, R_sin, Z_cos = m1_physical_to_constrained(
        R_cos, Z_sin, R_sin, Z_cos, modes=modes, lthreed=bool(res.lthreed), lasym=True, lconm1=True)
    zeros = np.zeros_like(R_cos)
    return SpectralState(R_cos=R_cos, R_sin=R_sin, Z_cos=Z_cos, Z_sin=Z_sin, L_cos=zeros, L_sin=zeros)


def vmex_projection(mpol, ntor, ns, kind="eta", directory=None):
    """(inp, state, wout) of the exact state through VMEX's own wout_from_state (no solve)."""
    import vmex
    inp = deck_for(mpol, ntor, ns, kind, directory=directory)
    state = vmex_state(inp, ns, kind)
    return inp, state, vmex.wout_from_state(inp=inp, state=state, fsqr=0.0, fsqz=0.0, fsql=0.0)


def table(ns_list, res_list, kind="eta", **kw):
    """Projection floor over (ns, (mpol, ntor)); returns the list of compact score rows."""
    case = sl.lopez_case()
    rows = []
    for mpol, ntor in res_list:
        for ns in ns_list:
            rec = sl.score_wout(oracle_wout(ns, mpol, ntor, kind), case, keep_profile=False, **kw)
            rows.append(dict(kind=kind, ns=ns, mpol=mpol, ntor=ntor, **compact(rec)))
    return rows


def compact(rec):
    v = rec["vacuum"]
    return dict(axis=rec["axis_offset"], flux=rec["flux_label_error_max"], B_s05=rec["B_relative_surface"]["value"],
                B_axis=rec["B_relative_near_axis"]["value"], B_vol=rec["B_relative_volume_l2"],
                iota=rec["iota_max_abs_error"], curl_s05=v["curl_over_B_over_a_rms_at_report"], curl_vol=v["curl_over_B_over_a_rms"],
                curl_outer=v["curl_over_B_over_a_rms_outer_s_ge_0.25"], Iencl=v["enclosed_current_over_aB_max"], div=v["div_over_B_over_a_rms"],
                Phi=v["toroidal_flux_rel_error"], circ=v["circulation_rel_error_max"], W=v["energy_rel_error"],
                V=v["volume_rel_error"], fsq=rec["fsq"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", action="store_true", help="oracle projection floor table")
    parser.add_argument("--kind", default="eta", choices=("eta", "alpha"))
    parser.add_argument("--ns", type=int, nargs="+", default=[17, 33, 65])
    parser.add_argument("--res", type=int, nargs="+", default=[10, 72], help="mpol ntor pairs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    res = list(zip(args.res[0::2], args.res[1::2]))
    rows = table(args.ns, res, args.kind)
    for r in rows:
        print({k: (float("%.3g" % v) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1, allow_nan=False) + "\n")
