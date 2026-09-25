"""Score any VMEC-format WOUT (VMEX, VMEC2000, VMEC++) against the exact equilibrium.

Uses only the WOUT file and the exact reference, with the same code for every
producer:

* flux-surface position: the exact flux label at points of each numerical
  surface (pure geometry, no field evaluation), and the distance of the
  numerical magnetic axis from the exact axis;
* B on native surfaces, from the WOUT field harmonics through one shared
  reader (``vmex.surface_field_data_from_wout``), relative to the exact B;
* iota on the full mesh relative to the exact transform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from analytic import cases, field, flux, iota, label_at_s, surface


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positions(w, j, theta, zeta):
    xm, xn = np.asarray(w.xm), np.asarray(w.xn)
    ang = theta[:, None, None]*xm - zeta[None, :, None]*xn
    R = np.sum(np.asarray(w.rmnc)[j]*np.cos(ang), -1)
    Z = np.sum(np.asarray(w.zmns)[j]*np.sin(ang), -1)
    if bool(w.lasym):
        R = R + np.sum(np.asarray(w.rmns)[j]*np.sin(ang), -1)
        Z = Z + np.sum(np.asarray(w.zmnc)[j]*np.cos(ang), -1)
    ph = np.broadcast_to(zeta[None, :], R.shape)
    return np.stack((R*np.cos(ph), R*np.sin(ph), Z), -1).reshape(-1, 3)


MU0 = 4e-7*np.pi


def exact_B_J(case, xyz):
    f = lambda x: field(case, x)[0]  # noqa: E731
    X = jnp.asarray(xyz)
    B = np.asarray(jax.vmap(f)(X))
    g = np.asarray(jax.vmap(jax.jacfwd(f))(X))
    curl = np.stack([g[:, 2, 1]-g[:, 1, 2], g[:, 0, 2]-g[:, 2, 0], g[:, 1, 0]-g[:, 0, 1]], -1)
    return B, curl/MU0


def wout_current(w, j, theta, zeta):
    """VMEC's own current: J = (currumnc e_u + currvmnc e_v)/sqrt(g) (sqrt(g) half mesh -> full)."""
    xm, xn = np.asarray(w.xm), np.asarray(w.xn)
    xmn, xnn = np.asarray(w.xm_nyq), np.asarray(w.xn_nyq)
    T, P = np.meshgrid(theta, zeta, indexing="ij")
    T, P = T.ravel(), P.ravel()
    ang, angn = T[:, None]*xm-P[:, None]*xn, T[:, None]*xmn-P[:, None]*xnn

    def cos_sum(name, angle, row=j):
        return np.sum(np.asarray(getattr(w, name))[row]*np.cos(angle), -1)

    def sin_sum(name, angle, row=j):
        return np.sum(np.asarray(getattr(w, name))[row]*np.sin(angle), -1)

    R, Z = cos_sum("rmnc", ang), sin_sum("zmns", ang)
    Ru = np.sum(-xm*np.asarray(w.rmnc)[j]*np.sin(ang), -1)
    Zu = np.sum(xm*np.asarray(w.zmns)[j]*np.cos(ang), -1)
    Rv = np.sum(xn*np.asarray(w.rmnc)[j]*np.sin(ang), -1)
    Zv = np.sum(-xn*np.asarray(w.zmns)[j]*np.cos(ang), -1)
    Ju, Jv = cos_sum("currumnc", angn), cos_sum("currvmnc", angn)
    gm = np.asarray(w.gmnc)
    upper = min(j+1, gm.shape[0]-1)
    sqrtg = np.sum(0.5*(gm[j]+gm[upper])*np.cos(angn), -1)
    if bool(w.lasym):
        R, Z = R+sin_sum("rmns", ang), Z+cos_sum("zmnc", ang)
        Ju, Jv = Ju+sin_sum("currumns", angn), Jv+sin_sum("currvmns", angn)
        raise NotImplementedError("LASYM WOUT current route: derivative terms not added")
    eR = np.stack((np.cos(P), np.sin(P), 0*P), -1)
    ep = np.stack((-np.sin(P), np.cos(P), 0*P), -1)
    ez = np.stack((0*P, 0*P, 1+0*P), -1)
    eu = Ru[:, None]*eR+Zu[:, None]*ez
    ev = Rv[:, None]*eR+R[:, None]*ep+Zv[:, None]*ez
    xyz = R[:, None]*eR+Z[:, None]*ez
    return xyz, (Ju[:, None]*eu+Jv[:, None]*ev)/sqrtg[:, None]


def native_field(w, deck):
    """VMEX's continuous interior field built from this WOUT (any producer)."""
    import vmex
    from vmex.core.restart import state_from_wout
    inp = vmex.VmecInput.from_file(deck)
    state = state_from_wout(w, inp=inp)
    return vmex.VmecInteriorField.from_state(inp, state)


def exact_s(case, xyz):
    labels = np.asarray(jax.vmap(lambda x: field(case, x)[1])(jnp.asarray(xyz)))
    edge_flux = float(flux(case, case.edge))
    return np.asarray([float(flux(case, max(l, 0.0))) for l in labels])/edge_flux


def score(wout_path, case_name, ntheta=32, nzeta=16, deck=None):
    import vmex
    case = cases()[case_name]
    w = vmex.read_wout(wout_path)
    ns = int(w.ns)
    s_mesh = np.linspace(0, 1, ns)
    theta = 2*np.pi*np.arange(ntheta)/ntheta
    zeta = 2*np.pi*np.arange(nzeta)/(nzeta*int(w.nfp))
    # Axis: numerical axis row versus the exact axis at the same toroidal angle.
    axis_xyz = positions(w, 0, np.zeros(1), zeta)
    exact_axis = np.asarray(surface(case, 0.0, np.zeros(nzeta), zeta), dtype=float).reshape(-1, 3)
    minor = np.sqrt(float(np.mean(np.sum((positions(w, ns-1, theta, zeta)
                                          - np.repeat(axis_xyz, ntheta, 0).reshape(ntheta, nzeta, 3)
                                            .reshape(-1, 3))**2, -1))))
    axis_offset = float(np.max(np.linalg.norm(axis_xyz-exact_axis, axis=1)))
    surf_rows = []
    for j in range(1, ns):
        s_err = exact_s(case, positions(w, j, theta, zeta))-s_mesh[j]
        surf_rows.append({"j": j, "s": float(s_mesh[j]), "flux_label_error_max": float(np.max(np.abs(s_err))),
                          "flux_label_error_rms": float(np.sqrt(np.mean(s_err**2)))})
    field_rows = []
    for j in sorted({1, 2, max(3, ns//16), ns//4, ns//2, 3*ns//4, ns-2}):
        sf = vmex.surface_field_data_from_wout(w, ntheta=16, nphi=8, s_index=j)
        xyz = np.moveaxis(np.asarray(sf.gamma), 0, -1).reshape(-1, 3)
        B = np.moveaxis(np.asarray(sf.B_total), 0, -1).reshape(-1, 3)
        area = np.linalg.norm(np.moveaxis(np.asarray(sf.area_vector), 0, -1), axis=-1).ravel()
        Bx = np.asarray(jax.vmap(lambda x: field(case, x)[0])(jnp.asarray(xyz)))
        rms = lambda q: np.sqrt(np.sum(area*np.sum(q*q, -1))/np.sum(area))  # noqa: E731
        field_rows.append({"j": j, "s": float(s_mesh[j]), "B_relative_surface_l2": float(rms(B-Bx)/rms(Bx))})
    # Current and field routes at identical points of native surfaces:
    # the standard WOUT current versus VMEX's continuous native field.
    route_rows = []
    native = native_field(w, deck) if deck is not None and not bool(w.lasym) else None
    th8, ze8 = 2*np.pi*np.arange(16)/16, 2*np.pi*np.arange(8)/(8*int(w.nfp))
    for j in sorted({1, 2, max(3, ns//16), ns//4, ns//2, 3*ns//4, ns-2}):
        row = {"j": j, "s": float(s_mesh[j])}
        if not bool(w.lasym):
            xyz, Jw = wout_current(w, j, th8, ze8)
            Bx, Jx = exact_B_J(case, xyz)
            row["J_wout_relative_l2"] = float(np.linalg.norm(Jw-Jx)/np.linalg.norm(Jx))
            if native is not None:
                pts = jnp.asarray(xyz)
                Bn = np.asarray(native.B(pts))
                # gradB[p, i, j] = dB_i/dx_j here; the transposed order gives exactly -J
                # (checked against the exact current on the axisymmetric case).
                gB = np.asarray(native.gradB(pts))
                Jn = np.stack([gB[:, 2, 1]-gB[:, 1, 2], gB[:, 0, 2]-gB[:, 2, 0], gB[:, 1, 0]-gB[:, 0, 1]], -1)/MU0
                row["B_native_relative_l2"] = float(np.linalg.norm(Bn-Bx)/np.linalg.norm(Bx))
                row["J_native_relative_l2"] = float(np.linalg.norm(Jn-Jx)/np.linalg.norm(Jx))
        route_rows.append(row)
    iota_w = np.asarray(w.iotaf)
    iota_x = np.asarray([float(iota(case, float(label_at_s(case, s)))) for s in s_mesh])
    return {
        "schema": 1, "evidence": "wout_vs_exact_code_neutral", "case": case_name,
        "wout_sha256": sha256(wout_path), "ns": ns, "mpol": int(w.mpol), "ntor": int(w.ntor),
        "lasym": bool(w.lasym), "fsq": [float(w.fsqr), float(w.fsqz), float(w.fsql)],
        "ier_flag": int(getattr(w, "ier_flag", -1)),
        "axis_offset_m": axis_offset, "minor_radius_m": minor, "axis_offset_over_minor": axis_offset/minor,
        "flux_label_error_max": float(max(r["flux_label_error_max"] for r in surf_rows)),
        "surfaces": surf_rows, "fields": field_rows,
        "B_relative_mid": float(np.median([r["B_relative_surface_l2"] for r in field_rows])),
        "B_relative_near_axis": field_rows[0]["B_relative_surface_l2"],
        "iota_max_abs_error": float(np.max(np.abs(iota_w[1:]-iota_x[1:]))),
        "routes": route_rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args(argv)
    receipt = json.loads((args.run_dir/"receipt.json").read_text())
    if not receipt.get("wout"):
        raise SystemExit("run produced no WOUT")
    deck = args.run_dir/f"input.{Path(receipt['wout']['path']).stem.removeprefix('wout_')}"
    record = score(args.run_dir/receipt["wout"]["path"], args.case, deck=deck)
    record["run_id"] = receipt["run_id"]
    record["code"] = receipt["code"]
    record["scorer_sha256"] = sha256(__file__)
    (args.run_dir/"exact_score.json").write_text(json.dumps(record, indent=2)+"\n")
    print(record["run_id"], "axis %.2e  flux %.2e  B(mid) %.2e  B(j=1) %.2e  iota %.2e  fsq %.1e" % (
        record["axis_offset_m"], record["flux_label_error_max"], record["B_relative_mid"],
        record["B_relative_near_axis"], record["iota_max_abs_error"], max(record["fsq"])), flush=True)
    for r in record["routes"]:
        print("   j=%d s=%.4f  J_wout %.2e  J_native %s  B_native %s" % (
            r["j"], r["s"], r.get("J_wout_relative_l2", float("nan")),
            "%.2e" % r["J_native_relative_l2"] if "J_native_relative_l2" in r else "-",
            "%.2e" % r["B_native_relative_l2"] if "B_native_relative_l2" in r else "-"))


if __name__ == "__main__":
    main()
