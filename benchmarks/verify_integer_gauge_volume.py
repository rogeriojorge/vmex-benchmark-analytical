"""Certify an exact compensated integer-family poloidal remap in the volume."""
import json
import sys
import resource
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from analytic import ROOT, cases, differential_fields, field, surface


def geometry(case, s, theta, phi, amplitude):
    u = amplitude*s*(1-s)*np.sin(2*theta)
    return surface(case, case.edge*s, theta-u, phi), u


def basis(case, s, theta, phi, amplitude, h):
    es = (geometry(case, s+h, theta, phi, amplitude)[0]
          - geometry(case, s-h, theta, phi, amplitude)[0])/(2*h)
    et = (geometry(case, s, theta+h, phi, amplitude)[0]
          - geometry(case, s, theta-h, phi, amplitude)[0])/(2*h)
    ep = (geometry(case, s, theta, phi+h, amplitude)[0]
          - geometry(case, s, theta, phi-h, amplitude)[0])/(2*h)
    jac = np.einsum("...i,...i->...", es, np.cross(et, ep))
    return es, et, ep, jac


def covariant_field(case, s, theta, phi, amplitude, hgeom):
    xyz, _ = geometry(case, s, theta, phi, amplitude)
    es, et, ep, jac = basis(case, s, theta, phi, amplitude, hgeom)
    B = np.asarray(field(case, xyz)[0])
    cov = np.stack((np.sum(B*es, axis=-1), np.sum(B*et, axis=-1),
                    np.sum(B*ep, axis=-1)), axis=-1)
    return xyz, es, et, ep, jac, B, cov


def periodic_derivative(values, axis, period):
    n = values.shape[axis]
    wave = 2*np.pi*np.fft.fftfreq(n, d=period/n)
    shape = [1]*values.ndim
    shape[axis] = n
    return np.fft.ifft(np.fft.fft(values, axis=axis)
                       * (1j*wave.reshape(shape)), axis=axis).real


def relative_l2(error, reference, weights):
    numerator = np.sum(weights*np.sum(error*error, axis=-1))
    denominator = np.sum(weights*np.sum(reference*reference, axis=-1))
    return float(np.sqrt(numerator/denominator))


def peak_rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/1024**2 if sys.platform == "darwin" else value/1024


def main():
    t0 = time.perf_counter()
    case = cases()["integer_3d"]
    ntheta, nphi = 32, 16
    theta = 2*np.pi*np.arange(ntheta)[:, None]*np.ones((1, nphi))/ntheta
    phi = 2*np.pi*np.ones((ntheta, 1))*np.arange(nphi)[None, :]/(nphi*case.nfp)
    surfaces = (0.1, 0.3, 0.5, 0.7, 0.9)
    hgeom = 2e-5
    radial_steps = (4e-4, 2e-4, 1e-4)
    charts = {"base": 0.0, "remapped": 0.1}
    records = []
    for chart, amplitude in charts.items():
        chart_rows = []
        for s in surfaces:
            xyz, _, et, ep, jac, B_exact, _ = covariant_field(
                case, s, theta, phi, amplitude, hgeom)
            _, u = geometry(case, s, theta, phi, amplitude)
            u_theta = 2*amplitude*s*(1-s)*np.cos(2*theta)
            du_old_dnew = 1-u_theta
            jbtheta = np.sum(B_exact*np.cross(ep, basis(
                case, s, theta, phi, amplitude, hgeom)[0]), axis=-1)
            es = basis(case, s, theta, phi, amplitude, hgeom)[0]
            jbphi = np.sum(B_exact*np.cross(es, et), axis=-1)
            chip, phip = float(np.mean(jbtheta)), float(np.mean(jbphi))
            lambda_theta = -phip*u_theta
            B_rebuilt = (chip/jac)[..., None]*et + ((phip+lambda_theta)/jac)[..., None]*ep
            e_B = relative_l2(B_rebuilt-B_exact, B_exact, np.abs(jac))
            row = {
                "s": s,
                "phip": phip,
                "chip": chip,
                "map_derivative_min": float(np.min(du_old_dnew)),
                "map_derivative_max": float(np.max(du_old_dnew)),
                "map_orientation_preserved": bool(np.all(du_old_dnew > 0)),
                "oriented_jacobian_min": float(np.min(jac)),
                "oriented_jacobian_max": float(np.max(jac)),
                "jacobian_sign_consistent": bool(np.all(jac < 0)),
                "reconstructed_B_relative_l2": e_B,
                "current_curl_radial_step_study": [],
            }
            for hrad in radial_steps:
                xyz, es, et, ep, jac, B_exact, cov = covariant_field(
                    case, s, theta, phi, amplitude, hgeom)
                _, _, _, _, _, _, cov_plus = covariant_field(
                    case, s+hrad, theta, phi, amplitude, hgeom)
                _, _, _, _, _, _, cov_minus = covariant_field(
                    case, s-hrad, theta, phi, amplitude, hgeom)
                ds_cov = (cov_plus-cov_minus)/(2*hrad)
                dt_cov = periodic_derivative(cov, 0, 2*np.pi)
                dp_cov = periodic_derivative(cov, 1, 2*np.pi/case.nfp)
                curl_s = (dt_cov[..., 2]-dp_cov[..., 1])/jac
                curl_t = (dp_cov[..., 0]-ds_cov[..., 2])/jac
                curl_p = (ds_cov[..., 1]-dt_cov[..., 0])/jac
                curl_cart = (curl_s[..., None]*es+curl_t[..., None]*et
                             +curl_p[..., None]*ep)
                _, curl_exact, _, _, _ = differential_fields(
                    case, xyz.reshape((-1, 3)))
                curl_exact = curl_exact.reshape(curl_cart.shape)
                weights = np.abs(jac)
                row["current_curl_radial_step_study"].append({
                    "radial_step": hrad,
                    "curlB_relative_l2": relative_l2(curl_cart-curl_exact,
                                                       curl_exact, weights),
                    "max_abs_curlB_error": float(np.max(np.linalg.norm(
                        curl_cart-curl_exact, axis=-1))),
                })
            chart_rows.append(row)
        records.append({"chart": chart, "amplitude": amplitude, "surfaces": chart_rows})

    remap_rows = records[1]["surfaces"]
    max_b = max(r["reconstructed_B_relative_l2"] for r in remap_rows)
    max_j = max(max(x["curlB_relative_l2"]
                    for x in r["current_curl_radial_step_study"])
                for r in remap_rows)
    min_map = min(r["map_derivative_min"] for r in remap_rows)
    orientation = all(r["map_orientation_preserved"] and r["jacobian_sign_consistent"]
                      for r in remap_rows)
    passed = orientation and min_map > 0 and max_b < 1e-6 and max_j < 1e-3
    result = {
        "schema": 1,
        "evidence": "analytical_consistency",
        "status": "passed" if passed else "failed",
        "case": case.name,
        "scope": "continuum Clebsch remap; B reconstructed and curl B checked from mapped covariant fields over five interior surfaces",
        "n_theta": ntheta,
        "n_phi": nphi,
        "h_geometry": hgeom,
        "radial_steps": list(radial_steps),
        "lambda_definition": "lambda=-phip*u; poloidal remap mode (m,n)=(2,0)",
        "max_remapped_B_relative_l2": max_b,
        "max_remapped_curlB_relative_l2": max_j,
        "minimum_map_derivative": min_map,
        "orientation_preserved_on_sample": orientation,
        "elapsed_s": time.perf_counter()-t0,
        "max_rss_mib": peak_rss_mib(),
        "surfaces": records,
        "no_nonlinear_solve": True,
    }
    out = ROOT/"results/projection/integer_3d_gauge_volume_continuum.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(out.read_text())
    if not passed:
        raise SystemExit("Continuum remap failed its sampled B/J/orientation gates")


if __name__ == "__main__":
    main()
