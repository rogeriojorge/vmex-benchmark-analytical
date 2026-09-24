"""Recover a field-consistent lambda on integer-family geometric surfaces.

This is an independent surface-level projection, before a VMEX state fit.
The lambda gradient is obtained from both contravariant components of the
exact Cartesian field and checked for angular integrability.
"""
import json

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from analytic import ROOT, cases, field, surface

NTHETA, NPHI = 64, 64
H = 1e-5
MAX_M, MAX_N = 12, 12
case = cases()["integer_3d"]
theta, phi = np.meshgrid(2*np.pi*np.arange(NTHETA)/NTHETA,
                         2*np.pi*np.arange(NPHI)/(NPHI*case.nfp), indexing="ij")
rows = []
for s in (0.2, 0.5, 0.8):
    label = case.edge*s
    xyz = surface(case, label, theta, phi)
    e_s = (surface(case, case.edge*(s+H), theta, phi)-
           surface(case, case.edge*(s-H), theta, phi))/(2*H)
    e_t = (surface(case, label, theta+H, phi)-
           surface(case, label, theta-H, phi))/(2*H)
    e_p = (surface(case, label, theta, phi+H)-
           surface(case, label, theta, phi-H))/(2*H)
    B = np.asarray(field(case, xyz)[0])
    jac = np.einsum("...i,...i->...", e_s, np.cross(e_t, e_p))
    JBu = np.einsum("...i,...i->...", B, np.cross(e_p, e_s))
    JBv = np.einsum("...i,...i->...", B, np.cross(e_s, e_t))
    phip, chip = float(np.mean(JBv)), float(np.mean(JBu))
    want_t, want_p = JBv-phip, chip-JBu
    ft, fp = np.fft.fftfreq(NTHETA, 1/NTHETA)[:, None], np.fft.fftfreq(NPHI, 1/NPHI)[None, :]*case.nfp
    t_hat, p_hat = np.fft.fft2(want_t), np.fft.fft2(want_p)
    k2 = ft*ft+fp*fp
    lam_hat = (-1j*ft*t_hat-1j*fp*p_hat)/np.where(k2 == 0, 1, k2)
    lam_hat[0, 0] = 0
    mask = (np.abs(ft) <= MAX_M) & (np.abs(fp/case.nfp) <= MAX_N)
    lam_hat = np.where(mask, lam_hat, 0)
    got_t = np.fft.ifft2(1j*ft*lam_hat).real
    got_p = np.fft.ifft2(1j*fp*lam_hat).real
    gradient_abs = np.sqrt(np.mean((got_t-want_t)**2+(got_p-want_p)**2))
    gradient_scale = np.hypot(phip, chip)
    gradient_error = gradient_abs/gradient_scale
    gradient_magnitude = np.sqrt(np.mean(want_t**2+want_p**2))/gradient_scale
    # Reconstruct B^theta and B^phi from the fitted lambda derivatives.
    bu = (chip-got_p)/jac
    bv = (phip+got_t)/jac
    B_fit = bu[..., None]*e_t+bv[..., None]*e_p
    field_error = np.sqrt(np.mean(np.sum((B_fit-B)**2, axis=-1)))/np.sqrt(
        np.mean(np.sum(B*B, axis=-1)))
    radial_error = np.sqrt(np.mean((np.einsum("...i,...i->...",B,np.cross(e_t,e_p))/jac)**2))
    rows.append(dict(s=s, phip=phip, chip=chip, chip_over_phip=chip/phip,
                     min_jacobian=float(jac.min()), max_jacobian=float(jac.max()),
                     lambda_gradient_absolute_rms=float(gradient_abs),
                     lambda_gradient_over_flux_scale=float(gradient_error),
                     measured_lambda_gradient_over_flux_scale=float(gradient_magnitude),
                     reconstructed_B_relative_l2=float(field_error),
                     B_radial_rms=float(radial_error)))
    print(rows[-1])
out = ROOT / "results/projection/integer_3d_surface.json"
passed = all(r["lambda_gradient_over_flux_scale"] < 1e-5 and
             r["reconstructed_B_relative_l2"] < 1e-5 and
             abs(r["chip_over_phip"]+2) < 1e-6 for r in rows)
out.write_text(json.dumps(dict(schema=1, evidence="analytic_projection", case=case.name,
    status="passed" if passed else "failed", no_nonlinear_solve=True,
    scope="angular lambda integrability and physical B on three exact surfaces; no radial state fit",
    ntheta=NTHETA, nphi=NPHI, finite_difference_step=H, max_m=MAX_M, max_n=MAX_N,
    rows=rows), indent=2, allow_nan=False)+"\n")
if not passed:
    raise SystemExit("Integer surface projection failed its field-consistency gates")
