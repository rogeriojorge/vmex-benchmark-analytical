"""Generate fixed-boundary INDATA decks and independent fit diagnostics.

These are candidate inputs, not certified VMEX equilibria. Run from anywhere.
Change the settings below rather than maintaining a second configuration API.
Both prescribed-iota and prescribed-current inputs are generated. Field/sign
recovery is a required local gate before accepting either closure.
"""
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from numpy.polynomial import Polynomial

from analytic import ROOT, cases, field, flux, iota, label_at_s, surface, validate

LENGTH_M = 1.0
FIELD_T = 1.0
MU0 = 4e-7*np.pi  # Fixed convention, recorded; not an unspecified CODATA constant.
MAX_M = 12
MAX_N = 12
FIT_GRID = 128
PROFILE_DEGREES = (8, 12, 16, 20)
PROFILE_TOL = 1e-8
OUT = ROOT / "inputs"


def is_axisymmetric(case):
    return (case.family == "solovev" or
            (case.family == "integer" and case.parameters[0] == case.parameters[1]) or
            (case.family == "sheared" and case.parameters[0] == 0))


def enclosed_current(case, labels, n=1024):
    """Toroidal current in mu0=1 units from an independent Ampere integral.

    A counterclockwise (R,Z) loop has normal -e_phi, hence the minus sign.
    """
    labels = np.atleast_1d(labels)
    theta = 2*np.pi*np.arange(n)/n
    xyz = surface(case, labels[:, None], theta[None, :], 0.)
    k = np.fft.fftfreq(n, 1/n)
    tangent = np.fft.ifft(1j*k[None, :, None]*np.fft.fft(xyz, axis=1), axis=1).real
    B = np.asarray(field(case, xyz)[0])
    return -2*np.pi*np.mean(np.sum(B*tangent, -1), -1)


def boundary_coefficients(case, max_m=MAX_M, max_n=MAX_N, grid=FIT_GRID):
    ntor = 0 if is_axisymmetric(case) else max_n
    theta = np.arange(grid)*2*np.pi/grid
    phi = np.arange(grid)*2*np.pi/(grid*case.nfp)
    xyz = surface(case, case.edge, theta[:, None], phi[None, :])
    R, Z = np.hypot(xyz[..., 0], xyz[..., 1]), xyz[..., 2]
    fr, fz = np.fft.fft2(R)/(grid*grid), np.fft.fft2(Z)/(grid*grid)
    rows = []
    for m in range(max_m+1):
        for n in range(0 if m == 0 else -ntor, ntor+1):
            factor = 1 if m == n == 0 else 2
            r, z = fr[m, -n], fz[m, -n]
            rows.append([m, n, factor*r.real, -factor*r.imag,
                         factor*z.real, -factor*z.imag])
    rows = np.asarray(rows)
    # Test the retained basis on unrelated, nonuniform points.
    rng = np.random.default_rng(551)
    th, ph = rng.uniform(0, 2*np.pi, (2, 1024))
    target = surface(case, case.edge, th, ph)
    phase = th[:, None]*rows[:, 0]-case.nfp*ph[:, None]*rows[:, 1]
    rfit = np.cos(phase)@rows[:, 2]+np.sin(phase)@rows[:, 3]
    zfit = np.cos(phase)@rows[:, 4]+np.sin(phase)@rows[:, 5]
    error = np.hypot(rfit-np.hypot(target[:, 0], target[:, 1]), zfit-target[:, 2])
    omitted = max(np.max(abs(rows[:, 3])), np.max(abs(rows[:, 4]))) if not case.lasym else 0.
    return rows, ntor, float(np.max(error)), float(omitted)


def fit_profiles(case):
    # Fit in normalized toroidal flux, not the pressure-surface label.
    test_s = (1-np.cos(np.linspace(0, np.pi, 193)))/2
    test_label = label_at_s(case, test_s)
    test_I = enclosed_current(case, test_label, n=2048)
    Iedge = float(test_I[-1])
    for degree in PROFILE_DEGREES:
        s = (1-np.cos(np.linspace(0, np.pi, 4*degree+1)))/2
        label = label_at_s(case, s)
        pressure = -case.pressure_slope*(case.edge-label)
        polp = Polynomial.fit(s, pressure, degree).convert()
        polp.coef[0] = -case.pressure_slope*case.edge
        polp.coef[1:] -= polp(1.)/degree
        poli = Polynomial.fit(s, iota(case, label), degree).convert()
        polI = Polynomial.fit(s, enclosed_current(case, label), degree).convert()
        polI.coef[0] = 0.
        polI.coef[1:] *= Iedge/polI(1.)
        errors = dict(pressure=float(np.max(abs(polp(test_s)+case.pressure_slope*(case.edge-test_label)))/abs(polp(0.))),
                      iota=float(np.max(abs(poli(test_s)-iota(case, test_label)))/max(1., np.max(abs(iota(case, test_label))))),
                      enclosed_current=float(np.max(abs(polI(test_s)-test_I))/abs(Iedge)))
        if max(errors.values()) < PROFILE_TOL:
            return polp, poli, polI, errors
    raise RuntimeError(f"{case.name}: profile fits unresolved: {errors}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    records = []
    for case in cases().values():
        validate(case)
        rows, ntor, geometry_error, symmetry_error = boundary_coefficients(case)
        polp, poli, polI, profile_errors = fit_profiles(case)
        for closure in (0, 1):
            suffix = "iota" if closure == 0 else "current"
            path = OUT / f"input.{case.name}_{suffix}"
            axis_symmetric = is_axisymmetric(case)
            lines = ["&INDATA", "  LFREEB = F", f"  LASYM = {'T' if case.lasym else 'F'}",
                     f"  NFP = {case.nfp}", f"  MPOL = {MAX_M+1}", f"  NTOR = {ntor}",
                     "  NS_ARRAY = 17 33 65", "  FTOL_ARRAY = 1e-10 1e-12 1e-14",
                     "  NITER_ARRAY = 10000 20000 30000", "  NSTEP = 200", "  DELT = 0.5",
                     "  GAMMA = 0", "  PRES_SCALE = 1", "  PMASS_TYPE = 'power_series'",
                     f"  PHIEDGE = {FIELD_T*LENGTH_M**2*float(flux(case, case.edge)):.17e}",
                     "  AM = "+" ".join(f"{x*FIELD_T**2/MU0:.17e}" for x in polp.coef),
                     f"  NCURR = {closure}"]
            if closure == 0:
                lines += ["  PIOTA_TYPE = 'power_series'", "  AI = "+" ".join(f"{x:.17e}" for x in poli.coef)]
            else:
                lines += ["  PCURR_TYPE = 'power_series'",
                          f"  CURTOR = {polI(1.)*FIELD_T*LENGTH_M/MU0:.17e}",
                          "  AC = "+" ".join(f"{x:.17e}" for x in polI.deriv().coef)]
            for m, n, rc, rs, zc, zs in rows:
                for key, value in (("RBC", rc), ("ZBS", zs), ("RBS", rs), ("ZBC", zc)):
                    if (case.lasym or key in ("RBC", "ZBS")) and abs(value) > 2e-15:
                        lines.append(f"  {key}({int(n)},{int(m)}) = {LENGTH_M*value:.17e}")
            lines.append("/")
            path.write_text("\n".join(lines)+"\n")
            records.append(dict(case=case.name, file=path.name, ncurr=closure,
                                axisymmetric=axis_symmetric, lasym=case.lasym,
                                boundary_max_error_m=geometry_error*LENGTH_M,
                                symmetric_omitted_coeff_m=symmetry_error*LENGTH_M,
                                profile_errors=profile_errors, vmex_run=False,
                                status="candidate_input_not_equilibrium",
                                resolution=dict(mpol=MAX_M+1, ntor=ntor, ns=[17,33,65])))
        print(f"{case.name:34s} boundary={geometry_error:.2e} profile={max(profile_errors.values()):.2e}")
    (OUT / "manifest.json").write_text(json.dumps(dict(schema=1, mu0=MU0,
        length_m=LENGTH_M, field_t=FIELD_T, reference_sign="CCW R-Z theta: iota negative; validate in VMEX",
        records=records), indent=2, allow_nan=False)+"\n")
