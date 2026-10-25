"""Score a VMEC-format WOUT against the exact Lopez configuration D vacuum field.

Works from WOUT harmonics only (no VmecInteriorField, so LASYM states are supported)
and does not need JAX. The same code scores VMEX, VMEC2000, a projected exact state,
or any other producer. The comparison is done in the exact (paper) frame: WOUT points
X_c = Rot (X - origin) are mapped back, B is rotated with the same proper rotation.

Quantities, all from the WOUT and the exact reference:

* B error: area-weighted surface L2 at native full-mesh surfaces (s = s_report and the
  first surface off the axis) and volume L2 over all half-mesh shells;
* flux label: exact s = s(h(lambda, beta)) at numerical surface points versus the mesh s,
  with lambda, beta the label-selected cubic-root first integrals (``lopez.labels``);
* transform: iotaf versus the exact constant transform;
* vacuum identities of the numerical field (no reference field needed): curl B
  (full vector, from the covariant components and the geometry), div B, toroidal flux
  through sections, circulation around toroidal loops, magnetic energy against
  2 pi Phi_edge, volume against the exact volume.

Conventions: surfaces are at s = j/(ns-1); half-mesh quantities (bsup*, gmn) carry index
j = 1..ns-1 at s = (j - 1/2)/(ns-1) like VMEC; geometry on the half mesh is the average
of the two bounding full-mesh rows, e_s the row difference. Angular derivatives are
spectral on a uniform (theta, zeta) grid built from the harmonic content; radial
derivatives of the covariant components use np.gradient with second-order edges.
curl B uses the signed Jacobian e_s . (e_theta x e_zeta); lengths are in the units of the
deck (the reference unit is used as SI), so the exact |B| scale is 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Callable, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lopez  # noqa: E402
import lopez_surfaces as ls  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
TWO_PI = 2*np.pi


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- case hook
class Case(NamedTuple):
    """Exact reference seen by the scorer.

    ``exact(xyz)`` takes points in the frame in which the reference is defined and returns
    (B, s, q, margin): field, exact normalized toroidal-flux label, certified-domain level
    (<= 1 inside the target domain) and the rival-root margin (>= 2 inside the domain).
    ``frame`` = (Rot, origin) with X_c = Rot (X - origin) the WOUT frame.
    ``axis(phi)`` gives the exact (R, Z) of the axis in the WOUT frame. Vacuum identities:
    toroidal circulation C and flux Phi_edge give int B^2 dV = C Phi_edge.
    """
    name: str
    frame: tuple
    exact: Callable
    iota: float
    flux_edge: float
    volume: float
    axis: Callable
    circulation: float  # int B . dl around a toroidal loop (2 pi for the unit-scale reference)


def lopez_case(frame=None, select="label", iota=None):
    """Configuration D. The frame defaults to the one recorded in the manifest (OPT_FRAME);
    ``iota`` overrides the integer transform of the poloidal-angle gauge of the state being scored
    (theta' = theta - k phi gives iota' = iota - k; the physical field is unchanged)."""
    record = next(r for r in json.loads((ROOT/"inputs/manifest.json").read_text())["records"]
                  if r["case"] == "lopez_D")["lopez"]
    if frame is None:
        frame = (np.array(record["frame_rotation"]), np.array(record["frame_origin"]))
    sigma, c = int(record["sigma"]), int(record["iota"])
    iota_scored = c if iota is None else float(iota)

    def exact(xyz):
        r = lopez.evaluate(xyz, select)
        b = (lopez._normal(r.zeta)[0]/(r.zeta*r.d)[..., None]).imag
        h = ((r.lam - ls.A0)/ls.A_RAD)**2 + ((r.beta - ls.B0)/ls.B_RAD)**2
        # the closed-form flux map needs h < (A0/A_RAD)^2; far-outside points (wrong root) are clipped there
        h = np.minimum(h, 0.99*(ls.A0/ls.A_RAD)**2)
        return b, ls.s_of_h(h), lopez.D_SCALE*r.q, lopez.D_SCALE*r.q_rival

    def axis(phi):
        return ls.boundary_rz(0., phi, frame, sigma, c, s=0.)

    return Case("lopez_D", (np.asarray(frame[0]), np.asarray(frame[1])), exact, float(iota_scored),
                float(record["phiedge"]), float(record["volume"]), axis, TWO_PI)


# ---------------------------------------------------------------- WOUT access and harmonics
NAMES = ("nfp", "ns", "mpol", "ntor", "lasym", "xm", "xn", "xm_nyq", "xn_nyq", "rmnc", "zmns", "rmns",
         "zmnc", "bsupumnc", "bsupvmnc", "bsupumns", "bsupvmns", "iotaf", "phi", "fsqr", "fsqz",
         "fsql", "ier_flag", "signgs")


def read_wout(path):
    """Read the fields used here from a VMEC WOUT (netCDF) into a namespace."""
    from netCDF4 import Dataset
    out = {}
    with Dataset(path) as nc:
        for name in NAMES:
            key = "lasym__logical__" if name == "lasym" else name
            if key in nc.variables:
                value = np.asarray(nc.variables[key][...])
                out[name] = bool(value) if name == "lasym" else value
    return as_wout(SimpleNamespace(**out))


def as_wout(w):
    """Namespace of numpy arrays from any WOUT-like object (netCDF reader, vmex.WoutData, ...)."""
    out = {}
    for name in NAMES:
        value = getattr(w, name, None)
        if value is None:
            continue
        out[name] = bool(value) if name == "lasym" else np.asarray(value)
    for name in ("nfp", "ns", "mpol", "ntor"):
        out[name] = int(out[name])
    out["lasym"] = bool(out.get("lasym", False))
    return SimpleNamespace(**out)


def _index(xm, xn, nfp, nth, nz):
    m = np.asarray(xm).astype(int)
    k = np.rint(np.asarray(xn)/nfp).astype(int)
    if np.abs(np.asarray(xn) - nfp*k).max() > 1e-9:
        raise ValueError("xn is not a multiple of nfp")
    if np.abs(m).max() >= nth//2 or np.abs(k).max() >= nz//2:
        raise ValueError(f"grid {nth}x{nz} too small for modes up to m={np.abs(m).max()}, n={np.abs(k).max()}")
    return m % nth, (-k) % nz, m, k


def synth(cos, sin, xm, xn, nfp, nth, nz, dtheta=False, dzeta=False):
    """f(theta_a, zeta_b) = sum cos_k cos(m th - n zeta) + sin_k sin(m th - n zeta) on the uniform
    grid theta_a = 2 pi a/nth, zeta_b = 2 pi b/(nfp nz); optionally its angle derivative.
    ``cos`` / ``sin`` may be None or have leading batch axes (..., nmodes)."""
    ip, iq, m, k = _index(xm, xn, nfp, nth, nz)
    lead = np.shape(cos if cos is not None else sin)[:-1]
    amp = np.zeros(lead + (len(m),), complex)
    if cos is not None:
        amp = amp + np.asarray(cos)
    if sin is not None:
        amp = amp - 1j*np.asarray(sin)
    if dtheta:
        amp = amp*(1j*m)
    if dzeta:
        amp = amp*(-1j*k*nfp)
    spec = np.zeros(lead + (nth, nz), complex)
    flat = spec.reshape((-1, nth, nz))
    for row, a in enumerate(amp.reshape((-1, len(m)))):
        np.add.at(flat[row], (ip, iq), a)
    return np.fft.ifft2(spec, axes=(-2, -1)).real*(nth*nz)


def analyze(f, xm, xn, nfp):
    """Inverse of ``synth``: (cos, sin) harmonics of a real grid function (..., nth, nz)."""
    nth, nz = f.shape[-2:]
    ip, iq, m, k = _index(xm, xn, nfp, nth, nz)
    spec = np.fft.fft2(f, axes=(-2, -1))/(nth*nz)
    a = spec[..., ip, iq]
    cos, sin = 2*a.real, -2*a.imag
    zero = (m == 0) & (k == 0)
    return np.where(zero, a.real, cos), np.where(zero, 0., sin)


def mode_table(mpol, ntor):
    """VMEC mode order: m = 0..mpol-1, n = 0..ntor for m = 0 else -ntor..ntor."""
    rows = [(m, n) for m in range(mpol) for n in range(0 if m == 0 else -ntor, ntor + 1)]
    return np.array([r[0] for r in rows], float), np.array([r[1] for r in rows], float)


def nyquist_table(mmax, nmax):
    return mode_table(mmax + 1, nmax)


def grid_for(w):
    """Smallest even grid resolving every geometry and Nyquist harmonic (with margin for products)."""
    mmax = max(np.abs(w.xm).max(), np.abs(w.xm_nyq).max())
    nmax = max(np.abs(np.rint(w.xn/w.nfp)).max(), np.abs(np.rint(w.xn_nyq/w.nfp)).max())
    even = lambda n: int(2*np.ceil(n/2))  # noqa: E731
    return even(max(32, 2*mmax + 4)), even(max(64, 2*nmax + 4))


class Surfaces(NamedTuple):
    """Fields on a set of full-mesh or half-mesh surfaces over the common grid: (nsurf, nth, nz)."""
    R: np.ndarray
    Z: np.ndarray
    Ru: np.ndarray
    Zu: np.ndarray
    Rv: np.ndarray
    Zv: np.ndarray
    Rs: np.ndarray | None
    Zs: np.ndarray | None
    bu: np.ndarray
    bv: np.ndarray
    s: np.ndarray
    phi: np.ndarray  # geometric toroidal angle zeta_b, (nz,)


def half_rows(a, xm, ds):
    """(coefficient, d coefficient/ds) of harmonic rows on the half mesh from full-mesh rows ``a``.

    Odd-m coefficients vanish like rho = sqrt(s) at the axis, so averaging and differencing are done
    on the smooth x = c / rho^(m mod 2) (VMEC's even/odd split); the axis row of odd modes is the
    linear extrapolation of x."""
    ns = a.shape[0]
    s_full = np.arange(ns)*ds
    s_half = (np.arange(1, ns) - 0.5)*ds
    p = (np.asarray(xm).astype(int) % 2)
    x = np.empty_like(a)
    x[1:] = a[1:]/np.sqrt(s_full[1:])[:, None]**p
    x[0] = np.where(p == 1, 2*x[1] - x[2], a[0])
    rho = np.sqrt(s_half)[:, None]**p
    coef = rho*0.5*(x[1:] + x[:-1])
    dcoef = rho*(x[1:] - x[:-1])/ds + p*coef/(2*s_half)[:, None]
    return coef, dcoef


def full_rows_of_half(a, xm, ds):
    """Half-mesh rows (index 0 unused, 1..ns-1) interpolated to the full mesh with the same even/odd
    scaling; the axis and edge rows are linear extrapolations of the smooth representation."""
    ns = a.shape[0]
    s_full = np.arange(ns)*ds
    s_half = (np.arange(1, ns) - 0.5)*ds
    p = (np.asarray(xm).astype(int) % 2)
    x = a[1:]/np.sqrt(s_half)[:, None]**p
    out = np.empty_like(a)
    out[1:-1] = 0.5*(x[:-1] + x[1:])
    out[-1] = 1.5*x[-1] - 0.5*x[-2]
    out[0] = 1.5*x[0] - 0.5*x[1]
    return out*np.sqrt(s_full)[:, None]**p


def evaluate_surfaces(w, mesh, grid=None):
    """Geometry and contravariant field on ``mesh`` = 'full' (rows 0..ns-1; bsup interpolated, no e_s)
    or 'half' (rows 1..ns-1)."""
    nfp, ns = w.nfp, w.ns
    nth, nz = grid or grid_for(w)
    ds = 1/(ns - 1)
    rc, rs = w.rmnc, (getattr(w, "rmns", None) if w.lasym else None)
    zc, zs = (getattr(w, "zmnc", None) if w.lasym else None), w.zmns

    def one(c, s, d=(False, False)):
        return synth(c, s, w.xm, w.xn, nfp, nth, nz, dtheta=d[0], dzeta=d[1])

    def geometry(c_r, s_r, c_z, s_z):
        return [one(c, s, d) for c, s in ((c_r, s_r), (c_z, s_z)) for d in ((0, 0), (1, 0), (0, 1))]
    bsup = {name: [getattr(w, f"bsup{name}mnc", None), getattr(w, f"bsup{name}mns", None) if w.lasym else None]
            for name in "uv"}
    if mesh == "full":
        R, Ru, Rv, Z, Zu, Zv = geometry(rc, rs, zc, zs)
        Rs = Zs = None
        s = np.arange(ns)*ds
        cu, su = (None if a is None else full_rows_of_half(a, w.xm_nyq, ds) for a in bsup["u"])
        cv, sv = (None if a is None else full_rows_of_half(a, w.xm_nyq, ds) for a in bsup["v"])
    else:
        hc, dhc = {}, {}
        for name, a in (("rc", rc), ("rs", rs), ("zc", zc), ("zs", zs)):
            hc[name], dhc[name] = (None, None) if a is None else half_rows(a, w.xm, ds)
        R, Ru, Rv, Z, Zu, Zv = geometry(hc["rc"], hc["rs"], hc["zc"], hc["zs"])
        Rs, Zs = one(dhc["rc"], dhc["rs"]), one(dhc["zc"], dhc["zs"])
        s = (np.arange(1, ns) - 0.5)*ds
        cu, su = (None if a is None else a[1:] for a in bsup["u"])
        cv, sv = (None if a is None else a[1:] for a in bsup["v"])
    bu = synth(cu, su, w.xm_nyq, w.xn_nyq, nfp, nth, nz)
    bv = synth(cv, sv, w.xm_nyq, w.xn_nyq, nfp, nth, nz)
    phi = TWO_PI*np.arange(nz)/(nfp*nz)
    return Surfaces(R, Z, Ru, Zu, Rv, Zv, Rs, Zs, bu, bv, s, phi)


def cartesian(f, vec=None):
    """(xyz, B) in the WOUT frame from a Surfaces object; B = bu e_theta + bv e_zeta."""
    cp, sp = np.cos(f.phi), np.sin(f.phi)
    xyz = np.stack((f.R*cp, f.R*sp, f.Z), -1)
    br, bp, bz = f.bu*f.Ru + f.bv*f.Rv, f.bv*f.R, f.bu*f.Zu + f.bv*f.Zv
    return xyz, np.stack((br*cp - bp*sp, br*sp + bp*cp, bz), -1)


def to_exact_frame(case, xyz, b=None):
    """WOUT frame -> frame of the exact reference: X = Rot^T X_c + origin, B = Rot^T B_c."""
    rot, origin = case.frame
    x = xyz @ rot + origin
    return (x, None) if b is None else (x, b @ rot)


# ---------------------------------------------------------------- vacuum identities of the numerical field
def sqrtg(f):
    """Signed e_s . (e_theta x e_zeta) in the local cylindrical basis: e_s = (Rs, 0, Zs),
    e_theta = (Ru, 0, Zu), e_zeta = (Rv, R, Zv), so the triple product is R (Zs Ru - Rs Zu)."""
    return f.R*(f.Zs*f.Ru - f.Rs*f.Zu)


def radial_derivative(a, s, ds, power):
    """d/ds of a covariant component on uniformly spaced shells at ``s`` (half mesh).

    Near the axis the m-th poloidal harmonic of a regular field component behaves as
    rho^power(m) x(s) with x smooth in s (rho = sqrt(s)); ``power(m)`` is that exponent
    (B_zeta: m mod 2, B_theta: (m + 1) mod 2, B_s: -(m mod 2)). Differencing is done on x
    harmonic by harmonic (np.gradient, second-order edges) and mapped back, so the first shells
    are not polluted by the sqrt singularity that plain differencing of c(s) would see."""
    nth = a.shape[-2]
    m = np.abs(np.fft.fftfreq(nth, 1/nth)).astype(int)
    q = np.asarray(power(m), float)[:, None]
    spec = np.fft.fft(a, axis=-2)
    rho = np.sqrt(s)[:, None, None]
    x = spec/rho**q
    dx = np.gradient(x, ds, axis=0, edge_order=2)
    out = rho**q*dx + q*spec/(2*s)[:, None, None]
    return np.fft.ifft(out, axis=-2).real


def curl_div(f, nfp, ds):
    """curl B (cylindrical-basis components) and div B on half-mesh surfaces."""
    g = sqrtg(f)
    g_uu, g_uv, g_vv = f.Ru**2 + f.Zu**2, f.Ru*f.Rv + f.Zu*f.Zv, f.Rv**2 + f.R**2 + f.Zv**2
    g_su, g_sv = f.Rs*f.Ru + f.Zs*f.Zu, f.Rs*f.Rv + f.Zs*f.Zv
    bs, bth, bze = f.bu*g_su + f.bv*g_sv, f.bu*g_uu + f.bv*g_uv, f.bu*g_uv + f.bv*g_vv
    nth, nz = g.shape[-2:]
    k_th = np.fft.fftfreq(nth, 1/nth)[:, None]
    k_ze = nfp*np.fft.fftfreq(nz, 1/nz)[None, :]
    nyq_th, nyq_ze = (nth//2), (nz//2)
    def d(a, axis):
        spec = np.fft.fft2(a, axes=(-2, -1))
        if axis == "th":
            fac = 1j*k_th
            fac = np.where(np.abs(k_th) == nyq_th, 0, fac)
        else:
            fac = 1j*k_ze
            fac = np.where(np.abs(k_ze) == nfp*nyq_ze, 0, fac)
        return np.fft.ifft2(spec*fac, axes=(-2, -1)).real
    ds_ = lambda a, q: radial_derivative(a, f.s, ds, q)  # noqa: E731
    cs = (d(bze, "th") - d(bth, "ze"))/g
    cu = (d(bs, "ze") - ds_(bze, lambda m: m % 2))/g
    cv = (ds_(bth, lambda m: (m + 1) % 2) - d(bs, "th"))/g
    curl_r = cs*f.Rs + cu*f.Ru + cv*f.Rv
    curl_p = cv*f.R
    curl_z = cs*f.Zs + cu*f.Zu + cv*f.Zv
    div = (d(g*f.bu, "th") + d(g*f.bv, "ze"))/g
    return np.sqrt(curl_r**2 + curl_p**2 + curl_z**2), div


def minor_radius_of(w, grid=None):
    """RMS distance of the edge from the axis point at the same toroidal angle."""
    nth, nz = grid or grid_for(w)
    def rz(row):
        c, s = w.rmnc[row], (w.rmns[row] if w.lasym else None)
        zc, zs = (w.zmnc[row] if w.lasym else None), w.zmns[row]
        return (synth(c, s, w.xm, w.xn, w.nfp, nth, nz), synth(zc, zs, w.xm, w.xn, w.nfp, nth, nz))
    Re, Ze = rz(-1)
    Ra, Za = rz(0)
    return float(np.sqrt(np.mean((Re - Ra[:1])**2 + (Ze - Za[:1])**2))), (Ra[0], Za[0])


# ---------------------------------------------------------------- scoring
def _rel_l2(err, ref, weight):
    return float(np.sqrt(np.sum(weight*np.sum(err**2, -1))/np.sum(weight*np.sum(ref**2, -1))))


def score_wout(w, case, s_report=0.5, grid=None, stride=(1, 1), keep_profile=True):
    """Score a WOUT-like object ``w`` against ``case``; returns a JSON-serializable dict."""
    w = as_wout(w)
    ns, nfp = w.ns, w.nfp
    ds = 1/(ns - 1)
    grid = grid or grid_for(w)
    sl = (slice(None), slice(None, None, stride[0]), slice(None, None, stride[1]))
    a_minor, axis_rz = minor_radius_of(w, grid)
    phi = TWO_PI*np.arange(grid[1])/(nfp*grid[1])

    # --- axis: numerical axis row versus exact axis at the same toroidal angle
    ax_r, ax_z = case.axis(phi)
    axis_offset = float(np.max(np.hypot(axis_rz[0] - ax_r, axis_rz[1] - ax_z)))

    # --- full-mesh surfaces: B, flux label
    full = evaluate_surfaces(w, "full", grid)
    surf_rows, field_rows = [], {}
    j_report = int(np.argmin(np.abs(np.arange(ns)*ds - s_report)))
    pick = sorted({1, 2, max(3, ns//16), ns//4, ns//2, 3*ns//4, ns - 2, ns - 1, j_report})
    flux_label_max, invalid_points, q_max, margin_min = 0., 0, 0., np.inf
    for j in range(1, ns):
        sub = Surfaces(*(None if a is None else (a[j:j+1][sl] if a.ndim == 3 else a) for a in full[:10]),
                       full.s[j:j+1], full.phi[::stride[1]])
        xyz, b = cartesian(sub)
        xe, be = to_exact_frame(case, xyz, b)
        bx, s_ex, q, margin = case.exact(xe)
        s_err = s_ex - full.s[j]
        invalid_points += int(np.sum(q > 1.0 + 1e-9))
        q_max, margin_min = max(q_max, float(q.max())), min(margin_min, float(margin.min()))
        flux_label_max = max(flux_label_max, float(np.abs(s_err).max()))
        row = dict(j=j, s=float(full.s[j]), flux_label_error_max=float(np.abs(s_err).max()),
                   flux_label_error_rms=float(np.sqrt(np.mean(s_err**2))))
        if j in pick:
            area = np.linalg.norm(np.cross(np.stack((sub.Ru*np.cos(sub.phi), sub.Ru*np.sin(sub.phi), sub.Zu), -1),
                                           np.stack((sub.Rv*np.cos(sub.phi) - sub.R*np.sin(sub.phi),
                                                     sub.Rv*np.sin(sub.phi) + sub.R*np.cos(sub.phi), sub.Zv), -1)), axis=-1)
            row["B_relative_surface_l2"] = _rel_l2(be - bx, bx, area)
            field_rows[j] = row["B_relative_surface_l2"]
        surf_rows.append(row)

    # --- half-mesh shells: volume metrics
    half = evaluate_surfaces(w, "half", grid)
    g = sqrtg(half)
    curl, div = curl_div(half, nfp, ds)
    xyz, b = cartesian(half)
    xe, be = to_exact_frame(case, xyz[sl], b[sl])
    bx, _, q_h, margin_h = case.exact(xe)
    weight = np.abs(g)[sl]
    cell = ds*TWO_PI*TWO_PI/(grid[0]*grid[1])  # nfp periods of width 2 pi/nfp
    b2_full = np.sum(b**2, -1)
    volume = float(np.sum(np.abs(g))*cell)
    energy = float(np.sum(np.abs(g)*b2_full)*cell)
    ref2 = np.sum(weight*np.sum(bx**2, -1))
    e_B = float(np.sqrt(np.sum(weight*np.sum((be - bx)**2, -1))/ref2))
    shell = lambda a: np.sum(np.abs(g)*a, axis=(-2, -1))  # noqa: E731  per-shell volume sums
    e_curl = float(a_minor*np.sqrt(np.sum(np.abs(g)*curl**2)/np.sum(np.abs(g)*b2_full)))
    outer = half.s >= 0.25
    e_curl_outer = float(a_minor*np.sqrt(shell(curl**2)[outer].sum()/shell(b2_full)[outer].sum()))
    e_curl_shell = a_minor*np.sqrt(shell(curl**2)/shell(b2_full))
    e_div = float(a_minor*np.sqrt(np.sum(np.abs(g)*div**2)/np.sum(np.abs(g)*b2_full)))
    curl_max_rel = float(a_minor*np.max(curl/np.sqrt(b2_full)))
    flux_sections = np.sum(np.sum(g*half.bv, axis=-2)*(TWO_PI/grid[0]), axis=0)*ds  # (nz,)
    flux = float(np.abs(flux_sections).mean())
    # circulation of B around the toroidal loop at fixed (shell, theta): int B_zeta dzeta
    g_uv = half.Ru*half.Rv + half.Zu*half.Zv
    g_vv = half.Rv**2 + half.R**2 + half.Zv**2
    b_zeta = half.bu*g_uv + half.bv*g_vv
    circulation = b_zeta.sum(-1)*(TWO_PI/grid[1])   # (nshell, ntheta), full 2 pi (nfp periods)
    c_exact = case.circulation
    profile = []
    if keep_profile:
        for jj in range(half.s.size):
            gj = np.abs(g[jj])
            profile.append(dict(s=float(half.s[jj]),
                                B_rel=float(np.sqrt(np.sum(np.abs(g[jj])[sl[1:]]*np.sum((be[jj] - bx[jj])**2, -1))
                                                   / np.sum(np.abs(g[jj])[sl[1:]]*np.sum(bx[jj]**2, -1)))),
                                curl_rel=float(a_minor*np.sqrt(np.sum(gj*curl[jj]**2)/np.sum(gj*b2_full[jj]))),
                                div_rel=float(a_minor*np.sqrt(np.sum(gj*div[jj]**2)/np.sum(gj*b2_full[jj])))))
    jh = int(np.argmin(np.abs(half.s - s_report)))
    # Ampere: int B_theta dtheta at fixed (shell, zeta) = mu0 x (toroidal current enclosed); zero in vacuum
    poloidal = (half.bu*(half.Ru**2 + half.Zu**2) + half.bv*(half.Ru*half.Rv + half.Zu*half.Zv)).sum(-2)*(
        TWO_PI/grid[0])
    b_rms = float(np.sqrt(np.sum(np.abs(g)*b2_full)/np.sum(np.abs(g))))
    iota_w = np.asarray(w.iotaf, float)
    iota_err = float(np.max(np.abs(iota_w[1:] - case.iota)))
    phiedge_w = float(abs(w.phi[-1])) if hasattr(w, "phi") else float("nan")
    nan = lambda v: None if not np.isfinite(v) else float(v)  # noqa: E731
    return {
        "schema": SCHEMA, "case": case.name, "ns": ns, "mpol": w.mpol, "ntor": w.ntor, "nfp": nfp, "lasym": w.lasym,
        "grid": list(grid), "stride": list(stride),
        "fsq": [nan(getattr(w, k, np.nan)) for k in ("fsqr", "fsqz", "fsql")],
        "ier_flag": int(getattr(w, "ier_flag", -1)),
        "minor_radius": a_minor, "axis_offset": axis_offset, "axis_offset_over_minor": axis_offset/a_minor,
        "flux_label_error_max": float(flux_label_max),
        "flux_label_error_at_report": next(r["flux_label_error_max"] for r in surf_rows if r["j"] == j_report),
        "B_relative_surface": dict(s=float(full.s[j_report]), s_requested=s_report, value=field_rows[j_report]),
        "B_relative_near_axis": dict(s=float(full.s[1]), value=field_rows[1]),
        "B_relative_volume_l2": e_B,
        "iota_max_abs_error": iota_err,
        "vacuum": {
            "curl_over_B_over_a_rms": e_curl, "curl_over_B_over_a_rms_outer_s_ge_0.25": e_curl_outer,
            "curl_over_B_over_a_rms_at_report": float(e_curl_shell[jh]), "curl_over_B_over_a_max": curl_max_rel,
            "enclosed_current_over_aB_max": float(np.abs(poloidal).max()/(TWO_PI*a_minor*b_rms)),
            "div_over_B_over_a_rms": e_div,
            "toroidal_flux": flux, "toroidal_flux_exact": case.flux_edge,
            "toroidal_flux_rel_error": abs(flux/case.flux_edge - 1),
            "toroidal_flux_section_spread": float((np.abs(flux_sections).max() - np.abs(flux_sections).min())/flux),
            "phiedge_wout_rel_error": nan(abs(phiedge_w/case.flux_edge - 1)),
            "circulation_over_exact_mean": float(np.abs(circulation).mean()/c_exact),
            "circulation_rel_error_max": float(np.abs(np.abs(circulation) - c_exact).max()/c_exact),
            "circulation_at_report_rel_error_max": float(np.abs(np.abs(circulation[jh]) - c_exact).max()/c_exact),
            "energy_b2_integral": energy, "energy_target_circulation_phi": case.circulation*case.flux_edge,
            "energy_rel_error": abs(energy/(case.circulation*case.flux_edge) - 1),
            "circulation_flux_energy_identity_rel_error": abs(
                energy/(float(np.abs(circulation).mean())*flux) - 1),
            "volume": volume, "volume_exact": case.volume, "volume_rel_error": abs(volume/case.volume - 1),
            "min_abs_jacobian": float(np.abs(g).min()), "jacobian_sign_changes": int(np.sum(np.sign(g) != np.sign(g.flat[0]))),
        },
        "domain": {"q_max_full_surfaces": q_max, "points_outside_target_domain_full": invalid_points,
                   "q_max_half": float(q_h.max()), "rival_margin_min": float(min(margin_min, margin_h.min())),
                   "note": "q <= 1 inside configuration D; points slightly outside use the label-selected root"},
        "surfaces": surf_rows, "profile": profile,
    }


# ---------------------------------------------------------------- command line
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="run directory with receipt.json (cross_code_run layout)")
    parser.add_argument("--wout", type=Path, help="score this WOUT directly")
    parser.add_argument("--out", type=Path, help="output JSON (default: <run-dir>/lopez_score_v1.json)")
    parser.add_argument("--select", default="label", choices=("label", "modulus"))
    args = parser.parse_args(argv)
    receipt = {}
    if args.run_dir:
        receipt = json.loads((args.run_dir/"receipt.json").read_text())
        if not receipt.get("wout"):
            raise SystemExit("run produced no WOUT")
        wout = args.run_dir/receipt["wout"]["path"]
    else:
        wout = args.wout
    record = score_wout(read_wout(wout), lopez_case(select=args.select))
    record.update(wout_sha256=sha256(wout), scorer_sha256=sha256(__file__), select=args.select,
                  run_id=receipt.get("run_id"), code=receipt.get("code"),
                  wall_seconds=receipt.get("wall_seconds"))
    target = args.out or args.run_dir/f"lopez_score_v{SCHEMA}.json"
    if target.exists():
        raise SystemExit(f"{target} exists; scores are not overwritten")
    target.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    v = record["vacuum"]
    print("%s  fsq %s  axis %.1e  flux %.1e  B(s) %.2e  B(axis) %.2e  B(V) %.2e  iota %.1e  curl %.1e  div %.1e  "
          "Phi %.1e  circ %.1e  W %.1e" % (record["run_id"], ["%.1e" % x if x else "-" for x in record["fsq"]],
          record["axis_offset"], record["flux_label_error_max"], record["B_relative_surface"]["value"],
          record["B_relative_near_axis"]["value"], record["B_relative_volume_l2"], record["iota_max_abs_error"],
          v["curl_over_B_over_a_rms_at_report"], v["div_over_B_over_a_rms"], v["toroidal_flux_rel_error"],
          v["circulation_rel_error_max"], v["energy_rel_error"]), flush=True)


if __name__ == "__main__":
    main()
