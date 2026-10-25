"""Field lines of a VMEC-format WOUT state of Lopez D, and the exact field on its surfaces. NumPy only.

Two closed-line diagnostics of a nested state, to set beside the MRX non-nested fields:

``vmec_line_closure``
    Lines of the state's own field on each full-mesh surface: d theta / d zeta = B^theta / B^zeta (the
    contravariant components of the WOUT, spectral interpolation). After one toroidal turn a line returns
    to theta + rotation(theta0); the surface is closed-line (iota = c) when rotation == c for every start.
    Reports the mean rotation, the spread over starts (the circle-map defect) and the WOUT iotaf.

``exact_field_on_surfaces``
    The exact vacuum field evaluated on the state's surfaces (``score_lopez`` conventions, exact frame):
    the area-weighted normal fraction ||B_exact . n|| / ||B_exact|| (zero iff the surface is a union of
    exact field lines, i.e. an exact foliation) and the exact label spread of the surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_lopez as sl  # noqa: E402

TWO_PI = 2*np.pi


def _spectrum(f):
    """2-D FFT of a (nth, nz) grid function over theta in turns and zeta in turns of the full torus."""
    return np.fft.fft2(f)/f.size


def _eval(spec, th, ze):
    """Spectral interpolation of the FFT ``spec`` at points (th, ze) in turns (arrays of equal shape)."""
    nth, nz = spec.shape
    kth = np.fft.fftfreq(nth, 1/nth)
    kz = np.fft.fftfreq(nz, 1/nz)
    keep_t = np.abs(kth) < nth/2
    keep_z = np.abs(kz) < nz/2
    sub = spec[np.ix_(keep_t, keep_z)]
    et = np.exp(1j*TWO_PI*np.outer(th, kth[keep_t]))          # (npts, mt)
    ez = np.exp(1j*TWO_PI*np.outer(ze, kz[keep_z]))
    return np.real(np.einsum("pm,mn,pn->p", et, sub, ez))


def vmec_line_closure(w, n_start=24, steps=240, surfaces=None, grid=None):
    """Closed-line statistics of the WOUT's own field on its full-mesh surfaces (see module doc)."""
    w = sl.as_wout(w)
    f = sl.evaluate_surfaces(w, "full", grid)
    ns = w.ns
    rows = []
    th0 = (np.arange(n_start) + 0.37)/n_start
    for j in (range(2, ns) if surfaces is None else surfaces):
        sb_u, sb_v = _spectrum(f.bu[j]), _spectrum(f.bv[j])
        th = th0.copy()
        h = 1.0/steps
        ze = np.zeros(n_start)

        def rhs(t, z, sb_u=sb_u, sb_v=sb_v):
            return _eval(sb_u, t, z)/_eval(sb_v, t, z)
        for _ in range(steps):
            k1 = rhs(th, ze)
            k2 = rhs(th + 0.5*h*k1, ze + 0.5*h)
            k3 = rhs(th + 0.5*h*k2, ze + 0.5*h)
            k4 = rhs(th + h*k3, ze + h)
            th = th + h*(k1 + 2*k2 + 2*k3 + k4)/6
            ze = ze + h
        rot = th - th0                         # poloidal turns per toroidal turn, per start
        rows.append(dict(j=int(j), s=float(f.s[j]), rotation_mean=float(rot.mean()),
                         rotation_spread=float(rot.max() - rot.min()), iotaf=float(w.iotaf[j])))
    return rows


def exact_field_on_surfaces(w, case=None, surfaces=None, grid=None, stride=(1, 1)):
    """Normal fraction of the exact field on the WOUT surfaces and the exact-label spread (module doc)."""
    w = sl.as_wout(w)
    case = case or sl.lopez_case()
    full = sl.evaluate_surfaces(w, "full", grid)
    ns = w.ns
    sl_ = (slice(None, None, stride[0]), slice(None, None, stride[1]))
    out = []
    for j in (range(1, ns) if surfaces is None else surfaces):
        sub = sl.Surfaces(*(None if a is None else (a[j][sl_] if a.ndim == 3 else a) for a in full[:10]),
                          full.s[j:j+1], full.phi[::stride[1]])
        sub = sub._replace(**{k: getattr(sub, k)[None] for k in ("R", "Z", "Ru", "Zu", "Rv", "Zv", "bu", "bv")})
        xyz, b = sl.cartesian(sub)
        xe, _ = sl.to_exact_frame(case, xyz, b)
        bx, s_ex, q, _ = case.exact(xe)
        cp, sp = np.cos(sub.phi), np.sin(sub.phi)
        e_th = np.stack((sub.Ru*cp, sub.Ru*sp, sub.Zu), -1)
        e_ze = np.stack((sub.Rv*cp - sub.R*sp, sub.Rv*sp + sub.R*cp, sub.Zv), -1)
        nrm = np.cross(e_th, e_ze)
        area = np.linalg.norm(nrm, axis=-1)
        n_hat = nrm/area[..., None]
        # exact field in the WOUT frame
        rot = case.frame[0]
        bw = bx @ rot.T
        bn = np.sum(bw*n_hat, -1)
        frac = np.sqrt(np.sum(area*bn**2)/np.sum(area*np.sum(bw**2, -1)))
        out.append(dict(j=int(j), s=float(full.s[j]), normal_fraction_rms=float(frac),
                        normal_fraction_max=float(np.max(np.abs(bn)/np.linalg.norm(bw, axis=-1))),
                        label_spread_max=float(np.abs(s_ex - full.s[j]).max())))
    return out
