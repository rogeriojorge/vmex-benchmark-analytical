"""Exact MHD references, in dimensionless units with mu0 = 1.

Landreman, arXiv:2609.26742v1, sections 2-3; plan.md gives the additional
stretch and asymmetric Solov'ev derivations. Angles in surface() are the
physical cylindrical phi and a counterclockwise R-Z parameter theta.
The latter need not be a straight-field-line angle.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    parameters: tuple[float, ...]
    nfp: int = 1
    lasym: bool = False
    phase: float = 0.0
    zshift: float = 0.0

    @property
    def edge(self):
        return self.parameters[-1]

    @property
    def pressure_slope(self):
        if self.family == "integer":
            return -2 * self.parameters[2]**2
        if self.family == "sheared":
            return -1 / self.parameters[2]**2
        return -8 * self.parameters[1]


def cases():
    data = json.loads((ROOT / "cases.json").read_text())
    return {r["name"]: Case(**{**r, "parameters": tuple(r["parameters"])})
            for r in data["cases"]}


def validate(case):
    p = np.asarray(case.parameters)
    if not np.isfinite(p).all() or case.edge <= 0:
        raise ValueError("Parameters must be finite and edge label positive.")
    if case.family == "integer":
        a, b, c, delta = p
        if min(a, b, c) <= 0 or abs((a*a-b*b)/(4*c*c))+np.sqrt(delta) >= 0.5:
            raise ValueError("Integer pressure torus leaves its smooth domain.")
    elif case.family == "sheared":
        eps, S, lam, delta = p
        if eps < 0 or lam <= 0 or 2*delta >= 1 or S <= np.arcsin(np.sqrt(2*delta)):
            raise ValueError("Invalid sheared-family domain.")
    elif case.family == "solovev":
        R0, b, g, chi, F0, edge = p
        if min(R0, b, g, F0) <= 0 or abs(chi) >= 1:
            raise ValueError("Solov'ev quadratic form must be positive definite.")
        if R0**2 <= np.sqrt(edge/(b*(1-chi*chi))) or F0**2 <= 4*g*edge:
            raise ValueError("Solov'ev torus reaches R=0 or F=0.")
    else:
        raise ValueError(f"Unknown family {case.family}.")
    if not case.lasym and (case.phase != 0 or case.zshift != 0):
        raise ValueError("A shifted symmetry origin requires the full Fourier basis.")


def rotate(x, angle, xp=jnp):
    c, s = xp.cos(angle), xp.sin(angle)
    return xp.stack((c*x[..., 0]-s*x[..., 1], s*x[..., 0]+c*x[..., 1], x[..., 2]), -1)


def integer_field(x, a, b, c):
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    q = (X/a)**2 + (Y/b)**2
    f = jnp.sqrt(2*q-q*q-4*(Z/c)**2)
    B = jnp.stack(((2*Z*X/c-a/b*f*Y)/q,
                   (2*Z*Y/c+b/a*f*X)/q, c*(1-q)), -1)
    Ha = (a*a+b*b)/2 - (a*a-b*b)**2/(8*c*c)
    H = (X*X+Y*Y+4*Z*Z+jnp.sum(B*B, -1))/2
    return B, (H-Ha)/(2*c*c)


def sheared_field(x, eps, S, lam):
    w, wc = x[..., 0]+1j*x[..., 1], x[..., 0]-1j*x[..., 1]
    K = wc*jnp.sqrt(1+eps/wc**2)
    phase = w*K+jnp.pi/2-S
    factor = jnp.exp(-1j*lam*x[..., 2])
    xy = 1j*factor*jnp.sin(phase)/(2*K)
    bz = jnp.real(factor*jnp.cos(phase))/lam
    B = jnp.stack((jnp.real(xy), jnp.imag(xy), bz), -1)
    return B, (jnp.sin(lam*x[..., 2])**2+(lam*bz)**2)/2


def solovev_field(x, R0, b, g, chi, F0):
    R = jnp.hypot(x[..., 0], x[..., 1])
    Z, U = x[..., 2], R*R-R0*R0
    h = chi*jnp.sqrt(b*g)
    psi = b*U*U+g*Z*Z+2*h*U*Z
    dR, dZ = 4*R*(b*U+h*Z), 2*(g*Z+h*U)
    br, bp, bz = dZ/R, jnp.sqrt(F0*F0-4*g*psi)/R, -dR/R
    return jnp.stack(((br*x[..., 0]-bp*x[..., 1])/R,
                       (br*x[..., 1]+bp*x[..., 0])/R, bz), -1), psi


def field(case, xyz):
    local = rotate(jnp.asarray(xyz)-jnp.array([0., 0., case.zshift]), -case.phase)
    functions = {"integer": integer_field, "sheared": sheared_field, "solovev": solovev_field}
    B, psi = functions[case.family](local, *case.parameters[:-1])
    return rotate(B, case.phase), psi


def pressure(case, xyz):
    return case.pressure_slope*(field(case, xyz)[1]-case.edge)


def shear_chart(k, theta, t, eps, S, lam, xp=np):
    """Explicit chart; t is NOT cylindrical phi. theta = -chi of the paper."""
    X, Y = -k*xp.cos(theta), -k*xp.sin(theta)
    nu = eps/2*xp.sin(2*t)
    sigma = S+xp.arctan(xp.tanh(nu)*Y/xp.sqrt(1-Y*Y))
    sigma -= xp.arcsin(X/xp.sqrt(xp.cosh(nu)**2-Y*Y))
    h = xp.sqrt(4*sigma*sigma+eps*eps)
    return xp.stack((xp.sqrt((h-eps)/2)*xp.cos(t),
                     xp.sqrt((h+eps)/2)*xp.sin(t), -xp.arcsin(Y)/lam), -1)


def validate_sheared_surface_chart_sampled(case, *, ntheta=64, nt=1025):
    """Reject a sampled toroidal-angle fold before using the quadrant inverse.

    The analytical smooth-domain condition is weaker than graph validity for
    physical toroidal angle. This numerical guard is not a global proof.
    """
    validate(case)
    if case.family != "sheared":
        return
    eps, S, lam, edge = case.parameters
    theta, t = np.broadcast_arrays(2*np.pi*np.arange(ntheta)[:, None]/ntheta,
                                    np.linspace(0, 2*np.pi, nt)[None, :])
    xyz = shear_chart(np.sqrt(2*edge), theta, t, eps, S, lam)
    phi = np.unwrap(np.arctan2(xyz[..., 1], xyz[..., 0]), axis=-1)
    if (not np.isfinite(xyz).all() or
        np.min(np.hypot(xyz[..., 0], xyz[..., 1])) <= 0 or
        np.min(np.diff(phi, axis=-1)) <= 0 or
        np.max(np.abs(phi[:, -1]-phi[:, 0]-2*np.pi)) > 1e-10):
        raise ValueError("Sampled sheared surface is not a single-valued physical-angle graph.")


def surface(case, label, theta, phi):
    """NumPy boundary sampler. Its bisection is NOT an AD input map.

    The local agent must implement the implicit derivative of phi(t)=phi_target
    before using sheared boundaries in exact-family adjoint tests (plan P4).
    """
    label, theta, phi = np.broadcast_arrays(label, theta, phi)
    phi0 = np.mod(phi-case.phase, 2*np.pi)
    if case.family == "integer":
        a, b, c, _ = case.parameters
        u0 = -(a*a-b*b)/(4*c*c)
        beta = 2*phi0+theta
        u, v = u0+np.sqrt(label)*np.cos(beta), np.sqrt(label)*np.sin(beta)
        ell = np.sqrt((1+np.sqrt(1-4*(u*u+v*v)))/2)
        A, C, D, E = a*(ell+u/ell), a*v/ell, b*v/ell, b*(ell-u/ell)
        ct, st = E*np.cos(phi0)-C*np.sin(phi0), A*np.sin(phi0)-D*np.cos(phi0)
        norm = np.hypot(ct, st)
        ct, st = ct/norm, st/norm
        xyz = np.stack((A*ct+C*st, D*ct+E*st, c*(v*(ct*ct-st*st)-2*u*st*ct)), -1)
    elif case.family == "sheared":
        eps, S, lam, _ = case.parameters
        lo = np.floor(phi0/(np.pi/2))*(np.pi/2)
        hi = lo+np.pi/2
        for _ in range(54):
            mid = (lo+hi)/2
            x = shear_chart(np.sqrt(2*label), theta, mid, eps, S, lam)
            cross = x[..., 0]*np.sin(phi0)-x[..., 1]*np.cos(phi0)
            lo, hi = np.where(cross > 0, mid, lo), np.where(cross > 0, hi, mid)
        xyz = shear_chart(np.sqrt(2*label), theta, (lo+hi)/2, eps, S, lam)
    else:
        R0, b, g, chi, _, _ = case.parameters
        U = np.sqrt(label/b)/np.sqrt(1-chi*chi)*np.cos(theta)
        Z = np.sqrt(label/g)*(np.sin(theta)-chi/np.sqrt(1-chi*chi)*np.cos(theta))
        R = np.sqrt(R0*R0+U)
        xyz = np.stack((R*np.cos(phi0), R*np.sin(phi0), Z), -1)
    return rotate(xyz, case.phase, np)+np.array([0., 0., case.zshift])


def shear_flux_rates(k, eps, S, lam, n=512, xp=np):
    """Q'(k)/k and poloidal A'(k)/k, regular at k=0; iota magnitude=A'/Q'."""
    k = xp.asarray(k)[..., None]
    u = xp.arange(n)*2*xp.pi/n
    c = xp.sqrt(1-k*k*xp.sin(u)**2)
    sigma = S+xp.arcsin(k*xp.cos(u)/c)
    q = xp.pi/lam*xp.mean(1/(xp.sqrt(4*sigma*sigma+eps*eps)*c), -1)
    nu = eps/2*xp.sin(2*u)
    sigma = S+xp.arcsin(k/xp.cosh(nu))
    h = xp.sqrt(4*sigma*sigma+eps*eps)
    g = (h+eps*xp.cos(2*u))/2
    a = 2*xp.pi/lam*xp.mean(g/(h*xp.sqrt(xp.cosh(nu)**2-k*k)), -1)
    return q, a


def flux_rate(case, label):
    """d(toroidal flux)/d(analytic pressure label)."""
    label = np.asarray(label)
    if case.family == "integer":
        a, b, c, _ = case.parameters
        return np.full_like(label, np.pi*a*b*c, dtype=float)
    if case.family == "sheared":
        return shear_flux_rates(np.sqrt(2*label), *case.parameters[:-1])[0]
    R0, b, g, chi, F0, _ = case.parameters
    return np.pi*np.sqrt(F0*F0-4*g*label)/(2*np.sqrt(b*g)*np.sqrt(1-chi*chi)
              *np.sqrt(R0**4-label/(b*(1-chi*chi))))


def flux(case, label, n=32):
    label = np.asarray(label)
    x, w = leggauss(n)
    return label/2*np.sum(w*flux_rate(case, label[..., None]*(x+1)/2), -1)


def label_at_s(case, s):
    s = np.asarray(s)
    if np.any((s < 0) | (s > 1)):
        raise ValueError("Normalized toroidal flux must be in [0,1].")
    edge_flux = float(flux(case, case.edge))
    if case.family == "integer":
        return case.edge*s
    result = [0. if x == 0 else case.edge if x == 1 else
              brentq(lambda psi: flux(case, psi)/edge_flux-x, 0., case.edge,
                     xtol=1e-14, rtol=1e-14) for x in s.ravel()]
    return np.asarray(result).reshape(s.shape)


def iota(case, label):
    """Signed transform for a counterclockwise R-Z poloidal convention."""
    if case.family == "integer":
        return np.zeros_like(label, dtype=float)-2.
    if case.family == "solovev":
        return -2*np.pi/flux_rate(case, label)
    q, a = shear_flux_rates(np.sqrt(2*np.asarray(label)), *case.parameters[:-1])
    return -a/q


def volume(case):
    if case.family == "integer":
        a, b, c, delta = case.parameters
        return 2*np.pi**2*a*b*c*delta
    if case.family == "solovev":
        _, b, g, chi, _, edge = case.parameters
        return np.pi**2*edge/(np.sqrt(b*g)*np.sqrt(1-chi*chi))
    return shear_averages(*case.parameters)["volume"]


def integer_targets(a, b, c, delta, xp=np):
    Ha = (a*a+b*b)/2-(a*a-b*b)**2/(8*c*c)
    return {"volume": 2*xp.pi**2*a*b*c*delta, "toroidal_flux": xp.pi*a*b*c*delta,
            "mean_B2": Ha+c*c*delta, "mean_pressure": c*c*delta,
            "beta": 2*c*c*delta/(Ha+c*c*delta)}


def shear_averages(eps, S, lam, delta, nr=80, nt=256):
    """Independent finite-radius pressure and field-energy quadratures."""
    n, w = leggauss(nr)
    d, alpha = np.arcsin(np.sqrt(2*delta)), np.pi*n/2
    xi = (d*np.sin(alpha))[:, None]
    weights = (d*np.pi/2*np.cos(alpha)*w)[:, None]
    t = np.arange(nt)[None, :]*2*np.pi/nt
    nu = eps/2*np.sin(2*t)
    f = np.cosh(nu)**2-np.sin(xi)**2
    u = np.arcsin(np.sqrt(np.maximum(0., (2*delta-np.sin(xi)**2)/f)))
    du = u-np.sin(2*u)/2
    c0 = -np.sin(xi)*np.cos(xi)/np.sqrt(f)
    c1 = np.cosh(nu)*np.sinh(nu)/np.sqrt(f)
    integral = lambda v: np.sum(weights*v)*2*np.pi/nt
    IU = integral(u)
    M = integral(2*np.sin(xi)**2*u+f*du)/(4*IU)
    C = integral(c0*c0*(2*u-du)+c1*c1*du)/(2*IU)
    A = integral(f*u/np.sqrt(4*(S+xi)**2+eps*eps))/(2*IU)
    meanp, meanb2 = (delta-M)/lam**2, A+C/lam**2
    return dict(volume=float(IU/lam), mean_pressure=float(meanp),
                mean_B2=float(meanb2), beta=float(2*meanp/meanb2))


def differential_fields(case, xyz):
    """B, curl B, grad p, dB_i/dx_j and analytic pressure label."""
    x = jnp.asarray(xyz)
    B, psi = field(case, x)
    db = jax.vmap(jax.jacfwd(lambda y: field(case, y)[0]))(x)
    curl = jnp.stack((db[:, 2, 1]-db[:, 1, 2], db[:, 0, 2]-db[:, 2, 0],
                      db[:, 1, 0]-db[:, 0, 1]), -1)
    gp = jax.vmap(jax.grad(lambda y: pressure(case, y)))(x)
    return tuple(np.asarray(y) for y in (B, curl, gp, db, psi))
