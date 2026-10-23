"""Lopez configuration D: exact cubic-root vacuum field, dimensionless (B in the paper frame).

Lopez, arXiv:2610.29921v1, eqs. (1)-(6), (13), (24)-(26). With w = x + iy,
G(zeta) = 2i zeta^3 + (conj(w) + 2) zeta^2 - 2 z zeta - w, D = G'(zeta),
N = (1 - zeta^2, i (1 + zeta^2), 2 zeta) and B = Im N/(zeta D).

The root is selected by its recovered field-line labels (lambda, beta), never by its
position in the zeta plane. The labels are first integrals of B, and the certified
domain is Q(lambda, beta) < 1 with Q = 900 (lambda - 3/20)^2 + 100 (beta - 1/50)^2;
configuration D is 100 Q < 1. Root-separation certificate (not reproduced here):
omlopez/cubicRootCertificate certify.py, git blob 008fe6730550649d18f627d3a4caaf8c1d0259b2,
PASS, 343037 certified, 21733 outside, depth 28: on Q <= 1 the two rival roots have
Q >= 2 and |D|^2 >= 1/100. It certifies root separation only. The cylindrical-chart
checks below are sampled evidence, not a certified chart.
"""
from typing import NamedTuple

import numpy as np

A0, B0 = 3/20, 1/50  # centre of the label-space ellipse (lambda, beta)
SEMI_LAM, SEMI_BETA = 1/300, 1/100  # configuration D semi-axes (not the principal domain)
D_SCALE = 100.0  # D is D_SCALE * Q < 1
ROT = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])  # proper rotation, det = +1
SHIFT = np.array([7/20, -17/1150, 0.])  # chart: (Xc, Yc, Zc) = ROT x + SHIFT, R = hypot(Xc, Yc)


def q_label(lam, beta):
    """Principal-domain level Q; the certified domain is Q < 1."""
    return 900*(lam-A0)**2 + 100*(beta-B0)**2


def _normal(zeta):
    one = np.ones_like(zeta)
    return np.stack((1-zeta**2, 1j*(1+zeta**2), 2*zeta), -1), np.stack((-2*zeta, 2j*zeta, 2*one), -1)


def root_labels(zeta, xyz):
    """(lambda, beta) of a root zeta = u + iv of the cubic at the point xyz."""
    u, v = zeta.real, zeta.imag
    lam = u*u + v*v
    nvec = np.stack((2*u, 2*v, lam-1), -1)/(1+lam)[..., None]
    return lam, np.sum(nvec*xyz, -1) + 4*lam*u*(1-v)/(1+lam)**2


def forward(lam, beta, phase):
    """Eq. (13): X(lambda, beta, phase) and its derivative along the source phase."""
    lam, beta, phase = np.broadcast_arrays(lam, beta, phase)
    zeta = np.sqrt(lam)*np.exp(1j*phase)
    u, v = zeta.real, zeta.imag
    n, dn = _normal(zeta)
    dn = dn*(1j*zeta)[..., None]
    h, dh = 2*zeta**2 + 2j*zeta**3, (4*zeta + 6j*zeta**2)*(1j*zeta)
    nu = np.stack((2*u, 2*v, lam-1), -1)/(1+lam)[..., None]
    dnu = np.stack((-2*v, 2*u, 0*u), -1)/(1+lam)[..., None]
    kappa = 4*lam*u*(1-v)/(1+lam)**2
    dkappa = 4*lam*(-v*(1-v)-u*u)/(1+lam)**2
    a, c, lp = n.real, n.imag, (1+lam)[..., None]**2
    x = (h.real[..., None]*a + h.imag[..., None]*c)/lp + (beta-kappa)[..., None]*nu
    xp = (dh.real[..., None]*a + h.real[..., None]*dn.real + dh.imag[..., None]*c
          + h.imag[..., None]*dn.imag)/lp - dkappa[..., None]*nu + (beta-kappa)[..., None]*dnu
    return x, xp


class Root(NamedTuple):
    zeta: np.ndarray  # selected root
    d: np.ndarray  # G'(zeta)
    residual: np.ndarray  # |G(zeta)| after Newton polishing
    lam: np.ndarray
    beta: np.ndarray
    q: np.ndarray  # Q of the selected root
    q_rival: np.ndarray  # smallest Q among the other two roots


def evaluate(xyz, select="label"):
    """Cubic roots at xyz (..., 3) and selection of one.

    select="label" takes the root of smallest Q(lambda, beta): inside the certified
    domain the selected root has Q < 1 and both rivals Q >= 2, so the choice is unique.
    select="modulus" takes the smallest |zeta|; it is the unqualified rule kept only as
    a negative control and is wrong for most points of D.
    """
    xyz = np.asarray(xyz, float)
    w, z = xyz[..., 0] + 1j*xyz[..., 1], xyz[..., 2]
    comp = np.zeros(w.shape + (3, 3), complex)
    comp[..., 0, 0], comp[..., 0, 1], comp[..., 0, 2] = -(w.conj()+2)/2j, 2*z/2j, w/2j
    comp[..., 1, 0] = comp[..., 2, 1] = 1
    roots = np.linalg.eigvals(comp)
    lam, beta = root_labels(roots, xyz[..., None, :])
    q = q_label(lam, beta)
    score = q if select == "label" else np.abs(roots) if select == "modulus" else None
    if score is None:
        raise ValueError("select must be 'label' or 'modulus'")
    k = np.argmin(score, -1)[..., None]
    zeta = np.take_along_axis(roots, k, -1)[..., 0]
    for _ in range(2):  # Newton polish; the selected root is simple on the certified domain
        g, d = _poly(zeta, w, z)
        zeta = zeta - g/d
    g, d = _poly(zeta, w, z)
    lam1, beta1 = root_labels(zeta, xyz)
    rival = np.sort(q, -1)[..., 1] if select == "label" else np.min(
        np.where(np.arange(3) == k, np.inf, q), -1)
    return Root(zeta, d, np.abs(g), lam1, beta1, q_label(lam1, beta1), rival)


def _poly(zeta, w, z):
    return (2j*zeta**3 + (w.conj()+2)*zeta**2 - 2*z*zeta - w,
            6j*zeta**2 + 2*(w.conj()+2)*zeta - 2*z)


def field(xyz, select="label"):
    """B(x, y, z) = Im N/(zeta D) in the paper frame."""
    r = evaluate(xyz, select)
    return (_normal(r.zeta)[0]/(r.zeta*r.d)[..., None]).imag


def labels(xyz, select="label"):
    """Field-line labels (lambda, beta) of the selected root."""
    r = evaluate(xyz, select)
    return r.lam, r.beta


def field_jacobian(xyz):
    """B and dB[..., i, j] = d B_i/d x_j from implicit differentiation of the selected root."""
    xyz = np.asarray(xyz, float)
    r = evaluate(xyz)
    zeta, d = r.zeta, r.d
    w = xyz[..., 0] + 1j*xyz[..., 1]
    n, nz = _normal(zeta)
    m = np.stack((2*zeta, -2j*zeta, -2*np.ones_like(zeta)), -1)
    dz = 12j*zeta + 2*(w.conj()+2)
    zj, den = n/d[..., None], zeta*d
    dq = nz[..., :, None]*zj[..., None, :]/den[..., None, None]
    dd = zj*(d+zeta*dz)[..., None] + zeta[..., None]*m
    dq = dq - n[..., :, None]*dd[..., None, :]/den[..., None, None]**2
    return (n/den[..., None]).imag, dq.imag


def label_gradients(xyz):
    """lambda, beta, grad lambda, grad beta (exact), with B = grad lambda x grad beta/(2 lambda)."""
    xyz = np.asarray(xyz, float)
    r = evaluate(xyz)
    zj = _normal(r.zeta)[0]/r.d[..., None]
    u, v = r.zeta.real, r.zeta.imag
    lam, (x, y, z) = r.lam, np.moveaxis(xyz, -1, 0)
    gu, gv, s = zj.real, zj.imag, 1+lam
    p = 2*u*x + 2*v*y + (lam-1)*z
    bu = (2*x+2*u*z)/s - 2*u*p/s**2 + 4*(1-v)*((lam+2*u*u)/s**2 - 4*lam*u*u/s**3)
    bv = (2*y+2*v*z)/s - 2*v*p/s**2 + 4*u*((2*v*(1-v)-lam)/s**2 - 4*lam*(1-v)*v/s**3)
    nvec = np.stack((2*u, 2*v, lam-1), -1)/s[..., None]
    return (lam, r.beta, 2*(u[..., None]*gu + v[..., None]*gv),
            nvec + bu[..., None]*gu + bv[..., None]*gv)


def to_chart(xyz):
    """Proper rigid frame of the cylindrical chart: (R, phi, Zc) and Xc = (Xc, Yc, Zc)."""
    xc = np.asarray(xyz) @ ROT.T + SHIFT
    return np.hypot(xc[..., 0], xc[..., 1]), np.arctan2(xc[..., 1], xc[..., 0]), xc[..., 2], xc


def sample_domain(nrho, nalpha, nphase, scale=0.1):
    """Grid on the closed domain Q_D <= 1 (scale = 0.1; scale = 1 is the principal domain)."""
    rho = np.linspace(0, 1, nrho)[:, None, None]
    alpha = np.linspace(0, 2*np.pi, nalpha, endpoint=False)[None, :, None]
    phase = np.linspace(0, 2*np.pi, nphase, endpoint=False)[None, None, :]
    return A0 + scale*rho*np.cos(alpha)/30, B0 + scale*rho*np.sin(alpha)/10, phase


def chart_check(nrho=5, nalpha=64, nphase=1024, scale=0.1):
    """Sampled checks of the cylindrical chart on Q_D <= 1; not an interval certificate.

    Along each source-phase line at fixed labels: R > 0, dphi/dphase > 0, the wrapped
    steps of phi are positive and sum to exactly one turn (single-valued, one toroidal
    turn per source period). At the same points the label-selected root reproduces the
    forward map and the rival roots stay outside the domain.
    """
    lam, beta, phase = sample_domain(nrho, nalpha, nphase, scale)
    x, xp = forward(lam, beta, phase)
    r, phi, _, xc = to_chart(x)
    xcp = xp @ ROT.T
    num = xc[..., 0]*xcp[..., 1] - xc[..., 1]*xcp[..., 0]
    step = np.diff(np.concatenate((phi, phi[..., :1]), -1), axis=-1)
    step = (step + np.pi) % (2*np.pi) - np.pi
    root = evaluate(x)
    dphase = (np.angle(root.zeta) - phase + np.pi) % (2*np.pi) - np.pi
    return dict(
        npoints=int(x.shape[0]*x.shape[1]*x.shape[2]), scale=scale, sampled_not_certified=True,
        r_min=float(r.min()), dphi_dphase_min=float((num/r**2).min()),
        dphi_dphase_max=float((num/r**2).max()), orientation_numerator_min=float(num.min()),
        wrapped_step_min=float(step.min()), wrapped_step_max=float(step.max()),
        winding_turns_min=float(step.sum(-1).min()/(2*np.pi)),
        winding_turns_max=float(step.sum(-1).max()/(2*np.pi)),
        root_error_max=float(np.abs(root.zeta - np.sqrt(lam)*np.exp(1j*phase)).max()),
        phase_error_max=float(np.abs(dphase).max()),
        label_error_max=float(max(np.abs(root.lam - lam).max(), np.abs(root.beta - beta).max())),
        q_d_max=float(D_SCALE*root.q.max()), q_d_rival_min=float(D_SCALE*root.q_rival.min()),
        d_squared_min=float((np.abs(root.d)**2).min()), residual_max=float(root.residual.max()))


def verify(npoints=2000, seed=947, steps=(1e-3, 1e-4, 1e-5), scale=0.1):
    """Field identities at random points of the domain, from exact derivatives.

    Reports max |div B|, max |curl B| and their ratios to the largest |grad B| entry, the
    exact Jacobian against label-reselecting central differences for each step,
    B . grad(label) and B - grad lambda x grad beta/(2 lambda), and tangent-field agreement
    B(X) = X_phase/|X_phase|^2 (an independent value path).
    """
    rng = np.random.default_rng(seed)
    rho = np.sqrt(rng.random(npoints))
    alpha, phase = rng.uniform(0, 2*np.pi, (2, npoints))
    lam, beta = A0 + scale*rho*np.cos(alpha)/30, B0 + scale*rho*np.sin(alpha)/10
    x, xp = forward(lam, beta, phase)
    b, db = field_jacobian(x)
    curl = np.stack((db[:, 2, 1]-db[:, 1, 2], db[:, 0, 2]-db[:, 2, 0], db[:, 1, 0]-db[:, 0, 1]), -1)
    div, scale_db = np.trace(db, axis1=1, axis2=2), np.abs(db).max()
    fd = {}
    for h in steps:
        jac = np.stack([(field(x + h*e) - field(x - h*e))/(2*h) for e in np.eye(3)], -1)
        fd[h] = float(np.abs(jac - db).max()/scale_db)
    lam_x, beta_x, gl, gb = label_gradients(x)
    bn = np.linalg.norm(b, axis=-1)
    q = q_label(lam, beta)
    gq = (1800*(lam_x-A0))[:, None]*gl + (200*(beta_x-B0))[:, None]*gb
    unit = lambda g: np.abs(np.sum(b*g, -1))/(bn*np.linalg.norm(g, axis=-1))
    return dict(
        npoints=npoints, max_abs_b=float(bn.max()), max_abs_grad_b=float(scale_db),
        max_abs_div=float(np.abs(div).max()), max_norm_curl=float(np.linalg.norm(curl, axis=-1).max()),
        div_over_grad=float(np.abs(div).max()/scale_db),
        curl_over_grad=float(np.linalg.norm(curl, axis=-1).max()/scale_db),
        jacobian_fd_error_by_step={f"{h:g}": e for h, e in fd.items()},
        b_dot_grad_lambda=float(unit(gl).max()), b_dot_grad_beta=float(unit(gb).max()),
        b_dot_grad_q=float(unit(gq).max()),
        clebsch_error=float((np.linalg.norm(b - np.cross(gl, gb)/(2*lam_x)[:, None], axis=-1)/bn).max()),
        label_error=float(max(np.abs(lam_x-lam).max()/np.abs(lam).max(), np.abs(beta_x-beta).max())),
        tangent_error=float((np.linalg.norm(b - xp/np.sum(xp*xp, -1)[:, None], axis=-1)/bn).max()),
        q_max=float(q.max()))


def modulus_failure(npoints=20000, seed=1, scale=0.1, nbins=16):
    """Negative control: where the smallest-|zeta| root is not the label-selected one.

    Returns the wrong fraction overall and per source-phase bin: the unqualified rule is
    right only on a window of source phases and wrong on all the others.
    """
    rng = np.random.default_rng(seed)
    rho = np.sqrt(rng.random(npoints))
    alpha, phase = rng.uniform(0, 2*np.pi, (2, npoints))
    lam, beta = A0 + scale*rho*np.cos(alpha)/30, B0 + scale*rho*np.sin(alpha)/10
    x = forward(lam, beta, phase)[0]
    true = np.sqrt(lam)*np.exp(1j*phase)
    wrong = np.abs(evaluate(x, "modulus").zeta - true) > 1e-8
    b = field(x)
    err = np.linalg.norm(field(x, "modulus") - b, axis=-1)/np.linalg.norm(b, axis=-1)
    bins = (phase/(2*np.pi)*nbins).astype(int)
    return dict(npoints=npoints, wrong_fraction=float(wrong.mean()),
                label_wrong_fraction=float((np.abs(evaluate(x).zeta - true) > 1e-8).mean()),
                median_field_error_when_wrong=float(np.median(err[wrong])),
                wrong_by_phase_bin=[float(wrong[bins == i].mean()) for i in range(nbins)])


if __name__ == "__main__":
    import json
    out = dict(verify=verify(), chart=chart_check(), modulus_control=modulus_failure())
    print(json.dumps(out, indent=2))
