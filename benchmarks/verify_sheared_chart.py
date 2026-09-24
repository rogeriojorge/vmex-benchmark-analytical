"""Check global physical-angle inversion and chart margins of sheared cases."""
import json
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from analytic import ROOT, cases, field, shear_chart, surface

NT = 513
NTHETA = 64
K_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
ANGLE_GATE = 1e-10
LABEL_GATE = 1e-10

rows = []
for case in cases().values():
    if case.family != "sheared":
        continue
    eps, S, lam, edge = case.parameters
    kb = np.sqrt(2*edge)
    t = np.linspace(0, 2*np.pi, NT)
    theta = 2*np.pi*np.arange(NTHETA)/NTHETA
    min_dphi, max_span_error, min_radius = np.inf, 0.0, np.inf
    for fraction in K_FRACTIONS:
        th_grid, t_grid = np.broadcast_arrays(theta[:, None], t[None, :])
        x = shear_chart(kb*fraction, th_grid, t_grid, eps, S, lam)
        phi = np.unwrap(np.arctan2(x[..., 1], x[..., 0]), axis=-1)
        steps = np.diff(phi, axis=-1)
        min_dphi = min(min_dphi, float(np.min(steps)/(2*np.pi/(NT-1))))
        max_span_error = max(max_span_error, float(np.max(abs(phi[:, -1]-phi[:, 0]-2*np.pi))))
        min_radius = min(min_radius, float(np.min(np.hypot(x[..., 0], x[..., 1]))))
    rng = np.random.default_rng(924)
    labels = edge*rng.uniform(0.01, 1.0, 1024)
    th, ph = rng.uniform(0, 2*np.pi, (2, 1024))
    xyz = surface(case, labels, th, ph)
    angle_delta = np.arctan2(xyz[:, 1], xyz[:, 0])-ph
    angle_error = float(np.max(np.abs(np.angle(np.exp(1j*angle_delta)))))
    label_error = float(np.max(abs(np.asarray(field(case, xyz)[1])-labels))/edge)
    status = "passed" if (min_dphi > 0 and max_span_error < ANGLE_GATE and
                          angle_error < ANGLE_GATE and label_error < LABEL_GATE and
                          min_radius > 0) else "failed"
    rows.append(dict(case=case.name, status=status, min_dphi_dt=min_dphi,
                     angle_span_max_abs=max_span_error, physical_angle_max_abs=angle_error,
                     label_max_over_edge=label_error, min_radius=min_radius,
                     S_minus_asin_kb=float(S-np.arcsin(kb))))
    print(rows[-1])
out = ROOT / "results/reference/sheared_chart.json"
out.write_text(json.dumps(dict(schema=1, evidence="analytic_reference_sampled",
    status="passed" if all(r["status"] == "passed" for r in rows) else "failed",
    t_samples=NT, theta_samples=NTHETA, random_inverse_samples=1024,
    radial_fractions=K_FRACTIONS, angle_gate=ANGLE_GATE, label_gate=LABEL_GATE,
    rows=rows), indent=2, allow_nan=False)+"\n")
if any(r["status"] != "passed" for r in rows):
    raise SystemExit("Sheared chart check failed")
