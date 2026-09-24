"""Generate reference figures only from recorded results and explicit geometry."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT, cases, surface

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
records = json.loads((ROOT / "results/reference/identities.json").read_text())["cases"]
fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
ax.semilogy(range(len(records)), [r["force_rms_over_gradp_rms"] for r in records], "o")
ax.set_xticks(range(len(records)), [r["name"].replace("_", " ") for r in records], rotation=65, ha="right")
ax.set(ylabel="Sampled force RMS / pressure-gradient RMS",
       title="Exact-reference identity checks (not VMEX solutions)")
ax.grid(True, which="both", alpha=.25)
fig.savefig(OUT / "reference_identities.png", dpi=170)
plt.close(fig)

rows = json.loads((ROOT / "results/reference/derivatives.json").read_text())["steps"]
fig, ax = plt.subplots(figsize=(7, 4.7), layout="constrained")
h = np.array([r["step"] for r in rows])
ax.loglog(h, [r["taylor_remainder"] for r in rows], "o-", label="First-order Taylor remainder")
ax.loglog(h, [max(r["central_fd_error"], 1e-18) for r in rows], "s-", label="Central difference error")
ax.loglog(h, 1e-2*h*h, "--", label="Slope 2 guide")
ax.set(xlabel="Parameter step", ylabel="Absolute error", title="Exact-family beta sensitivity; no equilibrium solve")
ax.legend()
ax.grid(True, which="both", alpha=.25)
fig.savefig(OUT / "reference_derivatives.png", dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5), layout="constrained")
for name in ("integer_axisymmetric", "integer_3d", "integer_3d_stretched"):
    case = cases()[name]
    x = surface(case, case.edge, np.linspace(0, 2*np.pi, 513), 0.)
    ax.plot(np.hypot(x[:, 0], x[:, 1]), x[:, 2], label=name.replace("_", " "))
ax.set(xlabel="R / length scale", ylabel="Z / length scale", title="Exact boundary sections at cylindrical phi = 0")
ax.set_aspect("equal", adjustable="box")
ax.legend()
fig.savefig(OUT / "reference_geometry.png", dpi=170)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5), layout="constrained")
for name in ("solovev_symmetric", "solovev_asymmetric"):
    case = cases()[name]
    x = surface(case, case.edge, np.linspace(0, 2*np.pi, 513), 0.)
    ax.plot(x[:, 0], x[:, 2], label=name.replace("_", " "))
ax.set(xlabel="R / length scale", ylabel="Z / length scale", title="An exact up-down-asymmetric benchmark")
ax.set_aspect("equal", adjustable="box")
ax.legend()
fig.savefig(OUT / "solovev_asymmetry.png", dpi=170)
plt.close(fig)

entries = json.loads((ROOT / "inputs/manifest.json").read_text())["records"][::2]
fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
ax.semilogy(range(len(entries)), [r["boundary_max_error_m"] for r in entries], "o")
ax.axhline(1e-6, linestyle="--", label="Initial smoke gate (not final accuracy target)")
ax.set_xticks(range(len(entries)), [r["case"].replace("_", " ") for r in entries], rotation=65, ha="right")
ax.set(ylabel="Maximum sampled boundary-fit error [m]", title="Initial Fourier truncation error; B and C need refinement")
ax.legend()
ax.grid(True, which="both", alpha=.25)
fig.savefig(OUT / "boundary_fit.png", dpi=170)
plt.close(fig)
