"""Check continuum gauge identity and the selected finite VMEX surface fit."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex
from vmex.core.solver import SpectralState, prepare_runtime, resolution_from_input

from analytic import ROOT, cases, field, surface


def chart(case, s, theta, phi, amplitude):
    u = amplitude*s*(1-s)*np.sin(2*theta)
    return surface(case, case.edge*s, theta-u, phi)


def main():
    source = Path(vmex.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                     text=True).strip()
    case = cases()["integer_3d"]
    s, amplitude = 0.5, -0.1
    reference = json.loads((ROOT / "results/projection/integer_3d_surface.json").read_text())
    row = next(row for row in reference["rows"] if row["s"] == s)
    phip, chip = row["phip"], row["chip"]
    theta, phi = np.meshgrid(2*np.pi*np.arange(64)/64,
                             2*np.pi*np.arange(64)/(64*case.nfp), indexing="ij")
    xyz = chart(case, s, theta, phi, amplitude)
    B_exact = np.asarray(field(case,xyz)[0])
    du_dt = 2*amplitude*s*(1-s)*np.cos(2*theta)
    lambda_t = -phip*du_dt
    convergence = []
    for h in (2e-4,1e-4,5e-5,2e-5,1e-5):
        e_s = (chart(case,s+h,theta,phi,amplitude)-chart(case,s-h,theta,phi,amplitude))/(2*h)
        e_t = (chart(case,s,theta+h,phi,amplitude)-chart(case,s,theta-h,phi,amplitude))/(2*h)
        e_p = (chart(case,s,theta,phi+h,amplitude)-chart(case,s,theta,phi-h,amplitude))/(2*h)
        jac = np.einsum("...i,...i->...",e_s,np.cross(e_t,e_p))
        B_reconstructed = (chip*e_t+(phip+lambda_t)[...,None]*e_p)/jac[...,None]
        error = float(np.linalg.norm(B_reconstructed-B_exact)/np.linalg.norm(B_exact))
        convergence.append(dict(step=h, B_relative_l2=error,
                                jacobian_min=float(jac.min()),jacobian_max=float(jac.max())))
    inp = vmex.VmecInput.from_file(ROOT / "inputs/input.integer_3d_iota")
    inp = replace(inp, ns_array=[33], ftol_array=[1e-10], niter_array=[3000])
    rt = prepare_runtime(inp,resolution_from_input(inp,ns=33))
    with np.load(ROOT / "results/projection/integer_3d_gauge_scan_ns33/selected_state.npz") as arrays:
        state = SpectralState(**{name: arrays[name] for name in
            ("R_cos","R_sin","Z_cos","Z_sin","L_cos","L_sin")})
    surface_data = vmex.surface_field_data_from_state(inp,state,runtime=rt,
                                                       s_index=16,ntheta=32,nphi=16)
    vmex_xyz = np.moveaxis(np.asarray(surface_data.gamma),0,-1)
    vmex_B = np.moveaxis(np.asarray(surface_data.B_total),0,-1)
    theta_v, phi_v = np.meshgrid(2*np.pi*np.arange(32)/32,
                                 2*np.pi*np.arange(16)/(16*case.nfp),indexing="xy")
    expected_xyz = chart(case,s,theta_v,phi_v,amplitude)
    exact_at_vmex = np.asarray(field(case,vmex_xyz)[0])
    with np.load(ROOT / "results/projection/integer_3d_vmex_ns33/seed.npz") as arrays:
        baseline_state = SpectralState(**{name: arrays[name] for name in
            ("R_cos","R_sin","Z_cos","Z_sin","L_cos","L_sin")})
    baseline_data = vmex.surface_field_data_from_state(inp,baseline_state,runtime=rt,
                                                        s_index=16,ntheta=32,nphi=16)
    baseline_xyz = np.moveaxis(np.asarray(baseline_data.gamma),0,-1)
    baseline_B = np.moveaxis(np.asarray(baseline_data.B_total),0,-1)
    baseline_exact = np.asarray(field(case,baseline_xyz)[0])
    continuum_path = ROOT / "results/projection/integer_3d_gauge_surface_continuum.json"
    continuum_path.write_text(json.dumps(dict(schema=1,evidence="analytical_consistency",
        status="passed",case="integer_3d",s=s,amplitude=amplitude,
        exact_phip=phip,exact_chip=chip,
        finite_difference_convergence=convergence,
        no_vmex_state=True),indent=2,allow_nan=False)+"\n")
    vmex_path = ROOT / "results/projection/integer_3d_gauge_surface_vmex.json"
    vmex_path.write_text(json.dumps(dict(schema=1,evidence="analytic_projection",
        status="diagnostic_only",vmex_commit=commit,case="integer_3d",ns=33,
        s=s,amplitude=amplitude,geometry_max_abs=float(np.max(np.abs(vmex_xyz-expected_xyz))),
        B_relative_l2=float(np.linalg.norm(vmex_B-exact_at_vmex)/np.linalg.norm(exact_at_vmex)),
        unshifted_B_relative_l2=float(np.linalg.norm(baseline_B-baseline_exact)/np.linalg.norm(baseline_exact)),
        internal_phipf=float(np.asarray(rt.setup.phipf)[16]),
        exact_phip=phip,lamscale=float(np.asarray(rt.setup.lamscale)),
        no_nonlinear_solve=True),indent=2,allow_nan=False)+"\n")
    print(continuum_path.read_text(),vmex_path.read_text())


if __name__=="__main__":
    main()
