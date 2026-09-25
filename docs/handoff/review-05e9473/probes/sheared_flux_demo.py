"""Test the flux-density reconstruction on an existing analytical reference.

Usage: python sheared_flux_demo.py --reference-root PATH --output FILE
PATH must contain benchmarks/analytic.py and cases.json. No VMEX is imported.
The reference implementation's hashes are recorded; no solver recovery is implied.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from flux_potential import derivative, recover_lambda


def run(reference_root: Path) -> dict:
    started = perf_counter()
    sys.path.insert(0, str(reference_root / "benchmarks"))
    from analytic import cases, surface, field, flux, iota, label_at_s
    case = cases()["sheared_A"]
    s = 0.5
    label = float(label_at_s(case, s))
    # This reference chart increases counterclockwise in R-Z; its Jacobian
    # is negative. Toroidal flux in the reference API is positively oriented.
    P = -float(flux(case, case.edge)) / (2*np.pi)
    T = P*float(iota(case, label))
    rows=[]
    for n in (33,65,97):
        th=2*np.pi*np.arange(n)[:,None]/n
        ph=2*np.pi*np.arange(n)[None,:]/(n*case.nfp)
        xyz=np.asarray(surface(case,label,th,ph))
        R,Z=np.hypot(xyz[...,0],xyz[...,1]),xyz[...,2]
        eR=np.broadcast_to(np.stack((np.cos(ph),np.sin(ph),np.zeros_like(ph)),axis=-1),xyz.shape)
        ephi=np.broadcast_to(np.stack((-np.sin(ph),np.cos(ph),np.zeros_like(ph)),axis=-1),xyz.shape)
        ez=np.zeros_like(xyz); ez[...,2]=1
        xt=derivative(R,0,2*np.pi)[...,None]*eR+derivative(Z,0,2*np.pi)[...,None]*ez
        xp=derivative(R,1,2*np.pi/case.nfp)[...,None]*eR+R[...,None]*ephi+derivative(Z,1,2*np.pi/case.nfp)[...,None]*ez
        B=np.asarray(field(case,xyz)[0])
        for h in (1e-4,2.5e-5,6.25e-6):
            xs=(np.asarray(surface(case,float(label_at_s(case,s+h)),th,ph))
                -np.asarray(surface(case,float(label_at_s(case,s-h)),th,ph)))/(2*h)
            jac=np.sum(xs*np.cross(xt,xp),axis=-1)
            if np.any(jac>=0):
                raise ValueError("Unexpected chart orientation")
            U=np.sum(B*np.cross(xp,xs),axis=-1)
            V=np.sum(B*np.cross(xs,xt),axis=-1)
            lam,cert=recover_lambda(U,V,poloidal=T,toroidal=P,nfp=case.nfp,tolerance=5e-5)
            # Reconstruct the full field, not only the field-line slope.
            dt=derivative(lam,0,2*np.pi)
            dp=derivative(lam,1,2*np.pi/case.nfp)
            recovered=((T-P*dp)[...,None]*xt+(P*(1+dt))[...,None]*xp)/jac[...,None]
            err=np.linalg.norm(recovered-B)/np.linalg.norm(B)
            rows.append(dict(grid=n,radial_difference_step=h,
                             B_relative_l2=float(err),
                             mean_U_relative_error=float(abs(U.mean()-T)/abs(T)),
                             mean_V_relative_error=float(abs(V.mean()-P)/abs(P)),
                             **cert))
    files=(reference_root/"benchmarks/analytic.py", reference_root/"cases.json")
    return dict(evidence="analytical_reference_reconstruction_only",vmex_executed=False,
                case=case.name,s=s,nfp=case.nfp,poloidal_signed=T,toroidal_signed=P,
                reference_hashes={str(p.relative_to(reference_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                python=sys.version.split()[0],numpy=np.__version__,jax=jax.__version__,
                elapsed_seconds=perf_counter()-started,rows=rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.reference_root.resolve())
    with args.output.open("x") as handle:
        json.dump(result,handle,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps(result,indent=2,allow_nan=False))
