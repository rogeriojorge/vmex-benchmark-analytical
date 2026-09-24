"""Sample a live VMEX state in Cartesian space for independent scoring.

The field, curl and pressure gradient all come from VMEX's native state.
Reference geometry supplies only held-out positions and volume weights.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from vmex.core import profiles
from vmex.core.extender import VmecInteriorField

from analytic import label_at_s, surface
from build_inputs import LENGTH_M, FIELD_T, MU0


def reference_points(case, *, nradial=3, ntheta=8, nphi=4):
    nodes, radial_weights = np.polynomial.legendre.leggauss(nradial)
    radial = (nodes+1)/2
    s, theta, phi = np.meshgrid(radial,
        2*np.pi*np.arange(ntheta)/ntheta,
        2*np.pi*np.arange(nphi)/(nphi*case.nfp), indexing="ij")
    label = label_at_s(case, s)
    xyz = LENGTH_M*surface(case, label, theta, phi)
    h = 1e-5
    xs = LENGTH_M*(surface(case, label_at_s(case, s+h), theta, phi)
                   - surface(case, label_at_s(case, s-h), theta, phi))/(2*h)
    xt = LENGTH_M*(surface(case, label, theta+h, phi)
                   - surface(case, label, theta-h, phi))/(2*h)
    xp = LENGTH_M*(surface(case, label, theta, phi+h)
                   - surface(case, label, theta, phi-h))/(2*h)
    jac = np.abs(np.sum(xs*np.cross(xt, xp), axis=-1))
    if not np.all(np.isfinite(jac)) or np.any(jac <= 0):
        raise ValueError("Reference chart has invalid volume weights")
    weights = jac*(radial_weights/2)[:, None, None]*(2*np.pi/ntheta)*(2*np.pi/(nphi*case.nfp))
    return xyz.reshape(-1, 3), weights.ravel(), s.ravel()


def sample_native(inp, state, case, *, runtime=None, chunk_size=8):
    """Return scorer-ready arrays from a solved native state.

    This routine deliberately rejects nonfinite field/inversion samples. It
    evaluates pressure from parsed VMEX coefficients at the inverted VMEX s.
    """
    field = VmecInteriorField.from_state(inp, state, runtime=runtime)
    xyz, weights, s_ref = reference_points(case)
    B_parts, J_parts, gp_parts, s_parts = [], [], [], []

    def pressure_at_s(s):
        return profiles.pressure(inp.pmass_type, inp.am, inp.am_aux_s,
            inp.am_aux_f, s, pres_scale=inp.pres_scale, bloat=inp.bloat,
            spres_ped=inp.spres_ped)

    pressure_slope = jax.grad(pressure_at_s)
    for first in range(0, len(xyz), chunk_size):
        points = jnp.asarray(xyz[first:first+chunk_size])
        B = np.asarray(field.B(points))
        grad = np.asarray(field.gradB(points))  # dB_i / dx_j
        coordinates = np.asarray(field.flux_coordinates(points))
        grad_s = np.asarray(jax.vmap(jax.jacfwd(
            lambda x: field.flux_coordinates(x[None, :])[0, 0]))(points))
        dp_ds = np.asarray(jax.vmap(pressure_slope)(jnp.asarray(coordinates[:, 0])))
        curl = np.stack((grad[:, 2, 1]-grad[:, 1, 2],
                         grad[:, 0, 2]-grad[:, 2, 0],
                         grad[:, 1, 0]-grad[:, 0, 1]), axis=-1)
        B_parts.append(B)
        J_parts.append(curl/MU0)
        gp_parts.append(dp_ds[:, None]*grad_s)
        s_parts.append(coordinates[:, 0])
    B, J, gp, s_native = (np.concatenate(parts) for parts in
                          (B_parts, J_parts, gp_parts, s_parts))
    if not all(np.isfinite(x).all() for x in (B, J, gp, s_native)):
        raise ValueError("VMEX native field or coordinate inversion returned nonfinite values")
    return dict(case_name=case.name, xyz=xyz, B=B, J=J, gradp=gp,
                weights=weights, s=s_ref, vmex_s=s_native,
                length_m=LENGTH_M, field_t=FIELD_T, mu0=MU0)
