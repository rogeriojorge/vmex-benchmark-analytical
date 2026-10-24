"""Stage 6 mechanism tools: one funct3d pass of an exact (projected) state, split into its parts.

The VMEC residual is  F = F_MHD + F_con  where F_con is the spectral-condensation (``tcon0``) force of
``forces.f`` / ``alias.f``: it penalises the m(m-1)-weighted deviation of the state from the baseline
``rcon0 = s * rcon(edge)`` (``funct3d.f``).  Nothing makes the exact states satisfy ``rcon = rcon0``, so
F_con is a bias at the exact root.  This module lets that bias be measured and removed one knob at a time:

* ``baseline`` of the constraint: ``vmec`` (``s * edge``, the VMEC rule), ``exact`` (the real-space
  ``rcon, zcon`` of the starting state, so F_con vanishes exactly at it), or ``blend:a`` (``a`` of the
  exact baseline plus ``1-a`` of the VMEC one);
* ``tcon`` scale (strength of the whole constraint), including 0 (F_con off);
* split of the scaled force into F(full), F(tcon=0) and their difference, per block, per m, per radius.

Everything here is VMEX ``caf09ee7`` internals (``solver._evaluate``, ``_force_pipeline``,
``constraint_force``); no VMEX file is modified.  The solver-level override of the baselines is
``use_baseline`` (a context manager that wraps ``solver._constraint_baselines_lane``).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]


def case_deck(case, ns, tcon0=None, deck=None, mpol=None, ntor=None, **changes):
    """The committed ``input.<case>_iota`` (or ``deck``) at one radial grid (single stage).

    ``mpol``/``ntor`` re-pad the boundary arrays (``project_integer_vmex.with_resolution``); ``tcon0`` and any other
    VmecInput field (e.g. ``ntheta``) are replaced.
    """
    import vmex
    from project_integer_vmex import with_resolution
    name = case if isinstance(case, str) else case.name
    inp = vmex.VmecInput.from_file(deck or ROOT/"inputs"/f"input.{name}_iota")
    inp = with_resolution(inp, ns, mpol, ntor)
    if tcon0 is not None:
        changes["tcon0"] = float(tcon0)
    return replace(inp, **changes) if changes else inp


def polar_rotation(state, inp, ns):
    """Rotate every surface's poloidal parametrisation so that R1s = Z1c on each surface (VMEC's m = 1 polar gauge).

    For the axisymmetric (n = 0) asymmetric cases the unrotated exact surfaces violate the polar constraint; VMEC (and VMEX's
    ``readin``) rotates the *boundary* parametrisation by delta = atan((R1s - Z1c)/(R1c + Z1s)) while the projected interior
    keeps the original angle and a lambda that belongs to it, so ``hot_restart_state`` shears the edge against the interior and
    the force path sees a field whose theta-structure is shifted by delta against the surfaces.  Here the SAME exact surfaces are
    re-parametrised by theta' = theta - delta(s) (R'(theta') = R(theta' + delta), same for Z and lambda; the lambda constant is a
    gauge), the deck boundary is rewritten in that angle, and the interior is then exactly polar.

    Returns (inp_polar, state_polar, deltas).  For ntor > 0 the same constant-per-shell rotation (delta from the n = 0, m = 1 pair,
    as VMEC does) is applied to every (m, n) pair, which rotates the combined phase m theta - n nfp phi by m delta; it does not
    enforce R1s = Z1c on the n != 0 pairs (that needs a phi-dependent omega, see lopez_polar).
    """
    import vmex.core.solver as S
    from vmex.core.fourier import mode_table, trig_tables
    from vmex.core.residuals import m1_constrained_to_physical, m1_physical_to_constrained
    from vmex.core.transforms import physical_to_internal_scale
    res = S.resolution_from_input(inp, ns=ns)
    modes = mode_table(res.mpol, res.ntor)
    trig = trig_tables(res)
    scale = np.asarray(physical_to_internal_scale(modes, trig))
    m = np.asarray(modes.m)
    Rc, Zs, Rs, Zc = (np.asarray(a) for a in m1_constrained_to_physical(
        state.R_cos, state.Z_sin, state.R_sin, state.Z_cos, modes=modes, lthreed=bool(res.lthreed), lasym=True, lconm1=True))
    Lc, Ls = np.asarray(state.L_cos), np.asarray(state.L_sin)
    n_modes = np.asarray(modes.n)
    i1 = int(np.where((m == 1) & (n_modes == 0))[0][0])
    delta = np.arctan2(Rs[:, i1]-Zc[:, i1], Rc[:, i1]+Zs[:, i1])
    delta[0] = 0.0                                   # axis row: only m = 0 content
    c, s = np.cos(np.outer(delta, m)), np.sin(np.outer(delta, m))

    def rot(cos_part, sin_part):
        return cos_part*c+sin_part*s, -cos_part*s+sin_part*c
    Rc2, Rs2 = rot(Rc, Rs)
    Zc2, Zs2 = rot(Zc, Zs)
    Lc2, Ls2 = rot(Lc, Ls)
    Rc2, Zs2, Rs2, Zc2 = (np.asarray(a) for a in m1_physical_to_constrained(
        Rc2, Zs2, Rs2, Zc2, modes=modes, lthreed=bool(res.lthreed), lasym=True, lconm1=True))
    # the deck boundary in the new angle: edge row, physical (unscaled) coefficients, shape (1, mpol)
    Rcp, Zsp, Rsp, Zcp = (np.asarray(a) for a in m1_constrained_to_physical(
        Rc2, Zs2, Rs2, Zc2, modes=modes, lthreed=bool(res.lthreed), lasym=True, lconm1=True))
    ntor, mpol = int(res.ntor), int(res.mpol)

    def edge(a):
        out = np.zeros((2*ntor+1, mpol))
        phys = a[-1]/scale
        for k in range(len(m)):
            out[int(n_modes[k])+ntor, int(m[k])] = phys[k]    # deck layout: [n + ntor, m], phase m theta - n nfp phi
        return out
    inp2 = replace(inp, rbc=edge(Rcp), zbs=edge(Zsp), rbs=edge(Rsp), zbc=edge(Zcp))
    from vmex.core.solver import SpectralState
    return inp2, SpectralState(R_cos=Rc2, R_sin=Rs2, Z_cos=Zc2, Z_sin=Zs2, L_cos=Lc2, L_sin=Ls2), delta


def exact_seed(case, ns, tcon0=None, lambda_mode="auto", deck=None, mpol=None, ntor=None, polar=False, **changes):
    """(inp, rt, state, info): deck, runtime with VMEC baselines of the hot-restarted exact seed, the seed.

    ``case`` is a name of ``cases.json`` or an ``analytic.Case`` (with its ``deck``).  ``polar=True`` re-parametrises the
    projected surfaces into VMEC's m = 1 polar gauge and rewrites the deck boundary accordingly (``polar_rotation``,
    axisymmetric asymmetric cases); ``info['polar_delta']`` is the rotation angle per shell."""
    import vmex.core.solver as S
    from analytic import cases
    from project_integer_vmex import project
    inp = case_deck(case, ns, tcon0, deck, mpol, ntor, **changes)
    case_obj = cases()[case] if isinstance(case, str) else case
    state, info = project(inp, case_obj, ns, lambda_mode=lambda_mode)
    if polar:
        inp, state, delta = polar_rotation(state, inp, ns)
        info = {**info, "polar_delta": [float(x) for x in delta]}
    resolution = S.resolution_from_input(inp, ns=ns)
    rt = S.prepare_runtime(inp, resolution, tcon0=float(inp.tcon0))
    state = S.hot_restart_state(rt, state)
    rt = S.runtime_with_baselines(rt, state)
    return inp, rt, state, info


def real_space_baselines(state, rt, kind="vmec"):
    """(rcon0, zcon0) arrays for ``kind`` in {vmec, exact, blend:a}; ``state`` is the reference state."""
    import vmex.core.solver as S
    from vmex.core.forces import constraint_force
    (Rc, Rs, Zc, Zs), geometry = S._geometry(state, rt)
    ns = int(rt.setup.s_full.shape[0])
    s = rt.setup.s_full
    zero = jnp.zeros((ns,), dtype=s.dtype)
    _, rcon, zcon, rcon0_v, zcon0_v = constraint_force(
        R_cos=Rc, R_sin=Rs, Z_cos=Zc, Z_sin=Zs, geometry=geometry, modes=rt.modes, trig=rt.trig, s=s,
        tcon=zero, signgs=rt.setup.signgs)
    if kind == "vmec":
        return rcon0_v, zcon0_v
    if kind == "exact":
        return rcon, zcon
    if kind.startswith("blend:"):
        a = float(kind.split(":")[1])
        return a*rcon+(1-a)*rcon0_v, a*zcon+(1-a)*zcon0_v
    raise ValueError(kind)


def with_baseline(rt, state, kind):
    r0, z0 = real_space_baselines(state, rt, kind)
    return replace(rt, rcon0=r0, zcon0=z0)


@contextmanager
def use_baseline_state(kind, seed, inp, tcon0):
    """Within the block every constraint-baseline binding of the solver returns the ``kind`` baselines of ``seed``.

    ``vmec`` leaves the solver alone.  For ``exact`` / ``blend:a`` the baselines are computed once from the
    (hot-restarted) exact seed and kept for the whole run, jacobian-retry rebinds included, so the constraint force
    is the one of the exact state throughout.
    """
    import vmex.core.solver as S
    if kind == "vmec":
        yield
        return
    ns = int(np.asarray(seed.R_cos).shape[0])
    rt = S.prepare_runtime(inp, S.resolution_from_input(inp, ns=ns), tcon0=float(tcon0))
    fixed = real_space_baselines(seed, rt, kind)
    original = S._constraint_baselines_lane
    S._constraint_baselines_lane = lambda state, rt, *, use_fft=False: fixed
    try:
        yield
    finally:
        S._constraint_baselines_lane = original


@jax.jit
def _scaled_lane(state, cache, rt, fsqz_previous, iteration):
    """The scalxc-scaled force (input of ``getfsq``) of ``state`` at a frozen ``cache`` (one XLA program)."""
    import vmex.core.solver as S
    s = rt.setup.s_full
    setup = rt.setup
    (R_cos, R_sin, Z_cos, Z_sin), geometry = S._geometry(state, rt)
    jacobian = S.half_mesh_jacobian(geometry, s=s)
    metrics = S.metric_elements(geometry, s=s)
    fields = S.magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig, s=s, phips=setup.phips,
        phipf=setup.phipf, chips=setup.chips, signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv)
    scaled, _, _ = S._force_pipeline(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields, R_cos=R_cos, R_sin=R_sin,
        Z_cos=Z_cos, Z_sin=Z_sin, cache=cache, rt=rt, iteration=iteration, fsqz_previous=fsqz_previous)
    return S._force_to_state(scaled, rt)


def fresh_cache(state, rt, fsqz_previous=1.0, iteration=100):
    """(cache refreshed at ``state``, residuals of the full force), as the first iteration of a solve sees them."""
    import vmex.core.solver as S
    res = S._evaluate_lane(state, S._zero_cache(rt), jnp.asarray(iteration), jnp.asarray(iteration),
                           jnp.asarray(float(fsqz_previous)), rt)
    return res.cache, res


def scaled_force(state, rt, tcon_scale=1.0, cache=None, fsqz_previous=1.0, iteration=100):
    """(scaled SpectralForce, residuals, cache).  ``tcon_scale`` multiplies the cache's tcon (0 = F_con off)."""
    import vmex.core.solver as S
    if cache is None:
        cache, _ = fresh_cache(state, rt, fsqz_previous, iteration)
    c = replace(cache, tcon=cache.tcon*tcon_scale)
    force = _scaled_lane(state, c, rt, jnp.asarray(float(fsqz_previous)), jnp.asarray(iteration+1))
    res = S._evaluate_lane(state, c, jnp.asarray(iteration+1), jnp.asarray(iteration),
                           jnp.asarray(float(fsqz_previous)), rt)
    return force, res, cache


@jax.jit
def _spectral_lane(state, cache, rt, fsqz_previous, iteration):
    """The raw spectral MHD force (before the m=1 rotation / gating / scalxc) as a dict of arrays."""
    import vmex.core.solver as S
    s = rt.setup.s_full
    setup = rt.setup
    (R_cos, R_sin, Z_cos, Z_sin), geometry = S._geometry(state, rt)
    jacobian = S.half_mesh_jacobian(geometry, s=s)
    metrics = S.metric_elements(geometry, s=s)
    fields = S.magnetic_fields(
        geometry=geometry, jacobian=jacobian, metrics=metrics, trig=rt.trig, s=s, phips=setup.phips,
        phipf=setup.phipf, chips=setup.chips, signgs=setup.signgs, gamma=rt.gamma, mass=setup.mass,
        ncurr=setup.ncurr, enclosed_current=setup.icurv)
    forces = S.mhd_forces(
        geometry=geometry, jacobian=jacobian, metrics=metrics, fields=fields, R_cos=R_cos, R_sin=R_sin,
        Z_cos=Z_cos, Z_sin=Z_sin, modes=rt.modes, trig=rt.trig, s=s, phipf=setup.phipf, tcon=cache.tcon,
        signgs=setup.signgs, rcon0=rt.rcon0, zcon0=rt.zcon0)
    sp = S.spectral_mhd_forces(forces, mpol=rt.resolution.mpol, ntor=rt.resolution.ntor, trig=rt.trig,
                               include_edge=bool(rt.lfreeb))
    return {k: v for k, v in sp.__dict__.items() if v is not None}


def defect_forces(state, rt, kind, cache=None, fsqz_previous=1.0, iteration=100):
    """Constant spectral forcing D (dict of arrays) taken from the exact state, for a defect-corrected solve.

    ``kind``: ``con`` (the whole constraint force F_con(x*) = F(tcon) - F(tcon=0)); ``mhd`` (F_MHD(x*));
    ``mhdaxis:K`` / ``mhdbulk:K`` (F_MHD(x*) on shells js < K / js >= K only); ``block:<field>[:<m>]`` (one spectral
    force block, optionally one m row, of F_MHD(x*)); ``all`` (F(x*): x* is then an exact root).  Subtracting D from the force makes the chosen part of the discrete residual of x* vanish.
    """
    if cache is None:
        cache, _ = fresh_cache(state, rt, fsqz_previous, iteration)
    args = (jnp.asarray(float(fsqz_previous)), jnp.asarray(iteration+1))
    full = _spectral_lane(state, cache, rt, *args)
    off = _spectral_lane(state, replace(cache, tcon=cache.tcon*0.0), rt, *args)
    if kind == "con":
        d = {k: full[k]-off[k] for k in full}
    elif kind == "all":
        d = dict(full)
    elif kind.startswith("block:"):
        # block:<field>[:<m>]  e.g. block:force_Z_cc:0 -> the m = 0 row of the Z_cc block of F_MHD(x*); several
        # blocks may be joined with '+'.  Spectral blocks are (ns, mpol, ntor+1): axis 1 is m.
        d = {k: jnp.zeros_like(v) for k, v in off.items()}
        for part in kind[len("block:"):].split("+"):
            bits = part.split(":")
            name = bits[0] if bits[0].startswith("force_") else "force_"+bits[0]
            if name not in off:
                raise ValueError(f"{name} is not a force block of this state: {sorted(off)}")
            if len(bits) > 1:
                row = int(bits[1])
                d[name] = d[name].at[:, row].set(off[name][:, row])
            else:
                d[name] = off[name]
    elif kind == "mhd" or kind.startswith("mhdaxis") or kind.startswith("mhdbulk"):
        d = dict(off)
        if kind != "mhd":
            K = int(kind.split(":")[1])
            for k, v in d.items():
                js = jnp.arange(v.shape[0])
                mask = (js < K) if kind.startswith("mhdaxis") else (js >= K)
                d[k] = jnp.where(mask.reshape((-1,)+(1,)*(v.ndim-1)), v, 0.0)
    else:
        raise ValueError(kind)
    return d


@contextmanager
def use_defect(d):
    """Within the block ``spectral_mhd_forces`` returns (force - D).  Clears the jit caches on entry and exit:
    D is a trace-time constant, so no executable compiled with another D may be reused."""
    import vmex.core.solver as S
    if d is None:
        yield
        return
    original = S.spectral_mhd_forces
    jax.clear_caches()

    def patched(forces, **kw):
        sp = original(forces, **kw)
        return replace(sp, **{k: getattr(sp, k)-d[k] for k in d})
    S.spectral_mhd_forces = patched
    try:
        yield
    finally:
        S.spectral_mhd_forces = original
        jax.clear_caches()


def block_arrays(force, rt):
    """Signed (ns, mnmax) arrays of a SpectralForce, weights restoring the sum of squares."""
    st = force
    n = np.asarray(rt.modes.n)
    w = np.where(n == 0, 1.0, 2.0)
    return {k: np.asarray(getattr(st, k)) for k in ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")}, w


def parts_report(state, rt, fsqz_previous=1.0):
    """Residual of ``state`` with F_con on and off, and the constraint part, per block / m / radius.

    Returns a dict of scalars and radial / per-m vectors.  ``fsq`` values are the VMEC invariants (``fsqr``,
    ``fsqz``, ``fsql``); ``*_con`` are the same sums of squares of the difference F(full) - F(tcon=0), which is
    the constraint contribution (the pipeline is linear in the added force at a fixed state), normalised
    with the same ``fnorm``.  ``frac_*`` split ``sum(block**2)`` over m and radial shell.
    """
    full, res_full, cache = scaled_force(state, rt, 1.0, fsqz_previous=fsqz_previous)
    off, res_off, _ = scaled_force(state, rt, 0.0, cache=cache, fsqz_previous=fsqz_previous)
    a_full, w = block_arrays(full, rt)
    a_off, _ = block_arrays(off, rt)
    m = np.asarray(rt.modes.m)
    fnorm = float(cache.fnorm)
    r1 = float(res_full.residuals.fsqr)/max(float(res_full.residuals.gcr2), 1e-300)
    out = {"fsqr": float(res_full.residuals.fsqr), "fsqz": float(res_full.residuals.fsqz),
           "fsql": float(res_full.residuals.fsql), "fsqr_off": float(res_off.residuals.fsqr),
           "fsqz_off": float(res_off.residuals.fsqz), "fsql_off": float(res_off.residuals.fsql),
           "fnorm": fnorm, "r1fnorm": r1}
    parts = {"full": a_full, "mhd": a_off, "con": {k: a_full[k]-a_off[k] for k in a_full}}
    ns = a_full["R_cos"].shape[0]
    jsmax = ns-1   # the residual sums run to ns-1 (edge row excluded for R, Z); lambda over all rows
    for name, arr in parts.items():
        e = {}
        for k in ("R", "Z"):
            e[k] = sum(w*(arr[f"{k}_cos"][:jsmax]**2 + arr[f"{k}_sin"][:jsmax]**2) for _ in [0])
        e["L"] = w*(arr["L_cos"]**2 + arr["L_sin"]**2)
        rz = e["R"]+e["Z"]
        out[f"fsq_rz_{name}"] = float(rz.sum()*r1)
        out[f"fsq_l_{name}"] = float(e["L"].sum()*float(cache.fnormL))
        tot = rz.sum()
        out[f"rz_by_m_{name}"] = [float(rz[:, m == k].sum()/tot) if tot > 0 else 0.0 for k in range(int(m.max())+1)]
        out[f"rz_by_js_{name}"] = [float(x) for x in rz.sum(1)*r1]
        out[f"rz_by_js_frac_{name}"] = [float(x) for x in rz.sum(1)/tot] if tot > 0 else []
        out[f"rz_by_m_abs_{name}"] = [float(rz[:, m == k].sum()*r1) for k in range(int(m.max())+1)]
        # per-component sums for lambda
        out[f"l_by_m_abs_{name}"] = [float(e["L"][:, m == k].sum()*float(cache.fnormL)) for k in range(int(m.max())+1)]
    return out
