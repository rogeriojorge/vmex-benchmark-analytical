"""Stage 6 diagnostics: the linear-algebra and bookkeeping pieces are checked without a VMEC solve; one small VMEX pass
checks the constraint split (``importorskip``).
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"benchmarks"))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import certify_projection as cp  # noqa: E402
import mech_solve as ms  # noqa: E402
from analytic import cases  # noqa: E402


def test_evolve_mask_counts_match_the_dense_control_of_the_sensitivity_track():
    import mech_linear as ml
    ns, m = 17, np.arange(13)
    mask = ml.evolve_mask(ns, m, ("R_cos", "Z_sin", "L_sin"))
    # R, Z: m = 0 on shells 0..ns-2, m >= 1 on 1..ns-2; lambda: shells 1..ns-1 for every m
    assert mask.sum() == 2*((ns-1)+12*(ns-2)) + 13*(ns-1)
    # structurally empty parities (Z m=0 sine, lambda m=0 sine) are removed numerically: 568 active dofs at NS = 17
    assert 2*((ns-1)+12*(ns-2)) + 12*(ns-1) - (ns-1) == 568   # R m0..12, Z m1..12, lambda m1..12
    assert not mask[(ns-1)*13:(ns)*13].any()   # R edge row is never evolved


def test_solve_active_solves_the_square_active_block_and_refuses_a_non_square_one():
    import mech_linear as ml
    rng = np.random.default_rng(3)
    n = 8
    A = rng.standard_normal((n, n)) + 4*np.eye(n)
    J = np.zeros((n+2, n+2))
    J[:n, :n] = A          # two structurally empty dofs / equations at the end
    mask = np.ones(n+2, bool)
    rhs = rng.standard_normal(n+2)
    rhs[n:] = 0
    (x,), info, rows, cols = ml.solve_active(J, mask, [rhs])
    assert np.allclose(J @ x, -rhs, atol=1e-12) and info["active_cols"] == n and info["cond"] > 1
    assert np.all(x[n:] == 0)
    J2 = J.copy()
    J2[0, n] = 1.0         # a dof that acts on the force (non-empty column) without an equation of its own (empty row)
    with pytest.raises(RuntimeError, match="not square"):
        ml.solve_active(J2, mask, [rhs])


def test_the_linear_split_is_exact_because_the_solve_is_linear_in_the_right_hand_side():
    import mech_linear as ml
    rng = np.random.default_rng(5)
    n = 12
    J = rng.standard_normal((n, n)) + 3*np.eye(n)
    g_mhd, g_con = rng.standard_normal(n), rng.standard_normal(n)
    mask = np.ones(n, bool)
    axis = np.arange(n) < 4
    (total, con, mhd, ax, bulk), *_ = ml.solve_active(J, mask, [g_mhd+g_con, g_con, g_mhd, np.where(axis, g_mhd, 0), np.where(~axis, g_mhd, 0)])
    assert np.allclose(total, con+mhd, atol=1e-13) and np.allclose(mhd, ax+bulk, atol=1e-13)


def test_pack_unpack_round_trip_and_state_distance():
    import mech_linear as ml
    from vmex.core.solver import SpectralState
    rng = np.random.default_rng(1)
    ns, mn = 5, 4
    arrays = {k: rng.standard_normal((ns, mn)) for k in ml.CHANNELS_ALL}
    state = SpectralState(**arrays)
    ch = ("R_cos", "Z_sin", "L_sin")
    back = ml.unpack(ml.pack(state, ch), state, ch)
    for k in ch:
        assert np.array_equal(np.asarray(getattr(back, k)), arrays[k])
    assert not np.any(np.asarray(back.R_sin))      # channels outside the packed set come back zero, not stale

    class Rt:                                       # state_distance only reads the mode table
        class modes:
            m = np.array([0, 1, 1, 2])
    other = SpectralState(**{k: v+0.1*(k == "R_cos") for k, v in arrays.items()})
    d = ms.state_distance(other, state, Rt)
    assert d["R_cos"]["abs"] == pytest.approx(0.1*np.sqrt(ns*mn)) and d["Z_cos"]["abs"] == 0.0
    assert d["R_cos"]["by_m_abs"][1] == pytest.approx(0.1*np.sqrt(ns*2))


class _Perfect:
    """VmecInteriorField stand-in: the exact field scaled by (1 + e(s)) with e = 1e-3 inside the first radial cell only."""

    def __init__(self, case, ns, eps):
        from analytic import field
        self.case, self.ns, self.eps, self.field = case, ns, eps, field

    def B(self, p):
        b = jax.vmap(lambda x: self.field(self.case, x)[0])(p)
        s = jax.vmap(lambda x: self.field(self.case, x)[1])(p)/self.case.edge
        first = (s < 1.0/(self.ns-1)).astype(b.dtype)[:, None]   # label ~ s: first cell
        return b*(1+self.eps*first)

    def gradB(self, p):
        return jax.vmap(jax.jacfwd(lambda x: self.B(x[None])[0]))(p)


def test_cell_profiles_localise_an_error_to_its_cell():
    case = cases()["solovev_symmetric"]
    ns = 6
    rule = cp.volume_rule(case, "m3", ns)
    prof = ms.cell_profiles(_Perfect(case, ns, 1e-3), rule, ns)
    b = np.array(prof["B_local"])
    assert b[0] > 1e-4 and np.all(b[2:] < 1e-12)      # the perturbation lives in cell 0 (cell 1 may share boundary points)
    assert prof["cell"] == list(range(ns-1)) and len(prof["s_mid"]) == ns-1


def test_cell_profiles_of_the_exact_field_vanish_in_every_cell():
    case = cases()["solovev_asymmetric"]
    ns = 5
    prof = ms.cell_profiles(_Perfect(case, ns, 0.0), cp.volume_rule(case, "m3", ns), ns)
    assert max(prof["B_local"]) < 1e-13 and max(prof["J_local"]) < 1e-10


def test_the_constraint_split_of_an_exact_seed_vanishes_for_the_exact_baseline_and_for_tcon0_zero():
    pytest.importorskip("vmex")
    import mech_core as M
    inp, rt, seed, _ = M.exact_seed("solovev_symmetric", 9, 1.0, mpol=8)
    vmec = M.parts_report(seed, M.with_baseline(rt, seed, "vmec"))
    exact = M.parts_report(seed, M.with_baseline(rt, seed, "exact"))
    # F = F_MHD + F_con at a fixed state: the default constraint force is the bulk of the residual of the exact state,
    # the exact baseline removes it to roundoff, and the MHD part does not depend on the baseline
    assert vmec["fsq_rz_con"] > 100*vmec["fsq_rz_mhd"]
    assert exact["fsq_rz_con"] < 1e-12*exact["fsq_rz_mhd"]
    assert exact["fsq_rz_mhd"] == pytest.approx(vmec["fsq_rz_mhd"], rel=1e-9)
    assert vmec["fsq_rz_full"] == pytest.approx(vmec["fsqr"] + vmec["fsqz"], rel=1e-6)
    assert vmec["fsqr_off"] == pytest.approx(exact["fsqr"], rel=1e-9)
    # the constraint acts on m >= 2 only (m(m-1) weight), none on m = 0, 1
    assert vmec["rz_by_m_abs_con"][0] == 0.0 and vmec["rz_by_m_abs_con"][1] < 1e-6*sum(vmec["rz_by_m_abs_con"])


def test_vmec2000_screen_parser_reads_the_first_iteration_row():
    import mech_vmec2000_iter1 as mv
    text = """
  NS =   17 NO. FOURIER MODES =   13 FTOLV =  1.000E-30 NITER =      2
  PROCESSOR COUNT - RADIAL:   1

  ITER    FSQR      FSQZ      FSQL    RAX(v=0)      DELT       WMHD
    1  7.60E-05  7.07E-04  4.00E-04  1.00E+00  5.00E-01  1.0000E-01
    2  1.0E-05   1.0E-04   1.0E-04   1.00E+00  5.00E-01  1.0000E-01
"""
    assert mv.parse_iter1(text) == [1, 7.6e-05, 7.07e-04, 4.0e-04]
    assert mv.parse_iter1("no table here") is None


def test_variational_row_masks_select_only_the_first_shell_m1_rz_dofs():
    import mech_variational as mvar
    ns, m = 9, np.arange(4)
    ch = ("R_cos", "Z_sin", "L_sin")
    mask = mvar.row_mask("m1_js1", ns, m, ch).reshape(3, ns, 4)
    assert mask[0, 1, 1] and mask[1, 1, 1] and mask.sum() == 2        # lambda never replaced, nothing else
    wide = mvar.row_mask("m1_js<=3", ns, m, ch).reshape(3, ns, 4)
    assert wide[0, 1:4, 1].all() and wide.sum() == 2*3
    assert mvar.row_mask("none", ns, m, ch).sum() == 0
    with pytest.raises(ValueError):
        mvar.row_mask("bogus", ns, m, ch)


def test_mirror_map_flips_exactly_the_odd_channels():
    import mech_symtests as st
    from vmex.core.solver import SpectralState
    a = {k: np.full((2, 2), i+1.0) for i, k in enumerate(("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin"))}
    b = st.mirror(SpectralState(**a))
    assert np.all(np.asarray(b.R_cos) == 1) and np.all(np.asarray(b.R_sin) == -2) and np.all(np.asarray(b.Z_sin) == 4)
    assert st.rel_violation(a, {k: -v if k in st.ODD else v for k, v in a.items()}, sign=True) == {k: 0.0 for k in a}


def test_polar_rotation_makes_the_m1_pair_polar_and_is_a_pure_reparametrisation():
    pytest.importorskip("vmex")
    import mech_core as M
    from analytic import cases
    from project_integer_vmex import project
    from vmex.core.fourier import mode_table, trig_tables
    from vmex.core.residuals import m1_constrained_to_physical
    from vmex.core.solver import resolution_from_input
    from vmex.core.transforms import physical_to_internal_scale
    ns = 9
    inp = M.case_deck("solovev_asymmetric", ns, mpol=8)
    state, _ = project(inp, cases()["solovev_asymmetric"], ns)
    inp2, rotated, delta = M.polar_rotation(state, inp, ns)
    res = resolution_from_input(inp, ns=ns)
    modes = mode_table(res.mpol, res.ntor)
    scale = np.asarray(physical_to_internal_scale(modes, trig_tables(res)))
    m = np.asarray(modes.m)

    def physical(st):
        return [np.asarray(a) for a in m1_constrained_to_physical(st.R_cos, st.Z_sin, st.R_sin, st.Z_cos, modes=modes, lthreed=False,
                                                                  lasym=True, lconm1=True)]
    Rc, Zs, Rs, Zc = physical(rotated)
    i1 = int(np.where(m == 1)[0][0])
    assert np.max(np.abs(Rs[1:, i1]-Zc[1:, i1])) < 1e-13                      # polar on every shell
    assert abs(delta[-1]) > 0.1 and delta[0] == 0.0                           # the asymmetric case is rotated by VMEC's shift
    theta = np.linspace(0, 2*np.pi, 7)

    def evaluate(st, shift):
        a = physical(st)
        Rcm, Zsm, Rsm, Zcm = (x/scale for x in a)
        th = theta[:, None]+shift
        R = np.sum(Rcm[-2]*np.cos(m*th)+Rsm[-2]*np.sin(m*th), -1)
        Z = np.sum(Zcm[-2]*np.cos(m*th)+Zsm[-2]*np.sin(m*th), -1)
        return R, Z
    a0 = [np.asarray(x) for x in m1_constrained_to_physical(state.R_cos, state.Z_sin, state.R_sin, state.Z_cos, modes=modes, lthreed=False, lasym=True, lconm1=True)]
    Rcm, _, Rsm, _ = (x/scale for x in a0)
    th = theta[:, None]+delta[-2]
    R_old = np.sum(Rcm[-2]*np.cos(m*th)+Rsm[-2]*np.sin(m*th), -1)    # original surface at theta' + delta
    R_new, _ = evaluate(rotated, 0.0)
    assert np.max(np.abs(R_new-R_old)) < 1e-12                              # same curve, new angle
    # the deck boundary was rewritten in the new angle: edge row == inp2 arrays
    assert np.allclose(np.asarray(inp2.rbc)[0], Rc[-1]/scale, atol=1e-14)
