"""C6 exterior-operator fixture: MGRID tabulation of an exact source-free field.

B = B0 R0 / R e_phi + a grad(R^2 - 2 Z^2): the potential is harmonic, so the
field is curl- and divergence-free, with a toroidal circulation carried by
the 1/R part (not by a single-valued potential).  The table is built by VMEX's
``tabulate_cartesian_field``, round-tripped through ``write_mgrid``/``read_mgrid``
and evaluated with ``MgridField`` trilinear interpolation on held-out points.
Needs VMEX; skipped in the reference-only CI.
"""
import numpy as np
import pytest

vmex_mgrid = pytest.importorskip("vmex.core.mgrid")

B0, R0, A = 1.3, 1.0, 0.07
RMIN, RMAX, ZMIN, ZMAX, NFP = 0.6, 1.4, -0.4, 0.4, 2


def exact_cylindrical(r, z):
    return 2*A*r, B0*R0/r, -4*A*z


def exact_cartesian(xyz):
    x, y, z = np.asarray(xyz).T
    r, phi = np.hypot(x, y), np.arctan2(y, x)
    br, bp, bz = exact_cylindrical(r, z)
    return np.column_stack([br*np.cos(phi)-bp*np.sin(phi), br*np.sin(phi)+bp*np.cos(phi), bz])


def _field(n, tmp_path):
    data = vmex_mgrid.tabulate_cartesian_field(exact_cartesian, rmin=RMIN, rmax=RMAX, zmin=ZMIN, zmax=ZMAX,
                                               ir=n, jz=n, kp=8, nfp=NFP, label="exact_harmonic")
    path = tmp_path/f"mgrid_{n}.nc"
    vmex_mgrid.write_mgrid(path, data)
    back = vmex_mgrid.read_mgrid(path)
    for name in ("br", "bp", "bz"):
        np.testing.assert_array_equal(np.asarray(getattr(back, name)), np.asarray(getattr(data, name)))
    return vmex_mgrid.MgridField.from_mgrid_data(back)


def test_mgrid_interpolation_of_exact_harmonic_field_is_second_order(tmp_path):
    rng = np.random.default_rng(7)
    r = rng.uniform(RMIN+0.05, RMAX-0.05, 400)
    z = rng.uniform(ZMIN+0.05, ZMAX-0.05, 400)
    phi = rng.uniform(0, 2*np.pi, 400)
    exact = np.column_stack(exact_cylindrical(r, z))
    errors = []
    for n in (17, 33, 65):
        br, bp, bz = _field(n, tmp_path)(r, phi, z)
        errors.append(np.linalg.norm(np.column_stack([br, bp, bz])-exact)/np.linalg.norm(exact))
    rates = np.log2(np.asarray(errors[:-1])/np.asarray(errors[1:]))
    assert errors[-1] < 1e-4
    assert np.all(rates > 1.8), (errors, rates)


def test_exact_fixture_is_source_free():
    # Independent check of the oracle itself: div B = 0 and curl B = 0 (R > 0).
    h = 1e-5
    r, z = 1.1, 0.13
    br = lambda rr, zz: exact_cylindrical(rr, zz)[0]  # noqa: E731
    bz = lambda rr, zz: exact_cylindrical(rr, zz)[2]  # noqa: E731
    div = (((r+h)*br(r+h, z)-(r-h)*br(r-h, z))/(2*h))/r + (bz(r, z+h)-bz(r, z-h))/(2*h)
    curl_phi = (br(r, z+h)-br(r, z-h))/(2*h) - (bz(r+h, z)-bz(r-h, z))/(2*h)
    assert abs(div) < 1e-9 and abs(curl_phi) < 1e-9
