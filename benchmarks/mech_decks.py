"""Decks for Stage 6 variants of the committed Solov'ev cases (tilt ``chi``, symmetric data in the full basis).

The writer is the one of ``build_inputs.py`` (same boundary fit, same profile fit, same units), applied to a
``Case`` that is not in ``cases.json``; ``write_deck`` returns the path and the Case.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path


import build_inputs as bi
from analytic import cases, flux, validate


def variant(base, name, lasym=None, **parameter_changes):
    """A Case derived from ``cases()[base]``; ``parameter_changes`` e.g. chi=0.1 (Solov'ev: index 3)."""
    case = cases()[base]
    params = list(case.parameters)
    names = {"chi": 3}
    for key, value in parameter_changes.items():
        params[names[key]] = value
    return replace(case, name=name, parameters=tuple(params), lasym=case.lasym if lasym is None else lasym)


def write_deck(case, path, closure=0, max_m=bi.MAX_M, max_n=bi.MAX_N):
    """INDATA at ``path`` for ``case`` (ns/ftol arrays are the build_inputs defaults; callers override them)."""
    validate(case)
    rows, ntor, geometry_error, symmetry_error = bi.boundary_coefficients(case, max_m=max_m, max_n=max_n,
                                                                           grid=max(bi.FIT_GRID, 4*max_m, 4*max_n))
    polp, poli, polI, profile_errors = bi.fit_profiles(case)
    lines = ["&INDATA", "  LFREEB = F", f"  LASYM = {'T' if case.lasym else 'F'}", f"  NFP = {case.nfp}",
             f"  MPOL = {max_m+1}", f"  NTOR = {ntor}", "  NS_ARRAY = 17 33 65", "  FTOL_ARRAY = 1e-10 1e-12 1e-14",
             "  NITER_ARRAY = 10000 20000 30000", "  NSTEP = 200", "  DELT = 0.5", "  GAMMA = 0", "  PRES_SCALE = 1",
             "  PMASS_TYPE = 'power_series'",
             f"  PHIEDGE = {bi.FIELD_T*bi.LENGTH_M**2*float(flux(case, case.edge)):.17e}",
             "  AM = "+" ".join(f"{x*bi.FIELD_T**2/bi.MU0:.17e}" for x in polp.coef), f"  NCURR = {closure}"]
    if closure == 0:
        lines += ["  PIOTA_TYPE = 'power_series'", "  AI = "+" ".join(f"{x:.17e}" for x in poli.coef)]
    else:
        lines += ["  PCURR_TYPE = 'power_series'", f"  CURTOR = {polI(1.)*bi.FIELD_T*bi.LENGTH_M/bi.MU0:.17e}",
                  "  AC = "+" ".join(f"{x:.17e}" for x in polI.deriv().coef)]
    for m, n, rc, rs, zc, zs in rows:
        for key, value in (("RBC", rc), ("ZBS", zs), ("RBS", rs), ("ZBC", zc)):
            if (case.lasym or key in ("RBC", "ZBS")) and abs(value) > 2e-15:
                lines.append(f"  {key}({int(n)},{int(m)}) = {bi.LENGTH_M*value:.17e}")
    lines.append("/")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines)+"\n")
    return path, dict(boundary_max_error=geometry_error, symmetric_omitted=symmetry_error, profile_errors=profile_errors)
