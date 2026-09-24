"""Check generated INDATA values against VMEX's pinned parser and profile setup.

This checks the specified physical problem before a nonlinear solve. It does
not certify a recovered equilibrium or the native interior field evaluator.
"""
import json
from pathlib import Path
import subprocess
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import vmex
from vmex.core import profiles
from vmex.core.setup import flux_profiles, radial_grids, boundary_from_input
from vmex.core.fourier import mode_table, trig_tables, Resolution

from analytic import ROOT, cases, iota, label_at_s, surface
from build_inputs import enclosed_current, MU0, LENGTH_M, FIELD_T


def source_commit():
    root = Path(vmex.__file__).resolve().parents[1]
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def boundary_error(inp, case, *, count=257):
    rng = np.random.default_rng(944)
    theta, phi = rng.uniform(0, 2*np.pi, (2, count))
    target = surface(case, case.edge, theta, phi)
    m = np.arange(inp.mpol)[None, None, :]
    n = np.arange(-inp.ntor, inp.ntor+1)[None, :, None]
    phase = theta[:, None, None]*m - inp.nfp*phi[:, None, None]*n
    rc, rs, zc, zs = (np.asarray(getattr(inp, key))[None, :, :] for key in
                      ("rbc", "rbs", "zbc", "zbs"))
    radius = np.sum(rc*np.cos(phase)+rs*np.sin(phase), axis=(1, 2))
    height = np.sum(zc*np.cos(phase)+zs*np.sin(phase), axis=(1, 2))
    return float(np.max(np.hypot(radius-LENGTH_M*np.hypot(target[:, 0], target[:, 1]),
                                 height-LENGTH_M*target[:, 2])))


def main():
    pins = {r["name"]: r["pin"] for r in json.loads((ROOT / "sources.json").read_text())["repositories"]}
    commit = source_commit()
    if commit != pins["vmex"]:
        raise SystemExit(f"Imported VMEX commit {commit} differs from sources.json pin")
    manifest = json.loads((ROOT / "inputs/manifest.json").read_text())
    results = []
    start = perf_counter()
    for row in manifest["records"]:
        case = cases()[row["case"]]
        inp = vmex.VmecInput.from_file(ROOT / "inputs" / row["file"])
        if inp.lfreeb or inp.ncurr != row["ncurr"] or inp.nfp != case.nfp or inp.lasym != case.lasym:
            raise AssertionError(f"{row['file']}: metadata changed in VMEX parser")
        geom_err = boundary_error(inp, case)
        s = np.linspace(0, 1, 33)
        labels = label_at_s(case, s)
        pressure_ref = -case.pressure_slope*(case.edge-labels)*FIELD_T**2/MU0
        pressure_vmex = np.asarray(profiles.pressure(inp.pmass_type, inp.am, inp.am_aux_s,
            inp.am_aux_f, s, pres_scale=inp.pres_scale, bloat=inp.bloat,
            spres_ped=inp.spres_ped))
        pressure_err = float(np.max(np.abs(pressure_vmex-pressure_ref)) /
                             max(1.0, np.max(np.abs(pressure_ref))))
        iota_err = None
        current_err = None
        if inp.ncurr == 0:
            iota_vmex = np.asarray(profiles.iota(inp.piota_type, inp.ai, inp.ai_aux_s,
                inp.ai_aux_f, s, bloat=inp.bloat))
            iota_err = float(np.max(np.abs(iota_vmex-iota(case, labels))))
        else:
            current_ref = enclosed_current(case, labels, n=512)*FIELD_T*LENGTH_M/MU0
            shape = np.asarray(profiles.current(inp.pcurr_type, inp.ac, inp.ac_aux_s,
                inp.ac_aux_f, s, bloat=inp.bloat))
            current_vmex = shape/shape[-1]*inp.curtor
            current_err = float(np.max(np.abs(current_vmex-current_ref)) /
                                max(1.0, np.max(np.abs(current_ref))))
        resolution = Resolution(ns=17, mpol=inp.mpol, ntor=inp.ntor,
                                ntheta=2*inp.mpol+1, nzeta=max(2, 4*inp.ntor+1),
                                nfp=inp.nfp, lasym=inp.lasym)
        boundary = boundary_from_input(inp, modes=mode_table(inp.mpol, inp.ntor),
                                       trig=trig_tables(resolution))
        radial = radial_grids(17)
        setup = flux_profiles(inp, radial, r00=float(np.asarray(boundary.r00)),
                              signgs=boundary.signgs, lflip=boundary.lflip)
        expected_psi = -inp.phiedge/(2*np.pi)
        if not np.isclose(float(np.asarray(setup["psi_edge"])), expected_psi, rtol=1e-12):
            raise AssertionError(f"{row['file']}: edge-flux sign or scaling mismatch")
        if boundary.lflip:
            raise AssertionError(f"{row['file']}: unexpected theta flip; sign map needs review")
        if inp.ncurr == 0:
            got = np.asarray(setup["iotaf"])
            expected = iota(case, label_at_s(case, np.asarray(radial.s_full)))
            if np.max(np.abs(got-expected)) > 1e-5:
                raise AssertionError(f"{row['file']}: signed setup iota mismatch")
        else:
            edge_shape = float(np.asarray(profiles.current(inp.pcurr_type, inp.ac,
                inp.ac_aux_s, inp.ac_aux_f, 1.0, bloat=inp.bloat)))
            half_shape = float(np.asarray(profiles.current(inp.pcurr_type, inp.ac,
                inp.ac_aux_s, inp.ac_aux_f, float(np.asarray(radial.s_half)[-1]),
                bloat=inp.bloat)))
            expected = -MU0*inp.curtor/(2*np.pi)*half_shape/edge_shape
            got = float(np.asarray(setup["icurv"])[-1])
            if not np.isclose(got, expected, rtol=1e-10, atol=1e-10):
                raise AssertionError(f"{row['file']}: signed setup current mismatch")
        results.append(dict(file=row["file"], case=case.name, ncurr=inp.ncurr,
                            boundary_max_error_m=geom_err, pressure_relative_error=pressure_err,
                            iota_max_abs_error=iota_err, current_relative_error=current_err,
                            lflip=boundary.lflip, signgs=boundary.signgs,
                            boundary_resolved=geom_err <= 1e-6))
    failed = [r["file"] for r in results if (
        r["boundary_max_error_m"] > 1e-6 or r["pressure_relative_error"] > 1e-8 or
        (r["iota_max_abs_error"] is not None and r["iota_max_abs_error"] > 1e-6) or
        (r["current_relative_error"] is not None and r["current_relative_error"] > 1e-8))]
    record = dict(schema=1, evidence="discrete_consistency", status="failed" if failed else "passed",
                  vmex_commit=commit, parser_and_setup_only=True, vmex_solved=False,
                  seconds=perf_counter()-start, failed_inputs=failed, records=results)
    out = ROOT / "results/inputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vmex_parser.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(f"Checked {len(results)} inputs; unresolved boundaries: " +
          ", ".join(r["file"] for r in results if not r["boundary_resolved"]))
    if failed:
        raise SystemExit("VMEX input conversion checks failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
