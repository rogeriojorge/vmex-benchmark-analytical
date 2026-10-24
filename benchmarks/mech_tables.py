"""Stage 6 tables (``results/mechanism/TABLES.md``) from the committed records.

Reads ``results/mechanism/{runs,linear,logs}/...`` and the single-file records, writes Markdown only; no solver
import, no recomputation.  Every number in the report comes from a record file named in the table header.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MECH = ROOT/"results/mechanism"


def load(path):
    return json.loads(Path(path).read_text())


def fmt(x, digits=2):
    if x is None:
        return "-"
    return f"{x:.{digits}e}" if isinstance(x, float) else str(x)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|"*len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def runs(pattern):
    return {Path(p).parent.name: load(p) for p in sorted(glob.glob(str(MECH/"runs"/pattern/"record.json")))}


def solved_table(case, variants, nss=(17, 33, 65, 129), suffix=""):
    rows = []
    for ns in nss:
        row = [ns]
        for tcon0, baseline in variants:
            key = f"{case}-ns{ns}-tcon{tcon0:g}-{baseline}{suffix}"
            p = MECH/"runs"/key/"record.json"
            if not p.exists():
                row += ["-"]*4
                continue
            r = load(p)
            h = r["headline"]
            row += [fmt(h["B"]), fmt(h["J"]), fmt(h["force"]), r["solve"]["iterations"]]
        rows.append(row)
    header = ["NS"]
    for tcon0, baseline in variants:
        tag = f"tcon0={tcon0:g}, {baseline}"
        header += [f"B ({tag})", "J", "force", "iterations"]
    return table(header, rows)


def iteration1_table(case, nss=(17, 33, 65, 129)):
    rows = []
    for ns in nss:
        p = MECH/"runs"/f"{case}-ns{ns}-tcon1-vmec"/"record.json"
        if not p.exists():
            continue
        it = load(p)["iteration1"]
        con_m = np.array(it["rz_by_m_abs_con"])
        top = int(con_m.argmax())
        rows.append([ns, fmt(it["fsq_rz_full"]), fmt(it["fsq_rz_mhd"]), fmt(it["fsq_rz_con"]), f"{it['fsq_rz_con']/it['fsq_rz_full']:.4f}",
                     f"m={top} ({con_m[top]/con_m.sum():.2f})", fmt(it["fsq_rz_con"]/it["fsq_rz_mhd"])])
    return table(["NS", "fsqr+fsqz (full)", "F_MHD only", "F_con part", "con share", "dominant m of F_con", "con/MHD"], rows)


def linear_table(case, nss=(17, 33, 65, 129)):
    rows = []
    for ns in nss:
        for baseline in ("vmec", "exact"):
            p = MECH/"linear"/f"{case}-ns{ns}-tcon1-{baseline}.json"
            if not p.exists():
                continue
            lin = load(p)
            sc = lin["scores"]
            run = MECH/"runs"/f"{case}-ns{ns}-tcon1-{baseline}"/"record.json"
            actual = load(run)["headline"]["B"] if run.exists() else None
            rows.append([ns, baseline, fmt(lin["solve"]["cond"]), fmt(sc["total"]["B"]), fmt(actual), fmt(sc["con"]["B"]), fmt(sc["mhd"]["B"]),
                         fmt(sc["mhd_axis"]["B"]), fmt(sc["mhd_bulk"]["B"]), fmt(lin["linearity"]["total_vs_con_plus_mhd"])])
    return table(["NS", "baseline", "cond(J)", "B predicted x*+delta", "B solved", "B of delta_con", "B of delta_mhd", "delta_mhd axis shells (<3)",
                  "delta_mhd bulk", "|delta - con - mhd|/|delta|"], rows)


def dose_table(case, nss=(65, 129), tcons=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0)):
    rows = []
    for ns in nss:
        for baseline in ("vmec", "exact"):
            row = [ns, baseline]
            for t in tcons:
                p = MECH/"runs"/f"{case}-ns{ns}-tcon{t:g}-{baseline}"/"record.json"
                if not p.exists() and t == 0.0:
                    p = MECH/"runs"/f"{case}-ns{ns}-tcon0-vmec"/"record.json"
                row.append(fmt(load(p)["headline"]["B"]) if p.exists() else "-")
            rows.append(row)
    return table(["NS", "baseline"] + [f"B at tcon0={t:g}" for t in tcons], rows)


def dose_layer_table(case, nss=(65, 129), tcons=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0)):
    rows = []
    for ns in nss:
        for baseline in ("vmec", "exact"):
            row = [ns, baseline]
            for t in tcons:
                p = MECH/"runs"/f"{case}-ns{ns}-tcon{t:g}-{baseline}"/"record.json"
                if not p.exists() and t == 0.0:
                    p = MECH/"runs"/f"{case}-ns{ns}-tcon0-vmec"/"record.json"
                if p.exists():
                    J = np.array(load(p)["profile_solved"]["J_local"], float)
                    row.append(f"{J.max():.1e} (cell {int(J.argmax())})")
                else:
                    row.append("-")
            rows.append(row)
    return table(["NS", "baseline"] + [f"peak J error at tcon0={t:g}" for t in tcons], rows)


def layer_table(case, variants=(("tcon1", "vmec"), ("tcon1", "exact"), ("tcon0", "vmec")), nss=(17, 33, 65, 129)):
    rows = []
    for ns in nss:
        for tc, baseline in variants:
            p = MECH/"runs"/f"{case}-ns{ns}-{tc}-{baseline}"/"record.json"
            if not p.exists():
                continue
            r = load(p)
            J = np.array(r["profile_solved"]["J_local"], float)
            seed = np.array(r["profile_seed"]["J_local"], float)
            bulk = slice(int(0.3*(ns-1)), None)
            rows.append([ns, f"{tc}, {baseline}", " ".join(f"{x:.1e}" for x in J[:4]), f"{J.max():.1e} (cell {int(J.argmax())})",
                         f"{np.median(J[bulk]):.1e}", f"{seed[0]:.1e}", fmt(r["regions"]["axis"]["B"]), fmt(r["regions"]["bulk"]["B"])])
    return table(["NS", "variant", "J error per cell, cells 0-3", "peak J error (cell)", "bulk median (s>0.3)", "seed J error cell 0",
                  "B error axis (s<0.02)", "B error bulk"], rows)


def asym_table(name="asym_residual.json"):
    p = MECH/name
    if not p.exists():
        return "(not run)"
    rows = []
    for r in load(p):
        e = r["tcon0_0"]
        top = e["top_modes"][0]
        z0 = e["Z_cos_by_m"][0]
        rows.append([r["label"], r["ns"], r["mpol_used"], r["ntheta_used"], fmt(e["fsqr"]), fmt(e["fsqz"]), fmt(e["fsql"]), fmt(e["even_sum"]), fmt(e["odd_sum"]),
                     f"{top['channel']} m={top['m']} ({top['value']:.1e})", fmt(z0)])
    return table(["configuration", "NS", "mpol", "ntheta", "fsqr", "fsqz", "fsql", "even-parity channels", "odd-parity channels", "largest single mode", "Z_cos m=0 share"], rows)


def named_asym_table(name="asym_residual_named.json"):
    p = MECH/name
    if not p.exists():
        return "(not run)"
    rows = []
    for r in load(p):
        for key in ("tcon0_0", "tcon0_1"):
            if key not in r:
                continue
            e = r[key]
            top = e["top_modes"][0]
            rows.append([r["label"], key.replace("tcon0_", "tcon0 = "), fmt(e["fsqr"]), fmt(e["fsqz"]), fmt(e["fsql"]), fmt(e["even_sum"]), fmt(e["odd_sum"]),
                         f"{top['channel']} m={top['m']} n={top['n']} ({top['value']:.1e})"])
    return table(["case", "constraint", "fsqr", "fsqz", "fsql", "even channels", "odd channels", "largest mode"], rows)


def crosscode_table():
    rows = []
    for case in ("solovev_symmetric", "solovev_asymmetric"):
        p = MECH/f"crosscode_{case}.json"
        if not p.exists():
            continue
        for r in load(p):
            ch = r["channels"]
            rows.append([case, r["ns"], r["code"], fmt(max(r["wout_fsq"])), fmt(r["vmex_fsq"][0]), fmt(r["vmex_fsq"][1]), fmt(r["vmex_fsq"][2]),
                         fmt(ch["even_sum"]), fmt(ch["odd_sum"])])
    return table(["case", "NS", "state written by", "its own max fsq", "VMEX fsqr", "VMEX fsqz", "VMEX fsql", "even channels", "odd channels"], rows) if rows else "(not run)"


def vmec2000_iter1_table():
    rows = []
    for sub in ("vmec2000_iter1", "vmec2000_iter1_asym"):
        p = MECH/sub/"summary.json"
        if not p.exists():
            continue
        for r in load(p):
            v = r["vmec2000_iter1"]
            rows.append([r["case"], r["ns"], f"{r['tcon0']:g}", " ".join(f"{x:.3e}" for x in v[1:]) if v else "-", " ".join(f"{x:.3e}" for x in r["vmex_iter1"]),
                         " ".join(f"{x:.3f}" for x in r["ratio_vmec2000_over_vmex"]) if "ratio_vmec2000_over_vmex" in r else "-",
                         fmt(max(r["roundtrip_max_abs"].values()))])
    return table(["case", "NS", "tcon0", "VMEC2000 iteration-1 fsqr fsqz fsql", "VMEX same seed", "ratio", "WOUT round trip max |dx|"], rows) if rows else "(not run)"


def gradcheck_table(name):
    p = MECH/f"gradcheck_{name}_ns17.json"
    if not p.exists():
        return "(not run)"
    data = load(p)
    rows = []
    by = {}
    for r in data["rows"]:
        by.setdefault((r["channel"], r["m"]), []).append(r)
    for (ch, m), rs in by.items():
        pert = [r for r in rs if r["state"] != "exact"]
        ex = [r for r in rs if r["state"] == "exact"]
        def ratio(rr, js):
            xs = [abs(r["ratio"][1]) for r in rr if r["js"] == js and r["ratio"][1] is not None]   # |c|: the overall sign of F against W flips with the sign of the orientation sum (abs in wb) in strongly perturbed states
            return "%.4g" % (np.median(xs)) if xs else "-"
        e1 = [r for r in ex if r["js"] == 1][0]
        e2 = [r for r in ex if r["js"] == 2][0]
        rows.append([f"{ch} m={m}", ratio(pert, 1), ratio(pert, 2), ratio(pert, 5), fmt(e1["force"]), fmt(e1["fd_grad"][1]), fmt(e2["force"]), fmt(e2["fd_grad"][1])])
    return table(["dof", "|dW/dx / F|, js=1 (perturbed states, median)", "js=2", "js=5", "F exact state js=1", "dW/dx exact js=1", "F exact js=2", "dW/dx exact js=2"], rows)


def defect_table(case, tag, nss=(33, 65, 129)):
    rows = []
    for ns in nss:
        row = [ns]
        for kind in tag:
            matches = sorted(glob.glob(str(MECH/"runs"/f"{case}-ns{ns}-*defect_{kind}*"/"record.json")))
            if not matches:
                row += ["-", "-"]
                continue
            r = load(matches[0])
            row += [fmt(r["headline"]["B"]), fmt(r["headline"]["J"])]
        rows.append(row)
    header = ["NS"]
    for kind in tag:
        header += [f"B ({kind})", "J"]
    return table(header, rows)


def variational_table():
    rows = []
    for p in sorted(glob.glob(str(MECH/"variational"/"*.json")), key=lambda q: (load(q)["ns"], load(q)["rows"])):
        r = load(p)
        if r["rows"] == "m1_js<=3" and False:
            continue
        sc = r["scores"]
        rows.append([r["ns"], r["rows"], r["n_replaced"], "%+.1f" % r["fitted_scale"], fmt(sc["B"]), fmt(sc["J"]), fmt(sc["axis_J"]),
                     f"{max(r['profile']['J_local']):.1e}", f"{np.argmax(r['profile']['J_local'])}", fmt(r["root_distance"])])
    return table(["NS", "rows replaced by scale x dW/dx", "n rows", "fitted scale", "B", "J", "J (s<0.02)", "peak J cell error", "cell", "|root - seed|"], rows) if rows else "(not run)"


def symtests_table():
    p = MECH/"symmetry_tests.json"
    if not p.exists():
        return "(not run)"
    r = load(p)
    rows = [["mirror covariance, perturbed state", " ".join(f"{k}:{v:.1e}" for k, v in r["mirror_covariance_perturbed"].items())],
            ["vertical translation, perturbed state", " ".join(f"{k}:{v:.1e}" for k, v in r["vertical_translation_perturbed"].items())],
            ["exact states of +chi and -chi (mirror images)", " ".join(f"{k}:{v:.1e}" for k, v in r["mirror_covariance_exact_pm_chi"].items())],
            ["Z_cos m=0 force, exact +chi, shells 1-4", " ".join("%.4e" % x for x in r["Z_cos_m0_force_exact"]["plus"])],
            ["Z_cos m=0 force, exact -chi", " ".join("%.4e" % x for x in r["Z_cos_m0_force_exact"]["minus"])],
            ["R_cos m=0 force, exact +chi", " ".join("%.4e" % x for x in r["R_cos_m0_force_exact"]["plus"])],
            ["R_cos m=0 force, exact -chi", " ".join("%.4e" % x for x in r["R_cos_m0_force_exact"]["minus"])]]
    return table(["test (NS %d, chi = %g)" % (r["ns"], r["chi"]), "largest violation / largest force per channel, or the values"], rows)


def lambda_table():
    rows = []
    for p in sorted(glob.glob(str(MECH/"lambda"/"*.json"))):
        r = load(p)
        a, b = r["seed_lambda"], r["resolved_lambda"]
        rows.append([r["ns"], " ".join("%.2e" % a[k] for k in ("fsqr", "fsqz", "fsql")), " ".join("%.2e" % b[k] for k in ("fsqr", "fsqz", "fsql")),
                     fmt(r["lambda_change"]["L_cos"]["change_norm"]), fmt(r["lambda_change"]["L_cos"]["seed_norm"]),
                     fmt(r["lambda_change"]["L_sin"]["change_norm"]), fmt(r["lambda_change"]["L_sin"]["seed_norm"]),
                     fmt(r["seed_lambda_score"]["B"]), fmt(r["resolved_lambda_score"]["B"])])
    return table(["NS", "fsqr fsqz fsql, seed lambda", "after lambda re-solve", "|dL_cos|", "|L_cos| seed", "|dL_sin|", "|L_sin| seed", "B error seed lambda", "B error re-solved lambda"], rows) if rows else "(not run)"


def fieldcheck_table():
    rows = []
    for p in sorted(glob.glob(str(MECH/"fieldcheck"/"*.json")), key=lambda q: (load(q)["case"], load(q).get("polar", False), load(q)["ns"])):
        r = load(p)
        mid = r["rows"][len(r["rows"])//2]
        rows.append([r["case"], "polar" if r.get("polar") else "own angle", r["ns"], fmt(r["max_theta_part_diff"]), fmt(mid["theta_part_rms_exact"]), fmt(r["max_ratio_spread"]),
                     "%.6f .. %.6f" % tuple(r["mean_ratio_range"])])
    return table(["case", "seed angle", "NS", "max |theta-part(B^2) force path - exact|", "theta-part rms of exact (mid shell)", "max spread of the ratio", "mean ratio range"], rows) if rows else "(not run)"


def eigen_table():
    rows = []
    for p in sorted(glob.glob(str(MECH/"eigen"/"*.json"))):
        r = load(p)
        rows.append([r["case"], r["ns"], r["baseline"], f"{r['tcon0']:g}", "polar" if r.get("polar") else "own angle",
                     "-" if r.get("ai0_override") is None else f"{r['ai0_override']:g}", r["n_active"], r["n_positive_real"],
                     " ".join("%.2e" % x[0] for x in r["largest_real"][:3]), fmt(r["max_abs_real"])])
    return table(["case", "NS", "baseline", "tcon0", "seed angle", "iota override", "active dofs", "eigenvalues with Re > 0", "three largest real parts", "max |Re|"], rows) if rows else "(not run)"


def lopez_table():
    rows = []
    for p in sorted(glob.glob(str(MECH/"lopez"/"*.json"))):
        r = load(p)
        sc = r.get("score", {})
        rows.append([r["baseline"], r["ns"], f"{r['tcon0']:g}", fmt(r["iteration1"]["fsq_rz_full"]), fmt(r["iteration1"]["fsq_rz_con"]), fmt(r["iteration1"]["fsq_rz_mhd"]),
                     r["iterations"], fmt(r["fsq_min"]), fmt(r["last_third"]["min"]), fmt(r["last_third"]["max"]), "%.1f" % r["last_third"]["log10_spread"],
                     fmt(sc.get("B_vol")), fmt(sc.get("axis"))])
    return table(["baseline", "NS", "tcon0", "iter-1 rz (full)", "F_con part", "F_MHD part", "iterations", "fsq min", "fsq last third min", "max", "decades", "B vol", "axis offset"], rows) if rows else "(not run)"


def hard_cases_table():
    rows = []
    for p in sorted(glob.glob(str(MECH/"runs"/"*"/"record.json")), key=lambda q: (load(q)["case"], load(q)["ns"], load(q).get("polar_seed", False), load(q)["baseline"])):
        r = load(p)
        if r["case"] in ("solovev_symmetric", "solovev_asymmetric") or "defect" in r["run_id"]:
            continue
        hist = r.get("fsq_history", [])
        f = np.array([v for _, v in hist]) if hist else np.array([np.nan])
        third = f[2*len(f)//3:]
        h = r["headline"]
        rows.append([r["case"], "polar" if r.get("polar_seed") else "own angle", r["ns"], r["baseline"], f"{r['tcon0']:g}", r["solve"]["iterations"], fmt(r["iteration1"]["fsq_rz_full"]) if "iteration1" in r else "-",
                     fmt(r["iteration1"]["fsq_rz_con"]) if "iteration1" in r else "-", fmt(r["solve"]["fsq_final"][0]), fmt(float(f.min())),
                     "%.1f" % np.log10(third.max()/third.min()), fmt(h["B"]), fmt(h["J"]), fmt(h["force"])])
    return table(["case", "seed angle", "NS", "baseline", "tcon0", "iterations", "iter-1 rz", "con part", "final fsqr", "fsq min over run", "sampled last-third decades", "B", "J", "force"], rows) if rows else "(not run)"


def main(argv=None):
    sections = ["# Stage 6 mechanism tables\n",
                "Generated by `benchmarks/mech_tables.py` from `results/mechanism/` records; VMEX `caf09ee7`, JAX x64, CPU.\n"]
    case = "solovev_symmetric"
    sections += ["## Symmetric Solov'ev: solved-state errors (exact seed, 1e-14, relative L2 on the exact chart)\n",
                 solved_table(case, [(1.0, "vmec"), (1.0, "exact"), (0.0, "vmec")]), "",
                 "## Iteration-1 residual of the exact seed split into F_MHD and F_con (default constraint)\n", iteration1_table(case), "",
                 "## Linear attribution: x* + delta against the nonlinear solve\n", linear_table(case), "",
                 "## Dose response in tcon0 (B error)\n", dose_table(case), "",
                 "## Dose response in tcon0 (peak per-cell J error of the near-axis layer)\n", dose_layer_table(case), "",
                 "## Near-axis layer in J (per radial cell, local relative L2)\n", layer_table(case), "",
                 "## Defect-corrected solves, symmetric Solov'ev, exact baseline (a constant forcing cancels that part of F_MHD(x*))\n",
                 defect_table(case, ["mhdaxis2", "mhdaxis4", "mhdbulk2"]), "",
                 "## Force versus energy gradient, symmetric Solov'ev NS 17 (|c| = |dW/dx / F| is 1/(NS-1) and state independent where the force is the gradient)\n", gradcheck_table("sym"), "",
                 "## Force versus energy gradient, asymmetric Solov'ev NS 17\n", gradcheck_table("asym"), "",
                 "## Asymmetric Solov'ev: solved-state errors (exact seed, 1e-14)\n",
                 solved_table("solovev_asymmetric", [(1.0, "vmec"), (1.0, "exact"), (0.0, "vmec")]), "",
                 "## Asymmetric Solov'ev with the exact seed re-parametrised into the polar gauge: solved-state errors\n",
                 solved_table("solovev_asymmetric", [(1.0, "vmec"), (1.0, "exact"), (0.0, "vmec")], suffix="-polar"), "",
                 "## Asymmetric iteration-1 residual at tcon0 = 0: scans (Solov'ev, tilt chi = 0.3 unless stated), exact seed in its own angle\n", asym_table(), "",
                 "## The same scans with the seed re-parametrised into the m = 1 polar gauge (`polar_rotation`)\n", asym_table("asym_residual_polar.json"), "",
                 "## Asymmetric named cases (committed decks), iteration-1 residual, own angle\n", named_asym_table(), "",
                 "## Asymmetric named cases, polar-gauge seed\n", named_asym_table("asym_residual_named_polar.json"), "",
                 "## VMEX operator evaluated at WOUT states written by VMEC2000 and by VMEX\n", crosscode_table(), "",
                 "## Spectrum of the preconditioned force Jacobian at the exact seed\n", eigen_table(), "",
                 "## Variational first shell: force on (js = 1, m = 1) R/Z dofs replaced by the exact energy gradient (Newton root, symmetric Solov'ev)\n", variational_table(), "",
                 "## Symmetry covariance of the force operator (asymmetric Solov'ev)\n", symtests_table(), "",
                 "## Lambda re-solved at fixed exact R, Z (asymmetric Solov'ev, tcon0 = 0)\n", lambda_table(), "",
                 "## Half-mesh |B|^2 of the force path against the exact |B|^2\n", fieldcheck_table(), "",
                 "## Lopez D polar-gauge seed, baseline of the exact state\n", lopez_table(), "",
                 "## Integer 3-D and sheared A exact-seed solves with the exact baseline\n", hard_cases_table(), ""]
    out = MECH/"TABLES.md"
    out.write_text("\n".join(sections)+"\n")
    print(out)


if __name__ == "__main__":
    main()
