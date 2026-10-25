"""Write results/index.json: every paper claim and figure panel mapped to exact record files.

The index is the one declared input of ``paper_figures.py``.  Each *record entry* lists the files a figure or a
claim reads (path and sha256), the producing scripts and command, the code identities (declared here and checked
against what the records themselves report), and, for large families, a digest over the per-run member files.
Raw states (WOUT, .h5, .npz) are not hashed here; ``raw_state_manifest.py`` lists them with office path and sha256.

    python benchmarks/build_index.py            # rewrite results/index.json
    python benchmarks/build_index.py --check    # fail if the file on disk differs from a fresh build

Nothing here recomputes a result; it only hashes what is already in ``results/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT/"results/index.json"
FIG_DIR = "figures/paper"

# ---- code identities --------------------------------------------------------------------------------------------
VMEX_CAF = "uwplasma/vmex caf09ee7faca2e1c74d139ef6910f05e02654514 (0.12.0)"
VMEX_3B7 = "uwplasma/vmex 3b73d6f88b832aaa6c63dc7574feb7de9e80d064 (0.11.x)"
VMEX_FCF = "uwplasma/vmex fcf211f20692e557345fa41fe53d83b411ad53a2 (0.11.10, projection only)"
VMEX_EC8 = "uwplasma/vmex ec8641f3c02ae45b4d1b055483b5ba6fa44f3aeb (projection only)"
VMEX_B5F = "uwplasma/vmex b5f5267efc0795c4a49a224e321e9b370975c14c (0.11.0)"
V2K_OFFICE = "VMEC2000 (STELLOPT build) 70c637a56a535de7b72a5721d4294a7c1a6cf739, office Linux build"
V2K_MAC = "STELLOPT ee175502dec1766c60c2027e1df5c74c343d9794, macOS arm64 build"
DESC32 = "DESC 0.17.3+32.gc9ec78e2d, JAX 0.6.2, Python 3.10.12 (office)"
DESC27 = "DESC 0.17.3+27.g4f48720be, JAX 0.6.2, Python 3.10.12 (office)"
VMECPP = "VMEC++ 0.5.2"
MRX = "rogeriojorge/mrx fork main 732f8249a8a1d39d0dabedc02e2ac0d743d3697a, JAX 0.11.0, Python 3.13.7, CPU float64"
JAX_OFFICE = "JAX 0.11.2 CPU float64, office (Linux x86-64, 36 cores shared)"
JAX_MAC = "JAX 0.11.0 CPU float64, macOS arm64 (Mac mini)"

# ---- record entries ---------------------------------------------------------------------------------------------
# files: globs of the records a figure/claim reads; members: globs hashed into one digest; raw: where the raw
# states of the entry are listed (a manifest, or committed .npz globs).
ENTRIES: dict[str, dict] = {}


def entry(key, title, label, files, producer, code, command="", members=(), raw=None, status="accepted", note=""):
    ENTRIES[key] = dict(title=title, label=label, status=status, files=list(files), members=list(members),
                        producer=list(producer), command=command, code=list(code), raw=raw or {}, note=note)


# References and certificates
entry("ref.identities", "Sampled identities and derivative reference of every exact family", "analytically verified reference",
      ["results/reference/identities.json", "results/reference/derivatives.json"],
      ["benchmarks/analytic.py", "benchmarks/verify_reference.py", "benchmarks/reference_derivatives.py"],
      [JAX_MAC], "python benchmarks/verify_reference.py; python benchmarks/reference_derivatives.py",
      note="192 sampled points per case, VMEX not executed; the exact fields are the oracle of every other record")
entry("ref.quadrature", "Exact-chart volume quadrature against the closed forms, and lambda reconstruction",
      "analytically verified reference",
      ["results/projection_certificate/*/reference.json"],
      ["benchmarks/certify_projection.py", "benchmarks/analytic.py"], [VMEX_FCF, VMEX_CAF],
      "python benchmarks/certify_projection.py --case <case> (reference block)",
      note="volume, <B^2> and force balance of the exact chart against analytic.shear_averages (1e-15 level)")
entry("ref.lopez_surfaces", "Lopez D exact flux map, chart, boundary series, section-flux and volume ladders",
      "analytically verified reference", ["results/lopez/surfaces.json"],
      ["benchmarks/lopez.py", "benchmarks/lopez_surfaces.py"], ["NumPy float64; no solver"],
      "python benchmarks/lopez_surfaces.py",
      note="chart/inversion entries are a sampled chart check (4000 samples); the upstream cubic-root certificate "
           "(certify.py blob 008fe673, PASS 343037/21733/depth 28) is logged in STATUS.md and benchmarks/lopez.py only")

# Projections
for case in ("integer_3d", "integer_3d_tau", "integer_3d_tau_mirror", "sheared_A", "sheared_issan", "solovev_symmetric",
             "solovev_asymmetric"):
    entry(f"proj.cert.{case}", f"Projection certificate ladder, {case}", "finite-basis projection",
          [f"results/projection_certificate/{case}/ns*.json"],
          ["benchmarks/certify_projection.py", "benchmarks/score_solved.py", "benchmarks/common_norms.py"],
          [VMEX_CAF] + ([VMEX_FCF] if case in ("sheared_A", "sheared_issan") else []),
          f"python benchmarks/certify_projection.py --case {case} --ns <NS> --mpol <m> --ntor <n>",
          note="exact state projected onto the VMEC basis, no solve; B, J, force vs the exact grad p on exact-chart "
               "points, relative L2 over the whole volume; gates B 1e-5, J 1e-3, force 1e-3; q* = fixed Gauss rule, "
               "m* = rule aligned with the state's radial mesh",
          )
entry("proj.cert.sheared_path", "Projection certificate along A -> Issan (t = 0.25, 0.5, 0.75)", "finite-basis projection",
      ["results/projection_certificate/sheared_path_t*/ns*.json"],
      ["benchmarks/certify_projection.py"], [VMEX_CAF], "python benchmarks/certify_projection.py --case sheared_path_t<25|50|75> --ns 33")
entry("proj.seed_states", "Projected exact states (records and seeds) used before the certificate", "finite-basis projection",
      ["results/projection/*/projection.json"], ["benchmarks/project_integer_vmex.py"], [VMEX_FCF, VMEX_EC8],
      "python benchmarks/project_integer_vmex.py <NS> --case <case>", status="supporting",
      raw={"glob": "results/projection/*/seed.npz"})
entry("proj.lopez.oracle", "Lopez D projection oracle (eta and alpha gauge) and VMEX WOUT read-out", "finite-basis projection",
      ["results/lopez/projection_oracle_eta.json", "results/lopez/projection_oracle_alpha.json",
       "results/lopez/projection_vmex_eta.json"],
      ["benchmarks/lopez_surfaces.py", "benchmarks/lopez_energy.py", "benchmarks/score_lopez.py"],
      [VMEX_3B7, "NumPy float64 oracle"], "python benchmarks/lopez_tables.py (inputs written by the projection scripts)",
      note="B volume error of the exact state truncated to (mpol, ntor) at NS; alpha gauge is the direction-only control")
entry("lopez_mrx.ladder", "MRX vacuum solve of Lopez D on the exact flux-surface chart, spline ladder", "physical error qualification",
      ["results/lopez_mrx/ladder/*.json", "results/lopez_mrx/exact_closure.json"],
      ["benchmarks/lopez_mrx.py", "benchmarks/lopez_mrx_run.py", "benchmarks/lopez_mrx_tables.py"], [MRX],
      "python benchmarks/lopez_mrx_run.py --chart label --twist -1 --ns 8 12 24 --p 3 --out <record>",
      note="linear solve for the harmonic 2-form, not a nonlinear equilibrium solve; interior foliation is the exact one")
entry("lopez_mrx.deck", "MRX vacuum solve on the VMEC-deck chart (cylindrical angle, truncated boundary)", "physical error qualification",
      ["results/lopez_mrx/cyl/*.json", "results/lopez_mrx/wout_lines.json"],
      ["benchmarks/lopez_mrx.py", "benchmarks/lopez_mrx_run.py", "benchmarks/lopez_wout_lines.py"], [MRX],
      "python benchmarks/lopez_mrx_run.py --chart cyl --mpol 10 --ntor 72 --ns 8 16 96 --p 3 --out <record>",
      raw={"manifest": "results/lopez_mrx/wout_lines.json"})
entry("lopez_mrx.perturb", "MRX boundary deformations, islands and added-current relaxation (closure experiments)", "sampled chart check",
      ["results/lopez_mrx/ham/*.json", "results/lopez_mrx/cylham/*.json", "results/lopez_mrx/relax/*.json"],
      ["benchmarks/lopez_mrx_perturb.py", "benchmarks/lopez_mrx_relax.py"], [MRX], "python benchmarks/lopez_mrx_perturb.py ...",
      status="diagnostic", raw={"glob": "results/lopez_mrx/*/*.npz"})

# Recovery
entry("rec.matrix", "Recovery matrix: 133 solves (VMEX, VMEC2000; cold, continued, exact seed, restart)", "physical error qualification",
      ["results/matrix/summary.json"],
      ["benchmarks/matrix_solve.py", "benchmarks/matrix_queue.py", "benchmarks/score_solved.py", "benchmarks/matrix_tables.py",
       "benchmarks/matrix_manifest.py"], [VMEX_CAF, V2K_OFFICE, JAX_OFFICE],
      "bash benchmarks/matrix_queue.sh <queue file>; python benchmarks/matrix_tables.py",
      members=["results/matrix/runs/*/*.json"], raw={"manifest": "results/matrix/wout_manifest.json"},
      note="OMP 4 threads, CPU; exact-seed rows of LASYM cases use the unrotated seed (see claims X3/W2)")
entry("rec.matrix.xbuild", "Same decks with the macOS arm64 VMEC2000 build (build dependence of stalled LASYM states)",
      "physical error qualification", ["results/matrix/xbuild/*/receipt.json", "results/matrix/xbuild/*/volume_score_v1.json"],
      ["benchmarks/matrix_solve.py", "benchmarks/score_solved.py"], [V2K_MAC, V2K_OFFICE],
      "python benchmarks/matrix_solve.py --code vmec2000 --xvmec <macOS xvmec> ...", status="diagnostic",
      raw={"manifest": "results/matrix/xbuild_wout_manifest.json"})
entry("rec.matrix.wp", "Work-precision: two passes of 46 configurations, repeats 2-3, load recorded", "physical error qualification",
      ["results/matrix/wp_summary.json"],
      ["benchmarks/matrix_wp_tables.py", "benchmarks/matrix_wp_queue.sh", "benchmarks/matrix_wp_score.sh"],
      [VMEX_CAF, V2K_OFFICE, JAX_OFFICE], "bash benchmarks/matrix_wp_queue.sh; python benchmarks/matrix_wp_tables.py",
      members=["results/matrix/wp_runs/*/*.json", "results/matrix/wp2_runs/*/*.json"],
      raw={"manifest": ["results/matrix/wp_wout_manifest.json", "results/matrix/wp2_wout_manifest.json"]},
      note="shared office box, 1-minute load 7-42 on 36 cores; timings are not benchmarks where load differs")
entry("rec.desc.matched", "DESC matched ladders and factor isolation on exact-chart common norms (142 solves)",
      "physical error qualification", ["results/desc/matched/index.json"],
      ["benchmarks/desc_matched.py", "benchmarks/desc_matched_jobs.py", "benchmarks/common_norms.py", "benchmarks/desc_matched_tables.py"],
      [DESC32], "bash benchmarks/desc_queue.sh; python benchmarks/desc_matched_tables.py",
      members=["results/desc/matched/*/record.json"], raw={"manifest": "results/desc/matched/h5_manifest.json"},
      note="success=False rows reached DESC's iteration cap: optimizer termination, not convergence")
entry("rec.desc.published", "Published sheared-A DESC protocol (vendored driver, unchanged) and its own checks",
      "nonlinear residual qualification",
      ["results/desc/published_protocol/sheared_A/landreman_sheared_desc_checks.json", "results/desc/published_protocol/sheared_A/record.json"],
      ["benchmarks/upstream/landreman_sheared_desc.py"], [DESC32], "python benchmarks/upstream/landreman_sheared_desc.py (driver unchanged)",
      raw={"extra": ["office:~/vab-shearD/run_default/landreman_sheared_desc.h5"]})
entry("rec.desc.general", "DESC general-case records (cold protocol, M ladders, Issan families), schema v2 rescoring",
      "physical error qualification", ["results/desc/general/*/record_v2.json"],
      ["benchmarks/desc_general_case.py", "benchmarks/score_wout_exact.py"], [DESC32, DESC27],
      "python benchmarks/desc_general_case.py --case <case> --M <M> --output-dir <dir>", status="supporting",
      note="superseded by rec.desc.matched where a case is in both; kept for the Issan families A and B",
      raw={"extra_dirs": ["office:~/vab-c26-rerun/results/desc/general"]})
entry("rec.cross_code", "Cross-code runs: VMEX, VMEC2000, VMEC++ on the same decks (September campaign, scorer v2)",
      "physical error qualification", ["results/cross_code/summary.json"],
      ["benchmarks/cross_code_run.py", "benchmarks/score_wout_exact.py", "benchmarks/cross_code_summary.py"],
      [VMEX_3B7, V2K_MAC, VMECPP], "python benchmarks/cross_code_run.py --deck <deck> --ns <NS> --code <code>",
      members=["results/cross_code/runs/*/receipt.json", "results/cross_code/runs/*/exact_score_v2.json"],
      status="supporting", note="flux-label and axis metrics only; whole-volume norms are in rec.matrix")
entry("lopez.early", "Lopez D recovery runs before the polar-gauge finding (cold, 'seed', 'alpha', VMEC2000)",
      "optimizer termination", ["results/lopez/summary.json"],
      ["benchmarks/lopez_solve.py", "benchmarks/lopez_tables.py", "benchmarks/score_lopez.py"], [VMEX_3B7, V2K_OFFICE, JAX_OFFICE],
      "bash benchmarks/lopez_run_queue.sh <queue>", members=["results/lopez/solves/*/*.json", "results/lopez/solves_vmec2000/*/*.json",
                                                           "results/lopez/runs/*/*.json"],
      status="partly superseded", raw={"manifest": "results/lopez/wout_manifest.json"},
      note="every 'seed' and 'alpha' row is a silent cold start (eta-gauge seed violates the m=1 polar constraint): invalid as seed runs; the cold rows and the VMEC2000 rows remain valid; see lopez.iotaF")
entry("lopez.iotaF", "Lopez D: iota scan, polar-gauge seeds, energy test, Solov'ev LASYM control", "optimizer termination",
      ["results/iotaF/collect_final.json", "results/iotaF/energy_all.json"],
      ["benchmarks/lopez_polar.py", "benchmarks/lopez_foliation.py", "benchmarks/lopez_energy.py", "benchmarks/lopez_iotaF_collect.py",
       "benchmarks/lopez_solve.py"], [VMEX_3B7, V2K_OFFICE, JAX_OFFICE], "bash benchmarks/lopez_iotaF_queue.sh <queue>",
      members=["results/iotaF/solves/*/*.json", "results/iotaF/control/*/*.json", "results/iotaF/res_*.json"],
      raw={"manifest": "results/iotaF/wout_manifest_iotaF.json"})

# Sensitivities
entry("sens.axisym", "Axisymmetric integer family, NS 17-257: root, tangent, transpose, FD ladder, dense control",
      "discrete derivative check", ["results/sensitivity/integer_axisymmetric_ns*.json"],
      ["benchmarks/sensitivity_vmex.py", "benchmarks/derivative_map.py"], [VMEX_CAF, JAX_OFFICE],
      "python benchmarks/sensitivity_vmex.py --case integer_axisymmetric --ns <NS> --steps ... (see results/sensitivity/RUNS.md)",
      raw={"glob": "results/sensitivity/integer_axisymmetric_ns*.npz"},
      note="NCURR=0 closure; iota is an input here; gates 1e-3 nonzero, 1e-5 null, root 1e-9 (pre-registered)")
entry("sens.3d", "3-D integer family NS 17, 25, 33 (+ augmented, tcon0 = 0 diagnostics)", "discrete derivative check",
      ["results/sensitivity/integer_3d_ns*.json"], ["benchmarks/sensitivity_vmex.py", "benchmarks/derivative_map.py"],
      [VMEX_CAF, JAX_OFFICE], "python benchmarks/sensitivity_vmex.py --case integer_3d --params c delta a --seed exact --ftol 1e-10 --ns <NS>",
      raw={"glob": "results/sensitivity/integer_3d_ns*.npz"}, status="negative",
      note="tangent certified against its own operator; physical response off by O(1); NS25 and tcon0=0 roots rejected")
entry("sens.regions", "Region-resolved response errors (all / axis s<0.1 / bulk s>=0.1, shell profile)", "continuum sensitivity check",
      ["results/sensitivity/regions.json", "results/sensitivity/orders.json"],
      ["benchmarks/sensitivity_regions.py", "benchmarks/sensitivity_table.py"], [VMEX_CAF],
      "python benchmarks/sensitivity_regions.py; python benchmarks/sensitivity_table.py",
      note="the axis/bulk split is a post-hoc diagnostic; the pre-registered gates apply to the whole sets and fail")
entry("sens.scalar", "Volume-mean |B| response (3-D scalar observable)", "discrete derivative check",
      ["results/sensitivity/scalar_mean_absB.json"], ["benchmarks/sensitivity_scalar.py"], [VMEX_CAF],
      "python benchmarks/sensitivity_scalar.py", status="diagnostic")

# Mechanisms
entry("mech.runs", "Exact-seed solves with the constraint taken apart: baseline, tcon0, polar seed, defect correction",
      "physical error qualification", ["results/mechanism/runs/*/record.json"],
      ["benchmarks/mech_solve.py", "benchmarks/mech_core.py", "benchmarks/score_solved.py", "benchmarks/mech_tables.py"],
      [VMEX_CAF, JAX_MAC, JAX_OFFICE],
      "python benchmarks/mech_solve.py --case <case> --ns <NS> --tcon0 <t> --baseline <vmec|exact> [--polar] --ftol 1e-14 --niter 20000",
      raw={"manifest": "results/mechanism/wout_manifest.json"},
      note="the jax version differs per record (0.11.0 Mac mini, 0.11.2 office), recorded in each file")
entry("mech.linear", "Dense-Jacobian linear attribution x* + delta (constraint / MHD / axis shell / bulk)", "discrete derivative check",
      ["results/mechanism/linear/*.json"], ["benchmarks/mech_linear.py"], [VMEX_CAF, JAX_MAC],
      "python benchmarks/mech_linear.py --case <case> --ns <NS> --baseline <vmec|exact>")
entry("mech.eigen", "Eigenvalues of the preconditioned force Jacobian at the exact seed", "discrete derivative check",
      ["results/mechanism/eigen/*.json"], ["benchmarks/mech_eigen.py"], [VMEX_CAF, JAX_MAC, JAX_OFFICE],
      "python benchmarks/mech_eigen.py --case <case> --ns <NS> --baseline <vmec|exact>",
      note="dense geev; 14392 evolved dofs for integer 3-D NS 17 (office)")
entry("mech.asym", "Iteration-1 residual of asymmetric exact seeds: own angle vs polar gauge, edge shear, field and lambda checks",
      "discrete derivative check",
      ["results/mechanism/asym_residual*.json", "results/mechanism/edge_shear.json", "results/mechanism/lambda/*.json",
       "results/mechanism/fieldcheck/*.json", "results/mechanism/symmetry_tests.json"],
      ["benchmarks/mech_asym.py", "benchmarks/mech_lambda.py", "benchmarks/mech_fieldcheck.py", "benchmarks/mech_symtests.py",
       "benchmarks/mech_core.py"], [VMEX_CAF, JAX_MAC], "python benchmarks/mech_asym.py [--polar]")
entry("mech.operator", "Operator checks: force vs energy gradient, variational Newton, VMEX operator at VMEC2000 states, hot-start reset",
      "discrete derivative check",
      ["results/mechanism/gradcheck_*.json", "results/mechanism/variational/*.json", "results/mechanism/crosscode_*.json",
       "results/mechanism/vmec2000_reset_check.json"],
      ["benchmarks/mech_gradcheck.py", "benchmarks/mech_variational.py", "benchmarks/mech_crosscode.py",
       "benchmarks/mech_vmec2000_iter1.py"], [VMEX_CAF, V2K_OFFICE, JAX_MAC])
entry("mech.lopez", "Lopez D polar seed with the exact constraint baseline (NS 17)", "optimizer termination",
      ["results/mechanism/lopez/*.json"], ["benchmarks/mech_lopez.py", "benchmarks/lopez_polar.py"], [VMEX_CAF, JAX_OFFICE],
      "python benchmarks/mech_lopez.py --ns 17 --baseline <vmec|exact>")

# ---- figures and claims -----------------------------------------------------------------------------------------
FIGURES = {
    "fig1_references_projection": {
        "title": "References and finite-basis projection convergence",
        "panels": {
            "a": ("Projection certificate, B error against NS (fixed m, n)", ["proj.cert.sheared_A", "proj.cert.sheared_issan", "proj.cert.integer_3d",
                                                                              "proj.cert.integer_3d_tau", "proj.cert.solovev_asymmetric"]),
            "b": ("Projection certificate, B error against angular resolution", ["proj.cert.sheared_A", "proj.cert.sheared_issan"]),
            "c": ("Lopez D projection oracle against (mpol, ntor) and NS; alpha-gauge control; VMEX WOUT read-out",
                  ["proj.lopez.oracle", "ref.lopez_surfaces"]),
            "d": ("MRX vacuum solve of Lopez D, spline ladder, exact-flux and VMEC-deck charts", ["lopez_mrx.ladder", "lopez_mrx.deck"]),
        }},
    "fig2_recovery_matrix": {
        "title": "Recovery matrix: solved error against the projection floor",
        "panels": {
            "a": ("VMEX and VMEC2000 by case and protocol (B error, floor, convergence)", ["rec.matrix", "mech.runs"]),
            "b": ("DESC by case and protocol against M (exact-chart norms)", ["rec.desc.matched"]),
            "c": ("Lopez D: cold, polar-seeded, MRX, projection floors", ["lopez.iotaF", "lopez.early", "mech.lopez", "lopez_mrx.ladder",
                                                                       "lopez_mrx.deck", "proj.lopez.oracle"]),
        }},
    "fig3_work_precision": {
        "title": "Work-precision at matched error (load caveat)",
        "panels": {
            "a": ("Solov'ev: VMEX, VMEC2000, DESC wall time against B error", ["rec.matrix.wp", "rec.desc.matched"]),
            "b": ("Asymmetric Solov'ev: VMEX, VMEC2000, DESC", ["rec.matrix.wp", "rec.desc.matched"]),
            "c": ("Integer 3-D: VMEX, VMEC2000, DESC", ["rec.matrix.wp", "rec.desc.matched"]),
            "d": ("DESC only: integer 3-D tau, sheared A, sheared Issan", ["rec.desc.matched"]),
        }},
    "fig4_sensitivities": {
        "title": "Sensitivities: bulk against whole volume, nonzero and null response, 3-D",
        "panels": {
            "a": ("Nonzero response (c): error against NS, bulk and whole volume", ["sens.regions"]),
            "b": ("Null response (delta): error against NS, bulk and whole volume", ["sens.regions"]),
            "c": ("3-D integer family: relative response error of the frozen-branch tangent and of reconverged FD", ["sens.3d"]),
        }},
    "fig5_constraint_bias": {
        "title": "Constraint-bias mechanism (symmetric Solov'ev)",
        "panels": {
            "a": ("B error against NS: default baseline, exact baseline, tcon0 = 0", ["mech.runs", "rec.matrix"]),
            "b": ("Dose response in tcon0 at NS 65 and 129", ["mech.runs"]),
            "c": ("Linear attribution: constraint and MHD parts of the first-order error", ["mech.linear", "mech.runs"]),
        }},
    "fig6_seed_gauge_stability": {
        "title": "Asymmetric seed gauge and unstable directions of the flow Jacobian",
        "panels": {
            "a": ("Iteration-1 residual at tcon0 = 0, own angle against polar gauge", ["mech.asym"]),
            "b": ("Solved B error, own-angle against polar seed", ["mech.runs"]),
            "c": ("Eigenvalues with positive real part at the exact seed", ["mech.eigen"]),
        }},
}

# claim id -> (status, label, short text, record keys); the prose lives in CLAIMS.md
CLAIMS = {
    "R1": ("supported", "analytically verified reference", "Exact fields, quadrature and derivative references of every family", ["ref.identities", "ref.quadrature"]),
    "R2": ("supported", "analytically verified reference", "Lopez D flux map, field, ladders; chart is a sampled chart check; upstream certificate only for its own domain", ["ref.lopez_surfaces"]),
    "P1": ("supported", "finite-basis projection", "Sheared A and sheared Issan projections meet the gates above stated resolution", ["proj.cert.sheared_A", "proj.cert.sheared_issan", "proj.cert.sheared_path"]),
    "P2": ("supported", "finite-basis projection", "Integer 3-D and Solov'ev projection floors fall with NS and sit far below solved errors", ["proj.cert.integer_3d", "proj.cert.integer_3d_tau", "proj.cert.integer_3d_tau_mirror", "proj.cert.solovev_symmetric", "proj.cert.solovev_asymmetric"]),
    "P3": ("supported", "finite-basis projection", "Lopez D projection floor set by (mpol, ntor) and NS; alpha gauge control", ["proj.lopez.oracle"]),
    "P4": ("qualified", "physical error qualification", "MRX non-nested vacuum solve recovers Lopez D; the VMEC-deck chart error is the boundary truncation", ["lopez_mrx.ladder", "lopez_mrx.deck"]),
    "M1": ("supported", "nonlinear residual qualification", "Axisymmetric Solov'ev (both parities) converges to fsq 1e-14 in VMEX and VMEC2000; B floor flat in NS", ["rec.matrix", "proj.cert.solovev_symmetric", "proj.cert.solovev_asymmetric"]),
    "M2": ("negative", "optimizer termination", "Integer 3-D, tau = +-0.5 and sheared A: no cold or laddered run reaches 1e-14 in 20000 iterations", ["rec.matrix"]),
    "M3": ("qualified", "physical error qualification", "Sheared A and the A -> Issan path are recovered only from the exact seed, not to the certificate gates", ["rec.matrix", "proj.cert.sheared_A", "proj.cert.sheared_path"]),
    "M4": ("qualified", "physical error qualification", "VMEX and VMEC2000 agree on symmetric cases; asymmetric Solov'ev differs at low NS; stalled LASYM states depend on the VMEC2000 build", ["rec.matrix", "rec.matrix.xbuild", "rec.cross_code", "mech.operator"]),
    "M5": ("qualified", "physical error qualification", "DESC converges spectrally on Solov'ev and the integer families; sheared A at the iteration cap; published protocol passes its checks", ["rec.desc.matched", "rec.desc.published"]),
    "M6": ("qualified", "optimizer termination", "Lopez D: nested recovery stays at 1e-3 with a valid seed and limit-cycles; cold starts stall at 1e-2", ["lopez.iotaF", "mech.lopez"]),
    "T1": ("qualified", "physical error qualification", "Work-precision: VMEC2000 faster by 1.1-2.7x on clean Solov'ev points; ratio unresolved under load", ["rec.matrix.wp", "rec.desc.matched"]),
    "S1": ("supported", "discrete derivative check", "Axisymmetric derivative machinery certified at every NS 17-257", ["sens.axisym"]),
    "S2": ("qualified", "continuum sensitivity check", "Bulk (s >= 0.1) responses pass at NS >= 129; whole-volume responses fail", ["sens.axisym", "sens.regions"]),
    "S3": ("negative", "discrete derivative check", "3-D: tangent consistent with its own operator, physical response off by O(1)", ["sens.3d", "sens.scalar"]),
    "X1": ("supported", "physical error qualification", "The NS-independent B floor of Solov'ev is the spectral-condensation baseline bias", ["mech.runs", "mech.linear"]),
    "X2": ("qualified", "physical error qualification", "Near-axis layer: a bias part growing with NS and a first-shell discretization part", ["mech.runs", "mech.linear", "mech.operator"]),
    "X3": ("qualified", "nonlinear residual qualification; physical error qualification", "Asymmetric seed-gauge defect: the unrotated seed is sheared against the interior; polar rotation removes the iteration-1 residual and, for asymmetric Solov'ev at tcon0 = 0, the solved-error excess; not shown to remove the 3-D tau error", ["mech.asym", "mech.runs"]),
    "X4": ("qualified", "discrete derivative check (dense AD Jacobian spectrum)", "3-D drift: the bias amplifies it; unstable directions of the flow Jacobian remain", ["mech.eigen", "mech.runs"]),
    "X5": ("supported", "optimizer termination", "Lopez D: iota = 1 refuted as the cause; earlier exact-seed runs were silent cold starts", ["lopez.iotaF", "lopez.early", "lopez_mrx.deck"]),
    "SUP1": ("withdrawn", "-", "README: a plateau at rational transform, with a neighbouring equilibrium as one hypothesis (asymmetric integer family, flux-label error 8e-4 at NS 33-129)", ["mech.asym", "mech.runs", "rec.matrix"]),
    "SUP2": ("superseded", "-", "Stage 4 findings 1 and 6: the asymmetric Solov'ev plateau is a distinct LASYM effect and tcon0 = 0 does not improve it; the flat floor as a property of the code", ["rec.matrix", "mech.runs", "mech.asym", "mech.linear"]),
    "SUP3": ("superseded", "-", "Stage 3/4: Lopez D exact-seed runs drift to a displaced 0.8 % attractor and iota = 1 makes the nested problem degenerate", ["lopez.early", "lopez.iotaF"]),
    "SUP4": ("withdrawn", "-", "Stage 1 README items: capped DESC runs called converged, unresolved volume J promoted (1.6e-4 midpoint rule against 2.3e-3 Gauss rule), 'degenerate neighbour' asserted", ["rec.desc.matched", "rec.desc.general"]),
}


# ---- builder ----------------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expand(patterns):
    paths = set()
    for pattern in patterns:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    return sorted(paths)


def observed_identities(paths):
    """Code identities that the JSON records themselves carry (checked against the declared ones)."""
    seen: dict[str, set] = {}

    def add(name, value):
        if isinstance(value, str) and value:
            seen.setdefault(name, set()).add(value)

    def walk(obj, depth=0):
        if not isinstance(obj, dict) or depth > 3:
            return
        for k, v in obj.items():
            if k == "vmex_commit":
                add("vmex_commit", v)
            elif k == "code_identity" and isinstance(v, dict):
                add("code_commit", v.get("commit"))
            elif k in ("desc_version", "jax", "jax_version"):
                add(k.replace("jax_version", "jax"), v)
            elif isinstance(v, dict):
                walk(v, depth + 1)

    for path in paths:
        if path.suffix != ".json" or path.stat().st_size > 5_000_000:
            continue
        try:
            data = json.loads(path.read_text())
        except ValueError:
            continue
        for item in (data if isinstance(data, list) else [data]):
            walk(item)
    return {k: sorted(v) for k, v in sorted(seen.items())}


def build():
    scripts = {}
    records = {}
    for key, spec in ENTRIES.items():
        files = expand(spec["files"])
        if not files:
            raise SystemExit(f"index entry {key}: no files match {spec['files']}")
        out = dict(title=spec["title"], label=spec["label"], status=spec["status"],
                   files=[{"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)} for p in files],
                   producer=dict(scripts=spec["producer"], command=spec["command"]),
                   code=spec["code"], observed=observed_identities(files))
        if spec["members"]:
            members = expand(spec["members"])
            lines = "".join(f"{p.relative_to(ROOT)} {sha256_file(p)}\n" for p in members)
            out["members"] = dict(patterns=spec["members"], count=len(members), digest=hashlib.sha256(lines.encode()).hexdigest())
        if spec["raw"]:
            out["raw"] = spec["raw"]
        if spec["note"]:
            out["note"] = spec["note"]
        for script in spec["producer"]:
            path = ROOT/script
            if not path.exists():
                raise SystemExit(f"index entry {key}: producer {script} does not exist")
            scripts[script] = sha256_file(path)
        records[key] = out
    figures = {}
    for name, spec in FIGURES.items():
        panels = {}
        for panel, (title, keys) in spec["panels"].items():
            missing = [k for k in keys if k not in ENTRIES]
            if missing:
                raise SystemExit(f"figure {name} panel {panel}: unknown records {missing}")
            panels[panel] = dict(title=title, records=keys)
        figures[name] = dict(title=spec["title"], files=[f"{FIG_DIR}/{name}.png", f"{FIG_DIR}/{name}.pdf"],
                             generator="benchmarks/paper_figures.py", panels=panels)
    claims = {}
    for cid, (status, label, text, keys) in CLAIMS.items():
        missing = [k for k in keys if k not in ENTRIES]
        if missing:
            raise SystemExit(f"claim {cid}: unknown records {missing}")
        claims[cid] = dict(status=status, label=label, text=text, records=keys)
    return {"schema": 1, "repository": "rogeriojorge/vmex-benchmark-analytical", "note": "generated by benchmarks/build_index.py; do not edit",
            "producer_script_sha256": dict(sorted(scripts.items())), "records": records, "figures": figures, "claims": claims}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = json.dumps(build(), indent=1, sort_keys=False, allow_nan=False)+"\n"
    if args.check:
        if not INDEX.exists() or INDEX.read_text() != text:
            print("results/index.json is out of date; run benchmarks/build_index.py", file=sys.stderr)
            return 1
        return 0
    INDEX.write_text(text)
    print(f"wrote {INDEX.relative_to(ROOT)}: {len(ENTRIES)} records, {len(FIGURES)} figures, {len(CLAIMS)} claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
