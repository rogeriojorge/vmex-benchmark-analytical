"""Energy test of settled Lopez D states against the exact vacuum.

Theorem (Thomson/Dirichlet principle for the multiply connected torus): among divergence-free fields tangent
to a given boundary with a given toroidal flux Phi, int B^2 dV is minimized by the curl-free (vacuum) field,
and for any admissible B = B_vac + b (same flux, so int B_vac . b dV = C * flux(b) = 0)

    int B^2 dV = int B_vac^2 dV + int b^2 dV  >=  int B_vac^2 dV = C Phi,   C = 2 pi (unit-scale reference).

A VMEC-family state is divergence-free by construction (B = grad s x grad(theta + lambda - iota phi)) and
tangent to the fixed boundary, with the prescribed flux. A *signed* energy below the exact value (beyond the
boundary-representation floor) therefore cannot belong to the stated problem: it would indicate a deck, scoring
or constraint defect. Correcting a flux mismatch to first order uses W ~ Phi^2 at fixed boundary:

    W_corr = W (Phi_edge / Phi_num)^2,    (W_corr - C Phi_edge) / (C Phi_edge)  ~  e_B^2   (e_B = relative L2 field error)

and the independent integral of the scorer is compared with the solver's own ``wb``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def energy_row(score, wb=None):
    v = score["vacuum"]
    target = v["energy_target_circulation_phi"]
    w_num = v["energy_b2_integral"]
    ratio_flux = v["toroidal_flux_exact"]/v["toroidal_flux"]
    w_corr = w_num*ratio_flux**2
    e_b2 = score["B_relative_volume_l2"]**2
    row = dict(run=score.get("run_id"), ns=score["ns"],
               signed_dW=w_num/target - 1, flux_rel_error=v["toroidal_flux"]/v["toroidal_flux_exact"] - 1,
               signed_dW_flux_corrected=w_corr/target - 1, eB2=e_b2,
               ratio_corrected_to_eB2=(w_corr/target - 1)/e_b2 if e_b2 > 0 else None,
               volume_rel_error=v["volume"]/v["volume_exact"] - 1, fsq=score.get("fsq"), wb=wb)
    return row


def read_wb(path):
    from netCDF4 import Dataset
    with Dataset(path) as nc:
        return float(np.asarray(nc.variables["wb"][...]))


def collect(roots):
    rows = []
    for root in roots:
        for f in sorted(Path(root).glob("**/lopez_score_v1.json")):
            score = json.loads(f.read_text())
            wouts = sorted(f.parent.glob("wout_*.nc"))
            wb = read_wb(wouts[0]) if wouts else None
            row = energy_row(score, wb)
            row["dir"] = str(f.parent)
            row["scorer_energy"] = score["vacuum"]["energy_b2_integral"]
            row["K"] = row["scorer_energy"]/wb if wb else None
            rows.append(row)
    ks = np.array([r["K"] for r in rows if r["K"]])
    # VMEC writes wb = int B^2/2 dV / (2 pi)^2 with mu0 absorbed: int B^2 dV = 2 (2 pi)^2 wb = K wb
    k_theory = 2*(2*np.pi)**2
    for r in rows:
        if r["wb"]:
            flux = json.loads(Path(r["dir"], "lopez_score_v1.json").read_text())["vacuum"]["toroidal_flux_exact"]
            r["wb_dW"] = r["wb"]*k_theory/(2*np.pi*flux) - 1          # solver's own energy against the exact 2 pi Phi_edge
            r["wb_vs_scorer"] = r["wb"]*k_theory/r["scorer_energy"] - 1
    return dict(rows=rows, K_theory=k_theory, K_median=float(np.median(ks)) if ks.size else None,
                K_spread=None if not ks.size else float((ks.max() - ks.min())/np.median(ks)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    res = collect(a.roots)
    a.out.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    print("K theory %.6f median %.6f spread %.2e" % (res["K_theory"], res["K_median"], res["K_spread"]))
    for r in res["rows"]:
        print("%-52s ns%3d signedW %+.2e flux %+.1e corrected %+.2e eB2 %.2e ratio %s wb_dW %s wb_vs_scorer %s" % (
            r["run"], r["ns"], r["signed_dW"], r["flux_rel_error"], r["signed_dW_flux_corrected"], r["eB2"],
            "%.2f" % r["ratio_corrected_to_eB2"] if r["ratio_corrected_to_eB2"] else "-",
            "%+.2e" % r["wb_dW"] if "wb_dW" in r else "-", "%+.2e" % r["wb_vs_scorer"] if "wb_vs_scorer" in r else "-"))
