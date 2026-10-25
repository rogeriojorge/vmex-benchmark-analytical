"""One MRX run on Lopez D: build, vacuum solve, score, closure diagnostics; one JSON record.

    python lopez_mrx_run.py --chart label --twist -1 --ns 8 12 24 --p 3 --out results/lopez_mrx/x.json
    python lopez_mrx_run.py --chart cyl --mpol 10 --ntor 72 --ns 8 16 96 --p 3 --out ...
    python lopez_mrx_run.py --chart label --twist -1 --pert 0.002,1,1,0 --ns 8 12 24 --p 3 ...

Run with PYTHONPATH at the MRX clone and from a directory without an ``mrx/`` folder.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lopez_mrx as lm  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chart", choices=("label", "cyl"), default="label")
    ap.add_argument("--twist", type=int, default=-1)
    ap.add_argument("--mpol", type=int, default=10)
    ap.add_argument("--ntor", type=int, default=72)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--ns", type=int, nargs=3, default=(8, 12, 24))
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--pert", action="append", default=[], help="eps,m,n,phase (label chart)")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--lines", type=int, nargs=2, default=(6, 8))
    ap.add_argument("--ham", action="store_true", help="also the effective-Hamiltonian analysis of the one-turn map")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--batch", type=int, default=0, help="MRX map batch size (0 = all points at once)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import mrx
    if a.batch:
        mrx.MAP_BATCH_SIZE_INNER = a.batch
    rec = dict(args=vars(a))
    t0 = time.time()
    if a.chart == "label":
        pert = tuple(tuple(float(v) for v in s.split(",")) for s in a.pert)
        phi_map = lm.LabelMap(pert=pert, twist=a.twist)
        rec["expected_rotation"] = float(-a.twist)
    else:
        phi_map, err, fit = lm.cyl_fourier_map(a.mpol, a.ntor, cache=a.cache)
        rec.update(boundary_truncation_error=err, radial_fit_error=fit, expected_rotation=1.0)
    seq = lm.build_sequence(a.ns, a.p, phi_map)
    rec["build_s"] = time.time() - t0
    dof, spread = lm.vacuum_field(seq)
    rec["flux_cut_spread"] = spread
    rec["dofs_2form"] = int(seq.n(2))
    rec["solve_done_s"] = time.time() - t0
    if not a.no_score:
        rec["score"] = lm.score(seq, dof)
    rec["score_done_s"] = time.time() - t0
    rec["closure"] = lm.closure_summary(seq, dof, *a.lines, a.turns, a.steps, expected_rotation=rec["expected_rotation"])
    if a.ham:
        import numpy as np
        rec["effective_hamiltonian"], (hu, hD, hH) = lm.hamiltonian_record(seq, dof)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(Path(a.out).with_suffix("")) + "_ham.npz", hgrid_u=hu, hgrid_D=hD, hgrid_H=hH)
    rec["total_s"] = time.time() - t0
    rec["max_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9 if sys.platform == "darwin" \
        else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float) + "\n")
    s = rec.get("score", {})
    print(a.chart, a.ns, a.p, "Bvol %.2e Bsurf %.2e curl %.2e E %.1e | defect %.1e rotdev %.1e | %.0fs %.1fGB" % (
        s.get("B_rel_volume_l2", float("nan")), s.get("B_rel_surface_l2", float("nan")),
        s.get("pointwise_curl_over_B_over_a", float("nan")), s.get("energy_rel_error", float("nan")),
        rec["closure"]["return_defect_max"], rec["closure"]["rotation_dev_max"], rec["total_s"], rec["max_rss_gb"]), flush=True)


if __name__ == "__main__":
    main()
