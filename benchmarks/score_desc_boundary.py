"""Independently score the saved native DESC LCFS against its exact target."""
import argparse
import json
import resource
import time
from pathlib import Path

import desc
desc.set_device("gpu", gpuid=0)
import numpy as np
from desc.grid import LinearGrid
from desc.io import load

from analytic import cases
from run_desc_coordinate import mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--case", choices=("integer_axisymmetric", "integer_3d"),
                        required=True)
    parser.add_argument("--chart", choices=("base", "remapped"), required=True)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    t0 = time.perf_counter()
    eq = load(str(args.state))
    case = cases()[args.case]
    grid = LinearGrid(rho=[1.0], M=4*eq.M+1, N=4*eq.N+1,
                      NFP=case.nfp, sym=False)
    target, _ = mapping(case, grid.nodes, args.chart, args.amplitude,
                        args.m, args.n)
    result = eq.compute(["R", "Z"], grid=grid)
    dr = np.asarray(result["R"])-target[:, 0]
    dz = np.asarray(result["Z"])-target[:, 2]
    error = np.hypot(dr, dz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "schema": 1,
        "evidence": "native_desc_boundary_fit",
        "case": args.case,
        "chart": args.chart,
        "desc_resolution": {"L": eq.L, "M": eq.M, "N": eq.N, "NFP": eq.NFP},
        "sample_count": int(len(error)),
        "max_boundary_error_m": float(np.max(error)),
        "rms_boundary_error_m": float(np.sqrt(np.mean(error**2))),
        "max_abs_R_error_m": float(np.max(np.abs(dr))),
        "max_abs_Z_error_m": float(np.max(np.abs(dz))),
        "elapsed_s": time.perf_counter()-t0,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "state_artifact": args.state.name,
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
