"""Job lists for ``desc_queue.sh``: factor isolation (sheared A) and the matched ladders.

Ladder protocols (tag ``ladder_<case>_<protocol>_M<M>``):

* ``strict``  the existing cold protocol: lambda = 0, prescribed iota, lsq-exact, ConcentricGrid, L = M + 2,
              N = M, ftol 1e-10 / gtol 1e-8, at most 150 iterations;
* ``loose``   as ``strict`` but the published termination (ftol 1e-8, gtol 1e-5): the time to the
              resolution-limited error without the strict-tolerance tail;
* ``exact``   ``loose`` with lambda seeded from the exact field;
* ``pub``     the published sheared-DESC protocol (exact seed, prescribed current, matrix-free TRF/LSMR,
              QuadratureGrid, N = 2M, L = M, published tolerances); stellarator-symmetric 3-D states only.
Usage: desc_matched_jobs.py factor|ladder > jobs.txt
"""
import sys

COMMON = "--levels {levels} --avail-mem-gb 6"
PROTOCOLS = {
    "strict": "--seed zero --closure iota --optimizer lsq-exact --grid default --tol ours",
    "loose": "--seed zero --closure iota --optimizer lsq-exact --grid default --tol published",
    "exact": "--seed exact --closure iota --optimizer lsq-exact --grid default --tol published",
    "pub": "--seed exact --closure current --optimizer matrix-free --grid quadrature --tol published",
}
# case: (ladder of M per protocol, axisymmetric, stellarator symmetric)
LADDERS = {
    "solovev_symmetric": {"loose": (4, 6, 8, 10, 12), "exact": (4, 6, 8, 10, 12), "strict": (4, 6, 8, 10, 12)},
    "solovev_asymmetric": {"loose": (4, 6, 8, 10, 12), "exact": (4, 6, 8, 10, 12), "strict": (4, 6, 8, 10, 12)},
    "integer_3d": {"loose": (4, 6, 8, 10), "exact": (4, 6, 8, 10), "pub": (4, 6, 8, 10), "strict": (4, 6, 8, 10, 12)},
    "integer_3d_tau": {"loose": (6, 8, 10), "exact": (6, 8, 10), "strict": (6, 8, 10)},
    "integer_3d_tau_mirror": {"loose": (6, 8, 10), "exact": (6, 8, 10), "strict": (6, 8, 10)},
    "sheared_A": {"loose": (6, 8, 10), "exact": (6, 8, 10), "pub": (6, 8, 10), "strict": (6,)},
    "sheared_issan": {"loose": (6, 8, 10), "exact": (6, 8, 10), "pub": (6, 8, 10), "strict": (6, 8)},
    "sheared_issan_tau": {"loose": (6, 8), "exact": (6, 8), "strict": (6,)},
    "sheared_issan_tau_mirror": {"loose": (6, 8), "exact": (6, 8), "strict": (6,)},
}
ORDER = ("loose", "exact", "pub", "strict")  # cheap first


def pub_args(case, M):
    axis = case.startswith("solovev")
    return f"--N {0 if axis else 2*M} --L {M}"


def ladder_jobs():
    jobs = []
    for proto in ORDER:
        for case, ladder in LADDERS.items():
            for M in ladder.get(proto, ()):
                extra = pub_args(case, M) if proto == "pub" else ""
                jobs.append(f"ladder_{case}_{proto}_M{M} --case {case} --M {M} {COMMON.format(levels='q2')} {PROTOCOLS[proto]} {extra}".strip())
    return jobs


def factor_jobs(Ms=(8, 10)):
    pub = dict(seed="exact", closure="current", optimizer="matrix-free", grid="quadrature", N="2M", L="M", tol="published")
    ours = dict(seed="zero", closure="iota", optimizer="lsq-exact", grid="default", N="M", L="M+2", tol="ours")
    flips = {"seed": "zero", "closure": "iota", "optimizer": "lsq-exact", "grid": "default", "N": "M", "L": "M+2", "tol": "ours"}
    fwd = {"seed": "exact", "closure": "current", "optimizer": "matrix-free", "grid": "quadrature", "N": "2M", "L": "M", "tol": "published"}

    def args(M, d):
        n, l = (2*M if d["N"] == "2M" else M), (M if d["L"] == "M" else M+2)
        return (f"--case sheared_A --levels q1 --avail-mem-gb 6 --M {M} --seed {d['seed']} --closure {d['closure']} "
                f"--optimizer {d['optimizer']} --grid {d['grid']} --tol {d['tol']} --N {n} --L {l}")
    jobs = []
    for M in Ms:
        jobs.append(f"factor_{M}_pub {args(M, pub)}")
        jobs += [f"factor_{M}_pub_minus_{k} {args(M, {**pub, k: flips[k]})}" for k in ("seed", "closure", "grid", "N", "L", "tol", "optimizer")]
        jobs.append(f"factor_{M}_base {args(M, ours)}")
        jobs += [f"factor_{M}_base_plus_{k} {args(M, {**ours, k: fwd[k]})}" for k in ("seed", "closure", "tol", "optimizer", "grid", "L", "N")]
    return jobs


if __name__ == "__main__":
    print("\n".join(factor_jobs() if sys.argv[1] == "factor" else ladder_jobs()))
