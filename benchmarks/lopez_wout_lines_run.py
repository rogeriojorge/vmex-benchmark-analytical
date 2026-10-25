"""Closed-line and exact-foliation diagnostics of WOUT states of Lopez D -> one JSON.

    python lopez_wout_lines_run.py out.json name=path.nc [name=path.nc ...]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lopez_wout_lines as wl  # noqa: E402
import score_lopez as sl  # noqa: E402


def main(argv):
    out, items = Path(argv[0]), argv[1:]
    rec = {}
    for item in items:
        name, path = item.split("=", 1)
        w = sl.read_wout(path)
        ns = w.ns
        pick = sorted({max(2, ns//8), ns//4, ns//2, 3*ns//4, ns - 1})
        lines = wl.vmec_line_closure(w, surfaces=pick)
        exact = wl.exact_field_on_surfaces(w, surfaces=pick, stride=(1, 2))
        rec[name] = dict(path=Path(path).name, sha256=sl.sha256(path), ns=ns, mpol=w.mpol, ntor=w.ntor,
                         fsq=[float(x) for x in (w.fsqr, w.fsqz, w.fsql)], lines=lines, exact_on_surfaces=exact)
        print(name, [(r["s"], round(r["rotation_mean"], 5), round(r["rotation_spread"], 4)) for r in lines], flush=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
