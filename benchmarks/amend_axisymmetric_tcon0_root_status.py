"""Add a residual-based status amendment without rewriting a saved probe."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from analytic import ROOT
from evidence import sha256_file, write_json


DEFAULT_REPORT = ROOT / (
    "results/audit/axisym_branch_tcon0/"
    "axisym-tcon0-branch-20260925T010824.995851Z/tcon0_branch_probe.json"
)
CERTIFICATE_TOLERANCE = 1e-11


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text())
    if report.get("evidence") != "matched_axisymmetric_delta_branch_tcon0_control":
        raise SystemExit("unexpected report type")
    recorded_script = ROOT / report["script"]
    current_script_sha256 = sha256_file(recorded_script)

    roots = []
    for row in report["roots"]:
        residual = float(row["root"]["residual_after_anchor"])
        roots.append({
            "case": row["case"],
            "recorded_status": row["status"],
            "corrected_status": ("root_certified" if residual <= CERTIFICATE_TOLERANCE
                                 else "residual_unresolved"),
            "residual_after_anchor": residual,
            "certificate_tolerance": CERTIFICATE_TOLERANCE,
            "residual_to_tolerance": residual / CERTIFICATE_TOLERANCE,
        })
    amendment = {
        "schema": 1,
        "amendment": "tcon0_zero_branch_root_status",
        "status": "diagnostic_only",
        "raw_report_unchanged": True,
        "source_report": report_path.relative_to(ROOT).as_posix(),
        "source_report_sha256": sha256_file(report_path),
        "root_source_sha256": report.get("script_sha256"),
        "current_probe_source": report["script"],
        "current_probe_source_sha256": current_script_sha256,
        "current_probe_source_matches_run": (
            current_script_sha256 == report.get("script_sha256")
        ),
        "certificate_rule": (
            "root_certified iff residual_after_anchor <= refine_tol; "
            "the saved probe used refine_tol=1e-11"
        ),
        "roots": roots,
        "all_roots_certified": all(r["corrected_status"] == "root_certified"
                                    for r in roots),
        "derivative_interpretation": (
            "The saved branch field finite difference is not a certified "
            "equilibrium response because every TCON0=0 root exceeds the "
            "fixed-point residual certificate tolerance."
        ),
        "script": Path(__file__).relative_to(ROOT).as_posix(),
        "script_sha256": sha256_file(Path(__file__)),
        "command": "python benchmarks/amend_axisymmetric_tcon0_root_status.py",
    }
    output = report_path.parent / "root_status_amendment.json"
    write_json(output, amendment, exclusive=True)
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(),
                      "sha256": sha256_file(output),
                      "all_roots_certified": amendment["all_roots_certified"],
                      "roots": roots}, indent=2))


if __name__ == "__main__":
    main()
