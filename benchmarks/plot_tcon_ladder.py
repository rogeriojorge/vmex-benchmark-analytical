"""Read historical TCON records and write separate, diagnostic summaries.

Raw run reports and native score files are treated as immutable inputs. Values
previously added by this plotter (iterations, software/device metadata, command,
and acceptance flags) are excluded because the reviewed commit cannot establish
whether they match the original run. Outputs are written under results/audit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT
from evidence import sha256_file, single_source, write_json


EXCLUDED_HISTORICAL_FIELDS = [
    "iterations",
    "jax_version",
    "numpy_version",
    "scipy_version",
    "accelerator",
    "command",
    "physical_gate",
    "native_score.accepted",
    "native_scores.json:accepted",
]
PLOT_METRICS = (
    ("field_relative_l2", r"$E_B$", 1e-5),
    ("current_relative_l2", r"$E_J$", 1e-3),
    ("force_pressure_scale", r"$\|J\times B-\nabla p\|_{rms}/\|\nabla p\|_{rms}$", 1e-3),
)


def source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_records(root: Path) -> list[dict]:
    """Load only stored observations; never repair or rewrite raw reports."""
    result = []
    for report_path in sorted((root / "results/vmex").glob("*/forward.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        input_name = str(report.get("input", ""))
        score = report.get("native_score")
        if (not input_name.endswith("input.integer_3d_iota") or
                not isinstance(score, dict) or report.get("initialization") not in {"cold", "projected"}):
            continue
        try:
            ns = int(report["ns_override"])
            tcon0 = float(report["tcon0_effective"])
            metrics = {key: float(score[key]) for key, _, _ in PLOT_METRICS}
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite([ns, tcon0, *metrics.values()]).all():
            continue
        sample_path = report_path.parent / "native_samples.npz"
        result.append({
            "ns": ns,
            "tcon0": tcon0,
            "initialization": report["initialization"],
            "converged": report.get("converged"),
            "source_commit": report.get("vmex_commit"),
            "metrics": metrics,
            "forward_record": report_path.relative_to(root).as_posix(),
            "forward_sha256": sha256_file(report_path),
            "native_samples": sample_path.relative_to(root).as_posix(),
            "native_samples_sha256": sha256_file(sample_path) if sample_path.is_file() else None,
            "reported_ftol": report.get("ftol"),
            "reported_niter_limit": report.get("niter_limit"),
            "reported_solve_seconds": report.get("solve_seconds_including_first_compile"),
            "reported_host_peak_rss_mib": report.get("peak_rss_mib"),
        })
    return result


def amendment_table(records: list[dict], root: Path) -> dict:
    rows = []
    for row in records:
        report = json.loads((root / row["forward_record"]).read_text(encoding="utf-8"))
        score_path = (root / row["forward_record"]).parent / "native_scores.json"
        rows.append({
            "record": row["forward_record"],
            "record_sha256": row["forward_sha256"],
            "overwritten_fields": EXCLUDED_HISTORICAL_FIELDS,
            "iteration_value_current": report.get("iterations"),
            "iteration_value_status": "unverified; excluded from derived summaries",
            "acceptance_value_current": (report.get("native_score") or {}).get("accepted"),
            "acceptance_value_status": "derived post hoc by historical plotting code; not a root certificate",
            "companion_score_sha256": sha256_file(score_path) if score_path.is_file() else None,
            "treatment": "Preserve bytes; report measured physical metrics as historical diagnostics only.",
        })
    old_plotter = subprocess.run(
        ["git", "-C", str(root), "show", "575f13f67b346118c3d7f05cc60cc6e29131359a:benchmarks/plot_tcon_ladder.py"],
        capture_output=True,
    )
    return {
        "schema": 1,
        "reviewed_commit": "575f13f67b346118c3d7f05cc60cc6e29131359a",
        "historical_plotter_source_sha256": hashlib.sha256(old_plotter.stdout).hexdigest()
            if old_plotter.returncode == 0 else None,
        "fields_injected_or_overwritten_by_historical_plotter": EXCLUDED_HISTORICAL_FIELDS,
        "entries": rows,
        "interpretation": (
            "Current raw bytes are preserved. Metrics in native_score were not rewritten by the "
            "historical plotting code, but their measurement resolution has not yet been independently certified. "
            "The table identifies overwritten metadata and excludes it from plotted/aggregated results."
        ),
    }


def summarized(records: list[dict], *, ns: int | None = None) -> dict:
    selected = [row for row in records if ns is None or row["ns"] == ns]
    common_commit = single_source(selected, "source_commit")
    return {
        "schema": 2,
        "status": "historical_diagnostic_only",
        "evidence": "legacy96_physical_score_not_independently_certified",
        "vmex_commit": common_commit,
        "sampling_points_per_state": 96,
        "sampling_resolution": "legacy96; no angular/radial refinement certificate",
        "records": selected,
        "excluded_fields": EXCLUDED_HISTORICAL_FIELDS,
    }


def render_tcon(records: list[dict], path: Path) -> None:
    selected = [row for row in records if row["ns"] == 33]
    if not selected:
        raise ValueError("no NS=33 historical TCON observations are available")
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.3), constrained_layout=True)
    series = (("cold", "#bd4b4b", "s"), ("projected", "#176b87", "o"))
    for ax, (key, title, gate) in zip(axes, PLOT_METRICS):
        for start, color, marker in series:
            rows = sorted((row for row in selected if row["initialization"] == start),
                          key=lambda row: row["tcon0"], reverse=True)
            if not rows:
                continue
            ax.plot([row["tcon0"] for row in rows],
                    [row["metrics"][key] for row in rows],
                    marker=marker, color=color, linewidth=1.6, markersize=5,
                    label="Cold" if start == "cold" else "Projected")
        ax.axhline(gate, color="#555555", linestyle="--", linewidth=1)
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_yscale("log")
        ax.set_xlabel("TCON0")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("relative L2")
    axes[1].set_ylabel("relative L2")
    axes[2].set_ylabel("normalized RMS")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Historical VMEX TCON0 runs (NS=33; diagnostic scores)", fontsize=11)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_resolution(records: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3), constrained_layout=True)
    colors = {1.0: "#bd4b4b", 0.0: "#176b87"}
    for ax, (key, title, gate) in zip(axes, PLOT_METRICS):
        for tcon0 in (1.0, 0.0):
            rows = sorted((row for row in records if row["initialization"] == "projected"
                           and row["tcon0"] == tcon0), key=lambda row: row["ns"])
            if rows:
                ax.plot([row["ns"] for row in rows], [row["metrics"][key] for row in rows],
                        marker="o", color=colors[tcon0], linewidth=1.6,
                        label=f"TCON0 = {tcon0:g}" if tcon0 == 0 else "TCON0 = deck default")
        ax.axhline(gate, color="#555555", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks([33, 65, 129])
        ax.set_xlabel("radial resolution NS")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("relative L2")
    axes[1].set_ylabel("relative L2")
    axes[2].set_ylabel("normalized RMS")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Historical projected VMEX states (NS ladder; NS129, TCON0=0 unrun)", fontsize=10.5)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def generate(root: Path) -> dict:
    records = collect_records(root)
    if not records:
        raise ValueError("no historical TCON observations found")
    # Reject cross-source pooling before writing any summary.
    single_source(records, "source_commit")
    out = root / "results/audit/tcon_reinterpretation"
    out.mkdir(parents=True, exist_ok=True)
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    amendments = amendment_table(records, root)
    write_json(out / "historical_amendments.json", amendments)
    summaries = {
        "ns33": summarized(records, ns=33),
        "resolution": summarized(records),
    }
    write_json(out / "ns33_summary.json", summaries["ns33"])
    write_json(out / "resolution_summary.json", summaries["resolution"])
    figures = {
        "ns33": figdir / "vmex_tcon_ladder_ns33_reinterpreted.png",
        "resolution": figdir / "vmex_tcon_resolution_ladder_reinterpreted.png",
    }
    render_tcon(records, figures["ns33"])
    render_resolution(records, figures["resolution"])
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            text=True, capture_output=True)
    worktree = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                              text=True, capture_output=True)
    script_path = Path(__file__).resolve()
    manifest = {
        "schema": 1,
        "status": "diagnostic_only",
        "generator_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "generator_worktree_clean": worktree.returncode == 0 and not worktree.stdout,
        "generator_source_sha256": source_digest(script_path),
        "input_records": [{"path": row["forward_record"],
                            "sha256": row["forward_sha256"]} for row in records],
        "amendments": {"path": (out / "historical_amendments.json").relative_to(root).as_posix(),
                       "sha256": sha256_file(out / "historical_amendments.json")},
        "summaries": {key: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
                      for key, path in (("ns33", out / "ns33_summary.json"),
                                        ("resolution", out / "resolution_summary.json"))},
        "figures": {key: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
                    for key, path in figures.items()},
    }
    write_json(out / "figure_manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.root.resolve()), indent=2), flush=True)


if __name__ == "__main__":
    main()
