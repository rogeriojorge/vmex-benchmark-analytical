"""Summarize and plot the NS33 VMEX constraint-strength experiment."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analytic import ROOT


CASES = (
    ("projected", 1.0, "integer_3d_iota_ns33_niter3000_projected_ftol1e-10_tcon0default", 311),
    ("projected", 0.1, "integer_3d_iota_ns33_niter3000_projected_ftol1e-10_tcon00.1", 309),
    ("projected", 0.0, "integer_3d_iota_ns33_niter3000_projected_ftol1e-10_tcon00", 95),
    ("cold", 1.0, "integer_3d_iota_ns33_niter3000_ftol1e-10_tcon0default", 366),
    ("cold", 0.1, "integer_3d_iota_ns33_niter3000_ftol1e-10_tcon00.1", 362),
    ("cold", 0.0, "integer_3d_iota_ns33_niter3000_ftol1e-10_tcon00", 498),
)
RESOLUTION_CASES = (
    (33, 1.0, "integer_3d_iota_ns33_niter3000_projected_ftol1e-10_tcon0default", 311),
    (33, 0.0, "integer_3d_iota_ns33_niter3000_projected_ftol1e-10_tcon00", 95),
    (65, 1.0, "integer_3d_iota_ns65_niter3000_projected_ftol1e-10_tcon0default", 335),
    (65, 0.0, "integer_3d_iota_ns65_niter3000_projected_ftol1e-10_tcon00", 67),
    (129, 1.0, "integer_3d_iota_ns129_niter3000_projected_ftol1e-10_tcon0default", 420),
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    for start, tcon0, dirname, iterations in CASES:
        run_dir = ROOT / "results/vmex" / dirname
        path = run_dir / "forward.json"
        report = json.loads(path.read_text())
        report["iterations"] = iterations
        report.update(jax_version="0.9.2", numpy_version="2.5.2",
                      scipy_version="1.18.1", accelerator="NVIDIA RTX A4000")
        report["command"] = (
            f"BENCH_FTOL=1e-10 BENCH_TCON0={'default' if tcon0 == 1 else f'{tcon0:g}'} "
            f"python benchmarks/run_vmex.py inputs/input.integer_3d_iota 33 3000"
            + (" results/projection/integer_3d_vmex_ns33/seed.npz" if start == "projected" else "")
        )
        score = report["native_score"]
        passed = (score["field_relative_l2"] <= 1e-5
                  and score["current_relative_l2"] <= 1e-3
                  and score["force_pressure_scale"] <= 1e-3)
        report["physical_gate"] = dict(
            field_relative_l2=1e-5,
            current_relative_l2=1e-3,
            force_pressure_scale=1e-3,
            passed=passed,
        )
        report["native_score"]["accepted"] = passed
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        score_path = run_dir / "native_scores.json"
        score_path.write_text(json.dumps(report["native_score"], indent=2, allow_nan=False) + "\n")
        sample_path = run_dir / "native_samples.npz"
        row = dict(
            initialization=start,
            tcon0=tcon0,
            iterations=iterations,
            command=report["command"],
            vmex_commit=report["vmex_commit"],
            vmex_version=report["vmex_version"],
            python=report["python"],
            platform=report["platform"],
            devices=report["devices"],
            jax_version=report["jax_version"],
            numpy_version=report["numpy_version"],
            scipy_version=report["scipy_version"],
            accelerator=report["accelerator"],
            ftol=report["ftol"],
            converged=report["converged"],
            fsqr=report["fsqr"],
            fsqz=report["fsqz"],
            fsql=report["fsql"],
            field_relative_l2=report["native_score"]["field_relative_l2"],
            current_relative_l2=report["native_score"]["current_relative_l2"],
            force_pressure_scale=report["native_score"]["force_pressure_scale"],
            vmex_s_minus_reference_max_abs=report["native_score"]["vmex_s_minus_reference_max_abs"],
            solve_seconds=report["solve_seconds_including_first_compile"],
            native_sampling_seconds=report["native_sample_and_score_seconds"],
            peak_rss_mib=report["peak_rss_mib"],
            native_sample_sha256=sha256(sample_path),
            forward_sha256=sha256(path),
            forward_record=path.relative_to(ROOT).as_posix(),
            native_samples=sample_path.relative_to(ROOT).as_posix(),
            accepted=passed,
        )
        rows.append(row)

    out_dir = ROOT / "results/vmex/tcon_ladder_ns33"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(
        schema=1,
        experiment="vmex_tcon0_ladder_integer_3d_ns33",
        evidence="analytic_recovery_diagnostic",
        status="diagnostic_only",
        vmex_commit=rows[0]["vmex_commit"],
        vmex_version=rows[0]["vmex_version"],
        case="integer_3d",
        radial_resolution=33,
        angular_modes="MPOL=13, NTOR=12 (313 Fourier modes)",
        ftol=1e-10,
        niter_limit=3000,
        starts="cold and analytically certified projected state",
        projected_seed="results/projection/integer_3d_vmex_ns33/seed.npz",
        environment=dict(
            python=rows[0].get("python"),
            platform=rows[0].get("platform"),
            devices=rows[0].get("devices"),
            jax="0.9.2",
            numpy="2.5.2",
            scipy="1.18.1",
            accelerator="CUDA",
            gpu="NVIDIA RTX A4000 x2, 16376 MiB each",
            gpu_memory_peak_measured=False,
        ),
        sample_count=96,
        target=dict(field_relative_l2=1e-5, current_relative_l2=1e-3,
                    force_pressure_scale=1e-3),
        rows=rows,
        sampling_batch_invariance_check=dict(
            comparison="projected, TCON0=1, same state and same 96 points; chunk size 8 versus 32",
            max_abs_B_difference_T=3.219646771412954e-15,
            max_abs_J_difference_A_per_m2=4.0978193283081055e-7,
            max_abs_gradp_difference_Pa_per_m=4.0046870708465576e-8,
            xyz_weights_and_reference_labels_identical=True,
            field_score_difference=1.23e-16,
            current_score_difference=7.0e-15,
            force_ratio_difference=9.0e-14,
            chunk8_seconds=147.99585654900875,
            chunk32_seconds=90.99310253901058,
        ),
        interpretation=(
            "All six discrete VMEX solves converged at FTOL=1e-10, but none met the "
            "physical B/J/force targets. Lower TCON0 improved the projected-start "
            "physical scores and coordinate drift; from cold start, zero TCON0 had "
            "larger flux-label drift than default or 0.1. This single-resolution "
            "comparison does not certify an accepted equilibrium or justify a default change."
        ),
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.9), constrained_layout=True)
    series = (("cold", "#bd4b4b", "s"), ("projected", "#176b87", "o"))
    x = np.arange(3)
    metrics = (
        ("field_relative_l2", r"$E_B$", 1e-5),
        ("current_relative_l2", r"$E_J$", 1e-3),
        ("force_pressure_scale", r"$\|J\times B-\nabla p\|_{rms}/\|\nabla p\|_{rms}$", 1e-3),
        ("vmex_s_minus_reference_max_abs", "max |sVMEX − sref|", None),
    )
    for ax, (key, title, gate) in zip(axes.flat, metrics):
        for start, color, marker in series:
            selected = [r for r in rows if r["initialization"] == start]
            selected.sort(key=lambda r: r["tcon0"], reverse=True)
            vals = [r[key] for r in selected]
            ax.plot(x, vals, marker=marker, color=color, linewidth=1.7,
                    markersize=5, label="Cold" if start == "cold" else "Projected")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xticks(x, ["1 (default)", "0.1", "0"])
        ax.set_xlabel("TCON0")
        ax.grid(True, which="both", alpha=0.23)
        if gate is not None:
            ax.axhline(gate, color="#555555", linestyle="--", linewidth=1)
            ax.text(0.98, gate * 1.12, f"target {gate:g}", ha="right", va="bottom",
                    fontsize=7, color="#444444", transform=ax.get_yaxis_transform())
    axes[0, 0].set_ylabel("relative L2")
    axes[0, 1].set_ylabel("relative L2")
    axes[1, 0].set_ylabel("normalized RMS")
    axes[1, 1].set_ylabel("normalized flux label")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.suptitle("VMEX integer 3-D: constraint strength and initial state (NS=33)",
                 fontsize=12, fontweight="semibold")
    figure_path = ROOT / "figures/vmex_tcon_ladder_ns33.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    manifest = dict(
        schema=1,
        evidence="generated_from_saved_tcon_ladder_scores",
        figure=figure_path.relative_to(ROOT).as_posix(),
        figure_sha256=sha256(figure_path),
        summary=summary_path.relative_to(ROOT).as_posix(),
        summary_sha256=sha256(summary_path),
        sample_count=96,
        data=[dict(
            initialization=r["initialization"], tcon0=r["tcon0"],
            sample_sha256=r["native_sample_sha256"],
            forward_sha256=r["forward_sha256"],
            forward_record=r["forward_record"],
            native_samples=r["native_samples"]) for r in rows],
    )
    (out_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    resolution_rows = []
    for ns, tcon0, dirname, iterations in RESOLUTION_CASES:
        run_dir = ROOT / "results/vmex" / dirname
        path = run_dir / "forward.json"
        report = json.loads(path.read_text())
        report.update(
            iterations=iterations,
            jax_version="0.9.2", numpy_version="2.5.2",
            scipy_version="1.18.1", accelerator="NVIDIA RTX A4000",
        )
        command = (
            f"BENCH_FTOL=1e-10 BENCH_TCON0={'default' if tcon0 == 1 else '0'} "
            f"python benchmarks/run_vmex.py inputs/input.integer_3d_iota {ns} 3000 "
            f"results/projection/integer_3d_vmex_ns{ns}/seed.npz"
        )
        report["command"] = command
        score = report["native_score"]
        passed = (score["field_relative_l2"] <= 1e-5
                  and score["current_relative_l2"] <= 1e-3
                  and score["force_pressure_scale"] <= 1e-3)
        report["physical_gate"] = dict(
            field_relative_l2=1e-5, current_relative_l2=1e-3,
            force_pressure_scale=1e-3, passed=passed,
        )
        score["accepted"] = passed
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        score_path = run_dir / "native_scores.json"
        score_path.write_text(json.dumps(score, indent=2, allow_nan=False) + "\n")
        sample_path = run_dir / "native_samples.npz"
        resolution_rows.append(dict(
            ns=ns, tcon0=tcon0, initialization="projected",
            iterations=iterations, command=command,
            python=report["python"], platform=report["platform"],
            devices=report["devices"], jax_version=report["jax_version"],
            numpy_version=report["numpy_version"], scipy_version=report["scipy_version"],
            accelerator=report["accelerator"],
            converged=report["converged"], fsqr=report["fsqr"],
            fsqz=report["fsqz"], fsql=report["fsql"],
            field_relative_l2=score["field_relative_l2"],
            current_relative_l2=score["current_relative_l2"],
            force_pressure_scale=score["force_pressure_scale"],
            vmex_s_minus_reference_max_abs=score["vmex_s_minus_reference_max_abs"],
            solve_seconds=report["solve_seconds_including_first_compile"],
            native_sampling_seconds=report["native_sample_and_score_seconds"],
            peak_rss_mib=report["peak_rss_mib"],
            native_sample_sha256=sha256(sample_path),
            forward_sha256=sha256(path),
            forward_record=path.relative_to(ROOT).as_posix(),
            native_samples=sample_path.relative_to(ROOT).as_posix(),
            accepted=passed,
        ))
    resolution_dir = ROOT / "results/vmex/tcon_resolution_ladder"
    resolution_dir.mkdir(parents=True, exist_ok=True)
    resolution_summary = dict(
        schema=1,
        experiment="vmex_tcon0_integer_3d_projected_resolution_ladder",
        evidence="analytic_recovery_diagnostic",
        status="partial",
        vmex_commit="b5f5267efc0795c4a49a224e321e9b370975c14c",
        vmex_source_commit="b5f5267efc0795c4a49a224e321e9b370975c14c",
        ftol=1e-10,
        niter_limit=3000,
        initialization="analytically certified projected state at matching radial resolution",
        sample_count=96,
        targets=dict(field_relative_l2=1e-5, current_relative_l2=1e-3,
                     force_pressure_scale=1e-3),
        rows=resolution_rows,
        unrun=[dict(ns=129, tcon0=0.0, initialization="projected",
                    status="not_run", reason="interrupted by user request")],
        next_exact_action=(
            "Run NS129 projected with BENCH_FTOL=1e-10 and BENCH_TCON0=0; "
            "score the same 96 held-out points and compare with the saved default case."
        ),
        interpretation=(
            "All five measured states converged discretely and all failed at least one "
            "physical recovery gate. Lowering TCON0 improved the projected NS33 and NS65 "
            "scores; the zero-strength NS129 result is unmeasured, so no monotone radial "
            "trend or constraint recommendation is inferred."
        ),
    )
    resolution_summary["vmex_version"] = "0.8.0"
    resolution_summary["angular_modes"] = "MPOL=13, NTOR=12 (313 Fourier modes)"
    resolution_summary["environment"] = dict(
        python=resolution_rows[0]["python"],
        platform=resolution_rows[0]["platform"],
        devices=resolution_rows[0]["devices"],
        jax=resolution_rows[0]["jax_version"],
        numpy=resolution_rows[0]["numpy_version"],
        scipy=resolution_rows[0]["scipy_version"],
        accelerator=resolution_rows[0]["accelerator"],
        gpu="NVIDIA RTX A4000 x2, 16376 MiB each",
        gpu_memory_peak_measured=False,
    )
    summary_path = resolution_dir / "summary.json"
    summary_path.write_text(json.dumps(resolution_summary, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), constrained_layout=True)
    colors = {1.0: "#bd4b4b", 0.0: "#176b87"}
    metrics = (
        ("field_relative_l2", r"$E_B$", 1e-5),
        ("current_relative_l2", r"$E_J$", 1e-3),
        ("force_pressure_scale", r"$\|J\times B-\nabla p\|_{rms}/\|\nabla p\|_{rms}$", 1e-3),
    )
    for ax, (key, label, target) in zip(axes, metrics):
        for tcon0 in (1.0, 0.0):
            points = sorted((r for r in resolution_rows if r["tcon0"] == tcon0),
                            key=lambda r: r["ns"])
            ax.plot([r["ns"] for r in points], [r[key] for r in points],
                    marker="o" if tcon0 == 0 else "s", markersize=5,
                    linewidth=1.7, color=colors[tcon0],
                    label="TCON0 = default" if tcon0 == 1 else "TCON0 = 0")
        ax.axhline(target, color="#555555", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xticks([33, 65, 129])
        ax.set_xlabel("radial resolution NS")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("relative L2")
    axes[1].set_ylabel("relative L2")
    axes[2].set_ylabel("normalized RMS")
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].text(0.97, 0.88, "TCON0 = 0 at NS129 not run", ha="right", va="top",
                 transform=axes[2].transAxes, fontsize=7.5, color="#444444",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    fig.suptitle("VMEX integer 3-D: projected-start TCON0 and radial resolution",
                 fontsize=11.5, fontweight="semibold")
    resolution_figure = ROOT / "figures/vmex_tcon_resolution_ladder.png"
    fig.savefig(resolution_figure, dpi=220, bbox_inches="tight")
    plt.close(fig)
    resolution_manifest = dict(
        schema=1,
        evidence="generated_from_saved_resolution_ladder_scores",
        figure=resolution_figure.relative_to(ROOT).as_posix(),
        figure_sha256=sha256(resolution_figure),
        summary=summary_path.relative_to(ROOT).as_posix(),
        summary_sha256=sha256(summary_path),
        measured_rows=[dict(ns=r["ns"], tcon0=r["tcon0"],
                            forward_sha256=r["forward_sha256"],
                            native_sample_sha256=r["native_sample_sha256"])
                       for r in resolution_rows],
        unrun=resolution_summary["unrun"],
    )
    (resolution_dir / "figure_manifest.json").write_text(
        json.dumps(resolution_manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "figure": str(figure_path),
                      "sha256": manifest["figure_sha256"],
                      "resolution_summary": str(summary_path),
                      "resolution_figure": str(resolution_figure),
                      "resolution_sha256": resolution_manifest["figure_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
