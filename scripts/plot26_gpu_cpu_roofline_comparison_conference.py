#!/usr/bin/env python3
"""
Conference-oriented compact variant of Plot 26.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot26_gpu_cpu_roofline_comparison import (
    DEFAULT_CPU_SUMMARY,
    DEFAULT_GPU_ROOT,
    build_summary,
    load_cpu_summary,
    load_gpu_kernel_summary,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
OUTPUT_STEM = RESULTS_DIR / "plot26_gpu_cpu_roofline_comparison_conference"
SUMMARY_CSV = DERIVED_DIR / "plot26_gpu_cpu_roofline_comparison_conference_summary.csv"

ROOF_DRAM_COLOR = "#b22222"
ROOF_COMPUTE_COLOR = "#1f4e79"
ROOF_SP_COLOR = "#6e6e6e"
CPU_SERIES_COLOR = "#1f4e79"
GPU_GRIDDER_COLOR = "#0b6e4f"
GPU_SUBFFT_COLOR = "#7a3db8"
GPU_HARDWARE_COLOR = "#2f2f2f"
GPU_L2_COLOR = "#5f6368"
GPU_L1_COLOR = "#8b9097"
THREAD_COLORS = {1: "#111111", 16: "#d95f02", 32: "#1b9e77", 64: "#c51b7d"}

AXIS_LABEL_FONTSIZE = 11.6
TITLE_FONTSIZE = 12.6
TICK_FONTSIZE = 9.6
LEGEND_FONTSIZE = 7.6
P_LEGEND_FONTSIZE = 7.4


plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)


def main() -> int:
    cpu_df = load_cpu_summary(DEFAULT_CPU_SUMMARY)
    _, gpu_df = load_gpu_kernel_summary(DEFAULT_GPU_ROOT)
    build_summary(cpu_df, gpu_df).to_csv(SUMMARY_CSV, index=False)

    cpu_dp_peak = float(cpu_df["dp_peak_gflops"].iloc[0])
    cpu_sp_peak = float(cpu_df["sp_peak_gflops"].iloc[0])
    cpu_bw = float(cpu_df["dram_bw_gbs"].iloc[0])
    cpu_dp_ridge = cpu_dp_peak / cpu_bw
    cpu_sp_ridge = cpu_sp_peak / cpu_bw

    # Empirical H100 SXM5 94 GB roofs from Nsight roofline screenshots.
    # Peak work is shared across the L1/L2/DRAM roofs; peak traffic values are
    # the reported empirical bandwidths for each memory level.
    h100_emp_peak = 54236.09636780428
    h100_dram_bw = 2446.84560203940
    h100_l2_bw = 9638.27343487540
    h100_l1_bw = 27118.04818390214
    h100_dram_ridge = h100_emp_peak / h100_dram_bw
    h100_l2_ridge = h100_emp_peak / h100_l2_bw
    h100_l1_ridge = h100_emp_peak / h100_l1_bw

    fig, (ax_roof, ax_cmp) = plt.subplots(1, 2, figsize=(7.25, 3.55), gridspec_kw={"width_ratios": [1.08, 1.0]})

    x_min = 0.02
    x_max = max(300.0, float(max(cpu_df["ai"].max(), gpu_df["ai_median"].max()) * 2.0))
    y_min = 12.0
    y_max = max(cpu_sp_peak, h100_emp_peak, float(gpu_df["gflops_median"].max())) * 1.35

    ai_cpu = np.logspace(np.log10(x_min), np.log10(cpu_sp_ridge), 300)
    ax_roof.plot(ai_cpu, cpu_bw * ai_cpu, color=ROOF_DRAM_COLOR, linewidth=1.8, label=f"CPU DRAM ({cpu_bw:.0f} GB/s)")
    ax_roof.hlines(cpu_dp_peak, cpu_dp_ridge, x_max, color=ROOF_COMPUTE_COLOR, linewidth=1.5, linestyles="-.", label="CPU FP64")
    ax_roof.hlines(cpu_sp_peak, cpu_sp_ridge, x_max, color=ROOF_SP_COLOR, linewidth=1.8, linestyles="--", label="CPU FP32")
    for bw, ridge, color, label, ls in [
        (h100_dram_bw, h100_dram_ridge, GPU_HARDWARE_COLOR, f"H100 DRAM ({h100_dram_bw/1000:.2f} TB/s)", ":"),
        (h100_l2_bw, h100_l2_ridge, GPU_L2_COLOR, f"H100 L2 ({h100_l2_bw/1000:.2f} TB/s)", "--"),
        (h100_l1_bw, h100_l1_ridge, GPU_L1_COLOR, f"H100 L1 ({h100_l1_bw/1000:.2f} TB/s)", "-."),
    ]:
        ai_gpu = np.logspace(np.log10(x_min), np.log10(ridge), 300)
        ax_roof.plot(ai_gpu, bw * ai_gpu, color=color, linewidth=1.35, linestyle=ls, label=label)
        ax_roof.hlines(h100_emp_peak, ridge, x_max, color=color, linewidth=1.35, linestyles=ls)

    cpu_df_line = cpu_df[cpu_df["threads"] >= 16]
    ax_roof.plot(cpu_df_line["ai"], cpu_df_line["gflops"], color=CPU_SERIES_COLOR, linewidth=1.2, alpha=0.9)
    ax_roof.scatter(
        cpu_df["ai"],
        cpu_df["gflops"],
        s=34,
        c=[THREAD_COLORS[int(v)] for v in cpu_df["threads"]],
        edgecolors=CPU_SERIES_COLOR,
        linewidth=0.95,
        marker="o",
        zorder=3,
    )
    gpu_styles = {
        "gridder": {"color": GPU_GRIDDER_COLOR, "marker": "^", "label": "GPU gridder"},
        "sub-fft": {"color": GPU_SUBFFT_COLOR, "marker": "s", "label": "GPU sub-FFT"},
    }
    for kind, sub in gpu_df.groupby("kind"):
        sub = sub.sort_values("threads")
        style = gpu_styles[kind]
        ax_roof.plot(sub["ai_median"], sub["gflops_median"], color=style["color"], linewidth=1.2, alpha=0.9)
        ax_roof.scatter(
            sub["ai_median"],
            sub["gflops_median"],
            s=38,
            c=[THREAD_COLORS[int(v)] for v in sub["threads"]],
            edgecolors=style["color"],
            linewidth=1.1,
            marker=style["marker"],
            zorder=3,
        )

    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(x_min, x_max)
    ax_roof.set_ylim(y_min, y_max)
    ax_roof.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_roof.set_ylabel("Throughput (GFLOP/s)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_roof.set_title("Roofline Comparison", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_roof.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    series_handles = [
        Line2D([0], [0], color=ROOF_DRAM_COLOR, linewidth=1.8, label=f"CPU DRAM ({cpu_bw:.0f} GB/s)"),
        Line2D([0], [0], color=ROOF_COMPUTE_COLOR, linewidth=1.5, linestyle="-.", label="CPU FP64"),
        Line2D([0], [0], color=ROOF_SP_COLOR, linewidth=1.8, linestyle="--", label="CPU FP32"),
        Line2D([0], [0], color=GPU_HARDWARE_COLOR, linewidth=1.35, linestyle=":", label=f"H100 DRAM ({h100_dram_bw/1000:.2f} TB/s)"),
        Line2D([0], [0], color=GPU_L2_COLOR, linewidth=1.35, linestyle="--", label=f"H100 L2 ({h100_l2_bw/1000:.2f} TB/s)"),
        Line2D([0], [0], color=GPU_L1_COLOR, linewidth=1.35, linestyle="-.", label=f"H100 L1 ({h100_l1_bw/1000:.2f} TB/s)"),
        Line2D([0], [0], color=CPU_SERIES_COLOR, marker="o", linestyle="-", linewidth=1.2, markersize=4.5, label="CPU WSClean"),
        Line2D([0], [0], color=GPU_GRIDDER_COLOR, marker="^", linestyle="-", linewidth=1.2, markersize=4.8, label="GPU gridder"),
        Line2D([0], [0], color=GPU_SUBFFT_COLOR, marker="s", linestyle="-", linewidth=1.2, markersize=4.5, label="GPU sub-FFT"),
    ]
    roof_legend = ax_roof.legend(
        series_handles,
        [h.get_label() for h in series_handles],
        loc="lower left",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        ncol=1,
    )
    ax_roof.add_artist(roof_legend)
    p_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=THREAD_COLORS[p], markeredgecolor="black", markeredgewidth=0.65, markersize=5.6, label=f"p={p}")
        for p in [1, 16, 32, 64]
    ]
    ax_roof.legend(
        handles=p_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=P_LEGEND_FONTSIZE,
        frameon=True,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    cmp_threads = sorted(set(cpu_df["threads"]).intersection(set(gpu_df["threads"])))
    cpu_cmp = cpu_df.set_index("threads").reindex(cmp_threads)
    x = np.arange(len(cmp_threads), dtype=float)
    ax_cmp_eff = ax_cmp.twinx()
    ax_cmp.plot(x, cpu_cmp["gflops"], marker="o", linewidth=1.8, color=CPU_SERIES_COLOR, label="CPU WSClean")
    for kind, style in gpu_styles.items():
        sub = gpu_df[gpu_df["kind"] == kind].set_index("threads").reindex(cmp_threads)
        ax_cmp.plot(x, sub["gflops_median"], marker=style["marker"], linewidth=1.8, color=style["color"], label=style["label"])
        ax_cmp_eff.plot(
            x,
            sub["mib_per_joule_median"],
            marker=style["marker"],
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            linewidth=1.3,
            linestyle="--",
            alpha=0.72,
            color=style["color"],
            label=f"{style['label']} MiB/J",
        )

    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels([f"p={int(v)}" for v in cmp_threads], fontsize=TICK_FONTSIZE)
    ax_cmp.set_xlim(-0.25, len(cmp_threads) - 0.75)
    ax_cmp.set_yscale("log")
    ax_cmp_eff.set_yscale("log")
    ax_cmp.set_ylabel("GFLOP/s", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_cmp_eff.set_ylabel("MiB/J", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_cmp.set_xlabel("Parallelism p", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_cmp.set_title("Matched-p Throughput and MiB/J", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_cmp.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_cmp_eff.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_cmp_eff.grid(False)
    ax_cmp.spines["top"].set_visible(False)
    ax_cmp_eff.spines["top"].set_visible(False)

    cmp_handles = [
        Line2D([0], [0], color=CPU_SERIES_COLOR, marker="o", linewidth=1.8, label="CPU WSClean"),
        Line2D([0], [0], color=GPU_GRIDDER_COLOR, marker="^", linewidth=1.8, label="GPU gridder"),
        Line2D([0], [0], color=GPU_SUBFFT_COLOR, marker="s", linewidth=1.8, label="GPU sub-FFT"),
        Line2D([0], [0], color=GPU_GRIDDER_COLOR, marker="^", markerfacecolor="white", markeredgewidth=1.0, linewidth=1.3, linestyle="--", alpha=0.72, label="Gridder MiB/J"),
        Line2D([0], [0], color=GPU_SUBFFT_COLOR, marker="s", markerfacecolor="white", markeredgewidth=1.0, linewidth=1.3, linestyle="--", alpha=0.72, label="Sub-FFT MiB/J"),
    ]
    ax_cmp.legend(
        cmp_handles,
        [h.get_label() for h in cmp_handles],
        loc="center right",
        bbox_to_anchor=(0.98, 0.60),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        ncol=1,
    )

    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.16, wspace=0.34)

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
