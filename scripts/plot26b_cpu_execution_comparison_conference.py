#!/usr/bin/env python3
"""
Conference-oriented compact variant of Plot 26b.

This version removes the host-side CPU summary from the CPU+GPU runs and keeps only
the CPU-only AMD uProf roofline summary in a compact layout.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
CPU_CSV = SCRIPT_DIR / "roofline_stacking_summary.csv"
OUTPUT_STEM = SCRIPT_DIR / "plot26b_cpu_execution_comparison_conference"
SUMMARY_CSV = SCRIPT_DIR / "plot26b_cpu_execution_comparison_conference_summary.csv"

ROOF_DRAM_COLOR = "#c44e52"
ROOF_COMPUTE_COLOR = "#4c72b0"
ROOF_SP_COLOR = "0.55"
CPU_SERIES_COLOR = plt.cm.cividis(0.25)
BANDWIDTH_COLOR = plt.cm.cividis(0.72)
UTIL_COLOR = "#dd8452"
THREAD_COLORS = {
    1: "#4c72b0",
    16: "#dd8452",
    32: "#55a868",
    64: "#8172b3",
}


plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)


def load_cpu_summary() -> pd.DataFrame:
    df = pd.read_csv(CPU_CSV).sort_values("requested_threads").reset_index(drop=True)
    df["threads"] = df["requested_threads"]
    df["ai"] = df["arithmetic_intensity_flop_per_byte"]
    df["gflops"] = df["throughput_gflops"]
    df["gbs"] = df["bandwidth_gbs"]
    df["util"] = df["profiler_utilization_pct"]
    return df


def main() -> int:
    cpu_df = load_cpu_summary()
    cpu_df.to_csv(SUMMARY_CSV, index=False)

    dp_peak = float(cpu_df["dp_peak_gflops"].iloc[0])
    sp_peak = float(cpu_df["sp_peak_gflops"].iloc[0])
    dram_bw = float(cpu_df["dram_bw_gbs"].iloc[0])
    dp_ridge = dp_peak / dram_bw
    sp_ridge = sp_peak / dram_bw

    fig = plt.figure(figsize=(7.1, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 0.82])
    ax_roof = fig.add_subplot(gs[:, 0])
    ax_metrics = fig.add_subplot(gs[0, 1])
    ax_util = fig.add_subplot(gs[1, 1], sharex=ax_metrics)

    x_min = 0.03
    x_max = 150.0
    y_min = 0.2
    y_max = sp_peak * 1.22
    ai_diag = np.logspace(np.log10(x_min), np.log10(sp_ridge), 300)

    ax_roof.plot(ai_diag, dram_bw * ai_diag, color=ROOF_DRAM_COLOR, linewidth=2.0, label=f"DRAM roof ({dram_bw:.0f} GB/s)")
    ax_roof.hlines(dp_peak, dp_ridge, x_max, color=ROOF_COMPUTE_COLOR, linestyle="-.", linewidth=1.8, label=f"FP64 peak ({dp_peak:.0f} GFLOP/s)")
    ax_roof.hlines(sp_peak, sp_ridge, x_max, color=ROOF_SP_COLOR, linestyle="--", linewidth=2.0, label=f"FP32 peak ({sp_peak:.0f} GFLOP/s)")
    ax_roof.axvline(dp_ridge, color="0.6", linestyle=":", linewidth=0.9, alpha=0.85)
    ax_roof.axvline(sp_ridge, color="0.6", linestyle=":", linewidth=0.9, alpha=0.85)
    ax_roof.plot(cpu_df["ai"], cpu_df["gflops"], color="0.45", linewidth=1.1, alpha=0.8)
    ax_roof.scatter(
        cpu_df["ai"],
        cpu_df["gflops"],
        s=52,
        c=[THREAD_COLORS[int(v)] for v in cpu_df["threads"]],
        edgecolors="black",
        linewidth=0.7,
        zorder=3,
    )

    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(x_min, x_max)
    ax_roof.set_ylim(y_min, y_max)
    ax_roof.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=10.5, fontweight="bold")
    ax_roof.set_ylabel("Throughput (GFLOP/s)", fontsize=10.5, fontweight="bold")
    ax_roof.set_title("CPU Roofline", fontsize=11.2, fontweight="bold")
    ax_roof.tick_params(axis="both", labelsize=8.6)

    roof_handles, roof_labels = ax_roof.get_legend_handles_labels()
    roof_legend = ax_roof.legend(roof_handles, roof_labels, loc="upper left", fontsize=7.6, frameon=True)
    ax_roof.add_artist(roof_legend)
    p_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=THREAD_COLORS[p], markeredgecolor="black", markeredgewidth=0.7, markersize=6.2, label=f"p={p}")
        for p in [1, 16, 32, 64]
    ]
    ax_roof.legend(
        handles=p_handles,
        loc="lower right",
        fontsize=7.4,
        frameon=True,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    x = np.arange(len(cpu_df), dtype=float)
    tick_labels = [f"p={int(v)}" for v in cpu_df["threads"]]

    ax_metrics.plot(x, cpu_df["gflops"], color=CPU_SERIES_COLOR, marker="o", linewidth=2.0, label="Throughput")
    ax_metrics_bw = ax_metrics.twinx()
    ax_metrics_bw.plot(x, cpu_df["gbs"], color=BANDWIDTH_COLOR, marker="s", linewidth=1.9, label="Bandwidth")
    ax_metrics.set_ylabel("GFLOP/s", fontsize=9.5, fontweight="bold", color=CPU_SERIES_COLOR)
    ax_metrics_bw.set_ylabel("GB/s", fontsize=9.5, fontweight="bold", color=BANDWIDTH_COLOR)
    ax_metrics.tick_params(axis="y", labelsize=8.2, colors=CPU_SERIES_COLOR)
    ax_metrics_bw.tick_params(axis="y", labelsize=8.2, colors=BANDWIDTH_COLOR)
    ax_metrics.set_title("Throughput and Bandwidth", fontsize=10.8, fontweight="bold")
    ax_metrics.grid(True, axis="y", alpha=0.25)
    ax_metrics_bw.grid(False)
    ax_metrics.spines["top"].set_visible(False)
    ax_metrics_bw.spines["top"].set_visible(False)
    metric_handles = [
        Line2D([0], [0], color=CPU_SERIES_COLOR, marker="o", linewidth=2.0, label="Throughput"),
        Line2D([0], [0], color=BANDWIDTH_COLOR, marker="s", linewidth=1.9, label="Bandwidth"),
    ]
    ax_metrics.legend(metric_handles, [h.get_label() for h in metric_handles], loc="upper left", fontsize=7.6, frameon=True)

    ax_util.plot(x, cpu_df["util"], color=UTIL_COLOR, marker="o", linewidth=2.0)
    ax_util.set_xticks(x)
    ax_util.set_xticklabels(tick_labels, fontsize=8.6)
    ax_util.set_xlabel("Parallelism p", fontsize=10.0, fontweight="bold")
    ax_util.set_ylabel("Utilization (%)", fontsize=9.5, fontweight="bold")
    ax_util.set_title("CPU Utilization", fontsize=10.8, fontweight="bold")
    ax_util.tick_params(axis="y", labelsize=8.2)
    ax_util.grid(True, axis="y", alpha=0.25)
    ax_util.spines["top"].set_visible(False)

    fig.suptitle("CPU AMD uProf Roofline Summary", fontsize=12.8, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.89, bottom=0.12, wspace=0.34, hspace=0.34)

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
