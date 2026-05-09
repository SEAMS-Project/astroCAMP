#!/usr/bin/env python3
"""
Plot 26d: Plot 26b variant with memory-efficiency bars on the throughput panel.
"""

from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
CPU_CSV = DERIVED_DIR / "roofline_stacking_summary.csv"
MEMORY_TABLE_CSV = DERIVED_DIR / "problem_size_memory_table.csv"
GPU_PROFILING_DIR = SCRIPT_DIR.parent.parent / "astroCAMP-bench" / "profiling_gpu2"
OUTPUT_STEM = RESULTS_DIR / "plot26d_cpu_gpu_execution_comparison_gibj"
SUMMARY_CSV = DERIVED_DIR / "plot26d_cpu_gpu_execution_comparison_gibj_summary.csv"

ROOF_DRAM_COLOR = "#c44e52"
ROOF_COMPUTE_COLOR = "#4c72b0"
ROOF_SP_COLOR = "0.55"
CPU_SERIES_COLOR = plt.cm.cividis(0.25)
GPU_SERIES_COLOR = "#6b8e23"
GPU_SERIES_DARK = "#556b2f"
THREAD_COLORS = {
    1: "#4c72b0",
    16: "#dd8452",
    32: "#55a868",
    64: "#8172b3",
}
FONT_DELTA = 3.0
AXIS_LABEL_FONTSIZE = 12.5 + FONT_DELTA
TITLE_FONTSIZE = 13.8 + FONT_DELTA
SUPTITLE_FONTSIZE = 16 + FONT_DELTA
TICK_FONTSIZE = 10.3 + FONT_DELTA
XTICK_FONTSIZE = 10.8 + FONT_DELTA
LEGEND_FONTSIZE = 9.0 + FONT_DELTA
VALUE_LABEL_FONTSIZE = 9.1 + FONT_DELTA
P_LEGEND_FONTSIZE = 8.9 + FONT_DELTA
PROBLEM_IMAGE_SIZE = 16384
PROBLEM_TIMESTEPS = 256
PROBLEM_CHANNELS = 256

# These GPU rows come from the user-provided Table 3 summary for the GPU runs.
GPU_TRANSFER_ROWS = [
    {
        "requested_threads": 1,
        "throughput_gflops": 0.3877,
        "bandwidth_gbs": 6.9725,
        "profiler_utilization_pct": 2.0872,
        "data_origin": "Table 3 host-to-device summary",
    },
    {
        "requested_threads": 16,
        "throughput_gflops": 0.3792,
        "bandwidth_gbs": 7.2271,
        "profiler_utilization_pct": 2.5937,
        "data_origin": "Table 3 host-to-device summary",
    },
    {
        "requested_threads": 32,
        "throughput_gflops": 0.4053,
        "bandwidth_gbs": 7.8045,
        "profiler_utilization_pct": 2.1071,
        "data_origin": "Table 3 host-to-device summary",
    },
    {
        "requested_threads": 64,
        "throughput_gflops": 0.3899,
        "bandwidth_gbs": 7.5698,
        "profiler_utilization_pct": 2.2229,
        "data_origin": "Table 3 host-to-device summary",
    },
]


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
    keep = [
        "requested_threads",
        "arithmetic_intensity_flop_per_byte",
        "throughput_gflops",
        "bandwidth_gbs",
        "profiler_utilization_pct",
        "dp_peak_gflops",
        "sp_peak_gflops",
        "dram_bw_gbs",
        "elapsed_s",
    ]
    df = df[keep].copy()
    df["series"] = "CPU-only execution"
    df["data_origin"] = "Local AMD uProf roofline summary"
    df["cpu_memory_gib_per_j"] = np.nan
    df["gpu_device_joules"] = np.nan
    df["problem_footprint_gib"] = np.nan
    df["gpu_device_gib_per_j"] = np.nan
    return df


def load_problem_footprint_gib() -> float:
    df = pd.read_csv(MEMORY_TABLE_CSV)
    row = df[
        (df["Image size"] == PROBLEM_IMAGE_SIZE)
        & (df["Timesteps"] == PROBLEM_TIMESTEPS)
        & (df["Channels"] == PROBLEM_CHANNELS)
    ]
    if row.empty:
        raise ValueError("Missing problem footprint for 16384,256,256")
    return float(row["Input + Output (GiB)"].iloc[0])


def parse_gpu_device_joules_by_threads() -> dict[int, float]:
    device_joules = {}
    for threads in [1, 16, 32, 64]:
        pattern = f"slurm-*t0-255_c0-255_16384pix_20deg_{threads}cores_uprof_collect_inst.txt"
        matches = sorted(GPU_PROFILING_DIR.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Missing GPU collect-inst log for p={threads}")
        total_j = 0.0
        for line in matches[0].read_text().splitlines():
            if "|device:" not in line:
                continue
            for token in [part.strip() for part in line.split(",")]:
                if token.endswith(" Joules"):
                    try:
                        total_j += float(token.split()[0])
                    except ValueError:
                        continue
        device_joules[threads] = total_j
    return device_joules


def load_gpu_transfer_summary(cpu_df: pd.DataFrame) -> pd.DataFrame:
    footprint_gib = load_problem_footprint_gib()
    device_joules = parse_gpu_device_joules_by_threads()

    df = pd.DataFrame(GPU_TRANSFER_ROWS).sort_values("requested_threads").reset_index(drop=True)
    df["arithmetic_intensity_flop_per_byte"] = df["throughput_gflops"] / df["bandwidth_gbs"]
    df["dp_peak_gflops"] = float(cpu_df["dp_peak_gflops"].iloc[0])
    df["sp_peak_gflops"] = float(cpu_df["sp_peak_gflops"].iloc[0])
    df["dram_bw_gbs"] = float(cpu_df["dram_bw_gbs"].iloc[0])
    df["elapsed_s"] = np.nan
    df["series"] = "GPU host-to-device transfer"
    df["gpu_device_joules"] = df["requested_threads"].map(device_joules)
    df["problem_footprint_gib"] = footprint_gib
    df["gpu_device_gib_per_j"] = df["problem_footprint_gib"] / df["gpu_device_joules"]
    df["data_origin"] = (
        df["data_origin"]
        + " + exact device joules from profiling_gpu2 collect-inst logs"
    )
    return df


def export_summary(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame) -> None:
    summary = pd.concat([cpu_df, gpu_df], ignore_index=True)
    summary = summary[
        [
            "series",
            "requested_threads",
            "arithmetic_intensity_flop_per_byte",
            "throughput_gflops",
            "bandwidth_gbs",
            "profiler_utilization_pct",
            "cpu_memory_gib_per_j",
            "gpu_device_joules",
            "problem_footprint_gib",
            "gpu_device_gib_per_j",
            "data_origin",
        ]
    ].rename(
        columns={
            "requested_threads": "threads",
            "profiler_utilization_pct": "utilization_pct",
        }
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary to {SUMMARY_CSV}")


def add_value_labels(ax: plt.Axes, xvals: np.ndarray, yvals: pd.Series, fmt: str, color: str, dy: float) -> None:
    for xval, yval in zip(xvals, yvals):
        ax.text(
            xval,
            yval + dy,
            fmt.format(yval),
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONTSIZE,
            color=color,
        )


def add_secondary_value_labels(
    ax: plt.Axes,
    xvals: np.ndarray,
    yvals: pd.Series,
    color: str,
    dy: float,
) -> None:
    for xval, yval in zip(xvals, yvals):
        ax.text(
            xval,
            yval + dy,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_FONTSIZE - 0.5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        )


def main() -> int:
    cpu_df = load_cpu_summary()
    gpu_df = load_gpu_transfer_summary(cpu_df)
    export_summary(cpu_df, gpu_df)

    dp_peak = float(cpu_df["dp_peak_gflops"].iloc[0])
    sp_peak = float(cpu_df["sp_peak_gflops"].iloc[0])
    dram_bw = float(cpu_df["dram_bw_gbs"].iloc[0])
    dp_ridge = dp_peak / dram_bw
    sp_ridge = sp_peak / dram_bw

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.0))
    ax_roof, ax_thr, ax_bw, ax_util = axes.flatten()

    x_min = 0.03
    x_max = 150.0
    y_min = 0.2
    y_max = sp_peak * 1.25
    ai_diag = np.logspace(np.log10(x_min), np.log10(sp_ridge), 400)

    ax_roof.plot(ai_diag, dram_bw * ai_diag, color=ROOF_DRAM_COLOR, linewidth=2.4, label=f"CPU DRAM roof ({dram_bw:.0f} GB/s)")
    ax_roof.hlines(dp_peak, dp_ridge, x_max, color=ROOF_COMPUTE_COLOR, linestyle="-.", linewidth=2.0, label=f"CPU FP64 peak ({dp_peak:.0f} GFLOP/s)")
    ax_roof.hlines(sp_peak, sp_ridge, x_max, color=ROOF_SP_COLOR, linestyle="--", linewidth=2.4, label=f"CPU FP32 peak ({sp_peak:.0f} GFLOP/s)")
    ax_roof.axvline(dp_ridge, color="0.6", linestyle=":", linewidth=1.0, alpha=0.85)
    ax_roof.axvline(sp_ridge, color="0.6", linestyle=":", linewidth=1.0, alpha=0.85)

    cpu_df_line = cpu_df[cpu_df["requested_threads"] >= 16]
    gpu_df_line = gpu_df[gpu_df["requested_threads"] >= 16]
    ax_roof.plot(
        cpu_df_line["arithmetic_intensity_flop_per_byte"],
        cpu_df_line["throughput_gflops"],
        color=CPU_SERIES_COLOR,
        linewidth=2.0,
        label="CPU-only execution",
    )
    ax_roof.plot(
        gpu_df_line["arithmetic_intensity_flop_per_byte"],
        gpu_df_line["throughput_gflops"],
        color=GPU_SERIES_COLOR,
        linewidth=2.0,
        label="GPU host-to-device transfer",
    )

    cpu_point_colors = [THREAD_COLORS[int(p)] for p in cpu_df["requested_threads"]]
    gpu_point_colors = [THREAD_COLORS[int(p)] for p in gpu_df["requested_threads"]]
    ax_roof.scatter(
        cpu_df["arithmetic_intensity_flop_per_byte"],
        cpu_df["throughput_gflops"],
        color=cpu_point_colors,
        marker="o",
        s=62,
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
    )
    ax_roof.scatter(
        gpu_df["arithmetic_intensity_flop_per_byte"],
        gpu_df["throughput_gflops"],
        color=gpu_point_colors,
        marker="s",
        s=58,
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
    )

    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(x_min, x_max)
    ax_roof.set_ylim(y_min, y_max)
    ax_roof.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_roof.set_ylabel("Throughput (GFLOP/s)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_roof.set_title("Roofline Summary Overlay", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_roof.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    roof_handles, roof_labels = ax_roof.get_legend_handles_labels()
    roof_legend = ax_roof.legend(roof_handles, roof_labels, loc="upper left", fontsize=LEGEND_FONTSIZE, frameon=True)
    ax_roof.add_artist(roof_legend)
    p_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=THREAD_COLORS[p],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=7.8,
            label=f"p={p}",
        )
        for p in [1, 16, 32, 64]
    ]
    ax_roof.legend(
        handles=p_handles,
        loc="lower right",
        fontsize=P_LEGEND_FONTSIZE,
        frameon=True,
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    x = np.arange(len(cpu_df), dtype=float)
    tick_labels = [f"p={int(v)}" for v in cpu_df["requested_threads"]]

    ax_thr.plot(x, cpu_df["throughput_gflops"], color=CPU_SERIES_COLOR, marker="o", linewidth=2.2, label="CPU-only throughput")
    ax_thr.plot(x, gpu_df["throughput_gflops"], color=GPU_SERIES_COLOR, marker="s", linewidth=2.2, label="GPU host-to-device throughput")
    add_value_labels(ax_thr, x, cpu_df["throughput_gflops"], "{:.1f}", CPU_SERIES_COLOR, dy=14.0)
    add_value_labels(ax_thr, x, gpu_df["throughput_gflops"], "{:.3f}", GPU_SERIES_DARK, dy=0.016)
    ax_thr.set_xticks(x)
    ax_thr.set_xticklabels(tick_labels, fontsize=XTICK_FONTSIZE)
    ax_thr.set_xlabel("Parallelism p", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_thr.set_ylabel("Throughput (GFLOP/s)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_thr.set_title("Throughput Summary", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_thr.tick_params(axis="y", labelsize=TICK_FONTSIZE)

    ax_thr_right = ax_thr.twinx()
    bar_width = 0.26
    bars_gpu = ax_thr_right.bar(
        x + 0.18,
        gpu_df["gpu_device_gib_per_j"],
        width=bar_width,
        color=GPU_SERIES_COLOR,
        alpha=0.35,
        edgecolor=GPU_SERIES_DARK,
        linewidth=1.2,
        label="GPU device memory efficiency",
        zorder=1,
    )
    add_secondary_value_labels(
        ax_thr_right,
        x + 0.18,
        gpu_df["gpu_device_gib_per_j"],
        GPU_SERIES_DARK,
        dy=0.0010,
    )
    ax_thr_right.set_ylabel("Memory efficiency (GiB/J)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold", color=GPU_SERIES_DARK)
    ax_thr_right.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors=GPU_SERIES_DARK)
    ax_thr_right.set_ylim(0.0, float(gpu_df["gpu_device_gib_per_j"].max()) * 1.45)

    thr_handles, thr_labels = ax_thr.get_legend_handles_labels()
    gibj_handles, gibj_labels = [bars_gpu], ["GPU device memory efficiency"]
    ax_thr.legend(
        thr_handles + gibj_handles,
        thr_labels + gibj_labels,
        loc="upper left",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
    )

    ax_bw.plot(x, cpu_df["bandwidth_gbs"], color=CPU_SERIES_COLOR, marker="o", linewidth=2.2, label="CPU-only execution")
    ax_bw.plot(x, gpu_df["bandwidth_gbs"], color=GPU_SERIES_COLOR, marker="s", linewidth=2.2, label="GPU host-to-device transfer")
    add_value_labels(ax_bw, x, cpu_df["bandwidth_gbs"], "{:.2f}", CPU_SERIES_COLOR, dy=0.23)
    add_value_labels(ax_bw, x, gpu_df["bandwidth_gbs"], "{:.2f}", GPU_SERIES_DARK, dy=0.23)
    ax_bw.set_xticks(x)
    ax_bw.set_xticklabels(tick_labels, fontsize=XTICK_FONTSIZE)
    ax_bw.set_xlabel("Parallelism p", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_bw.set_ylabel("Bandwidth (GB/s)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_bw.set_title("Bandwidth Summary", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_bw.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_bw.legend(loc="upper left", fontsize=LEGEND_FONTSIZE, frameon=True)

    ax_util.plot(x, cpu_df["profiler_utilization_pct"], color=CPU_SERIES_COLOR, marker="o", linewidth=2.2, label="CPU-only execution")
    ax_util.plot(x, gpu_df["profiler_utilization_pct"], color=GPU_SERIES_COLOR, marker="s", linewidth=2.2, label="GPU host-to-device transfer")
    add_value_labels(ax_util, x, cpu_df["profiler_utilization_pct"], "{:.2f}%", CPU_SERIES_COLOR, dy=0.45)
    add_value_labels(ax_util, x, gpu_df["profiler_utilization_pct"], "{:.2f}%", GPU_SERIES_DARK, dy=0.18)
    ax_util.set_xticks(x)
    ax_util.set_xticklabels(tick_labels, fontsize=XTICK_FONTSIZE)
    ax_util.set_xlabel("Parallelism p", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_util.set_ylabel("Utilization (%)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_util.set_title("Utilization Summary", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_util.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_util.legend(loc="upper left", fontsize=LEGEND_FONTSIZE, frameon=True)

    fig.suptitle(
        "CPU-Only Roofline Summary vs GPU Host-to-Device Transfer Summary",
        fontsize=SUPTITLE_FONTSIZE,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.09, wspace=0.28, hspace=0.30)

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
