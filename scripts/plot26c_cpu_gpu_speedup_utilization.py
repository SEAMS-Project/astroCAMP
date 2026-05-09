#!/usr/bin/env python3
"""
Alternative CPU/GPU comparison with two panels:
1) roofline overlay
2) dual-axis speedup and utilization versus parallelism
"""

import os
import tempfile
from pathlib import Path

TMP_CACHE_ROOT = Path(tempfile.gettempdir()) / "astrocamp-mpl"
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE_ROOT / "xdg-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CPU_CSV = SCRIPT_DIR / "roofline_stacking_summary.csv"
OUTPUT_STEM = SCRIPT_DIR / "plot26c_cpu_gpu_speedup_utilization"
SUMMARY_CSV = SCRIPT_DIR / "plot26c_cpu_gpu_speedup_utilization_summary.csv"

CPU_COLOR = plt.cm.cividis(0.25)
CPU_DARK = plt.cm.cividis(0.10)
GPU_COLOR = plt.cm.cividis(0.78)
GPU_DARK = plt.cm.cividis(0.92)
ROOF_COLOR = "#c44e52"
ROOF_SP_COLOR = "0.55"
FONT_DELTA = 3.0

AXIS_LABEL_FONTSIZE = 12.5 + FONT_DELTA
TITLE_FONTSIZE = 14 + FONT_DELTA
SUPTITLE_FONTSIZE = 16 + FONT_DELTA
TICK_FONTSIZE = 10.4 + FONT_DELTA
LEGEND_FONTSIZE = 9.0 + FONT_DELTA
ANNOTATION_FONTSIZE = 9.2 + FONT_DELTA
BAR_TEXT_FONTSIZE = 8.7 + FONT_DELTA
UTIL_TEXT_FONTSIZE = 8.8 + FONT_DELTA
THREAD_COLOR_MAP = {
    1: plt.cm.cividis(0.15),
    16: plt.cm.cividis(0.38),
    32: plt.cm.cividis(0.62),
    64: plt.cm.cividis(0.86),
}

GPU_TRANSFER_ROWS = [
    {"requested_threads": 1, "throughput_gflops": 0.3877, "bandwidth_gbs": 6.9725, "profiler_utilization_pct": 2.0872},
    {"requested_threads": 16, "throughput_gflops": 0.3792, "bandwidth_gbs": 7.2271, "profiler_utilization_pct": 2.5937},
    {"requested_threads": 32, "throughput_gflops": 0.4053, "bandwidth_gbs": 7.8045, "profiler_utilization_pct": 2.1071},
    {"requested_threads": 64, "throughput_gflops": 0.3899, "bandwidth_gbs": 7.5698, "profiler_utilization_pct": 2.2229},
]

CPU_TABLE2_TIMES = {
    1: "3:13:10",
    16: "1:08:28",
    32: "1:08:10",
    64: "1:08:06",
}

GPU_TABLE2_TIMES = {
    1: "1:09:27",
    16: "1:09:57",
    32: "1:09:14",
    64: "1:08:45",
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


def parse_elapsed_to_seconds(text: str) -> float:
    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported elapsed time format: {text}")


def load_cpu_summary() -> pd.DataFrame:
    cpu_df = pd.read_csv(CPU_CSV).sort_values("requested_threads").reset_index(drop=True)
    cpu_df = cpu_df[
        [
            "requested_threads",
            "elapsed_s",
            "arithmetic_intensity_flop_per_byte",
            "throughput_gflops",
            "bandwidth_gbs",
            "profiler_utilization_pct",
            "dp_peak_gflops",
            "sp_peak_gflops",
            "dram_bw_gbs",
        ]
    ].copy()
    cpu_df["elapsed_s"] = cpu_df["requested_threads"].map(lambda p: parse_elapsed_to_seconds(CPU_TABLE2_TIMES[int(p)]))
    base = float(cpu_df.loc[cpu_df["requested_threads"] == 1, "elapsed_s"].iloc[0])
    cpu_df["speedup"] = base / cpu_df["elapsed_s"]
    cpu_df["series"] = "CPU-only application"
    return cpu_df


def load_gpu_walltimes() -> pd.DataFrame:
    rows = [
        {
            "requested_threads": threads,
            "gpu_elapsed_label": label,
            "gpu_elapsed_s": parse_elapsed_to_seconds(label),
        }
        for threads, label in sorted(GPU_TABLE2_TIMES.items())
    ]
    return pd.DataFrame(rows)


def load_gpu_summary(cpu_df: pd.DataFrame) -> pd.DataFrame:
    gpu_df = pd.DataFrame(GPU_TRANSFER_ROWS).sort_values("requested_threads").reset_index(drop=True)
    gpu_df["arithmetic_intensity_flop_per_byte"] = gpu_df["throughput_gflops"] / gpu_df["bandwidth_gbs"]
    wall_df = load_gpu_walltimes()
    gpu_df = gpu_df.merge(wall_df, on="requested_threads", how="left")
    base = float(gpu_df.loc[gpu_df["requested_threads"] == 1, "gpu_elapsed_s"].iloc[0])
    gpu_df["elapsed_s"] = gpu_df["gpu_elapsed_s"]
    gpu_df["speedup"] = base / gpu_df["elapsed_s"]
    gpu_df["dp_peak_gflops"] = float(cpu_df["dp_peak_gflops"].iloc[0])
    gpu_df["sp_peak_gflops"] = float(cpu_df["sp_peak_gflops"].iloc[0])
    gpu_df["dram_bw_gbs"] = float(cpu_df["dram_bw_gbs"].iloc[0])
    gpu_df["series"] = "GPU-accelerated execution (host-side summary)"
    return gpu_df


def export_summary(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame) -> None:
    export_df = pd.concat([cpu_df, gpu_df], ignore_index=True)
    export_df = export_df[
        [
            "series",
            "requested_threads",
            "elapsed_s",
            "speedup",
            "arithmetic_intensity_flop_per_byte",
            "throughput_gflops",
            "bandwidth_gbs",
            "profiler_utilization_pct",
        ]
    ].rename(
        columns={
            "requested_threads": "threads",
            "profiler_utilization_pct": "utilization_pct",
        }
    )
    export_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary to {SUMMARY_CSV}")


def plot_roofline_points(ax: plt.Axes, df: pd.DataFrame, marker: str, label: str) -> None:
    point_colors = [THREAD_COLOR_MAP[int(p)] for p in df["requested_threads"]]
    ax.scatter(
        df["arithmetic_intensity_flop_per_byte"],
        df["throughput_gflops"],
        color=point_colors,
        marker=marker,
        s=76,
        edgecolors="black",
        linewidth=0.8,
        label=label,
        zorder=3,
    )


def contrast_text_color(facecolor: tuple[float, float, float, float]) -> str:
    r, g, b = facecolor[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.52 else "white"


def main() -> int:
    cpu_df = load_cpu_summary()
    gpu_df = load_gpu_summary(cpu_df)
    export_summary(cpu_df, gpu_df)

    dp_peak = float(cpu_df["dp_peak_gflops"].iloc[0])
    sp_peak = float(cpu_df["sp_peak_gflops"].iloc[0])
    dram_bw = float(cpu_df["dram_bw_gbs"].iloc[0])
    dp_ridge = dp_peak / dram_bw
    sp_ridge = sp_peak / dram_bw

    fig = plt.figure(figsize=(21.0, 8.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.24, 0.12, 1.18])
    ax_roof = fig.add_subplot(grid[0, 0])
    ax_cmp = fig.add_subplot(grid[0, 2])

    x_min = 0.03
    x_max = 150.0
    y_min = 0.2
    y_max = sp_peak * 1.22
    ai_diag = np.logspace(np.log10(x_min), np.log10(sp_ridge), 400)

    ax_roof.plot(ai_diag, dram_bw * ai_diag, color=ROOF_COLOR, linewidth=2.4, label=f"CPU DRAM roof ({dram_bw:.0f} GB/s)")
    ax_roof.hlines(dp_peak, dp_ridge, x_max, color=CPU_DARK, linestyle="-.", linewidth=2.0, label=f"CPU FP64 peak ({dp_peak:.0f} GFLOP/s)")
    ax_roof.hlines(sp_peak, sp_ridge, x_max, color=ROOF_SP_COLOR, linestyle="--", linewidth=2.4, label=f"CPU FP32 peak ({sp_peak:.0f} GFLOP/s)")
    ax_roof.axvline(dp_ridge, color="0.6", linestyle=":", linewidth=1.0, alpha=0.85)
    ax_roof.axvline(sp_ridge, color="0.6", linestyle=":", linewidth=1.0, alpha=0.85)

    plot_roofline_points(
        ax_roof,
        cpu_df,
        marker="o",
        label="CPU-only application",
    )
    plot_roofline_points(
        ax_roof,
        gpu_df,
        marker="s",
        label="GPU-accelerated execution (host-side transfer summary)",
    )
    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(x_min, x_max)
    ax_roof.set_ylim(y_min, y_max)
    ax_roof.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_roof.set_ylabel("Throughput (GFLOP/s)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_roof.set_title("Roofline Comparison", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_roof.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    series_handles, series_labels = ax_roof.get_legend_handles_labels()
    series_legend = ax_roof.legend(
        series_handles,
        series_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.8,
    )
    ax_roof.add_artist(series_legend)
    thread_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=THREAD_COLOR_MAP[p],
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=8.2,
            label=f"p={p}",
        )
        for p in [1, 16, 32, 64]
    ]
    ax_roof.legend(
        handles=thread_handles,
        loc="lower right",
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        columnspacing=1.1,
        handletextpad=0.5,
    )

    threads = cpu_df["requested_threads"].astype(int).tolist()
    x = np.arange(len(threads), dtype=float)
    width = 0.34

    cpu_bar = ax_cmp.bar(x - width / 2, cpu_df["speedup"], width=width, color=CPU_COLOR, edgecolor="black", linewidth=0.6, label="CPU-only speedup")
    gpu_bar = ax_cmp.bar(x + width / 2, gpu_df["speedup"], width=width, color=GPU_COLOR, edgecolor="black", linewidth=0.6, label="GPU-accelerated speedup")
    for bars, frame in ((cpu_bar, cpu_df), (gpu_bar, gpu_df)):
        for rect, (_, row) in zip(bars, frame.iterrows()):
            height = rect.get_height()
            elapsed_h = row["elapsed_s"] / 3600.0
            text_color = contrast_text_color(rect.get_facecolor())
            ax_cmp.text(
                rect.get_x() + rect.get_width() / 2,
                max(height * 0.55, 0.16),
                f"{height:.2f}x\n{elapsed_h:.2f} h",
                ha="center",
                va="center",
                fontsize=BAR_TEXT_FONTSIZE,
                fontweight="bold",
                color=text_color,
            )

    ax_util = ax_cmp.twinx()
    ax_util.plot(x, cpu_df["profiler_utilization_pct"], color=CPU_DARK, marker="o", linewidth=2.1, alpha=0.6, label="CPU-only host utilization")
    ax_util.plot(x, gpu_df["profiler_utilization_pct"], color=GPU_DARK, marker="s", linewidth=2.1, alpha=0.6, label="GPU-accelerated host utilization")
    cpu_util_offsets = {1: (-24, 8), 16: (-26, 22), 32: (-26, 28), 64: (-26, 34)}
    gpu_util_offsets = {1: (20, 52), 16: (24, -24), 32: (24, -30), 64: (24, -36)}
    for xv, thread, yv in zip(x, threads, cpu_df["profiler_utilization_pct"]):
        offset = cpu_util_offsets.get(int(thread), (-10, 10))
        ax_util.annotate(
            f"{yv:.2f}%",
            (xv, yv),
            textcoords="offset points",
            xytext=offset,
            color="black",
            fontsize=UTIL_TEXT_FONTSIZE,
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] >= 0 else "top",
            annotation_clip=True,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8", "linewidth": 0.5},
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.6, "shrinkA": 2, "shrinkB": 4},
        )
    for xv, thread, yv in zip(x, threads, gpu_df["profiler_utilization_pct"]):
        offset = gpu_util_offsets.get(int(thread), (10, -10))
        ax_util.annotate(
            f"{yv:.2f}%",
            (xv, yv),
            textcoords="offset points",
            xytext=offset,
            color="black",
            fontsize=UTIL_TEXT_FONTSIZE,
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] >= 0 else "top",
            annotation_clip=True,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8", "linewidth": 0.5},
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.6, "shrinkA": 2, "shrinkB": 4},
        )

    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels([f"p={p}" for p in threads], fontsize=TICK_FONTSIZE)
    ax_cmp.set_xlabel("Parallelism p", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_cmp.set_ylabel("Wall-time speedup vs p=1", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_util.set_ylabel("CPU utilization (%)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    ax_cmp.set_title("Speedup and Host Utilization vs Parallelism", fontsize=TITLE_FONTSIZE, fontweight="bold")
    ax_cmp.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_util.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_cmp.set_xlim(-0.55, len(threads) - 0.45)
    ax_cmp.set_ylim(0.0, max(cpu_df["speedup"].max(), gpu_df["speedup"].max()) * 1.28)
    ax_util.set_ylim(0.0, max(cpu_df["profiler_utilization_pct"].max(), gpu_df["profiler_utilization_pct"].max()) * 1.42)

    handles1, labels1 = ax_cmp.get_legend_handles_labels()
    handles2, labels2 = ax_util.get_legend_handles_labels()
    ax_cmp.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.8,
    )

    fig.suptitle(
        "WSClean + IDG on CPU vs WSClean + IDG on GPU: 16k images, 256 c, 256 t | Roofline, Speedup, and CPU Utilization",
        fontsize=SUPTITLE_FONTSIZE,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.045, right=0.965, top=0.88, bottom=0.24)

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
