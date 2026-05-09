#!/usr/bin/env python3
"""
New Plot 25 variant:
1. CPU roofline for WSClean stacking on the dual-socket EPYC node.
2. Comparative speedup and host utilization for CPU-only vs GPU-accelerated
   WSClean + IDG execution, using the same execution-time and host-side summary
   data as plot26c.
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
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
CPU_CSV = DERIVED_DIR / "roofline_stacking_summary.csv"
OUTPUT_STEM = RESULTS_DIR / "plot25c_cpu_roofline_scalability_comparison"
SUMMARY_CSV = DERIVED_DIR / "plot25c_cpu_roofline_scalability_comparison_summary.csv"

CPU_COLOR = plt.cm.cividis(0.25)
CPU_DARK = plt.cm.cividis(0.10)
GPU_COLOR = plt.cm.cividis(0.78)
GPU_DARK = plt.cm.cividis(0.92)
ROOF_DRAM_COLOR = "#c44e52"
ROOF_DP_COLOR = "#4c72b0"
ROOF_SP_COLOR = "0.55"
EFFICIENCY_COLOR = "#dd8452"

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


def contrast_text_color(facecolor: tuple[float, float, float, float]) -> str:
    r, g, b = facecolor[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.52 else "white"


def parse_elapsed_to_seconds(text: str) -> float:
    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported elapsed time format: {text}")


def load_cpu_roofline_summary() -> pd.DataFrame:
    return pd.read_csv(CPU_CSV).sort_values("requested_threads").reset_index(drop=True)


def load_cpu_scaling(cpu_roof_df: pd.DataFrame) -> pd.DataFrame:
    df = cpu_roof_df[
        [
            "requested_threads",
            "elapsed_s",
            "profiler_utilization_pct",
        ]
    ].copy()
    df["elapsed_s"] = df["requested_threads"].map(lambda p: parse_elapsed_to_seconds(CPU_TABLE2_TIMES[int(p)]))
    base = float(df.loc[df["requested_threads"] == 1, "elapsed_s"].iloc[0])
    df["speedup"] = base / df["elapsed_s"]
    df["series"] = "CPU-only execution"
    return df


def load_gpu_scaling(cpu_roof_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(GPU_TRANSFER_ROWS).sort_values("requested_threads").reset_index(drop=True)
    df["arithmetic_intensity_flop_per_byte"] = df["throughput_gflops"] / df["bandwidth_gbs"]
    df["elapsed_s"] = df["requested_threads"].map(lambda p: parse_elapsed_to_seconds(GPU_TABLE2_TIMES[int(p)]))
    base = float(df.loc[df["requested_threads"] == 1, "elapsed_s"].iloc[0])
    df["speedup"] = base / df["elapsed_s"]
    df["series"] = "GPU-accelerated execution"
    df["processor_name"] = str(cpu_roof_df["processor_name"].iloc[0]).strip()
    return df


def export_summary(cpu_roof_df: pd.DataFrame, cpu_scale_df: pd.DataFrame, gpu_scale_df: pd.DataFrame) -> None:
    cpu_export = cpu_roof_df[
        [
            "requested_threads",
            "elapsed_s",
            "arithmetic_intensity_flop_per_byte",
            "throughput_gflops",
            "bandwidth_gbs",
            "profiler_utilization_pct",
            "walltime_speedup_vs_base",
            "walltime_parallel_efficiency_pct",
            "dp_peak_fraction_pct",
            "dram_bw_fraction_pct",
        ]
    ].copy()
    cpu_export["series"] = "CPU roofline"

    scale_export = pd.concat([cpu_scale_df, gpu_scale_df], ignore_index=True)[
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
    ].rename(columns={"profiler_utilization_pct": "host_utilization_pct"})

    cpu_export = cpu_export.rename(columns={"requested_threads": "threads"})
    scale_export = scale_export.rename(columns={"requested_threads": "threads"})
    export_df = pd.concat([cpu_export, scale_export], ignore_index=True, sort=False)
    export_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved summary to {SUMMARY_CSV}")


def plot_figure(cpu_roof_df: pd.DataFrame, cpu_scale_df: pd.DataFrame, gpu_scale_df: pd.DataFrame) -> None:
    processor = str(cpu_roof_df["processor_name"].iloc[0]).strip()
    dp_peak = float(cpu_roof_df["dp_peak_gflops"].iloc[0])
    sp_peak = float(cpu_roof_df["sp_peak_gflops"].iloc[0])
    dram_bw = float(cpu_roof_df["dram_bw_gbs"].iloc[0])
    ridge = float(cpu_roof_df["ridge_point_flop_per_byte"].iloc[0])

    best_roofline_row = cpu_roof_df.loc[cpu_roof_df["throughput_gflops"].idxmax()]
    colors = plt.cm.cividis(np.linspace(0.15, 0.85, len(cpu_roof_df)))

    fig, (ax_roof, ax_cmp) = plt.subplots(
        1,
        2,
        figsize=(18.8, 8.0),
        gridspec_kw={"width_ratios": [1.38, 1.12]},
    )

    x_min = min(0.04, float(cpu_roof_df["arithmetic_intensity_flop_per_byte"].min()) / 1.8)
    x_max = max(150.0, float(cpu_roof_df["arithmetic_intensity_flop_per_byte"].max()) * 2.0)
    y_min = min(30.0, float(cpu_roof_df["throughput_gflops"].min()) * 0.75)
    y_max = sp_peak * 1.15
    ai_diag = np.logspace(np.log10(x_min), np.log10(ridge), 300)

    ax_roof.plot(
        ai_diag,
        dram_bw * ai_diag,
        color=ROOF_DRAM_COLOR,
        linewidth=2.6,
        label=f"DRAM bandwidth roof ({dram_bw:.1f} GB/s)",
    )
    ax_roof.hlines(
        dp_peak,
        ridge,
        x_max,
        color=ROOF_DP_COLOR,
        linewidth=2.6,
        label=f"Double-precision peak ({dp_peak:.0f} GFLOP/s)",
    )
    ax_roof.hlines(
        sp_peak,
        sp_peak / dram_bw,
        x_max,
        color=ROOF_SP_COLOR,
        linewidth=1.8,
        linestyles="--",
        label=f"Single-precision peak ({sp_peak:.0f} GFLOP/s)",
    )
    ax_roof.axvline(ridge, color="0.6", linestyle=":", linewidth=1.1)
    ax_roof.plot(
        cpu_roof_df["arithmetic_intensity_flop_per_byte"],
        cpu_roof_df["throughput_gflops"],
        color="0.35",
        linewidth=1.2,
        alpha=0.85,
        zorder=2,
    )

    label_offsets = {1: (-34, -10), 16: (-42, 18), 32: (16, 18), 64: (18, -18)}
    for color, row in zip(colors, cpu_roof_df.to_dict("records")):
        ax_roof.scatter(
            row["arithmetic_intensity_flop_per_byte"],
            row["throughput_gflops"],
            s=95,
            color=color,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )
        offset = label_offsets.get(int(row["requested_threads"]), (8, 8))
        ax_roof.annotate(
            f"p={int(row['requested_threads'])}",
            (row["arithmetic_intensity_flop_per_byte"], row["throughput_gflops"]),
            textcoords="offset points",
            xytext=offset,
            fontsize=11.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "0.75",
                "linewidth": 0.6,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "0.45",
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    ax_roof.plot(
        gpu_scale_df["arithmetic_intensity_flop_per_byte"],
        gpu_scale_df["throughput_gflops"],
        color=GPU_COLOR,
        linewidth=1.0,
        alpha=0.7,
        zorder=2,
    )
    ax_roof.scatter(
        gpu_scale_df["arithmetic_intensity_flop_per_byte"],
        gpu_scale_df["throughput_gflops"],
        s=88,
        marker="s",
        color=GPU_COLOR,
        edgecolors="black",
        linewidth=0.8,
        zorder=3,
        label="GPU host-device transfer summary",
    )

    roofline_text = (
        f"Arithmetic intensity remains {cpu_roof_df['arithmetic_intensity_flop_per_byte'].min():.1f}-"
        f"{cpu_roof_df['arithmetic_intensity_flop_per_byte'].max():.1f} FLOP/B\n"
        f"All CPU runs stay on the compute side of the ridge ({ridge:.2f} FLOP/B)\n"
        f"Best CPU roofline point: p={int(best_roofline_row['requested_threads'])} = "
        f"{best_roofline_row['throughput_gflops']:.1f} GFLOP/s\n"
        f"That is {best_roofline_row['dp_peak_fraction_pct']:.1f}% of DP peak and "
        f"{best_roofline_row['dram_bw_fraction_pct']:.2f}% of peak DRAM bandwidth\n"
        f"GPU host-device transfer points stay near AI {gpu_scale_df['arithmetic_intensity_flop_per_byte'].min():.3f}-"
        f"{gpu_scale_df['arithmetic_intensity_flop_per_byte'].max():.3f} FLOP/B and "
        f"{gpu_scale_df['throughput_gflops'].min():.3f}-{gpu_scale_df['throughput_gflops'].max():.3f} GFLOP/s"
    )
    ax_roof.text(
        0.03,
        0.05,
        roofline_text,
        transform=ax_roof.transAxes,
        fontsize=11.4,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(x_min, x_max)
    ax_roof.set_ylim(y_min, y_max)
    ax_roof.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=14, fontweight="bold")
    ax_roof.set_ylabel("Throughput (GFLOP/s)", fontsize=14, fontweight="bold")
    ax_roof.set_title("CPU Roofline", fontsize=15, fontweight="bold")
    ax_roof.tick_params(axis="both", labelsize=12)
    roof_legend = ax_roof.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=4,
        fontsize=10.7,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.8,
        borderaxespad=0.0,
    )

    threads = cpu_scale_df["requested_threads"].astype(int).tolist()
    x = np.arange(len(threads), dtype=float)
    width = 0.34

    cpu_bars = ax_cmp.bar(
        x - width / 2,
        cpu_scale_df["speedup"],
        width=width,
        color=CPU_COLOR,
        edgecolor="black",
        linewidth=0.7,
        label="CPU-only speedup",
    )
    gpu_bars = ax_cmp.bar(
        x + width / 2,
        gpu_scale_df["speedup"],
        width=width,
        color=GPU_COLOR,
        edgecolor="black",
        linewidth=0.7,
        label="GPU-accelerated speedup",
    )

    for bars, frame in ((cpu_bars, cpu_scale_df), (gpu_bars, gpu_scale_df)):
        for bar, row in zip(bars, frame.to_dict("records")):
            text_color = contrast_text_color(bar.get_facecolor())
            ax_cmp.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(bar.get_height() * 0.55, 0.16),
                f"{row['speedup']:.2f}x\n{row['elapsed_s'] / 3600.0:.2f} h",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color=text_color,
            )

    ax_util = ax_cmp.twinx()
    cpu_line = ax_util.plot(
        x,
        cpu_scale_df["profiler_utilization_pct"],
        color=CPU_DARK,
        marker="o",
        linewidth=2.1,
        alpha=0.62,
        label="CPU-only host utilization",
    )
    gpu_line = ax_util.plot(
        x,
        gpu_scale_df["profiler_utilization_pct"],
        color=GPU_DARK,
        marker="s",
        linewidth=2.1,
        alpha=0.62,
        label="GPU-accelerated host utilization",
    )

    cpu_util_offsets = {1: (-22, 4), 16: (-26, 20), 32: (-26, 26), 64: (-26, 32)}
    gpu_util_offsets = {1: (20, 50), 16: (24, -24), 32: (24, -30), 64: (24, -36)}
    for xv, thread, yv in zip(x, threads, cpu_scale_df["profiler_utilization_pct"]):
        offset = cpu_util_offsets[int(thread)]
        ax_util.annotate(
            f"{yv:.2f}%",
            (xv, yv),
            textcoords="offset points",
            xytext=offset,
            color="black",
            fontsize=10.3,
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] >= 0 else "top",
            annotation_clip=True,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8", "linewidth": 0.5},
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.6, "shrinkA": 2, "shrinkB": 4},
        )
    for xv, thread, yv in zip(x, threads, gpu_scale_df["profiler_utilization_pct"]):
        offset = gpu_util_offsets[int(thread)]
        ax_util.annotate(
            f"{yv:.2f}%",
            (xv, yv),
            textcoords="offset points",
            xytext=offset,
            color="black",
            fontsize=10.3,
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] >= 0 else "top",
            annotation_clip=True,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8", "linewidth": 0.5},
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.6, "shrinkA": 2, "shrinkB": 4},
        )

    compare_text = (
        f"CPU-only scaling: {cpu_scale_df['speedup'].max():.2f}x at p={int(cpu_scale_df.loc[cpu_scale_df['speedup'].idxmax(), 'requested_threads'])}\n"
        f"GPU-accelerated run time stays near {gpu_scale_df['elapsed_s'].min() / 3600.0:.2f}-"
        f"{gpu_scale_df['elapsed_s'].max() / 3600.0:.2f} h across p\n"
        f"GPU panel utilization is host-side only, taken from the transfer summary"
    )
    ax_cmp.text(
        0.03,
        0.95,
        compare_text,
        transform=ax_cmp.transAxes,
        fontsize=11.4,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels([f"p={p}" for p in threads], fontsize=13)
    ax_cmp.set_xlabel("Parallelism p  (-j threads)", fontsize=14, fontweight="bold")
    ax_cmp.set_ylabel(r"Wall-time speedup $S(p)=T_1/T_p$", fontsize=14, fontweight="bold")
    ax_cmp.set_title("Comparative Scalability and Host Utilization", fontsize=15, fontweight="bold")
    ax_cmp.tick_params(axis="y", labelsize=12)
    ax_cmp.set_xlim(-0.55, len(threads) - 0.45)
    ax_cmp.set_ylim(0.0, max(cpu_scale_df["speedup"].max(), gpu_scale_df["speedup"].max()) * 1.28)

    ax_util.set_ylabel("CPU utilization (%)", fontsize=14, fontweight="bold")
    ax_util.tick_params(axis="y", labelsize=12)
    ax_util.set_ylim(0.0, max(cpu_scale_df["profiler_utilization_pct"].max(), gpu_scale_df["profiler_utilization_pct"].max()) * 1.42)

    handles = [cpu_bars, gpu_bars, cpu_line[0], gpu_line[0]]
    labels = [
        "CPU-only speedup",
        "GPU-accelerated speedup",
        "CPU-only host utilization",
        "GPU-accelerated host utilization",
    ]
    scale_legend = ax_cmp.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        fontsize=10.7,
        frameon=True,
        columnspacing=1.3,
        handlelength=2.7,
        borderaxespad=0.0,
    )

    fig.suptitle(
        "Roofline Analysis and Comparative Scalability for 16384^2, t=256, c=256 WSClean + IDG Stacking\n"
        f"Dual-socket node: 2 x {processor} | p denotes WSClean thread count (-j) | CPU roofline + GPU host-device transfer + CPU/GPU host-side scaling",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    for legend in (roof_legend, scale_legend):
        legend.get_frame().set_edgecolor("0.75")
        legend.get_frame().set_alpha(0.95)

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.27, top=0.84, wspace=0.12)

    for suffix in (".png", ".pdf"):
        out_path = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out_path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
    plt.close(fig)


def main() -> int:
    cpu_roof_df = load_cpu_roofline_summary()
    cpu_scale_df = load_cpu_scaling(cpu_roof_df)
    gpu_scale_df = load_gpu_scaling(cpu_roof_df)
    export_summary(cpu_roof_df, cpu_scale_df, gpu_scale_df)
    plot_figure(cpu_roof_df, cpu_scale_df, gpu_scale_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
