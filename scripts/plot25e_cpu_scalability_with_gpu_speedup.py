#!/usr/bin/env python3
"""
Single-panel Plot 25 variant with grouped bars for CPU-only and heterogeneous
CPU+GPU wall-time speedup, plus the CPU parallel-efficiency line.
"""

import argparse
import os
import sys
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

from plot25_cpu_roofline_stacking import (
    CPU_BAR_COLOR,
    DEFAULT_INPUT_ROOT,
    EFFICIENCY_COLOR,
    contrast_text_color,
    load_roofline_dataframe,
)
from plot26c_cpu_gpu_speedup_utilization import GPU_TABLE2_TIMES, parse_elapsed_to_seconds


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
DEFAULT_OUTPUT_STEM = RESULTS_DIR / "plot25e_cpu_scalability_with_gpu_speedup"
DEFAULT_SUMMARY_CSV = DERIVED_DIR / "plot25e_cpu_scalability_with_gpu_speedup_summary.csv"
GPU_BAR_COLOR = plt.cm.cividis(0.78)

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a grouped scalability figure with CPU-only and CPU+GPU speedup."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Directory containing roofline report.json files (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help=f"Output path stem for PNG/PDF files (default: {DEFAULT_OUTPUT_STEM})",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Output CSV for the derived summary table (default: {DEFAULT_SUMMARY_CSV})",
    )
    return parser.parse_args()


def build_gpu_speedup_dataframe(cpu_df: pd.DataFrame) -> pd.DataFrame:
    threads = cpu_df["requested_threads"].astype(int).tolist()
    rows = []
    for p in threads:
        cpu_elapsed_s = float(cpu_df.loc[cpu_df["requested_threads"] == int(p), "elapsed_s"].iloc[0])
        elapsed_s = parse_elapsed_to_seconds(GPU_TABLE2_TIMES[int(p)])
        rows.append(
            {
                "requested_threads": int(p),
                "elapsed_s": elapsed_s,
                "speedup": cpu_elapsed_s / elapsed_s,
                "series": "CPU+GPU execution",
            }
        )
    return pd.DataFrame(rows)


def export_summary(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame, summary_csv: Path) -> None:
    cpu_export = cpu_df[
        [
            "requested_threads",
            "elapsed_s",
            "walltime_speedup_vs_base",
            "walltime_parallel_efficiency_pct",
            "throughput_gflops",
        ]
    ].copy()
    cpu_export["series"] = "CPU-only execution"
    cpu_export = cpu_export.rename(
        columns={
            "requested_threads": "threads",
            "walltime_speedup_vs_base": "speedup",
            "walltime_parallel_efficiency_pct": "parallel_efficiency_pct",
        }
    )

    gpu_export = gpu_df.rename(columns={"requested_threads": "threads"})
    gpu_export["parallel_efficiency_pct"] = np.nan
    gpu_export["throughput_gflops"] = np.nan

    export_df = pd.concat([cpu_export, gpu_export], ignore_index=True, sort=False)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")


def plot_scalability_with_gpu(df: pd.DataFrame, gpu_df: pd.DataFrame, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    best_scaling_row = df.loc[df["elapsed_s"].idxmin()]
    fig, ax_scale = plt.subplots(1, 1, figsize=(10.4, 7.6))
    ax_eff = ax_scale.twinx()

    x = np.arange(len(df))
    width = 0.34
    cpu_bars = ax_scale.bar(
        x - width / 2,
        df["walltime_speedup_vs_base"],
        width=width,
        color=CPU_BAR_COLOR,
        edgecolor="black",
        linewidth=0.7,
        label="CPU-only wall-time speedup",
    )
    gpu_bars = ax_scale.bar(
        x + width / 2,
        gpu_df["speedup"],
        width=width,
        color=GPU_BAR_COLOR,
        edgecolor="black",
        linewidth=0.7,
        label="CPU+GPU acceleration over CPU-only",
    )

    eff_line = ax_eff.plot(
        x,
        df["walltime_parallel_efficiency_pct"],
        color=EFFICIENCY_COLOR,
        marker="o",
        linewidth=3.0,
        markersize=6.5,
        alpha=0.78,
        label=r"CPU parallel efficiency $E(p)=S(p)/p$",
    )
    eff_ref = ax_eff.axhline(
        100.0,
        color="0.45",
        linestyle=":",
        linewidth=1.4,
        label="Ideal efficiency",
    )

    for bar, row in zip(cpu_bars, df.to_dict("records")):
        text_color = contrast_text_color(bar.get_facecolor())
        ax_scale.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(bar.get_height() * 0.52, 0.15),
            f"{row['walltime_speedup_vs_base']:.2f}x\n{row['elapsed_s'] / 3600.0:.2f} h",
            ha="center",
            va="center",
            fontsize=10.1,
            fontweight="bold",
            color=text_color,
        )
    for bar, row in zip(gpu_bars, gpu_df.to_dict("records")):
        text_color = contrast_text_color(bar.get_facecolor())
        ax_scale.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(bar.get_height() * 0.52, 0.15),
            f"{row['speedup']:.2f}x\n{row['elapsed_s'] / 3600.0:.2f} h",
            ha="center",
            va="center",
            fontsize=10.1,
            fontweight="bold",
            color=text_color,
        )

    scaling_text = (
        f"CPU-only best speedup: {best_scaling_row['walltime_speedup_vs_base']:.2f}x at "
        f"p={int(best_scaling_row['requested_threads'])}\n"
        f"Matched-p CPU+GPU runtime stays near {gpu_df['elapsed_s'].min() / 3600.0:.2f}-"
        f"{gpu_df['elapsed_s'].max() / 3600.0:.2f} h across p\n"
        r"CPU+GPU bars use $T_{\mathrm{CPU}}(p) / T_{\mathrm{CPU+GPU}}(p)$"
    )
    ax_scale.text(
        0.03,
        0.95,
        scaling_text,
        transform=ax_scale.transAxes,
        fontsize=11.4,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    ax_scale.set_xticks(x)
    ax_scale.set_xticklabels([f"p={int(v)}" for v in df["requested_threads"]], fontsize=13)
    ax_scale.set_xlabel("Parallelism p  (-j threads)", fontsize=14, fontweight="bold")
    ax_scale.set_ylabel(r"Wall-time speedup $S(p)=T_1/T_p$", fontsize=14, fontweight="bold")
    ax_scale.set_ylim(0.0, max(float(df["walltime_speedup_vs_base"].max()), float(gpu_df["speedup"].max())) * 1.34)
    ax_scale.set_title("Scalability Summary with CPU+GPU Acceleration", fontsize=15, fontweight="bold")
    ax_scale.grid(axis="y", alpha=0.3, linestyle="--")
    ax_scale.tick_params(axis="y", labelsize=12)

    ax_eff.set_ylabel(
        r"CPU parallel efficiency $E(p)=S(p)/p$ (%)",
        fontsize=14,
        fontweight="bold",
        color=EFFICIENCY_COLOR,
    )
    ax_eff.tick_params(axis="y", labelcolor=EFFICIENCY_COLOR, labelsize=12)
    ax_eff.set_ylim(0.0, 105.0)

    legend_handles = [cpu_bars, gpu_bars, eff_line[0], eff_ref]
    legend_labels = [
        "CPU-only wall-time speedup",
        "CPU+GPU acceleration over CPU-only",
        "CPU parallel efficiency",
        "Ideal efficiency",
    ]
    legend = ax_scale.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=10.6,
        frameon=True,
        columnspacing=1.4,
        handlelength=2.6,
        borderaxespad=0.0,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_alpha(0.95)

    processor = str(df["processor_name"].iloc[0]).strip()
    fig.suptitle(
        "Scalability Summary for 16384^2, t=256, c=256 CPU Stacking Runs\n"
        f"Dual-socket node: 2 x {processor} | p denotes WSClean thread count (-j)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.25, top=0.82)

    saved_paths = []
    for suffix in (".png", ".pdf"):
        out_path = output_stem.with_suffix(suffix)
        fig.savefig(out_path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def main() -> int:
    args = parse_args()
    if not args.input_root.exists():
        print(f"Skipping CPU+GPU scalability plot: input directory not found: {args.input_root}")
        return 0

    df, _ = load_roofline_dataframe(args.input_root)
    gpu_df = build_gpu_speedup_dataframe(df)
    export_summary(df, gpu_df, args.summary_csv)
    for path in plot_scalability_with_gpu(df, gpu_df, args.output_stem):
        print(f"Saved figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
