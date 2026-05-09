#!/usr/bin/env python3
"""
Plot 25 variant that merges the original CPU roofline panel with the standalone
scaling-gap panel derived from matched CPU-only profiling logs.
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
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot25_cpu_roofline_stacking import (
    CPU_BAR_COLOR,
    DEFAULT_INPUT_ROOT,
    EFFICIENCY_COLOR,
    load_roofline_dataframe,
)
from plot31_cpu_scaling_phase_limits import (
    DEFAULT_PROFILING_ROOT,
    GRIDDING_SPEEDUP_COLOR,
    build_summary_dataframe,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
DEFAULT_OUTPUT_STEM = RESULTS_DIR / "plot25f_cpu_roofline_with_scaling_gap"
DEFAULT_SUMMARY_CSV = DERIVED_DIR / "plot25f_cpu_roofline_with_scaling_gap_summary.csv"

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
        description="Merge the Plot 25 CPU roofline with the standalone scaling-gap panel."
    )
    parser.add_argument(
        "--roofline-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Directory containing Plot 25 AMD uProf roofline exports (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=DEFAULT_PROFILING_ROOT,
        help=f"Directory containing matched CPU-only .out and collect_inst logs (default: {DEFAULT_PROFILING_ROOT})",
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


def export_summary(roofline_df: pd.DataFrame, gap_df: pd.DataFrame, summary_csv: Path) -> None:
    export_df = roofline_df.merge(
        gap_df[
            [
                "requested_threads",
                "gridding_host_speedup_vs_base",
                "gridding_median_mvis_per_s",
                "inversion_median_s",
                "fixed_phase_share_pct",
            ]
        ],
        on="requested_threads",
        how="left",
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")


def plot_merged_figure(
    roofline_df: pd.DataFrame, gap_df: pd.DataFrame, output_stem: Path
) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    dp_peak = float(roofline_df["dp_peak_gflops"].iloc[0])
    sp_peak = float(roofline_df["sp_peak_gflops"].iloc[0])
    dram_bw = float(roofline_df["dram_bw_gbs"].iloc[0])
    ridge = float(roofline_df["ridge_point_flop_per_byte"].iloc[0])
    processor = str(roofline_df["processor_name"].iloc[0]).strip()

    best_roofline_row = roofline_df.loc[roofline_df["throughput_gflops"].idxmax()]
    colors = plt.cm.cividis(np.linspace(0.15, 0.85, len(roofline_df)))

    fig, (ax_roof, ax_gap) = plt.subplots(
        1,
        2,
        figsize=(18.2, 8.0),
        gridspec_kw={"width_ratios": [1.28, 1.02]},
    )
    ax_eff = ax_gap.twinx()

    x_min = min(0.04, float(roofline_df["arithmetic_intensity_flop_per_byte"].min()) / 1.8)
    x_max = max(150.0, float(roofline_df["arithmetic_intensity_flop_per_byte"].max()) * 2.0)
    y_min = min(30.0, float(roofline_df["throughput_gflops"].min()) * 0.75)
    y_max = sp_peak * 1.15

    ai_diag = np.logspace(np.log10(x_min), np.log10(ridge), 300)
    ax_roof.plot(
        ai_diag,
        dram_bw * ai_diag,
        color="#c44e52",
        linewidth=2.6,
        label=f"DRAM bandwidth roof ({dram_bw:.1f} GB/s)",
    )
    ax_roof.hlines(
        dp_peak,
        ridge,
        x_max,
        color="#4c72b0",
        linewidth=2.6,
        label=f"Double-precision peak ({dp_peak:.0f} GFLOP/s)",
    )
    ax_roof.hlines(
        sp_peak,
        sp_peak / dram_bw,
        x_max,
        color="0.55",
        linewidth=1.8,
        linestyles="--",
        label=f"Single-precision peak ({sp_peak:.0f} GFLOP/s)",
    )
    ax_roof.axvline(ridge, color="0.6", linestyle=":", linewidth=1.1)
    ax_roof.plot(
        roofline_df["arithmetic_intensity_flop_per_byte"],
        roofline_df["throughput_gflops"],
        color="0.35",
        linewidth=1.2,
        alpha=0.85,
        zorder=2,
    )

    label_offsets = {
        1: (-34, -10),
        16: (-42, 18),
        32: (16, 18),
        64: (18, -18),
    }
    for color, row in zip(colors, roofline_df.to_dict("records")):
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

    ax_roof.text(
        ridge * 1.05,
        dp_peak * 0.35,
        f"ridge = {ridge:.2f} FLOP/B",
        color="0.35",
        fontsize=11,
        rotation=90,
        va="center",
    )

    summary_text = (
        f"Arithmetic intensity stays at {roofline_df['arithmetic_intensity_flop_per_byte'].min():.1f}-"
        f"{roofline_df['arithmetic_intensity_flop_per_byte'].max():.1f} FLOP/B\n"
        f"All runs land on the compute side of the ridge\n"
        f"Best roofline point: p={int(best_roofline_row['requested_threads'])} = "
        f"{best_roofline_row['throughput_gflops']:.1f} GFLOP/s\n"
        f"That is {best_roofline_row['dp_peak_fraction_pct']:.1f}% of the double-precision peak and "
        f"{best_roofline_row['dram_bw_fraction_pct']:.2f}% of peak DRAM bandwidth"
    )
    ax_roof.text(
        0.03,
        0.05,
        summary_text,
        transform=ax_roof.transAxes,
        fontsize=11.5,
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
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=10.9,
        frameon=True,
        columnspacing=1.4,
        handlelength=2.8,
        borderaxespad=0.0,
    )
    roof_legend.get_frame().set_edgecolor("0.75")
    roof_legend.get_frame().set_alpha(0.95)

    x = np.arange(len(gap_df))
    labels = [f"p={int(value)}" for value in gap_df["requested_threads"]]
    ideal_line = ax_gap.plot(
        x,
        gap_df["requested_threads"],
        color="0.45",
        linestyle=":",
        linewidth=1.8,
        label="Ideal speedup",
    )
    wall_line = ax_gap.plot(
        x,
        gap_df["walltime_speedup_vs_base"],
        color=CPU_BAR_COLOR,
        marker="o",
        linewidth=3.2,
        markersize=8.0,
        label="End-to-end wall-time speedup",
    )
    grid_line = ax_gap.plot(
        x,
        gap_df["gridding_host_speedup_vs_base"],
        color=GRIDDING_SPEEDUP_COLOR,
        marker="s",
        linewidth=3.2,
        markersize=7.7,
        label="Profiled gridding-stage speedup",
    )
    eff_line = ax_eff.plot(
        x,
        gap_df["walltime_parallel_efficiency_pct"],
        color=EFFICIENCY_COLOR,
        marker="o",
        linewidth=3.0,
        markersize=6.8,
        alpha=0.82,
        label="Parallel efficiency",
    )

    wall_offsets = [(-10, 10), (-2, 12), (0, 12), (2, 12)]
    grid_offsets = [(10, -14), (6, 8), (6, 8), (6, 8)]
    eff_offsets = [(8, -18), (8, -18), (8, -18), (8, -18)]

    for i, row in enumerate(gap_df.to_dict("records")):
        ax_gap.annotate(
            f"{row['walltime_speedup_vs_base']:.2f}x",
            (x[i], row["walltime_speedup_vs_base"]),
            xytext=wall_offsets[i],
            textcoords="offset points",
            fontsize=11.0,
            fontweight="bold",
            color=CPU_BAR_COLOR,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
            clip_on=True,
        )
        ax_gap.annotate(
            f"{row['gridding_host_speedup_vs_base']:.1f}x",
            (x[i], row["gridding_host_speedup_vs_base"]),
            xytext=grid_offsets[i],
            textcoords="offset points",
            fontsize=11.0,
            fontweight="bold",
            color=GRIDDING_SPEEDUP_COLOR,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
            clip_on=True,
        )
        ax_eff.annotate(
            f"{row['walltime_parallel_efficiency_pct']:.1f}%",
            (x[i], row["walltime_parallel_efficiency_pct"]),
            xytext=eff_offsets[i],
            textcoords="offset points",
            fontsize=10.8,
            fontweight="bold",
            color=EFFICIENCY_COLOR,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
            clip_on=True,
        )

    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(labels, fontsize=13)
    ax_gap.set_xlabel("Parallelism p  (-j threads)", fontsize=14, fontweight="bold")
    ax_gap.set_ylabel("Speedup relative to p=1", fontsize=14, fontweight="bold")
    ax_gap.set_yscale("log")
    ax_gap.set_ylim(0.9, 72.0)
    ax_gap.set_yticks([1, 2, 4, 8, 16, 32, 64])
    ax_gap.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_gap.tick_params(axis="y", labelsize=12)
    ax_gap.grid(axis="y", alpha=0.3, linestyle="--", which="major")
    ax_gap.set_title("Gridding-stage scaling gap", fontsize=15, fontweight="bold")

    ax_eff.set_ylabel(
        r"Parallel efficiency $E(p)=S(p)/p$ (%)",
        fontsize=14,
        fontweight="bold",
        color=EFFICIENCY_COLOR,
    )
    ax_eff.tick_params(axis="y", labelcolor=EFFICIENCY_COLOR, labelsize=12)
    ax_eff.set_ylim(0.0, 105.0)

    gap_legend = ax_gap.legend(
        [ideal_line[0], wall_line[0], grid_line[0], eff_line[0]],
        [
            "Ideal speedup",
            "End-to-end wall-time speedup",
            "Profiled gridding-stage speedup",
            "Parallel efficiency",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=10.8,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.7,
        borderaxespad=0.0,
    )
    gap_legend.get_frame().set_edgecolor("0.75")
    gap_legend.get_frame().set_alpha(0.95)

    fig.suptitle(
        "CPU Roofline and Scaling Gap for WSClean Stacking: 16384$^2$, t=256, c=256\n"
        f"Dual-socket node: 2 x {processor} | p denotes WSClean thread count (-j)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.24, top=0.84, wspace=0.29)

    saved_paths = []
    for suffix in (".png", ".pdf"):
        out_path = output_stem.with_suffix(suffix)
        fig.savefig(out_path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def main() -> int:
    args = parse_args()
    if not args.roofline_root.exists():
        print(f"Skipping Plot 25f: roofline directory not found: {args.roofline_root}")
        return 0
    if not args.profiling_root.exists():
        print(f"Skipping Plot 25f: profiling directory not found: {args.profiling_root}")
        return 0

    roofline_df, _ = load_roofline_dataframe(args.roofline_root)
    gap_df = build_summary_dataframe(args.profiling_root, args.roofline_root)
    export_summary(roofline_df, gap_df, args.summary_csv)
    for path in plot_merged_figure(roofline_df, gap_df, args.output_stem):
        print(f"Saved figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
