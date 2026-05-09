#!/usr/bin/env python3
"""
Standalone single-panel version of the scaling-gap panel from Plot 31, intended
for paper placement after the kernel-breakdown figure.
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

from plot31_cpu_scaling_phase_limits import (
    DEFAULT_PROFILING_ROOT,
    DEFAULT_INPUT_ROOT,
    build_summary_dataframe,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
DEFAULT_OUTPUT_STEM = RESULTS_DIR / "plot31b_cpu_scaling_gap_only"
DEFAULT_SUMMARY_CSV = DERIVED_DIR / "plot31b_cpu_scaling_gap_only_summary.csv"

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)

IDEAL_COLOR = "0.30"
WALL_SPEEDUP_COLOR = "#1f4e79"
GRID_SPEEDUP_COLOR = "#2ca02c"
EFFICIENCY_PLOT_COLOR = "#d62728"

TITLE_FONT = 36
AXIS_LABEL_FONT = 32
TICK_FONT = 30
LEGEND_FONT = 24
ANNOTATION_FONT = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a standalone scaling-gap figure from the matched Plot 31 data."
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=DEFAULT_PROFILING_ROOT,
        help=f"Directory containing matched CPU-only .out and collect_inst logs (default: {DEFAULT_PROFILING_ROOT})",
    )
    parser.add_argument(
        "--roofline-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Directory containing Plot 25 AMD uProf roofline exports (default: {DEFAULT_INPUT_ROOT})",
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


def export_summary(df, summary_csv: Path) -> None:
    export_df = df[
        [
            "requested_threads",
            "gridding_host_speedup_vs_base",
            "gridding_median_mvis_per_s",
            "walltime_speedup_vs_base",
            "walltime_parallel_efficiency_pct",
            "elapsed_s",
            "throughput_gflops",
        ]
    ].copy()
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")


def plot_scaling_gap_only(df, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(17.2, 12.6))
    ax_eff = ax.twinx()

    x = df["requested_threads"].astype(float).to_list()
    labels = [f"p={int(value)}" for value in x]
    x_ideal = np.geomspace(min(x), max(x), 256)

    ideal_line = ax.plot(
        x_ideal,
        x_ideal,
        color=IDEAL_COLOR,
        linestyle=":",
        linewidth=3.0,
        label="Ideal speedup",
        zorder=1,
    )
    wall_line = ax.plot(
        x,
        df["walltime_speedup_vs_base"],
        color=WALL_SPEEDUP_COLOR,
        marker="o",
        linewidth=3.2,
        markersize=12.0,
        label="End-to-end wall-time speedup",
        zorder=3,
    )
    grid_line = ax.plot(
        x,
        df["gridding_host_speedup_vs_base"],
        color=GRID_SPEEDUP_COLOR,
        marker="s",
        linewidth=3.2,
        markersize=11.6,
        label="Profiled gridding-stage speedup",
        zorder=3,
    )
    eff_line = ax_eff.plot(
        x,
        df["walltime_parallel_efficiency_pct"],
        color=EFFICIENCY_PLOT_COLOR,
        marker="o",
        linewidth=3.0,
        markersize=10.8,
        alpha=0.82,
        label="Parallel efficiency",
        zorder=4,
    )

    wall_offsets = [(16, 16), (-40, 18), (-40, 18), (-42, 20)]
    wall_align = [("left", "bottom"), ("right", "bottom"), ("right", "bottom"), ("right", "bottom")]
    grid_offsets = [(18, -28), (18, 12), (18, 12), (-28, 12)]
    grid_align = [("left", "top"), ("left", "bottom"), ("left", "bottom"), ("right", "bottom")]
    eff_offsets = [(12, -38), (14, 16), (14, 16), (-24, -30)]
    eff_align = [("left", "top"), ("left", "bottom"), ("left", "bottom"), ("right", "top")]

    for i, row in enumerate(df.to_dict("records")):
        ax.annotate(
            f"{row['walltime_speedup_vs_base']:.2f}x",
            (x[i], row["walltime_speedup_vs_base"]),
            xytext=wall_offsets[i],
            textcoords="offset points",
            fontsize=ANNOTATION_FONT,
            fontweight="bold",
            color=WALL_SPEEDUP_COLOR,
            ha=wall_align[i][0],
            va=wall_align[i][1],
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.95, "edgecolor": "0.55"},
            clip_on=True,
            zorder=6,
        )
        ax.annotate(
            f"{row['gridding_host_speedup_vs_base']:.1f}x",
            (x[i], row["gridding_host_speedup_vs_base"]),
            xytext=grid_offsets[i],
            textcoords="offset points",
            fontsize=ANNOTATION_FONT,
            fontweight="bold",
            color=GRID_SPEEDUP_COLOR,
            ha=grid_align[i][0],
            va=grid_align[i][1],
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.95, "edgecolor": "0.55"},
            clip_on=True,
            zorder=6,
        )
        ax_eff.annotate(
            f"{row['walltime_parallel_efficiency_pct']:.1f}%",
            (x[i], row["walltime_parallel_efficiency_pct"]),
            xytext=eff_offsets[i],
            textcoords="offset points",
            fontsize=ANNOTATION_FONT,
            fontweight="bold",
            color=EFFICIENCY_PLOT_COLOR,
            ha=eff_align[i][0],
            va=eff_align[i][1],
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.95, "edgecolor": "0.55"},
            clip_on=True,
            zorder=7,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlim(0.65, 112.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FONT)
    ax.set_xlabel("Parallelism p  (-j threads)", fontsize=AXIS_LABEL_FONT, fontweight="bold")
    ax.set_ylabel("Speedup relative to p=1", fontsize=AXIS_LABEL_FONT, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(0.82, 220.0)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="x", labelsize=TICK_FONT)
    ax.tick_params(axis="y", labelsize=TICK_FONT)
    ax.grid(axis="both", alpha=0.3, linestyle="--", which="major")
    ax.set_title("Gridding-stage scaling versus end-to-end scaling", fontsize=TITLE_FONT, fontweight="bold")

    ax_eff.set_ylabel(
        r"Parallel efficiency $E(p)=S(p)/p$ (%)",
        fontsize=AXIS_LABEL_FONT,
        fontweight="bold",
        color=EFFICIENCY_PLOT_COLOR,
    )
    ax_eff.tick_params(axis="y", labelcolor=EFFICIENCY_PLOT_COLOR, labelsize=TICK_FONT)
    ax_eff.set_ylim(0.0, 165.0)

    legend = ax.legend(
        [ideal_line[0], wall_line[0], grid_line[0], eff_line[0]],
        [
            "Ideal speedup",
            "End-to-end wall-time speedup",
            "Profiled gridding-stage speedup",
            "Parallel efficiency",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=LEGEND_FONT,
        frameon=True,
        columnspacing=1.3,
        handlelength=2.7,
    )
    legend.get_frame().set_edgecolor("0.75")
    legend.get_frame().set_alpha(0.95)

    fig.suptitle(
        "CPU-only WSClean stacking strong-scaling gap: 16384$^2$, t=256, c=256",
        fontsize=TITLE_FONT + 2,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.28, top=0.84)

    saved_paths = []
    for suffix in (".png", ".pdf"):
        out_path = output_stem.with_suffix(suffix)
        fig.savefig(out_path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def main() -> int:
    args = parse_args()
    if not args.profiling_root.exists():
        print(f"Skipping Plot 31b: profiling directory not found: {args.profiling_root}")
        return 0
    if not args.roofline_root.exists():
        print(f"Skipping Plot 31b: roofline directory not found: {args.roofline_root}")
        return 0

    df = build_summary_dataframe(args.profiling_root, args.roofline_root)
    export_summary(df, args.summary_csv)
    for path in plot_scaling_gap_only(df, args.output_stem):
        print(f"Saved figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
IDEAL_COLOR = "0.30"
WALL_SPEEDUP_COLOR = "#1f4e79"
GRID_SPEEDUP_COLOR = "#2ca02c"
EFFICIENCY_PLOT_COLOR = "#d62728"
