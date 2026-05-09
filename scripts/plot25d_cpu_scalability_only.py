#!/usr/bin/env python3
"""
Single-panel variant of Plot 25 that keeps only the scalability summary from the
CPU roofline figure for the WSClean stacking runs.
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

from plot25_cpu_roofline_stacking import (
    CPU_BAR_COLOR,
    DEFAULT_INPUT_ROOT,
    EFFICIENCY_COLOR,
    contrast_text_color,
    load_roofline_dataframe,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "plot25d_cpu_scalability_only"
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "plot25d_cpu_scalability_only_summary.csv"

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
        description="Create a single-panel scalability figure from the Plot 25 roofline summary."
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


def plot_scalability_only(df, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    best_scaling_row = df.loc[df["elapsed_s"].idxmin()]
    fig, ax_scale = plt.subplots(1, 1, figsize=(9.4, 7.2))
    ax_eff = ax_scale.twinx()

    x = np.arange(len(df))
    bars = ax_scale.bar(
        x,
        df["walltime_speedup_vs_base"],
        width=0.62,
        color=CPU_BAR_COLOR,
        edgecolor="black",
        linewidth=0.7,
        label=r"Wall-time speedup $S(p)=T_1/T_p$",
    )
    eff_line = ax_eff.plot(
        x,
        df["walltime_parallel_efficiency_pct"],
        color=EFFICIENCY_COLOR,
        marker="o",
        linewidth=3.0,
        markersize=6.5,
        alpha=0.78,
        label=r"Parallel efficiency $E(p)=S(p)/p$",
    )
    eff_ref = ax_eff.axhline(
        100.0,
        color="0.45",
        linestyle=":",
        linewidth=1.4,
        label="Ideal efficiency",
    )

    for bar, row in zip(bars, df.to_dict("records")):
        text_color = contrast_text_color(bar.get_facecolor())
        ax_scale.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() / 2.0,
            f"T={row['elapsed_s']:.0f} s\n{row['throughput_gflops']:.0f} GFLOP/s",
            ha="center",
            va="center",
            fontsize=10.8,
            fontweight="bold",
            color=text_color,
        )

    scaling_text = (
        r"Scalability uses profiler elapsed time $T_p$, not sampled GFLOP/s" "\n"
        f"Best wall-time speedup: {best_scaling_row['walltime_speedup_vs_base']:.2f}x at "
        f"p={int(best_scaling_row['requested_threads'])}\n"
        f"Runs at p=16..64 all stay near "
        f"{df.loc[df['requested_threads'] >= 16, 'elapsed_s'].min():.0f}-"
        f"{df.loc[df['requested_threads'] >= 16, 'elapsed_s'].max():.0f} s"
    )
    ax_scale.text(
        0.03,
        0.95,
        scaling_text,
        transform=ax_scale.transAxes,
        fontsize=11.6,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    ax_scale.set_xticks(x)
    ax_scale.set_xticklabels([f"p={int(v)}" for v in df["requested_threads"]], fontsize=13)
    ax_scale.set_xlabel("Parallelism p  (-j threads)", fontsize=14, fontweight="bold")
    ax_scale.set_ylabel(r"Wall-time speedup $S(p)=T_1/T_p$", fontsize=14, fontweight="bold")
    ax_scale.set_ylim(0.0, max(3.6, float(df["walltime_speedup_vs_base"].max()) * 1.30))
    ax_scale.set_title("Scalability Summary", fontsize=15, fontweight="bold")
    ax_scale.grid(axis="y", alpha=0.3, linestyle="--")
    ax_scale.tick_params(axis="y", labelsize=12)

    ax_eff.set_ylabel(
        r"Parallel efficiency $E(p)=S(p)/p$ (%)",
        fontsize=14,
        fontweight="bold",
        color=EFFICIENCY_COLOR,
    )
    ax_eff.tick_params(axis="y", labelcolor=EFFICIENCY_COLOR, labelsize=12)
    ax_eff.set_ylim(0.0, 105.0)

    legend_handles = [bars, eff_line[0], eff_ref]
    legend_labels = [
        "Wall-time speedup",
        "Parallel efficiency",
        "Ideal efficiency",
    ]
    legend = ax_scale.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=10.9,
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
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.24, top=0.82)

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
        print(f"Skipping scalability-only plot: input directory not found: {args.input_root}")
        return 0

    df, _ = load_roofline_dataframe(args.input_root)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.summary_csv, index=False)
    print(f"Saved summary CSV to {args.summary_csv}")
    for path in plot_scalability_only(df, args.output_stem):
        print(f"Saved figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
