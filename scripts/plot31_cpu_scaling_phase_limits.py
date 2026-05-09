#!/usr/bin/env python3
"""
Illustrate why the Plot 25 CPU stacking run shows low parallel efficiency by
combining matched CPU-only phase timings with the existing roofline scalability
summary.
"""

import argparse
import os
import re
import sys
import tempfile
from datetime import datetime
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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PROFILING_ROOT = REPO_ROOT.parent / "astroCAMP-bench" / "profiling_cpu"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "plot31_cpu_scaling_phase_limits"
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "plot31_cpu_scaling_phase_limits_summary.csv"

PHASE_COLORS = {
    "weights_grid_s": plt.cm.cividis(0.72),
    "average_beam_s": plt.cm.cividis(0.54),
    "remaining_inversion_s": plt.cm.cividis(0.26),
}
GRIDDING_SPEEDUP_COLOR = plt.cm.cividis(0.88)

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
        description="Plot matched CPU-only phase timings to explain Plot 25's weak scalability."
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


def extract_threads(path: Path) -> int:
    match = re.search(r"_(\d+)cores(?:_uprof_collect_inst)?\.txt$|_(\d+)cores\.out$", path.name)
    if not match:
        raise ValueError(f"Could not parse requested threads from {path.name}")
    for group in match.groups():
        if group is not None:
            return int(group)
    raise ValueError(f"Could not parse requested threads from {path.name}")


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%b-%d %H:%M:%S.%f")


def parse_duration_pairs(text: str, start_label: str, end_label: str) -> list[float]:
    pattern = (
        rf"(\d{{4}}-\w{{3}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}\.\d+)\s+{re.escape(start_label)}"
        rf".*?"
        rf"(\d{{4}}-\w{{3}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}\.\d+)\s+{re.escape(end_label)}"
    )
    durations = []
    for start_text, end_text in re.findall(pattern, text, flags=re.S):
        durations.append((parse_timestamp(end_text) - parse_timestamp(start_text)).total_seconds())
    return durations


def parse_inversion_durations(text: str) -> list[float]:
    durations = []
    for hours, minutes, seconds in re.findall(
        r"Inversion:\s+(\d+):(\d+):(\d+\.\d+)", text
    ):
        durations.append(int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds))
    return durations


def summarize_out_log(out_path: Path) -> dict:
    text = out_path.read_text()
    weights_grid = parse_duration_pairs(text, "== weights->Grid() start", "== weights->Grid() end")
    average_beam = parse_duration_pairs(
        text,
        "Computing average beam.",
        "Finished computing average beam.",
    )
    inversion = parse_inversion_durations(text)
    if not weights_grid or not average_beam or not inversion:
        raise ValueError(f"Missing expected phase timings in {out_path}")

    weights_median = float(np.median(weights_grid))
    beam_median = float(np.median(average_beam))
    inversion_median = float(np.median(inversion))
    remaining = inversion_median - weights_median - beam_median

    return {
        "requested_threads": extract_threads(out_path),
        "out_path": out_path.as_posix(),
        "weights_grid_s": weights_median,
        "average_beam_s": beam_median,
        "inversion_median_s": inversion_median,
        "remaining_inversion_s": remaining,
        "weights_grid_samples": len(weights_grid),
        "average_beam_samples": len(average_beam),
        "inversion_samples": len(inversion),
    }


def summarize_collect_inst(inst_path: Path) -> dict:
    text = inst_path.read_text()
    host_values = [
        float(value) for value in re.findall(r"^\|host:\s+([0-9.e+-]+)\s+s$", text, flags=re.M)
    ]
    gridding_values = [
        float(value)
        for value in re.findall(
            r"^\|gridding:\s+([0-9.]+)\s+Mvisibilities/s$", text, flags=re.M
        )
    ]
    if not host_values or not gridding_values:
        raise ValueError(f"Missing |host or |gridding summaries in {inst_path}")

    return {
        "requested_threads": extract_threads(inst_path),
        "collect_inst_path": inst_path.as_posix(),
        "gridding_host_median_s": float(np.median(host_values)),
        "gridding_host_samples": len(host_values),
        "gridding_median_mvis_per_s": float(np.median(gridding_values)),
        "gridding_samples": len(gridding_values),
    }


def load_phase_dataframe(profiling_root: Path) -> pd.DataFrame:
    out_paths = sorted(
        profiling_root.glob("slurm-*_wsc_dirty_t0-255_c0-255_16384pix_20deg_*cores.out")
    )
    inst_paths = sorted(
        profiling_root.glob(
            "slurm-*_wsc_dirty_t0-255_c0-255_16384pix_20deg_*cores_uprof_collect_inst.txt"
        )
    )
    if not out_paths:
        raise FileNotFoundError(f"No matched .out logs found under {profiling_root}")
    if not inst_paths:
        raise FileNotFoundError(f"No matched collect_inst logs found under {profiling_root}")

    out_df = pd.DataFrame([summarize_out_log(path) for path in out_paths])
    inst_df = pd.DataFrame([summarize_collect_inst(path) for path in inst_paths])
    df = out_df.merge(inst_df, on="requested_threads", how="inner")
    df = df.sort_values("requested_threads").reset_index(drop=True)

    fixed = df["weights_grid_s"] + df["average_beam_s"]
    df["fixed_phase_share_pct"] = 100.0 * fixed / df["inversion_median_s"]
    base_host = float(df.loc[0, "gridding_host_median_s"])
    base_gridding = float(df.loc[0, "gridding_median_mvis_per_s"])
    base_inversion = float(df.loc[0, "inversion_median_s"])
    df["gridding_host_speedup_vs_base"] = base_host / df["gridding_host_median_s"]
    df["gridding_throughput_gain_vs_base"] = (
        df["gridding_median_mvis_per_s"] / base_gridding
    )
    df["inversion_speedup_vs_base"] = base_inversion / df["inversion_median_s"]
    return df


def build_summary_dataframe(profiling_root: Path, roofline_root: Path) -> pd.DataFrame:
    phase_df = load_phase_dataframe(profiling_root)
    roofline_df, _ = load_roofline_dataframe(roofline_root)
    merged = phase_df.merge(
        roofline_df[
            [
                "requested_threads",
                "elapsed_s",
                "walltime_speedup_vs_base",
                "walltime_parallel_efficiency_pct",
                "throughput_gflops",
            ]
        ],
        on="requested_threads",
        how="inner",
    )
    merged["walltime_vs_profiled_inversion_delta_pct"] = 100.0 * (
        merged["elapsed_s"] - merged["inversion_median_s"]
    ) / merged["elapsed_s"]
    return merged.sort_values("requested_threads").reset_index(drop=True)


def plot_scaling_phase_limits(df: pd.DataFrame, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_phase, ax_gap) = plt.subplots(
        1,
        2,
        figsize=(17.6, 7.9),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )
    ax_eff = ax_gap.twinx()

    x = np.arange(len(df))
    labels = [f"p={int(value)}" for value in df["requested_threads"]]

    weights_h = df["weights_grid_s"] / 3600.0
    beam_h = df["average_beam_s"] / 3600.0
    remaining_h = df["remaining_inversion_s"] / 3600.0
    total_h = df["inversion_median_s"] / 3600.0

    weights_bars = ax_phase.bar(
        x,
        weights_h,
        width=0.66,
        color=PHASE_COLORS["weights_grid_s"],
        edgecolor="black",
        linewidth=0.7,
        label="Median weights->Grid()",
    )
    beam_bars = ax_phase.bar(
        x,
        beam_h,
        width=0.66,
        bottom=weights_h,
        color=PHASE_COLORS["average_beam_s"],
        edgecolor="black",
        linewidth=0.7,
        label="Median average beam",
    )
    remaining_bars = ax_phase.bar(
        x,
        remaining_h,
        width=0.66,
        bottom=weights_h + beam_h,
        color=PHASE_COLORS["remaining_inversion_s"],
        edgecolor="black",
        linewidth=0.7,
        label="Remaining inversion time",
    )

    for xi, total, fixed_share in zip(x, total_h, df["fixed_phase_share_pct"]):
        ax_phase.text(
            xi,
            total + 0.05,
            f"{total:.2f} h\n{fixed_share:.0f}% fixed",
            ha="center",
            va="bottom",
            fontsize=11.1,
            fontweight="bold",
        )

    ax_phase.set_xticks(x)
    ax_phase.set_xticklabels(labels, fontsize=13)
    ax_phase.set_xlabel("Parallelism p  (-j threads)", fontsize=14, fontweight="bold")
    ax_phase.set_ylabel("Median inversion time (hours)", fontsize=14, fontweight="bold")
    ax_phase.set_ylim(0.0, max(3.7, float(total_h.max()) * 1.15))
    ax_phase.set_title("Median inversion-time decomposition", fontsize=15, fontweight="bold")
    ax_phase.grid(axis="y", alpha=0.3, linestyle="--")
    ax_phase.tick_params(axis="y", labelsize=12)

    ideal_line = ax_gap.plot(
        x,
        df["requested_threads"],
        color="0.45",
        linestyle=":",
        linewidth=1.7,
        label="Ideal speedup",
    )
    total_speedup_line = ax_gap.plot(
        x,
        df["walltime_speedup_vs_base"],
        color=CPU_BAR_COLOR,
        marker="o",
        linewidth=3.0,
        markersize=7.4,
        label="Plot 25 wall-time speedup",
    )
    grid_speedup_line = ax_gap.plot(
        x,
        df["gridding_host_speedup_vs_base"],
        color=GRIDDING_SPEEDUP_COLOR,
        marker="s",
        linewidth=3.0,
        markersize=7.1,
        label="Profiled gridding-stage speedup",
    )
    efficiency_line = ax_eff.plot(
        x,
        df["walltime_parallel_efficiency_pct"],
        color=EFFICIENCY_COLOR,
        marker="o",
        linewidth=3.0,
        markersize=6.4,
        alpha=0.82,
        label="Parallel efficiency",
    )

    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(labels, fontsize=13)
    ax_gap.set_xlabel("Parallelism p  (-j threads)", fontsize=14, fontweight="bold")
    ax_gap.set_ylabel("Speedup relative to p=1", fontsize=14, fontweight="bold")
    ax_gap.set_yscale("log")
    ax_gap.set_ylim(0.9, 72.0)
    ax_gap.set_yticks([1, 2, 4, 8, 16, 32, 64])
    ax_gap.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_gap.set_title("Gridding speedup outruns end-to-end speedup", fontsize=15, fontweight="bold")
    ax_gap.grid(axis="y", alpha=0.3, linestyle="--", which="major")
    ax_gap.tick_params(axis="y", labelsize=12)

    ax_eff.set_ylabel(
        r"Parallel efficiency $E(p)=S(p)/p$ (%)",
        fontsize=14,
        fontweight="bold",
        color=EFFICIENCY_COLOR,
    )
    ax_eff.tick_params(axis="y", labelcolor=EFFICIENCY_COLOR, labelsize=12)
    ax_eff.set_ylim(0.0, 105.0)

    phase_legend = ax_phase.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=10.8,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.3,
    )
    phase_legend.get_frame().set_edgecolor("0.75")
    phase_legend.get_frame().set_alpha(0.95)

    gap_legend = ax_gap.legend(
        [ideal_line[0], total_speedup_line[0], grid_speedup_line[0], efficiency_line[0]],
        [
            "Ideal speedup",
            "Plot 25 wall-time speedup",
            "Profiled gridding-stage speedup",
            "Parallel efficiency",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=10.8,
        frameon=True,
        columnspacing=1.2,
        handlelength=2.6,
    )
    gap_legend.get_frame().set_edgecolor("0.75")
    gap_legend.get_frame().set_alpha(0.95)

    fig.suptitle(
        "CPU Scaling Limits for WSClean Stacking: 16384$^2$, t=256, c=256\n"
        "Matched CPU-only profiling logs explain the low parallel efficiency seen in Plot 25",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.24, top=0.84, wspace=0.30)

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
        print(f"Skipping Plot 31: profiling directory not found: {args.profiling_root}")
        return 0
    if not args.roofline_root.exists():
        print(f"Skipping Plot 31: roofline directory not found: {args.roofline_root}")
        return 0

    df = build_summary_dataframe(args.profiling_root, args.roofline_root)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.summary_csv, index=False)
    print(f"Saved summary CSV to {args.summary_csv}")
    for path in plot_scaling_phase_limits(df, args.output_stem):
        print(f"Saved figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
