#!/usr/bin/env python3
"""
Plot 17 variant: kernel-time breakdown with an additional right-axis line for
problem-size disk footprint (input + output).
"""

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

TMP_CACHE_ROOT = Path(tempfile.gettempdir()) / "astrocamp-mpl"
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE_ROOT / "xdg-cache"))

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LOG_ROOT = REPO_ROOT.parent / "astroCAMP-bench" / "pasc25_16c"
DEFAULT_MEMORY_TABLE = SCRIPT_DIR / "problem_size_memory_table.csv"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "plot17b_kernel_time_breakdown_with_disk"
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "plot17b_kernel_time_breakdown_with_disk_summary.csv"

MEMORY_COLOR = "#8c6d31"

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
        description="Create a Plot 17 variant with a disk-footprint overlay."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"Directory containing pasc25_16c raw logs (default: {DEFAULT_LOG_ROOT})",
    )
    parser.add_argument(
        "--memory-table",
        type=Path,
        default=DEFAULT_MEMORY_TABLE,
        help=f"CSV with input/output size estimates (default: {DEFAULT_MEMORY_TABLE})",
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
        help=f"Output CSV for the merged summary table (default: {DEFAULT_SUMMARY_CSV})",
    )
    return parser.parse_args()


def parse_hms_to_seconds(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60.0 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    raise ValueError(f"Unsupported elapsed time format: {value}")


def parse_kernel_breakdown_from_logs(log_root: Path) -> pd.DataFrame:
    root = Path(log_root)
    if not root.exists():
        return pd.DataFrame()

    run_re = re.compile(r"slurm-(\d+)_wsc_dirty_t0-(\d+)_c0-(\d+)_([0-9]+)p_.*\.log$")
    timed_re = re.compile(r"^\s*(gridder|sub-fft|wtiling):\s*([0-9.eE+-]+)\s*s,")
    elapsed_re = re.compile(r"Elapsed .*?:\s*([0-9:.]+)\s*$")

    rows = []
    for path in sorted(root.glob("slurm-*_wsc_dirty_t0-*_c0-*_*p_*.log")):
        match = run_re.match(path.name)
        if not match:
            continue

        rec = {
            "run_id": int(match.group(1)),
            "timesteps": int(match.group(2)),
            "channels": int(match.group(3)),
            "image_size": int(match.group(4)),
            "gridder_s": 0.0,
            "sub_fft_s": 0.0,
            "wtiling_s": 0.0,
            "wall_s": np.nan,
            "log_file": str(path),
        }

        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                m_time = timed_re.match(line)
                if m_time:
                    label = m_time.group(1)
                    secs = float(m_time.group(2))
                    if label == "gridder":
                        rec["gridder_s"] += secs
                    elif label == "sub-fft":
                        rec["sub_fft_s"] += secs
                    elif label == "wtiling":
                        rec["wtiling_s"] += secs
                    continue

                m_elapsed = elapsed_re.search(line)
                if m_elapsed:
                    rec["wall_s"] = parse_hms_to_seconds(m_elapsed.group(1))

        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["known_kernel_s"] = out["gridder_s"] + out["sub_fft_s"] + out["wtiling_s"]
    return out.sort_values(["image_size", "timesteps", "channels", "run_id"]).reset_index(drop=True)


def load_memory_table(memory_table_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(memory_table_csv)
    return df.rename(
        columns={
            "Image size": "image_size",
            "Timesteps": "timesteps",
            "Channels": "channels",
            "Input on-disk est. (GiB)": "input_disk_gib",
            "Output (GiB)": "output_gib",
            "Input + Output (GiB)": "total_disk_gib",
        }
    )[
        [
            "image_size",
            "timesteps",
            "channels",
            "input_disk_gib",
            "output_gib",
            "total_disk_gib",
        ]
    ].copy()


def build_summary_dataframe(log_root: Path, memory_table_csv: Path) -> pd.DataFrame:
    kernel_df = parse_kernel_breakdown_from_logs(log_root)
    if kernel_df.empty:
        return kernel_df

    kernel_summary = (
        kernel_df.groupby(["image_size", "timesteps", "channels"], as_index=False)
        .agg(
            gridder_s=("gridder_s", "median"),
            sub_fft_s=("sub_fft_s", "median"),
            wtiling_s=("wtiling_s", "median"),
            known_kernel_s=("known_kernel_s", "median"),
            wall_s=("wall_s", "median"),
            n_runs=("run_id", "count"),
        )
        .sort_values(["image_size", "timesteps", "channels"])
        .reset_index(drop=True)
    )

    memory_df = load_memory_table(memory_table_csv)
    merged = kernel_summary.merge(
        memory_df,
        on=["image_size", "timesteps", "channels"],
        how="left",
    )
    merged["other_wsclean_s"] = np.maximum(merged["wall_s"] - merged["known_kernel_s"], 0.0)
    return merged


def build_timestep_group_layout(sub: pd.DataFrame, gap: float = 0.55):
    x_positions = []
    channel_labels = []
    groups = []
    cursor = 0.0
    for tval, tsub in sub.groupby("timesteps", sort=True):
        start = cursor
        for _, row in tsub.iterrows():
            x_positions.append(cursor)
            channel_labels.append(f"c={int(row['channels'])}")
            cursor += 1.0
        end = cursor - 1.0
        groups.append((int(tval), start, end))
        cursor += gap
    return np.array(x_positions, dtype=float), channel_labels, groups


def add_timestep_subgroups(ax, groups, labelsize: int = 16):
    for idx in range(1, len(groups)):
        left = groups[idx - 1][2]
        right = groups[idx][1]
        ax.axvline(0.5 * (left + right), linestyle=":", linewidth=1.2, color="0.45", alpha=0.95)

    secax = ax.secondary_xaxis("top")
    secax.set_xticks([0.5 * (start + end) for _, start, end in groups])
    secax.set_xticklabels([f"t={tval}" for tval, _, _ in groups])
    secax.tick_params(axis="x", labelsize=labelsize, length=0, pad=2)
    return secax


def format_zero_decade_tick(value, _position) -> str:
    if np.isclose(value, 0.0):
        return "0"
    if value <= 0.0:
        return ""

    exponent = np.log10(value)
    if np.isclose(exponent, round(exponent)):
        return rf"$10^{{{int(round(exponent))}}}$"
    return ""


def apply_zero_decade_ticks(axis, upper_limit: float) -> None:
    decade_ticks = [0.0, 1.0, 10.0, 100.0, 1000.0]
    axis.set_yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
    axis.set_ylim(0.0, upper_limit)
    axis.yaxis.set_major_locator(FixedLocator(decade_ticks))
    axis.yaxis.set_major_formatter(FuncFormatter(format_zero_decade_tick))
    axis.yaxis.set_minor_formatter(NullFormatter())


def plot_figure(df: pd.DataFrame, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    image_sizes = sorted(df["image_size"].unique())
    fig, axes = plt.subplots(len(image_sizes), 1, figsize=(7.2, 3.6 * len(image_sizes)), squeeze=False)

    memory_line_handle = None

    for row, img in enumerate(image_sizes):
        ax = axes[row][0]
        ax_mem = ax.twinx()
        sub = df[df["image_size"] == img].copy().sort_values(["timesteps", "channels"]).reset_index(drop=True)
        x, channel_labels, t_groups = build_timestep_group_layout(sub)

        bw = 0.38
        stack_offset = bw * 0.06
        wall_offset = bw * 0.12
        stack_width = bw * 0.62
        wall_width = bw * 0.68

        k0 = sub["gridder_s"].values
        k1 = sub["sub_fft_s"].values
        k2 = sub["wtiling_s"].values
        wall = sub["wall_s"].values
        total_disk = sub["total_disk_gib"].values

        ax.bar(
            x + wall_offset,
            wall,
            width=wall_width,
            color="0.6",
            alpha=0.28,
            label="WSClean total",
            zorder=1,
        )
        ax.bar(x - stack_offset, k0, width=stack_width, color="#1f77b4", label="gridder", zorder=2)
        ax.bar(x - stack_offset, k1, width=stack_width, bottom=k0, color="#ff7f0e", label="sub-FFT", zorder=2)
        ax.bar(x - stack_offset, k2, width=stack_width, bottom=k0 + k1, color="#2ca02c", label="w-tiling", zorder=2)

        memory_line = ax_mem.plot(
            x,
            total_disk,
            color=MEMORY_COLOR,
            marker="D",
            markersize=4.8,
            linewidth=1.9,
            label="Input+output on disk",
            zorder=3,
        )[0]
        memory_line_handle = memory_line

        apply_zero_decade_ticks(ax, max(1000.0, float(np.nanmax(wall)) * 1.08))
        apply_zero_decade_ticks(ax_mem, max(100.0, float(np.nanmax(total_disk)) * 1.12))

        ax.set_ylabel("Time (s)", fontsize=16)
        ax_mem.set_ylabel("Input+output on disk (GiB)", fontsize=16, color=MEMORY_COLOR)
        ax_mem.tick_params(axis="y", labelsize=16, labelcolor=MEMORY_COLOR)
        ax.tick_params(axis="y", labelsize=16)

        ax.set_xticks(x)
        ax.set_xticklabels(channel_labels, fontsize=16, rotation=90, ha="center", va="top")
        ax.set_xlabel("Channels", fontsize=16)
        ax.set_title(f"Image {int(img)}", fontsize=16, pad=8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)
        add_timestep_subgroups(ax, t_groups, labelsize=16)

    handles, labels = axes[0][0].get_legend_handles_labels()
    uniq = {}
    for handle, label in zip(handles, labels):
        if label not in uniq:
            uniq[label] = handle
    if memory_line_handle is not None:
        uniq["Input+output on disk"] = memory_line_handle

    fig.legend(
        list(uniq.values()),
        list(uniq.keys()),
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        fontsize=16,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.suptitle("IDG Kernels, WSClean Total, and Disk Footprint", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    saved_paths = []
    for suffix in (".png", ".pdf"):
        out_path = output_stem.with_suffix(suffix)
        fig.savefig(out_path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def main() -> int:
    args = parse_args()
    if not args.log_root.exists():
        print(f"Skipping Plot 17b: log directory not found: {args.log_root}")
        return 0
    if not args.memory_table.exists():
        print(f"Skipping Plot 17b: memory table not found: {args.memory_table}")
        return 0

    df = build_summary_dataframe(args.log_root, args.memory_table)
    if df.empty:
        print("Skipping Plot 17b: no workload rows parsed")
        return 0

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.summary_csv, index=False)
    print(f"Saved summary CSV to {args.summary_csv}")
    for path in plot_figure(df, args.output_stem):
        print(f"Saved figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
