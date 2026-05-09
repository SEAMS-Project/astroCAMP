#!/usr/bin/env python3
"""
Plot direct host/device data-movement efficiency for the pasc25_16c profiling runs.

The metric is log-derived IDG data movement efficiency:
    (sum of |gridder| and |sub-fft| transferred GB) / direct profiled Joules

Energy comes directly from the `|host:` and `|device:` lines in the raw logs.
Data movement comes directly from the `|gridder:` and `|sub-fft:` summaries in
the same logs, using `GB/s * seconds`.
"""

from __future__ import annotations

import argparse
import os
import re
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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LOG_ROOT = REPO_ROOT.parent / "astroCAMP-bench" / "pasc25_16c"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "plot32b_idg_data_movement_gb_per_joule"
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "plot32b_idg_data_movement_gb_per_joule_summary.csv"

HOST_COLOR = plt.cm.cividis(0.26)
DEVICE_COLOR = plt.cm.cividis(0.72)

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
        description="Create a GB/J figure from direct host/device joules and stage bytes in pasc25_16c logs."
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f"Directory containing pasc25_16c raw logs (default: {DEFAULT_LOG_ROOT})",
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
        help=f"Output CSV for the aggregated summary (default: {DEFAULT_SUMMARY_CSV})",
    )
    return parser.parse_args()


def parse_direct_efficiency_logs(log_root: Path) -> pd.DataFrame:
    root = Path(log_root)
    if not root.exists():
        return pd.DataFrame()

    run_re = re.compile(r"slurm-(\d+)_wsc_dirty_t0-(\d+)_c0-(\d+)_([0-9]+)p_.*\.log$")
    host_re = re.compile(
        r"^\|host:\s+[0-9.eE+-]+\s+s,\s+[0-9.]+\s+Watt,\s+(?P<joules>[0-9.]+)\s+Joules$"
    )
    device_re = re.compile(
        r"^\|device:\s+[0-9.eE+-]+\s+s,\s+[0-9.]+\s+Watt,\s+(?P<joules>[0-9.]+)\s+Joules$"
    )
    stage_re = re.compile(
        r"^\|(?P<stage>gridder|sub-fft):\s+"
        r"(?P<seconds>[0-9.eE+-]+)\s+s,\s+"
        r"[0-9.]+\s+GFLOPS,\s+"
        r"(?P<gbs>[0-9.]+)\s+GB/s,.*$"
    )

    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("slurm-*_wsc_dirty_t0-*_c0-*_*p_*.log")):
        match = run_re.match(path.name)
        if not match:
            continue

        host_joules = []
        device_joules = []
        gridder_gb = 0.0
        subfft_gb = 0.0
        gridder_count = 0
        subfft_count = 0

        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()

            host_match = host_re.match(stripped)
            if host_match:
                host_joules.append(float(host_match.group("joules")))
                continue

            device_match = device_re.match(stripped)
            if device_match:
                device_joules.append(float(device_match.group("joules")))
                continue

            stage_match = stage_re.match(stripped)
            if stage_match:
                seconds = float(stage_match.group("seconds"))
                gbs = float(stage_match.group("gbs"))
                moved_gb = seconds * gbs
                if stage_match.group("stage") == "gridder":
                    gridder_gb += moved_gb
                    gridder_count += 1
                else:
                    subfft_gb += moved_gb
                    subfft_count += 1

        total_stage_gb = gridder_gb + subfft_gb
        host_j_total = float(np.sum(host_joules)) if host_joules else np.nan
        device_j_total = float(np.sum(device_joules)) if device_joules else np.nan

        rows.append(
            {
                "run_id": int(match.group(1)),
                "timesteps": int(match.group(2)),
                "channels": int(match.group(3)),
                "image_size": int(match.group(4)),
                "host_joules_total": host_j_total,
                "device_joules_total": device_j_total,
                "gridder_gb_total": gridder_gb,
                "subfft_gb_total": subfft_gb,
                "idg_stage_gb_total": total_stage_gb,
                "host_gb_per_j": total_stage_gb / host_j_total if host_j_total and np.isfinite(host_j_total) else np.nan,
                "device_gb_per_j": total_stage_gb / device_j_total if device_j_total and np.isfinite(device_j_total) else np.nan,
                "host_event_count": len(host_joules),
                "device_event_count": len(device_joules),
                "gridder_event_count": gridder_count,
                "subfft_event_count": subfft_count,
                "log_file": str(path),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["image_size", "timesteps", "channels", "run_id"]).reset_index(drop=True)


def build_summary_dataframe(log_root: Path) -> pd.DataFrame:
    raw_df = parse_direct_efficiency_logs(log_root)
    if raw_df.empty:
        return raw_df

    summary = (
        raw_df.groupby(["image_size", "timesteps", "channels"], as_index=False)
        .agg(
            idg_stage_gb_total=("idg_stage_gb_total", "median"),
            gridder_gb_total=("gridder_gb_total", "median"),
            subfft_gb_total=("subfft_gb_total", "median"),
            host_joules_total=("host_joules_total", "median"),
            device_joules_total=("device_joules_total", "median"),
            host_gb_per_j=("host_gb_per_j", "median"),
            device_gb_per_j=("device_gb_per_j", "median"),
            host_event_count=("host_event_count", "median"),
            device_event_count=("device_event_count", "median"),
            gridder_event_count=("gridder_event_count", "median"),
            subfft_event_count=("subfft_event_count", "median"),
            n_runs=("run_id", "count"),
        )
        .sort_values(["image_size", "timesteps", "channels"])
        .reset_index(drop=True)
    )
    return summary


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


def add_timestep_subgroups(ax, groups, labelsize: int = 12):
    for idx in range(1, len(groups)):
        left = groups[idx - 1][2]
        right = groups[idx][1]
        ax.axvline(0.5 * (left + right), linestyle=":", linewidth=1.1, color="0.45", alpha=0.9)

    secax = ax.secondary_xaxis("top")
    secax.set_xticks([0.5 * (start + end) for _, start, end in groups])
    secax.set_xticklabels([f"t={tval}" for tval, _, _ in groups])
    secax.tick_params(axis="x", labelsize=labelsize, length=0, pad=2)
    return secax


def plot_summary(summary_df: pd.DataFrame, output_stem: Path) -> None:
    image_sizes = sorted(summary_df["image_size"].unique())
    if len(image_sizes) != 4:
        raise ValueError(f"Expected 4 image sizes, found {len(image_sizes)}: {image_sizes}")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.6), sharey=True)
    axes = axes.flatten()

    for ax, image_size in zip(axes, image_sizes):
        sub = (
            summary_df[summary_df["image_size"] == image_size]
            .sort_values(["timesteps", "channels"])
            .reset_index(drop=True)
        )
        xvals, channel_labels, groups = build_timestep_group_layout(sub)

        ax.plot(
            xvals,
            sub["host_gb_per_j"],
            color=HOST_COLOR,
            marker="o",
            linewidth=2.1,
            markersize=5.5,
            label="CPU",
        )
        ax.plot(
            xvals,
            sub["device_gb_per_j"],
            color=DEVICE_COLOR,
            marker="s",
            linewidth=2.1,
            markersize=5.2,
            label="GPU",
        )

        ax.set_title(f"Image size = {image_size}$^2$", fontsize=14)
        ax.set_xticks(xvals)
        ax.set_xticklabels(channel_labels, rotation=90, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlim(xvals.min() - 0.55, xvals.max() + 0.55)
        ax.grid(True, which="major", axis="y", alpha=0.25)
        add_timestep_subgroups(ax, groups)

    for idx in (0, 2):
        axes[idx].set_ylabel("Memory efficiency (GB/J)", fontsize=13)
    for idx in (2, 3):
        axes[idx].set_xlabel("Channels", fontsize=13)

    ymax = float(np.nanmax(summary_df[["host_gb_per_j", "device_gb_per_j"]].to_numpy()))
    yhi = ymax * 1.08
    for ax in axes:
        ax.set_ylim(0.0, yhi)

    for ax in axes:
        ax.legend(loc="upper right", fontsize=11, frameon=True)
    # fig.suptitle(
    #     "IDG Data-Movement Efficiency",
    #     fontsize=16,
    #     fontweight="bold",
    #     y=0.995,
    # )
    fig.tight_layout(rect=[0.03, 0.04, 1.0, 0.95])

    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    summary_df = build_summary_dataframe(args.log_root)
    if summary_df.empty:
        raise FileNotFoundError(f"No direct |host: / |device: / |gridder: / |sub-fft: data found under {args.log_root}")

    summary_df.to_csv(args.summary_csv, index=False)
    plot_summary(summary_df, args.output_stem)

    print(f"Wrote summary CSV to {args.summary_csv}")
    print(f"Wrote figure to {args.output_stem.with_suffix('.png')}")
    print(f"Wrote figure to {args.output_stem.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
