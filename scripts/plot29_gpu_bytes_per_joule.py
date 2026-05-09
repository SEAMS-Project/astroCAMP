#!/usr/bin/env python3
"""
Plot profiler-reported GPU efficiency metrics from the instruction-collection
logs in a single figure:

1. Stage-level data movement efficiency in GB/J from the representative
   `|gridder:` and `|sub-fft:` lines.
2. End-to-end device-side gridding efficiency in Mvis/J from the paired
   `|device:` and `|gridding:` lines.

These are relative profiler-derived efficiency views, not full hardware DRAM or
node-level energy-efficiency metrics.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INPUT_ROOT = REPO_ROOT.parent / "astroCAMP-bench" / "profiling_gpu2"
OUTPUT_STEM = SCRIPT_DIR / "plot29_gpu_bytes_per_joule"
CSV_PATH = SCRIPT_DIR / "gpu_bytes_per_joule_summary.csv"

STAGES = ["gridder", "sub-fft"]
COLORS = {
    "gridder": plt.cm.cividis(0.30),
    "sub-fft": plt.cm.cividis(0.62),
    "gridder + sub-fft": plt.cm.cividis(0.85),
    "device gridding": plt.cm.cividis(0.48),
}


def load_stage_efficiency(root: Path) -> pd.DataFrame:
    pattern = re.compile(
        r"^\|(?P<kind>[a-zA-Z0-9_-]+):\s+"
        r"(?P<time>[0-9.e+-]+)\s+s,\s+"
        r"(?P<gflops>[0-9.]+)\s+GFLOPS,\s+"
        r"(?P<bandwidth>[0-9.]+)\s+GB/s,\s+"
        r"(?P<power>[0-9.]+)\s+Watt"
        r"(?:,\s+[0-9.]+\s+GFLOPS/W,\s+(?P<joules>[0-9.]+)\s+Joules)?"
    )
    rows: list[dict[str, float | int | str]] = []

    for path in sorted(root.glob("*_uprof_collect_inst.txt")):
        match = re.search(r"_(\d+)cores_", path.name)
        if not match:
            continue
        threads = int(match.group(1))
        counters = {stage: 0 for stage in STAGES}
        for line in path.read_text(errors="ignore").splitlines():
            parsed = pattern.match(line.strip())
            if not parsed:
                continue
            kind = parsed.group("kind")
            if kind not in STAGES:
                continue

            time_s = float(parsed.group("time"))
            bandwidth_gbs = float(parsed.group("bandwidth"))
            power_w = float(parsed.group("power"))
            joules = parsed.group("joules")
            counters[kind] += 1
            energy_j = float(joules) if joules else power_w * time_s
            gb_per_joule = bandwidth_gbs / power_w if power_w > 0 else np.nan
            bytes_per_joule_gb = bandwidth_gbs * time_s / float(joules) if joules else np.nan

            rows.append(
                {
                    "run_label": path.stem,
                    "threads": threads,
                    "stage": kind,
                    "sample_index": counters[kind],
                    "time_s": time_s,
                    "bandwidth_gbs": bandwidth_gbs,
                    "power_w": power_w,
                    "joules": float(joules) if joules else np.nan,
                    "bytes_gb": bandwidth_gbs * time_s,
                    "energy_j": energy_j,
                    "gb_per_joule": gb_per_joule,
                    "gb_per_joule_from_time_energy": bytes_per_joule_gb,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No stage lines with both bandwidth and power found under {root}")
    return df


def load_device_efficiency(root: Path) -> pd.DataFrame:
    device_pattern = re.compile(
        r"^\|device:\s+"
        r"(?P<time>[0-9.e+-]+)\s+s,\s+"
        r"(?P<power>[0-9.]+)\s+Watt,\s+"
        r"(?P<joules>[0-9.]+)\s+Joules"
    )
    gridding_pattern = re.compile(r"^\|gridding:\s+(?P<mvis>[0-9.]+)\s+Mvisibilities/s")

    rows: list[dict[str, float | int | str]] = []
    for path in sorted(root.glob("*_uprof_collect_inst.txt")):
        match = re.search(r"_(\d+)cores_", path.name)
        if not match:
            continue
        threads = int(match.group(1))
        pending_device: dict[str, float | int | str] | None = None
        sample_index = 0
        for line in path.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            device_match = device_pattern.match(stripped)
            if device_match:
                sample_index += 1
                pending_device = {
                    "run_label": path.stem,
                    "threads": threads,
                    "sample_index": sample_index,
                    "device_time_s": float(device_match.group("time")),
                    "device_power_w": float(device_match.group("power")),
                    "device_energy_j": float(device_match.group("joules")),
                }
                continue

            gridding_match = gridding_pattern.match(stripped)
            if gridding_match and pending_device is not None:
                mvis_per_s = float(gridding_match.group("mvis"))
                rows.append(
                    {
                        **pending_device,
                        "gridding_mvis_per_s": mvis_per_s,
                        "mvis_per_joule": (
                            mvis_per_s * float(pending_device["device_time_s"]) / float(pending_device["device_energy_j"])
                            if float(pending_device["device_energy_j"]) > 0
                            else np.nan
                        ),
                    }
                )
                pending_device = None

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No paired |device: / |gridding: lines found under {root}")
    return df


def summarize_stage(df: pd.DataFrame) -> pd.DataFrame:
    stage_summary = (
        df.groupby(["threads", "stage"], as_index=False)
        .agg(
            sample_count=("gb_per_joule", "size"),
            median_gb_per_joule=("gb_per_joule", "median"),
            mean_gb_per_joule=("gb_per_joule", "mean"),
            median_bandwidth_gbs=("bandwidth_gbs", "median"),
            median_power_w=("power_w", "median"),
            median_time_s=("time_s", "median"),
            median_gb_per_joule_from_time_energy=("gb_per_joule_from_time_energy", "median"),
        )
        .sort_values(["stage", "threads"])
        .reset_index(drop=True)
    )
    total_df = (
        df.groupby(["threads", "run_label", "sample_index"], as_index=False)
        .agg(
            bytes_gb=("bytes_gb", "sum"),
            energy_j=("energy_j", "sum"),
            bandwidth_gbs=("bandwidth_gbs", "sum"),
            power_w=("power_w", "sum"),
            time_s=("time_s", "sum"),
        )
        .assign(stage="gridder + sub-fft")
    )
    total_df["gb_per_joule"] = total_df["bytes_gb"] / total_df["energy_j"]

    total_summary = (
        total_df.groupby(["threads", "stage"], as_index=False)
        .agg(
            sample_count=("gb_per_joule", "size"),
            median_gb_per_joule=("gb_per_joule", "median"),
            mean_gb_per_joule=("gb_per_joule", "mean"),
            median_bandwidth_gbs=("bandwidth_gbs", "median"),
            median_power_w=("power_w", "median"),
            median_time_s=("time_s", "median"),
            median_gb_per_joule_from_time_energy=("gb_per_joule", "median"),
        )
        .sort_values(["stage", "threads"])
        .reset_index(drop=True)
    )

    return pd.concat([stage_summary, total_summary], ignore_index=True)


def summarize_device(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("threads", as_index=False)
        .agg(
            sample_count=("mvis_per_joule", "size"),
            median_mvis_per_joule=("mvis_per_joule", "median"),
            mean_mvis_per_joule=("mvis_per_joule", "mean"),
            median_gridding_mvis_per_s=("gridding_mvis_per_s", "median"),
            median_device_power_w=("device_power_w", "median"),
            median_device_time_s=("device_time_s", "median"),
        )
        .reset_index(drop=True)
    )

def plot_summary(stage_summary: pd.DataFrame, device_summary: pd.DataFrame) -> None:
    threads = sorted(stage_summary["threads"].unique())

    fig, (ax_stage, ax_device) = plt.subplots(
        1,
        2,
        figsize=(15.2, 7.5),
        gridspec_kw={"width_ratios": [1.2, 0.95]},
    )

    for stage in STAGES + ["gridder + sub-fft"]:
        sub = stage_summary[stage_summary["stage"] == stage].set_index("threads").reindex(threads)
        ax_stage.plot(
            threads,
            sub["median_gb_per_joule"],
            marker="o",
            markersize=8,
            linewidth=2.0,
            color=COLORS[stage],
            label=stage,
        )
        for idx, thread in enumerate(threads):
            row = sub.loc[thread]
            if stage == "gridder":
                dx, dy = 8, -12 if idx % 2 else 10
            elif stage == "sub-fft":
                dx, dy = -8, -14 if idx % 2 else 10
            else:
                dx, dy = 10, 16 if idx % 2 == 0 else -18
            ax_stage.annotate(
                f"{row['median_gb_per_joule']:.2f}",
                (thread, row["median_gb_per_joule"]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha="left" if dx > 0 else "right",
                va="bottom" if dy > 0 else "top",
                fontsize=10.0,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.75", "linewidth": 0.6},
                arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.8},
            )

    ax_stage.set_xticks(threads)
    ax_stage.set_xticklabels([f"p={p}" for p in threads], fontsize=11)
    ax_stage.set_yscale("log")
    ax_stage.set_xlabel("Parallelism p", fontsize=12.5, fontweight="bold")
    ax_stage.set_ylabel("Profiler-reported data movement efficiency (GB/J)", fontsize=12.5, fontweight="bold")
    ax_stage.set_title("Stage-Level GB/J", fontsize=14, fontweight="bold")
    ax_stage.grid(True, axis="y", alpha=0.25)
    ax_stage.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        fontsize=10.0,
        handlelength=2.8,
        columnspacing=1.4,
    )

    ax_device.plot(
        device_summary["threads"],
        device_summary["median_mvis_per_joule"],
        color=COLORS["device gridding"],
        marker="o",
        markersize=8,
        linewidth=2.0,
        label="device gridding",
    )
    for idx, row in device_summary.iterrows():
        dx = 8 if idx % 2 == 0 else -8
        dy = 12 if idx < 2 else -14
        ax_device.annotate(
            f"{row['median_mvis_per_joule']:.2f}",
            (row["threads"], row["median_mvis_per_joule"]),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="left" if dx > 0 else "right",
            va="bottom" if dy > 0 else "top",
            fontsize=10.0,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.75", "linewidth": 0.6},
            arrowprops={"arrowstyle": "-", "color": "0.45", "lw": 0.8},
        )

    ax_device.set_xticks(threads)
    ax_device.set_xticklabels([f"p={p}" for p in threads], fontsize=11)
    ax_device.set_xlabel("Parallelism p", fontsize=12.5, fontweight="bold")
    ax_device.set_ylabel("Device-side gridding efficiency (Mvis/J)", fontsize=12.5, fontweight="bold")
    ax_device.set_title("End-to-End Device Mvis/J", fontsize=14, fontweight="bold")
    ax_device.grid(True, axis="y", alpha=0.25)
    for _, row in device_summary.iterrows():
        ax_device.text(
            row["threads"],
            row["median_mvis_per_joule"] * 0.94,
            f"{row['median_gridding_mvis_per_s']:.0f} Mvis/s\n{row['median_device_power_w']:.0f} W",
            ha="center",
            va="top",
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.90, "edgecolor": "0.82", "linewidth": 0.55},
        )

    note = (
        "Left: `gridder`, `sub-fft`, and their combined total use the representative `|stage:` lines with "
        "GB/J computed from bytes = (GB/s x time) and energy = power x time.\n"
        "Right: `|device:` and `|gridding:` are paired to compute device-side Mvis/J. These are profiler-derived "
        "efficiency views, not full-system energy metrics."
    )
    fig.text(
        0.5,
        0.03,
        note,
        ha="center",
        fontsize=10.1,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
    )

    fig.suptitle(
        "Profiler-Reported GPU Efficiency for WSClean + IDG",
        fontsize=15.5,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.22, wspace=0.28)

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)


def main() -> int:
    stage_raw_df = load_stage_efficiency(INPUT_ROOT)
    stage_summary = summarize_stage(stage_raw_df)
    device_raw_df = load_device_efficiency(INPUT_ROOT)
    device_summary = summarize_device(device_raw_df)
    summary = pd.concat(
        [
            stage_summary.assign(metric="gb_per_joule"),
            device_summary.assign(
                stage="device gridding",
                metric="mvis_per_joule",
                median_gb_per_joule=device_summary["median_mvis_per_joule"],
                mean_gb_per_joule=device_summary["mean_mvis_per_joule"],
                median_bandwidth_gbs=np.nan,
                median_power_w=device_summary["median_device_power_w"],
                median_time_s=device_summary["median_device_time_s"],
                median_gb_per_joule_from_time_energy=np.nan,
            )[[
                "threads",
                "stage",
                "sample_count",
                "median_gb_per_joule",
                "mean_gb_per_joule",
                "median_bandwidth_gbs",
                "median_power_w",
                "median_time_s",
                "median_gb_per_joule_from_time_energy",
                "metric",
            ]],
        ],
        ignore_index=True,
    )
    summary.to_csv(CSV_PATH, index=False)
    print(f"Saved summary to {CSV_PATH}")
    plot_summary(stage_summary, device_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
