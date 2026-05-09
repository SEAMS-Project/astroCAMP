#!/usr/bin/env python3
"""
Composite rebuttal figure:
1. H100 kernel roofline from the first completed rebuttal GPU run.
2. CPU-side host hotspots from WSClean timestamped phases.
3. GPU kernel hotspots from profiler summary lines in the same log.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from datetime import datetime
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
DEFAULT_LOG_PATH = (
    SCRIPT_DIR.parent
    / "../astroCAMP-bench/profiling_rebuttal/slurm-2654179_wsc_dirty_t0-0_c0-7_8192pix_20deg_8cores.out"
).resolve()
OUTPUT_STEM = SCRIPT_DIR / "plot30_rebuttal_gpu_roofline_hotspots"
SUMMARY_CSV = SCRIPT_DIR / "plot30_rebuttal_gpu_roofline_hotspots_summary.csv"

ROOF_DRAM_COLOR = "#c44e52"
ROOF_COMPUTE_COLOR = "#4c72b0"
GPU_GRIDDER_COLOR = plt.cm.cividis(0.32)
GPU_SUBFFT_COLOR = plt.cm.cividis(0.60)
CPU_HOTSPOT_COLOR = plt.cm.cividis(0.22)
GPU_HOTSPOT_COLORS = {
    "average-beam": plt.cm.cividis(0.18),
    "wtiling": plt.cm.cividis(0.40),
    "gridder": plt.cm.cividis(0.62),
    "sub-fft": plt.cm.cividis(0.82),
}
H100_FP32_PEAK_GFLOPS = 67000.0

TIMESTAMP_RE = re.compile(r"^(?P<ts>\d{4}-[A-Za-z]{3}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+(?P<msg>.*)$")
GPU_STAGE_RE = re.compile(
    r"^\|(?P<kind>[a-zA-Z0-9_-]+):\s+(?P<time>[0-9.e+-]+)\s+s"
    r"(?:,\s+(?P<gflops>[0-9.]+)\s+GFLOPS,\s+(?P<gbs>[0-9.]+)\s+GB/s"
    r"(?:,\s+(?P<power>[0-9.]+)\s+Watt(?:,\s+[0-9.]+\s+GFLOPS/W(?:,\s+(?P<joules>[0-9.]+)\s+Joules)?)?)?)?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create rebuttal roofline + CPU/GPU hotspot figure.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH, help=f"Path to rebuttal .out log (default: {DEFAULT_LOG_PATH})")
    parser.add_argument("--output-stem", type=Path, default=OUTPUT_STEM, help=f"Output path stem (default: {OUTPUT_STEM})")
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV, help=f"Output CSV (default: {SUMMARY_CSV})")
    return parser.parse_args()


def first_completed_run_section(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(line)
        if re.search(r"\bGridding:\s+\d", line):
            break
    return "\n".join(lines)


def parse_h100_bandwidth(section: str) -> float:
    match = re.search(r"Mem bandwidth\s*:\s*([0-9.]+)\s+GB/s", section)
    if not match:
        raise ValueError("Could not parse H100 memory bandwidth from log")
    return float(match.group(1))


def parse_cpu_hotspots(section: str) -> pd.DataFrame:
    phase_specs = [
        ("Average beam", "Computing average beam.", "Finished computing average beam."),
        ("weights->Grid", "== weights->Grid() start", "== weights->Grid() end"),
        ("Min/max w and beam", "Determining min and max w & theoretical beam size start", "Determining min and max w & theoretical beam size DONE"),
        ("StartInversion", "IdgMsGridder::StartInversion() start", "IdgMsGridder::StartInversion() DONE"),
        ("Map rows", "Mapping measurement set rows start", "Mapping measurement set rows DONE"),
        ("weights->FinishGridding", "== weights->FinishGridding() start", "== weights->FinishGridding() end"),
        ("Weight cache", "== imageweightcache Get() start", "== imageweightcache Get() end"),
    ]
    start_times: dict[str, datetime] = {}
    rows: list[dict[str, float | str]] = []
    recorded = set()
    for line in section.splitlines():
        match = TIMESTAMP_RE.match(line)
        if not match:
            continue
        ts = datetime.strptime(match.group("ts"), "%Y-%b-%d %H:%M:%S.%f")
        msg = match.group("msg")
        for label, start_token, end_token in phase_specs:
            if label in recorded:
                continue
            if start_token in msg and label not in start_times:
                start_times[label] = ts
            elif end_token in msg and label in start_times:
                rows.append({"category": "cpu_hotspot", "label": label, "duration_s": (ts - start_times[label]).total_seconds()})
                recorded.add(label)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No CPU hotspot phases parsed from rebuttal log")
    return df.sort_values("duration_s", ascending=False).reset_index(drop=True)


def parse_gpu_hotspots(section: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    for line in section.splitlines():
        match = GPU_STAGE_RE.match(line.strip())
        if not match:
            continue
        kind = match.group("kind")
        rows.append(
            {
                "kind": kind,
                "time_s": float(match.group("time")),
                "throughput_gflops": float(match.group("gflops")) if match.group("gflops") else np.nan,
                "bandwidth_gbs": float(match.group("gbs")) if match.group("gbs") else np.nan,
                "power_w": float(match.group("power")) if match.group("power") else np.nan,
                "joules": float(match.group("joules")) if match.group("joules") else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No GPU hotspot lines parsed from rebuttal log")

    hotspot_df = (
        df.groupby("kind", as_index=False)
        .agg(
            sample_count=("time_s", "size"),
            median_time_s=("time_s", "median"),
            median_throughput_gflops=("throughput_gflops", "median"),
            median_bandwidth_gbs=("bandwidth_gbs", "median"),
        )
        .sort_values("median_time_s", ascending=False)
        .reset_index(drop=True)
    )
    roofline_df = hotspot_df.dropna(subset=["median_throughput_gflops", "median_bandwidth_gbs"]).copy()
    roofline_df["arithmetic_intensity_flop_per_byte"] = roofline_df["median_throughput_gflops"] / roofline_df["median_bandwidth_gbs"]
    return hotspot_df, roofline_df


def export_summary(cpu_hotspots: pd.DataFrame, gpu_hotspots: pd.DataFrame, gpu_roofline: pd.DataFrame, summary_csv: Path) -> None:
    cpu_export = cpu_hotspots.copy()
    gpu_hot_export = gpu_hotspots.rename(columns={"kind": "label", "median_time_s": "duration_s"}).copy()
    gpu_hot_export["category"] = "gpu_hotspot"
    gpu_roof_export = gpu_roofline.rename(
        columns={
            "kind": "label",
            "median_throughput_gflops": "throughput_gflops",
            "median_bandwidth_gbs": "bandwidth_gbs",
        }
    ).copy()
    gpu_roof_export["category"] = "gpu_roofline"
    summary = pd.concat([cpu_export, gpu_hot_export, gpu_roof_export], ignore_index=True, sort=False)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV to {summary_csv}")


def plot_figure(cpu_hotspots: pd.DataFrame, gpu_hotspots: pd.DataFrame, gpu_roofline: pd.DataFrame, h100_bw_gbs: float, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    ridge = H100_FP32_PEAK_GFLOPS / h100_bw_gbs

    fig, (ax_roof, ax_cpu, ax_gpu) = plt.subplots(
        1,
        3,
        figsize=(18.5, 7.6),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
    )

    x_min = 0.5
    x_max = 40.0
    y_min = 1.0
    y_max = H100_FP32_PEAK_GFLOPS * 1.15
    ai_diag = np.logspace(np.log10(x_min), np.log10(ridge), 300)
    ax_roof.plot(ai_diag, h100_bw_gbs * ai_diag, color=ROOF_DRAM_COLOR, linewidth=2.6, label=f"H100 memory roof ({h100_bw_gbs:.0f} GB/s)")
    ax_roof.hlines(H100_FP32_PEAK_GFLOPS, ridge, x_max, color=ROOF_COMPUTE_COLOR, linewidth=2.6, label="H100 FP32 peak (67 TFLOP/s)")
    ax_roof.axvline(ridge, color="0.6", linestyle=":", linewidth=1.1)

    point_colors = {
        "gridder": GPU_GRIDDER_COLOR,
        "sub-fft": GPU_SUBFFT_COLOR,
    }
    for row in gpu_roofline.to_dict("records"):
        label = str(row["kind"])
        color = point_colors.get(label, plt.cm.cividis(0.7))
        ax_roof.scatter(
            row["arithmetic_intensity_flop_per_byte"],
            row["median_throughput_gflops"],
            s=95,
            color=color,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )
        ax_roof.annotate(
            label,
            (row["arithmetic_intensity_flop_per_byte"], row["median_throughput_gflops"]),
            textcoords="offset points",
            xytext=(12, 10 if label == "gridder" else -14),
            fontsize=10.8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.75", "linewidth": 0.6},
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.8, "shrinkA": 2, "shrinkB": 4},
        )

    roof_text = (
        f"Representative rebuttal GPU run: 8192^2, t=1, c=8, p=8\n"
        f"Kernel points from profiler summary lines in the completed run\n"
        f"gridder: {gpu_roofline.loc[gpu_roofline['kind']=='gridder', 'median_throughput_gflops'].iloc[0]:.1f} GFLOP/s, "
        f"{gpu_roofline.loc[gpu_roofline['kind']=='gridder', 'arithmetic_intensity_flop_per_byte'].iloc[0]:.2f} FLOP/B\n"
        f"sub-fft: {gpu_roofline.loc[gpu_roofline['kind']=='sub-fft', 'median_throughput_gflops'].iloc[0]:.0f} GFLOP/s, "
        f"{gpu_roofline.loc[gpu_roofline['kind']=='sub-fft', 'arithmetic_intensity_flop_per_byte'].iloc[0]:.2f} FLOP/B"
    )
    ax_roof.text(
        0.03,
        0.05,
        roof_text,
        transform=ax_roof.transAxes,
        fontsize=10.6,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(x_min, x_max)
    ax_roof.set_ylim(y_min, y_max)
    ax_roof.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=13.0, fontweight="bold")
    ax_roof.set_ylabel("Throughput (GFLOP/s)", fontsize=13.0, fontweight="bold")
    ax_roof.set_title("H100 Kernel Roofline", fontsize=14.5, fontweight="bold")
    ax_roof.tick_params(axis="both", labelsize=11)
    ax_roof.legend(loc="upper left", fontsize=10.0, frameon=True)

    cpu_top = cpu_hotspots.head(5).iloc[::-1]
    y_cpu = np.arange(len(cpu_top), dtype=float)
    ax_cpu.barh(y_cpu, cpu_top["duration_s"], color=CPU_HOTSPOT_COLOR, edgecolor="black", linewidth=0.6)
    ax_cpu.set_yticks(y_cpu)
    ax_cpu.set_yticklabels(cpu_top["label"], fontsize=11)
    ax_cpu.set_xlabel("Duration (s)", fontsize=13.0, fontweight="bold")
    ax_cpu.set_title("Host CPU Hotspots", fontsize=14.5, fontweight="bold")
    ax_cpu.grid(True, axis="x", alpha=0.25)
    for yi, duration in zip(y_cpu, cpu_top["duration_s"]):
        ax_cpu.text(duration * 1.01, yi, f"{duration:.1f} s", va="center", fontsize=10.0)

    gpu_plot = gpu_hotspots[gpu_hotspots["kind"].isin(["average-beam", "wtiling", "gridder", "sub-fft"])].copy()
    gpu_plot = gpu_plot.sort_values("median_time_s", ascending=True).reset_index(drop=True)
    y_gpu = np.arange(len(gpu_plot), dtype=float)
    ax_gpu.barh(
        y_gpu,
        gpu_plot["median_time_s"],
        color=[GPU_HOTSPOT_COLORS.get(k, plt.cm.cividis(0.6)) for k in gpu_plot["kind"]],
        edgecolor="black",
        linewidth=0.6,
    )
    ax_gpu.set_yticks(y_gpu)
    ax_gpu.set_yticklabels(gpu_plot["kind"], fontsize=11)
    ax_gpu.set_xscale("log")
    ax_gpu.set_xlabel("Median kernel time (s)", fontsize=13.0, fontweight="bold")
    ax_gpu.set_title("GPU Kernel Hotspots", fontsize=14.5, fontweight="bold")
    ax_gpu.grid(True, axis="x", alpha=0.25)
    for yi, duration in zip(y_gpu, gpu_plot["median_time_s"]):
        ax_gpu.text(duration * 1.04, yi, f"{duration:.3g} s", va="center", fontsize=10.0)

    fig.suptitle(
        "Rebuttal Profiling: H100 Roofline with CPU and GPU Hotspots\n"
        "WSClean + IDG GPU run on Kuma H100, 8192^2 image, t=1, c=8, p=8",
        fontsize=16.0,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.12, wspace=0.28)

    for suffix in (".png", ".pdf"):
        out_path = output_stem.with_suffix(suffix)
        fig.savefig(out_path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out_path}")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.log_path.exists():
        print(f"Skipping rebuttal roofline plot: log not found: {args.log_path}")
        return 0

    text = args.log_path.read_text(errors="ignore")
    section = first_completed_run_section(text)
    h100_bw_gbs = parse_h100_bandwidth(section)
    cpu_hotspots = parse_cpu_hotspots(section)
    gpu_hotspots, gpu_roofline = parse_gpu_hotspots(section)
    export_summary(cpu_hotspots, gpu_hotspots, gpu_roofline, args.summary_csv)
    plot_figure(cpu_hotspots, gpu_hotspots, gpu_roofline, h100_bw_gbs, args.output_stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
