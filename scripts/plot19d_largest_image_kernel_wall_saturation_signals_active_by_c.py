#!/usr/bin/env python3
"""
Standalone channel-grouped companion to plot19b using the exported kernel summary CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "kernel_breakdown_pasc25_16c_summary.csv"
OUT_PNG = BASE_DIR / "plot19d_largest_image_kernel_wall_saturation_signals_active_by_c.png"
OUT_PDF = BASE_DIR / "plot19d_largest_image_kernel_wall_saturation_signals_active_by_c.pdf"


def add_channel_group_guides(ax: plt.Axes, c_series: np.ndarray) -> None:
    c_series = np.asarray(c_series)
    breaks = [i - 0.5 for i in range(1, len(c_series)) if c_series[i] != c_series[i - 1]]
    groups = []
    start = 0
    for i in range(1, len(c_series) + 1):
        if i == len(c_series) or c_series[i] != c_series[i - 1]:
            groups.append((int(c_series[start]), start, i - 1))
            start = i

    for xb in breaks:
        ax.axvline(xb, linestyle=":", linewidth=1.1, color="0.45", alpha=0.9)

    secax = ax.secondary_xaxis("top")
    secax.set_xticks([0.5 * (i0 + i1) for _, i0, i1 in groups])
    secax.set_xticklabels([f"c={cval}" for cval, _, _ in groups])
    secax.tick_params(axis="x", labelsize=8, length=0, pad=2)


def main() -> int:
    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise SystemExit(f"No data in {INPUT_CSV}")

    largest_img = int(df["image_size"].max())
    sub = (
        df[df["image_size"] == largest_img]
        .sort_values(["channels", "timesteps"])
        .reset_index(drop=True)
        .copy()
    )
    sub["t_label"] = sub["timesteps"].map(lambda v: f"t={int(v)}")

    x = np.arange(len(sub), dtype=float)
    bw = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4))

    # Left panel: same structure as plot19b, regrouped by channels.
    g = sub["gridder_s"].values
    f = sub["sub_fft_s"].values
    w = sub["wtiling_s"].values
    wall = sub["wall_s_mean"].values
    c_series_all = sub["channels"].values

    ax1.bar(x - bw / 2, g, width=bw, label="gridder", color="#1f77b4")
    ax1.bar(x - bw / 2, f, width=bw, bottom=g, label="sub-fft", color="#ff7f0e")
    ax1.bar(x - bw / 2, w, width=bw, bottom=g + f, label="wtiling", color="#2ca02c")
    ax1.bar(x + bw / 2, wall, width=bw, label="WSClean wall time", color="0.6", alpha=0.45)
    ax1.set_yscale("log")
    ax1.set_ylabel("Time (s)")
    ax1.set_xlabel("Workload (timesteps, channels)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["t_label"].tolist(), rotation=90, ha="center")
    ax1.set_title("Kernel breakdown merged with total wall time")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)
    add_channel_group_guides(ax1, c_series_all)

    # Right panel: active-window metrics.
    ax2r = ax2.twinx()
    thr_mean = sub["throughput_mvis_s_mean"].values
    sm_util_active = sub["gpu_sm_util_active_mean_pct"].values
    mem_util_active = sub["gpu_mem_util_active_mean_pct"].values
    mem_active_gb = sub["gpu_mem_used_sum_mb_active_mean"].values / 1024.0

    ax2.plot(
        x,
        thr_mean,
        marker="o",
        linewidth=1.5,
        color="black",
        label="throughput mean",
    )

    finite_mem = np.where(np.isfinite(mem_active_gb), mem_active_gb, np.nan)
    ms = None
    if np.any(np.isfinite(finite_mem)):
        mmin = np.nanmin(finite_mem)
        mmax = np.nanmax(finite_mem)
        ms = 40.0 + 120.0 * (finite_mem - mmin) / (mmax - mmin + 1e-12)
        ax2.scatter(x, thr_mean, s=ms, color="black", alpha=0.28, label="active memory footprint (marker size)")

    ax2.set_ylabel("Average throughput (Mvis/s)")
    ax2.set_xlabel("Workload (timesteps, channels)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub["t_label"].tolist(), rotation=90, ha="center")
    ax2.set_title("Throughput with active-window GPU saturation indicators")
    ax2.grid(True, axis="y", alpha=0.25)
    add_channel_group_guides(ax2, c_series_all)

    ax2r.plot(x, sm_util_active, marker="s", linewidth=1.2, color="#d62728", label="GPU SM util active mean (%)")
    ax2r.plot(x, mem_util_active, marker="^", linewidth=1.2, color="#9467bd", label="GPU mem util active mean (%)")
    ax2r.set_ylabel("GPU utilization during active windows (%)")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    legend_lines = ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
    ax2.add_artist(legend_lines)

    if ms is not None and np.any(np.isfinite(mem_active_gb)):
        mvals = np.array([np.nanmin(mem_active_gb), np.nanmedian(mem_active_gb), np.nanmax(mem_active_gb)], dtype=float)
        mlabels = [f"{v:.1f} GB" for v in mvals]
        msizes = 40.0 + 120.0 * (mvals - np.nanmin(mem_active_gb)) / (np.nanmax(mem_active_gb) - np.nanmin(mem_active_gb) + 1e-12)
        bubble_handles = [ax2.scatter([], [], s=float(s), color="black", alpha=0.28) for s in msizes]
        legend_bubbles = ax2.legend(
            bubble_handles,
            mlabels,
            title="Active memory footprint",
            frameon=False,
            fontsize=8,
            title_fontsize=8,
            loc="upper left",
            bbox_to_anchor=(0.0, 0.62),
        )
        ax2.add_artist(legend_bubbles)

    ax2.text(
        0.99,
        0.02,
        "Active window = any GPU has SM util > 0 or mem util > 0",
        transform=ax2.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )

    fig.suptitle(
        f"Largest Image ({largest_img}): Kernel-Wall Coupling with Active-Window Saturation Signals Grouped by Channels",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
