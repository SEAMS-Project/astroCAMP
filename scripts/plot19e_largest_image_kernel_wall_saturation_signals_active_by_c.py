#!/usr/bin/env python3
"""
Variant of plot19d with left panel on a linear time scale and side-by-side
(grouped) bars for each kernel component and WSClean wall time.
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
OUT_PNG = BASE_DIR / "plot19e_largest_image_kernel_wall_saturation_signals_active_by_c.png"
OUT_PDF = BASE_DIR / "plot19e_largest_image_kernel_wall_saturation_signals_active_by_c.pdf"


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
    secax.tick_params(axis="x", labelsize=16, length=0, pad=2)


def main() -> int:
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 18,
    })

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

    fig = plt.figure(figsize=(24.0, 7.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    g = sub["gridder_s"].values
    f = sub["sub_fft_s"].values
    w = sub["wtiling_s"].values
    wall = sub["wall_s_mean"].values
    c_series_all = sub["channels"].values

    bw = 0.5

    # Left panel: WSClean total wall time only.
    ax1.bar(x, wall, width=bw, label="WSClean wall time", color="0.6", alpha=0.75)
    ax1.set_ylabel("Time (s)")
    ax1.set_xlabel("Workload (timesteps, channels)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["t_label"].tolist(), rotation=90, ha="center")
    ax1.tick_params(axis="x", labelsize=15)
    ax1.set_title("WSClean total wall time")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(frameon=False, fontsize=16)
    add_channel_group_guides(ax1, c_series_all)

    # Middle panel: IDG sub-kernel components only (much smaller timescale).
    bw_sub = 0.24
    offsets_sub = np.array([-1.0, 0.0, 1.0]) * bw_sub
    ax2.bar(x + offsets_sub[0], g, width=bw_sub, label="gridder",  color="#1f77b4")
    ax2.bar(x + offsets_sub[1], f, width=bw_sub, label="sub-fft",  color="#ff7f0e")
    ax2.bar(x + offsets_sub[2], w, width=bw_sub, label="wtiling",  color="#2ca02c")
    ax2.set_ylabel("Time (s)")
    ax2.set_xlabel("Workload (timesteps, channels)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sub["t_label"].tolist(), rotation=90, ha="center")
    ax2.tick_params(axis="x", labelsize=15)
    ax2.set_title("IDG sub-kernel components")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(frameon=False, fontsize=16)
    add_channel_group_guides(ax2, c_series_all)

    # Right panel: active-window metrics (unchanged from plot19d).
    ax3r = ax3.twinx()
    thr_mean = sub["throughput_mvis_s_mean"].values
    sm_util_active = sub["gpu_sm_util_active_mean_pct"].values
    mem_util_active = sub["gpu_mem_util_active_mean_pct"].values
    mem_active_gb = sub["gpu_mem_used_sum_mb_active_mean"].values / 1024.0

    ax3.plot(
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
        ms = 200.0 + 600.0 * (finite_mem - mmin) / (mmax - mmin + 1e-12)
        ax3.scatter(x, thr_mean, s=ms, color="steelblue", alpha=0.35, zorder=5, label="active memory footprint (marker size)")

    ax3.set_ylabel("Average throughput (Mvis/s)")
    ax3.set_xlabel("")
    ax3.set_xticks(x)
    ax3.set_xticklabels(sub["t_label"].tolist(), rotation=90, ha="center")
    ax3.set_title(
        "Throughput & active-window GPU saturation\n"
        "(bubble size \u221d active memory footprint in GiB;"
        " active window: any GPU SM or mem util > 0)"
    )
    ax3.grid(True, axis="y", alpha=0.25)
    add_channel_group_guides(ax3, c_series_all)

    ax3r.plot(x, sm_util_active, marker="s", linewidth=1.2, color="#d62728", label="GPU SM util active mean (%)")
    ax3r.plot(x, mem_util_active, marker="^", linewidth=1.2, color="#f08080", label="GPU mem util active mean (%)")
    ax3r.set_ylabel("GPU utilization during active windows (%)", color="#d62728")
    ax3r.tick_params(axis="y", colors="#d62728")
    ax3r.spines["right"].set_edgecolor("#d62728")

    # Single combined legend placed below the panel so nothing overlaps the data.
    h1, l1 = ax3.get_legend_handles_labels()
    h2, l2 = ax3r.get_legend_handles_labels()
    ax3.legend(
        h1 + h2, l1 + l2,
        frameon=True,
        framealpha=0.92,
        fontsize=14,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=2,
        borderaxespad=0,
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
