#!/usr/bin/env python3
"""
Plot 33: all-workload memory and measured I/O overview from the Darshan companion table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "problem_size_memory_darshan_table.csv"
OUT_PNG = BASE_DIR / "plot33_problem_sizes_memory_io_overview.png"
OUT_PDF = BASE_DIR / "plot33_problem_sizes_memory_io_overview.pdf"


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def workload_ticks(df: pd.DataFrame) -> tuple[list[int], list[str]]:
    tick_positions = []
    tick_labels = []
    grouped = df.groupby("Image size", sort=True)
    for image_size, group in grouped:
        center = int(round(group.index.to_numpy().mean()))
        tick_positions.append(center)
        tick_labels.append(f"{image_size}")
    return tick_positions, tick_labels


def add_group_spans(ax: plt.Axes, df: pd.DataFrame) -> None:
    colors = ["#f6f6f6", "#ffffff"]
    for idx, (_, group) in enumerate(df.groupby("Image size", sort=True)):
        left = group.index.min() - 0.5
        right = group.index.max() + 0.5
        ax.axvspan(left, right, color=colors[idx % 2], zorder=0)
        ax.axvline(right, color="#bbbbbb", linewidth=0.8, alpha=0.8)


def main() -> int:
    df = pd.read_csv(INPUT_CSV).sort_values(["Image size", "Timesteps", "Channels"]).reset_index(drop=True)
    x = np.arange(len(df))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.3], "hspace": 0.18},
        constrained_layout=True,
    )

    fig.suptitle(
        "All Workloads: Problem Footprint Estimates and Darshan I/O Measurements",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Top panel: analytical footprint plus measured transfer volumes where available.
    top_bar_width = 0.26
    ax_top.bar(
        x - top_bar_width,
        df["Input payload est. (GiB)"],
        width=top_bar_width,
        color="#8db8e8",
        edgecolor="white",
        linewidth=0.25,
        label="Input payload est.",
        zorder=2,
    )
    ax_top.bar(
        x,
        df["Output payload est. (GiB)"],
        width=top_bar_width,
        color="#f1c27d",
        edgecolor="white",
        linewidth=0.25,
        label="Output payload est.",
        zorder=2,
    )
    ax_top.bar(
        x + top_bar_width,
        df["Total payload est. (GiB)"],
        width=top_bar_width,
        color="#4f81bd",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.8,
        label="Total payload est.",
        zorder=2,
    )

    measured_series = [
        ("CPU Darshan MS read (GiB)", "#c94c4c", "o", "CPU MS read"),
        ("GPU Darshan MS read (GiB)", "#6b8e23", "s", "GPU MS read"),
        ("CPU Darshan named output writes (GiB)", "#7b52ab", "^", "CPU output writes"),
        ("GPU Darshan named output writes (GiB)", "#2f9c95", "D", "GPU output writes"),
    ]
    for column, color, marker, label in measured_series:
        mask = df[column].notna()
        ax_top.scatter(
            x[mask],
            df.loc[mask, column],
            s=44,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=4,
        )

    add_group_spans(ax_top, df)
    style_axis(ax_top)
    ax_top.set_yscale("log")
    ax_top.set_ylabel("Footprint / I/O volume (GiB)", fontsize=12)
    ax_top.set_title("Analytical payload sizes and measured transfer volumes", fontsize=13)
    ax_top.legend(
        ncol=4,
        fontsize=9,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        columnspacing=1.1,
        handletextpad=0.5,
    )

    # Bottom panel: measured throughput only, with missing workloads shown as gaps.
    throughput_series = [
        ("CPU Darshan POSIX throughput (MiB/s)", "#c94c4c", "CPU POSIX"),
        ("GPU Darshan POSIX throughput (MiB/s)", "#6b8e23", "GPU POSIX"),
        ("CPU Darshan STDIO throughput (MiB/s)", "#7b52ab", "CPU STDIO"),
        ("GPU Darshan STDIO throughput (MiB/s)", "#2f9c95", "GPU STDIO"),
    ]
    for column, color, label in throughput_series:
        ax_bottom.plot(
            x,
            df[column],
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            alpha=0.9,
            label=label,
            zorder=3,
        )

    add_group_spans(ax_bottom, df)
    style_axis(ax_bottom)
    ax_bottom.set_ylabel("Measured throughput (MiB/s)", fontsize=12)
    ax_bottom.set_title("Darshan-derived POSIX and STDIO throughput", fontsize=13)
    ax_bottom.legend(ncol=4, fontsize=9, frameon=True, loc="upper left")

    tick_positions, tick_labels = workload_ticks(df)
    ax_bottom.set_xticks(tick_positions)
    ax_bottom.set_xticklabels(tick_labels, fontsize=11)
    ax_bottom.set_xlabel("Image size group (workloads sorted by image size, timesteps, channels)", fontsize=12)

    # Top annotations for image-size groups.
    for image_size, group in df.groupby("Image size", sort=True):
        center = group.index.to_numpy().mean()
        ax_top.text(
            center,
            1.02,
            f"{image_size} px",
            transform=ax_top.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#444444",
        )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
