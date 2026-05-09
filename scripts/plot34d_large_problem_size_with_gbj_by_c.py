#!/usr/bin/env python3
"""
Plot 34d: stacked problem size with IDG GB/J on a secondary y-axis for 8k and 32k workloads,
grouped by channels rather than timesteps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = BASE_DIR.parent / "results"
SIZE_CSV = DERIVED_DIR / "problem_size_memory_table.csv"
GBJ_CSV = DERIVED_DIR / "plot32b_idg_data_movement_gb_per_joule_summary.csv"
OUT_PNG = RESULTS_DIR / "plot34d_large_problem_size_with_gbj_by_c.png"
OUT_PDF = RESULTS_DIR / "plot34d_large_problem_size_with_gbj_by_c.pdf"
OUT_SUMMARY = DERIVED_DIR / "plot34d_large_problem_size_with_gbj_by_c_summary.csv"

HOST_COLOR = plt.cm.cividis(0.26)
DEVICE_COLOR = plt.cm.cividis(0.72)


def main() -> int:
    size_df = (
        pd.read_csv(SIZE_CSV)
        .query("`Image size` in [8192, 32768]")
        .sort_values(["Image size", "Channels", "Timesteps"])
        .reset_index(drop=True)
    )
    gbj_df = (
        pd.read_csv(GBJ_CSV)
        .query("image_size in [8192, 32768]")
        .rename(
            columns={
                "image_size": "Image size",
                "timesteps": "Timesteps",
                "channels": "Channels",
            }
        )
    )

    merged = size_df.merge(gbj_df, on=["Image size", "Timesteps", "Channels"], how="left")
    merged.to_csv(OUT_SUMMARY, index=False)

    x = np.arange(len(merged), dtype=float) * 1.18
    fig, ax = plt.subplots(figsize=(11.8, 6.8), constrained_layout=True)

    ax.bar(
        x,
        merged["Input on-disk est. (GiB)"],
        width=0.9,
        color="#8db8e8",
        edgecolor="white",
        linewidth=0.25,
        label="Input on-disk est.",
        zorder=2,
    )
    ax.bar(
        x,
        merged["Output (GiB)"],
        bottom=merged["Input on-disk est. (GiB)"],
        width=0.9,
        color="#f1c27d",
        edgecolor="white",
        linewidth=0.25,
        label="Output",
        zorder=3,
    )

    colors = ["#f7f7f7", "#ffffff"]
    for idx, (image_size, group) in enumerate(merged.groupby("Image size", sort=True)):
        left = x[group.index.min()] - 0.6
        right = x[group.index.max()] + 0.6
        ax.axvspan(left, right, color=colors[idx % 2], zorder=0)
        ax.axvline(right, color="#c0c0c0", linewidth=0.8, alpha=0.8)
        center = x[group.index].mean()
        ax.text(
            center,
            1.02,
            f"{int(image_size)} px",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12,
            color="#444444",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"t={int(v)}" for v in merged["Timesteps"]], rotation=90, fontsize=10)

    c_positions = []
    c_labels = []
    c_groups = list(merged.groupby(["Image size", "Channels"], sort=True))
    for (_, channels), group in c_groups:
        c_positions.append(x[group.index].mean())
        c_labels.append(f"c={int(channels)}")

    for idx in range(1, len(c_groups)):
        prev_key, prev_group = c_groups[idx - 1]
        curr_key, curr_group = c_groups[idx]
        if prev_key[0] == curr_key[0]:
            boundary_x = 0.5 * (x[prev_group.index.max()] + x[curr_group.index.min()])
            ax.axvline(
                boundary_x,
                color="#7a7a7a",
                linewidth=1.0,
                linestyle=":",
                alpha=0.9,
                zorder=1,
            )

    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(c_positions)
    secax.set_xticklabels(c_labels, fontsize=12)
    secax.spines["bottom"].set_position(("outward", 48))
    secax.tick_params(axis="x", length=0, pad=3)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        merged["host_gb_per_j"],
        color=HOST_COLOR,
        marker="o",
        linewidth=2.0,
        markersize=4.8,
        label="CPU GB/J",
        zorder=5,
    )
    ax2.plot(
        x,
        merged["device_gb_per_j"],
        color=DEVICE_COLOR,
        marker="s",
        linewidth=2.0,
        markersize=4.8,
        label="GPU GB/J",
        zorder=5,
    )

    ax.set_xlim(x.min() - 0.75, x.max() + 0.75)
    ax.set_ylim(0, merged["Input + Output (GiB)"].max() * 1.06)
    finite_gbj = pd.concat([merged["host_gb_per_j"], merged["device_gb_per_j"]], ignore_index=True).dropna()
    if not finite_gbj.empty:
        ax2.set_ylim(0, float(finite_gbj.max()) * 1.08)

    ax.set_ylabel("Problem size (GiB)", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax2.set_ylabel("IDG data movement (GB/J)", fontsize=12)
    ax2.tick_params(axis="y", labelsize=12)

    ax.grid(axis="y", color="#d6d6d6", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, ncol=4, fontsize=11, frameon=True, loc="upper left")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
