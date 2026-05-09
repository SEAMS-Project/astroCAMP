#!/usr/bin/env python3
"""
Plot 34f: single-panel variant of Plot 34e — largest image size only.
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
OUT_PNG = RESULTS_DIR / "plot34f_largest_problem_size_with_gbj_by_c.png"
OUT_PDF = RESULTS_DIR / "plot34f_largest_problem_size_with_gbj_by_c.pdf"

INPUT_COLOR = "#8db8e8"
OUTPUT_COLOR = "#f1c27d"
HOST_COLOR = "#d62728"
DEVICE_COLOR = "#f08080"
EFF_AXIS_COLOR = "#d62728"
GIB_TO_GB = (1024**3) / 1e9

MEASURED_INPUT_GIB = {
    (1, 1): 13 / 1024,
    (1, 8): 41 / 1024,
    (1, 64): 269 / 1024,
    (1, 128): 528 / 1024,
    (1, 256): 1.1,
    (8, 1): 97 / 1024,
    (8, 8): 324 / 1024,
    (8, 64): 2.1,
    (8, 128): 4.2,
    (8, 256): 8.2,
    (64, 1): 772 / 1024,
    (64, 8): 2.6,
    (64, 64): 17.0,
    (64, 128): 33.0,
    (64, 256): 66.0,
    (128, 1): 1.6,
    (128, 8): 5.1,
    (128, 64): 34.0,
    (128, 128): 66.0,
    (128, 256): 131.0,
    (256, 1): 3.1,
    (256, 8): 11.0,
    (256, 64): 67.0,
    (256, 128): 132.0,
    (256, 256): 262.0,
}


def main() -> int:
    size_df = (
        pd.read_csv(SIZE_CSV)
        .sort_values(["Image size", "Channels", "Timesteps"])
    )
    largest_img = int(size_df["Image size"].max())
    size_df = size_df[size_df["Image size"] == largest_img].reset_index(drop=True)

    gbj_df = (
        pd.read_csv(GBJ_CSV)
        .query(f"image_size == {largest_img}")
        .rename(columns={"image_size": "Image size", "timesteps": "Timesteps", "channels": "Channels"})
    )

    sub = size_df.merge(gbj_df, on=["Image size", "Timesteps", "Channels"], how="left")
    sub["Input on-disk measured (GiB)"] = sub.apply(
        lambda row: MEASURED_INPUT_GIB.get(
            (int(row["Timesteps"]), int(row["Channels"])),
            float(row["Input on-disk est. (GiB)"]),
        ),
        axis=1,
    )
    sub["Input + Output measured (GiB)"] = sub["Input on-disk measured (GiB)"] + sub["Output (GiB)"]

    x = np.arange(len(sub), dtype=float) * 1.04
    input_gb = sub["Input on-disk measured (GiB)"] * GIB_TO_GB
    output_gb = sub["Output (GiB)"] * GIB_TO_GB
    total_gb = sub["Input + Output measured (GiB)"] * GIB_TO_GB
    host_gb_per_j = sub["host_gb_per_j"]
    device_gb_per_j = sub["device_gb_per_j"]

    fig, ax = plt.subplots(figsize=(4.8, 2.56), constrained_layout=True)

    ax.bar(x, input_gb, width=0.82, color=INPUT_COLOR, edgecolor="black", linewidth=0.4, label="Input on-disk", zorder=2, hatch="///")
    ax.bar(x, output_gb, bottom=input_gb, width=0.82, color=OUTPUT_COLOR, edgecolor="black", linewidth=0.4, label="Output", zorder=3, hatch="...")

    # Channel group separators and secondary x-axis
    c_groups = list(sub.groupby("Channels", sort=True))
    c_positions: list[float] = []
    c_labels: list[str] = []
    for idx, (channels, group) in enumerate(c_groups):
        c_positions.append(x[group.index].mean())
        c_labels.append(f"c={int(channels)}")
        if idx > 0:
            prev_group = c_groups[idx - 1][1]
            boundary_x = 0.5 * (x[prev_group.index.max()] + x[group.index.min()])
            ax.axvline(boundary_x, color="#7a7a7a", linewidth=0.9, linestyle=":", alpha=0.9, zorder=1)

    ax2 = ax.twinx()
    ax2.plot(x, host_gb_per_j, color=HOST_COLOR, marker="o", linewidth=1.6, markersize=3.8, label="CPU GB/J", zorder=5)
    ax2.plot(x, device_gb_per_j, color=DEVICE_COLOR, marker="s", linewidth=1.6, markersize=3.8, label="GPU GB/J", zorder=5)

    ax.set_xlim(x.min() - 0.65, x.max() + 0.65)
    ax.set_ylim(0, total_gb.max() * 1.08)
    finite_gbj = pd.concat([host_gb_per_j, device_gb_per_j], ignore_index=True).dropna()
    if not finite_gbj.empty:
        ax2.set_ylim(0, float(finite_gbj.max()) * 1.12)

    ax.set_xticks(x)
    ax.set_xticklabels([f"t={int(v)}" for v in sub["Timesteps"]], rotation=90, fontsize=7)
    ax.tick_params(axis="x", pad=1)
    ax.tick_params(axis="y", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8, colors=EFF_AXIS_COLOR)
    ax2.spines["right"].set_edgecolor(EFF_AXIS_COLOR)
    ax2.spines["top"].set_visible(False)
    ax.spines["top"].set_visible(False)

    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(c_positions)
    secax.set_xticklabels(c_labels, fontsize=8)
    secax.spines["bottom"].set_position(("outward", 30))
    secax.tick_params(axis="x", length=0, pad=2)

    ax.grid(axis="y", color="#d6d6d6", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)

    ax.text(0.01, 0.98, f"{largest_img}$^2$", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, fontweight="bold", color="#444444")

    ax.set_ylabel("Input and output size (GB)", fontsize=9)
    ax2.set_ylabel("IDG memory efficiency (GB/J)", fontsize=9, color=EFF_AXIS_COLOR)

    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    ax.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        ncol=2, fontsize=7, frameon=True, loc="upper right",
        columnspacing=0.9, handletextpad=0.4, borderpad=0.3, labelspacing=0.3,
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
