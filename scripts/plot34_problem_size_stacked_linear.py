#!/usr/bin/env python3
"""
Plot 34: stacked input/output problem sizes on a linear y-axis.
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
INPUT_CSV = DERIVED_DIR / "problem_size_memory_table.csv"
OUT_PNG = RESULTS_DIR / "plot34_problem_size_stacked_linear.png"
OUT_PDF = RESULTS_DIR / "plot34_problem_size_stacked_linear.pdf"


def main() -> int:
    df = pd.read_csv(INPUT_CSV).sort_values(["Image size", "Timesteps", "Channels"]).reset_index(drop=True)
    x = np.arange(len(df), dtype=float) * 1.14

    fig, ax = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)

    ax.bar(
        x,
        df["Input on-disk est. (GiB)"],
        width=0.86,
        color="#8db8e8",
        edgecolor="white",
        linewidth=0.25,
        label="Input on-disk est.",
        zorder=2,
    )
    ax.bar(
        x,
        df["Output (GiB)"],
        bottom=df["Input on-disk est. (GiB)"],
        width=0.86,
        color="#f1c27d",
        edgecolor="white",
        linewidth=0.25,
        label="Output",
        zorder=3,
    )

    colors = ["#f7f7f7", "#ffffff"]
    for idx, (image_size, group) in enumerate(df.groupby("Image size", sort=True)):
        left = x[group.index.min()] - 0.57
        right = x[group.index.max()] + 0.57
        ax.axvspan(left, right, color=colors[idx % 2], zorder=0)
        ax.axvline(right, color="#c0c0c0", linewidth=0.8, alpha=0.8)
        center = x[group.index].mean()
        ax.text(
            center,
            1.02,
            f"{image_size} px",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12,
            color="#444444",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={int(v)}" for v in df["Channels"]], rotation=90, fontsize=10)

    t_positions = []
    t_labels = []
    for (_, timesteps), group in df.groupby(["Image size", "Timesteps"], sort=True):
        t_positions.append(x[group.index].mean())
        t_labels.append(f"t={int(timesteps)}")

    secax = ax.secondary_xaxis("bottom")
    secax.set_xticks(t_positions)
    secax.set_xticklabels(t_labels, fontsize=12)
    secax.spines["bottom"].set_position(("outward", 48))
    secax.tick_params(axis="x", length=0, pad=3)

    ax.set_xlim(x.min() - 0.72, x.max() + 0.72)
    ax.set_ylim(0, df["Input + Output (GiB)"].max() * 1.06)
    ax.set_ylabel("Problem size (GiB)", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.legend(ncol=2, fontsize=12, frameon=True, loc="upper left")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
