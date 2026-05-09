#!/usr/bin/env python3
"""
Plot 33b: compact normalized heatmap overview for all-workload memory and I/O metrics.
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
INPUT_CSV = DERIVED_DIR / "problem_size_memory_darshan_table.csv"
OUT_PNG = RESULTS_DIR / "plot33b_problem_sizes_memory_io_heatmap.png"
OUT_PDF = RESULTS_DIR / "plot33b_problem_sizes_memory_io_heatmap.pdf"


METRICS = [
    ("Input payload est. (GiB)", "Input est. (GiB)"),
    ("Output payload est. (GiB)", "Output est. (GiB)"),
    ("Total payload est. (GiB)", "Total est. (GiB)"),
    ("CPU Darshan MS read (GiB)", "CPU MS read (GiB)"),
    ("GPU Darshan MS read (GiB)", "GPU MS read (GiB)"),
    ("CPU Darshan named output writes (GiB)", "CPU out (GiB)"),
    ("GPU Darshan named output writes (GiB)", "GPU out (GiB)"),
    ("CPU Darshan POSIX throughput (MiB/s)", "CPU POSIX (MiB/s)"),
    ("GPU Darshan POSIX throughput (MiB/s)", "GPU POSIX (MiB/s)"),
    ("CPU Darshan STDIO throughput (MiB/s)", "CPU STDIO (MiB/s)"),
    ("GPU Darshan STDIO throughput (MiB/s)", "GPU STDIO (MiB/s)"),
]


def normalize_row(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.full_like(values, np.nan, dtype=float)
    vmax = np.nanmax(values[finite])
    if vmax <= 0:
        return np.full_like(values, np.nan, dtype=float)
    return values / vmax


def main() -> int:
    df = pd.read_csv(INPUT_CSV).sort_values(["Image size", "Timesteps", "Channels"]).reset_index(drop=True)

    heatmap_rows = []
    row_labels = []
    row_max_labels = []
    for column, label in METRICS:
        values = df[column].to_numpy(dtype=float)
        normalized = normalize_row(values)
        heatmap_rows.append(normalized)
        row_labels.append(label)
        finite = np.isfinite(values)
        if finite.any():
            vmax = np.nanmax(values[finite])
            row_max_labels.append(f"max {vmax:.2f}")
        else:
            row_max_labels.append("no data")

    heatmap = np.vstack(heatmap_rows)
    cmap = plt.cm.cividis.copy()
    cmap.set_bad(color="#d9d9d9")

    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    im = ax.imshow(heatmap, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_title(
        "All Workloads: Normalized Memory and I/O Overview\n"
        "Each row is normalized by its own maximum; gray cells indicate missing measurements",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Metric", fontsize=12)
    ax.set_xlabel("Workloads sorted by image size, timesteps, channels", fontsize=12)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([f"{name}  ({extra})" for name, extra in zip(row_labels, row_max_labels)], fontsize=10)

    group_centers = []
    group_labels = []
    for image_size, group in df.groupby("Image size", sort=True):
        left = group.index.min() - 0.5
        right = group.index.max() + 0.5
        ax.axvline(right, color="white", linewidth=0.7, alpha=0.6)
        group_centers.append(group.index.to_numpy().mean())
        group_labels.append(str(image_size))

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Within-metric normalized value", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
