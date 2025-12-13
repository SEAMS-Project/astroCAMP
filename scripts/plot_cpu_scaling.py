#!/usr/bin/env python3
"""Plot CPU scalability (speedup vs threads) for fixed image sizes.
Reads data from cpu_scaling.csv and produces a speedup plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

plt.rcParams.update({
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "axes.titleweight": "bold",
})


def plot_scaling(csv_path: Path, results_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))

    image_sizes = sorted(df["image_size"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(image_sizes)))

    # Calculate max speedup for consistent y-axis
    max_speedup = df["speedup"].max() * 1.1  # Add 10% headroom

    for im, color in zip(image_sizes, colors):
        subset = df[df["image_size"] == im].sort_values("threads")
        ax.plot(
            subset["threads"],
            subset["speedup"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=color,
            label=f"im={im}",
        )
        # Annotate tail point
        tail = subset.iloc[-1]
        ax.text(
            tail["threads"],
            tail["speedup"] * 1.02,
            f"{tail['speedup']:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Ideal linear scaling reference
    max_threads = df["threads"].max()
    ideal_threads = sorted(df["threads"].unique())
    ax.plot(
        ideal_threads,
        ideal_threads,
        linestyle=":",
        color="black",
        linewidth=2,
        label="Ideal linear",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(df["threads"].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("CPU Threads", fontsize=13, fontweight="bold")
    ax.set_ylabel("Speedup (× vs 1 thread)", fontsize=13, fontweight="bold")
    ax.set_title("CPU Scalability: Speedup vs Threads", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max_speedup)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=10, title="Image Size", title_fontsize=11)

    outfile = results_dir / "cpu_scalability.png"
    results_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved CPU scalability plot to {outfile}")


def plot_time_speedup_dual(csv_path: Path, results_dir: Path) -> None:
    """Dual-axis plot: bars for absolute time, lines for speedup."""
    df = pd.read_csv(csv_path)
    threads = sorted(df["threads"].unique())
    image_sizes = sorted(df["image_size"].unique())

    fig, ax_time = plt.subplots(figsize=(5.5, 3.5))
    ax_speed = ax_time.twinx()

    colors = plt.cm.viridis(np.linspace(0, 1, len(image_sizes)))
    bar_width = 0.5 / max(len(image_sizes), 1)
    x = np.arange(len(threads), dtype=float)

    # Calculate max speedup for consistent right y-axis
    max_speedup = df["speedup"].max() * 1.1  # Add 10% headroom

    bar_handles = []
    line_handles = []

    for idx, (im, color) in enumerate(zip(image_sizes, colors)):
        subset = df[df["image_size"] == im].sort_values("threads")
        # Align data to full thread list
        time_vals = []
        speed_vals = []
        for t in threads:
            row = subset[subset["threads"] == t].iloc[0]
            time_vals.append(row["real_seconds"])
            speed_vals.append(row["speedup"])

        offset = (idx - (len(image_sizes) - 1) / 2) * bar_width
        bars = ax_time.bar(
            x + offset,
            time_vals,
            width=bar_width,
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.6,
            label=f"Time im={im}",
        )
        bar_handles.append(bars)

        line, = ax_speed.plot(
            x,
            speed_vals,
            color=color,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=f"Speedup im={im}",
        )
        line_handles.append(line)

        # Annotate last point speedup
        ax_speed.text(
            x[-1],
            speed_vals[-1] * 1.03,
            f"{speed_vals[-1]:.2f}×",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=color,
        )

    # Ideal linear reference for speedup
    ax_speed.plot(
        x,
        threads,
        linestyle=":",
        color="black",
        linewidth=2,
        label="Ideal linear",
    )

    ax_time.set_xlabel("CPU Threads", fontsize=13, fontweight="bold")
    ax_time.set_ylabel("Absolute Time (seconds)", fontsize=13, fontweight="bold")
    ax_speed.set_ylabel("Speedup (× vs 1 thread)", fontsize=13, fontweight="bold")
    ax_time.set_title("CPU Scalability: Time (bars) and Speedup (line)", fontsize=14, fontweight="bold")

    ax_time.set_xticks(x)
    ax_time.set_xticklabels([str(t) for t in threads], fontsize=14)
    ax_time.tick_params(axis="y", labelsize=11)
    ax_speed.tick_params(axis="y", labelsize=13)
    ax_speed.set_ylim(0, max_speedup)

    # Legends: combine bars and lines
    first_bar = bar_handles[0]
    handles = [first_bar] + line_handles + [
        plt.Line2D([0], [0], color="black", linestyle=":", linewidth=1.8, label="Ideal linear")
    ]
    labels = ["Time (bars)"] + [f"im={im}" for im in image_sizes] + ["Ideal linear"]
    ax_time.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=10,
        title_fontsize=11,
    )

    results_dir.mkdir(exist_ok=True)
    outfile = results_dir / "cpu_scalability_time_speedup.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved CPU time+speedup plot to {outfile}")


def main() -> None:
    csv_path = BASE_DIR / "cpu_scaling.csv"
    results_dir = BASE_DIR / "results"
    plot_scaling(csv_path, results_dir)
    plot_time_speedup_dual(csv_path, results_dir)


if __name__ == "__main__":
    main()
