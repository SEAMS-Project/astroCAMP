#!/usr/bin/env python3
"""
Generate four publication-friendly figures from benchmark data:
1) Pareto: Throughput (Mvis/s) vs Energy (kWh)
2) Iso-performance heatmaps: throughput over (n_times, n_chans) per image size
3) Energy breakdown: static vs dynamic for best-throughput configs
4) Carbon vs Cost: scatter with Pareto highlight
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from results_dataframe import generate_results_dataframe

# Matplotlib style tweaks
plt.rcParams.update({
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "axes.titleweight": "bold",
})

BASE_DIR = Path(__file__).resolve().parent


def load_data(lifetime_years: int, location_id: str) -> pd.DataFrame:
    """Load benchmarks, machines, and locations; compute derived metrics using helper function."""
    # Generate results DataFrame using helper function
    results_df = generate_results_dataframe(
        benchmarks_csv_path=BASE_DIR / 'benchmarks.csv',
        machines_csv_path=BASE_DIR / 'machines.csv',
        locations_csv_path=BASE_DIR / 'locations.csv',
        lifetime_years=lifetime_years,
        location_ids=[location_id]
    )
    
    # Filter to selected location
    results_df = results_df[results_df['Location'] == location_id].copy()
    
    # Rename columns to match expected names in plotting functions
    df = results_df.copy()
    
    # Map results DataFrame columns to expected column names
    df['im_size'] = df['Image Size']
    df['n_times'] = df['Timesteps']
    df['n_chans'] = df['Channels']
    df['time_s'] = df['Time (s)']
    df['time_h'] = df['Time (s)'] / 3600.0
    df['mvis'] = df['Mvis']
    
    # Convert energy from Wh to kWh
    df['energy_static_kwh'] = df['Static Energy (Wh)'] / 1000.0
    df['energy_dynamic_kwh'] = df['Dynamic Energy (Wh)'] / 1000.0
    df['energy_kwh'] = df['Energy (Wh)'] / 1000.0
    
    # Performance & efficiency
    df['throughput_mvis_s'] = df['Mvis'] / (df['Time (s)'] / 3600.0)  # Mvis per hour / 3600
    df['efficiency_mvis_kwh'] = df['Mvis/kWh']
    
    # Cost & carbon (convert from grams to kg where needed)
    df['operational_cost_$'] = df['Operational Cost ($)']
    df['operational_carbon_kg'] = df['Operational Carbon (g CO2)'] / 1000.0
    df['capital_cost_$'] = df['Capital Cost ($)']
    df['capital_carbon_kg'] = df['Embodied Carbon (g CO2)'] / 1000.0
    df['total_cost_$'] = df['Total Cost ($)']
    df['total_carbon_kg'] = df['Total Carbon (g CO2)'] / 1000.0
    
    # Regime indicator for visual encoding
    df['regime'] = df.apply(
        lambda row: "Time-heavy"
        if row['n_times'] > row['n_chans']
        else "Channel-heavy"
        if row['n_chans'] > row['n_times']
        else "Balanced",
        axis=1,
    )

    return df


def pareto_front(df: pd.DataFrame, energy_col: str, perf_col: str) -> pd.DataFrame:
    """Return nondominated points minimizing energy and maximizing performance."""
    ordered = df.sort_values([energy_col, perf_col], ascending=[True, False])
    best_perf = -np.inf
    front_indices: List[int] = []
    for idx, row in ordered.iterrows():
        perf = row[perf_col]
        if perf > best_perf:
            front_indices.append(idx)
            best_perf = perf
    return df.loc[front_indices].sort_values(energy_col)


def plot_pareto(df: pd.DataFrame, results_dir: Path) -> None:
    print("\n" + "="*80)
    print("FIGURE: Pareto Front (Throughput vs Energy)")
    print("="*80)
    print("Description: Scatter plot showing trade-off between energy consumption and")
    print("throughput. Color represents image size; marker shape represents n_times bucket.")
    print("Pareto front (black line) highlights non-dominated configurations.")
    print("Interpretation: Points on the Pareto front offer best trade-offs; larger")
    print("image sizes and more timesteps generally improve both metrics.")
    print()
    
    fig, ax = plt.subplots(figsize=(10, 7))

    # Encodings
    im_sizes = sorted(df["im_size"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(im_sizes)))
    color_map = dict(zip(im_sizes, colors))

    # Marker by n_times bucket (small/med/large)
    def times_bucket(n):
        if n <= 4:
            return "small"
        if n <= 32:
            return "medium"
        return "large"

    markers = {"small": "o", "medium": "s", "large": "^"}

    for _, row in df.iterrows():
        bucket = times_bucket(row["n_times"])
        ax.scatter(
            row["energy_kwh"] * 3.6e3,  # kWh -> kJ
            row["throughput_mvis_s"],
            color=color_map[row["im_size"]],
            marker=markers[bucket],
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.7,
        )

    # Pareto front
    front = pareto_front(df, "energy_kwh", "throughput_mvis_s")
    ax.plot(
        front["energy_kwh"] * 3.6e3,
        front["throughput_mvis_s"],
        color="black",
        linewidth=2.5,
        label="Pareto front",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Energy (kJ)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Throughput (Mvis/s)", fontsize=14, fontweight="bold")
    ax.set_title("Pareto Front: Throughput vs Energy (kJ)", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)

    color_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"im={im}",
                   markerfacecolor=color_map[im], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=9)
        for im in im_sizes
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=m, color="gray", label=lab,
                   markerfacecolor="gray", markeredgecolor="black",
                   markeredgewidth=0.5, linestyle="None", markersize=9)
        for lab, m in markers.items()
    ]
    legend1 = ax.legend(handles=color_handles, title="Image Size", loc="upper left", fontsize=10, title_fontsize=11)
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles + [plt.Line2D([0], [0], color="black", linewidth=2.5, label="Pareto front")],
              title="n_times bucket", loc="lower right", fontsize=10, title_fontsize=11)

    outfile = results_dir / "pareto_throughput_energy.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved pareto plot to {outfile}\n")


def plot_iso_heatmaps(df: pd.DataFrame, results_dir: Path) -> None:
    print("="*80)
    print("FIGURE: Iso-Performance Heatmaps (Throughput over n_times × n_chans)")
    print("="*80)
    print("Description: 2×2 grid of heatmaps showing throughput (Mvis/s) across all")
    print("(n_times, n_chans) combinations for each image size. Log color scale.")
    print("Interpretation: Green regions show high throughput; red regions show low.")
    print("Larger image sizes and balanced (equal) timesteps/channels yield best results.")
    print()
    
    im_sizes = sorted(df["im_size"].unique())
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 11), sharex=False, sharey=False)
    axes = axes.flatten()

    for ax, im in zip(axes, im_sizes):
        subset = df[df["im_size"] == im]
        pivot = subset.pivot_table(
            index="n_times", columns="n_chans", values="throughput_mvis_s", aggfunc="max"
        )
        # Ensure sorted axes
        pivot = pivot.sort_index().sort_index(axis=1)
        times = pivot.index.to_list()
        chans = pivot.columns.to_list()
        mesh = ax.pcolormesh(
            chans,
            times,
            pivot.values,
            norm=LogNorm(),
            cmap="viridis",
            shading="auto",
        )
        cbar = fig.colorbar(mesh, ax=ax, label="Throughput (Mvis/s)")
        cbar.ax.tick_params(labelsize=11)
        ax.set_title(f"Image Size = {im}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Channels (n_chans)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Timesteps (n_times)", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.tick_params(axis="both", labelsize=11)

        # Annotate top-3 throughput cells
        top3 = subset.nlargest(3, "throughput_mvis_s")
        for _, row in top3.iterrows():
            ax.text(
                row["n_chans"],
                row["n_times"],
                f"{row['throughput_mvis_s']:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5, linewidth=0),
            )

    # Hide unused axes if fewer than 4 image sizes
    for ax in axes[len(im_sizes) :]:
        ax.axis("off")

    plt.tight_layout()
    outfile = results_dir / "iso_performance_heatmaps.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved iso-performance heatmaps to {outfile}\n")


def plot_energy_breakdown(df: pd.DataFrame, results_dir: Path) -> None:
    print("="*80)
    print("FIGURE: Energy Breakdown (Best Throughput Configurations)")
    print("="*80)
    print("Description: Stacked bar chart showing static vs dynamic energy for best-")
    print("throughput configuration per (image_size, n_times) pair.")
    print("Interpretation: Dynamic energy dominates for larger workloads. Static")
    print("energy becomes more significant for small jobs. Total scales with workload.")
    print()
    
    # Best throughput per (im_size, n_times)
    idx = df.groupby(["im_size", "n_times"])["throughput_mvis_s"].idxmax()
    best = df.loc[idx].sort_values(["im_size", "n_times"])

    labels = [f"im{row.im_size}\nt{int(row.n_times)}\nc{int(row.n_chans)}" for _, row in best.iterrows()]
    x = np.arange(len(best))

    fig, ax = plt.subplots(figsize=(14, 6))
    static_kj = best["energy_static_kwh"] * 3.6e3
    dynamic_kj = best["energy_dynamic_kwh"] * 3.6e3
    ax.bar(x, static_kj, label="Static", color="#b0c4de", width=0.8)
    ax.bar(x, dynamic_kj, bottom=static_kj, label="Dynamic", color="#1f77b4", width=0.8)

    ax.set_ylabel("Energy (kJ)", fontsize=14, fontweight="bold")
    ax.set_title("Energy Breakdown (kJ): Static vs Dynamic (Best Throughput Configs)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="right", fontsize=9)
    ax.legend(fontsize=12, loc="upper left")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    outfile = results_dir / "energy_breakdown_best_configs.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved energy breakdown to {outfile}\n")


def plot_energy_breakdown_grouped(df: pd.DataFrame, results_dir: Path, group_by: str = "im_size") -> None:
    """Stacked energy bars per configuration grouped by `group_by` (im_size, n_times, or n_chans).
    Adds text markers above each bar with total carbon (g CO2).
    Uses consistent y-axis range across all subplots for easy comparison.
    Right y-axis shows wall time in seconds.
    """
    print("="*80)
    print(f"FIGURE: Energy Breakdown Grouped by {group_by.upper()}")
    print("="*80)
    print(f"Description: Stacked bars (dynamic vs static energy) for all configurations,")
    print(f"grouped by {group_by}. Carbon (g CO2) labels above each bar.")
    print(f"Right y-axis shows wall time (seconds) for each configuration.")
    print(f"Interpretation: Consistent y-axis enables visual comparison across {group_by}")
    print(f"groups. Larger workloads show higher energy and carbon. Balance is key.")
    print()
    
    assert group_by in {"im_size", "n_times", "n_chans"}
    groups = sorted(df[group_by].unique())

    # Build labels per configuration: just n_times x n_chans (im_size is obvious from grouping)
    def label_row(row):
        return f"{int(row.n_times)}×{int(row.n_chans)}"

    # Compute global y-axis limits (in kiloJoules) for consistent comparison across subplots
    all_totals_kj = (df["energy_static_kwh"].values + df["energy_dynamic_kwh"].values) * 3.6e3
    y_min = 0.0
    y_max = all_totals_kj.max() * 1.30  # Add 30% headroom for carbon labels and margins

    # Compute global wall time limits for consistent right y-axis
    time_min = 0.0
    time_max = (df["time_s"].max() / 60.0) * 1.25  # Add 25% headroom, convert to minutes

    # Create a multi-row figure to keep bars readable
    nrows = len(groups)
    # Calculate figure height based on number of bars per subplot
    bars_per_subplot = len(df) // nrows if nrows > 0 else len(df)
    fig_height = max(3 + (bars_per_subplot * 0.2), 4 * nrows)
    fig_width = max(8, 2 + bars_per_subplot * 0.4)
    fig, axes = plt.subplots(nrows, 1, figsize=(fig_width, fig_height), sharex=False)
    if nrows == 1:
        axes = [axes]

    for idx, (ax, g) in enumerate(zip(axes, groups)):
        subset = df[df[group_by] == g].copy()
        # Sort for consistent display
        if group_by == "im_size":
            subset = subset.sort_values(["n_times", "n_chans"]).reset_index(drop=True)
        else:
            subset = subset.sort_values(["im_size", "n_times", "n_chans"]).reset_index(drop=True)

        x = np.arange(len(subset))
        labels = [label_row(r) for _, r in subset.iterrows()]

        # Energy in kiloJoules
        static_kj = subset["energy_static_kwh"] * 3.6e3
        dynamic_kj = subset["energy_dynamic_kwh"] * 3.6e3
        ax.bar(x, static_kj, label="Static", color="#b0c4de", width=0.8)
        ax.bar(x, dynamic_kj, bottom=static_kj, label="Dynamic", color="#1f77b4", width=0.8)

        # Carbon markers above bars (g CO2)
        carbon_g = subset["total_carbon_kg"].values * 1000.0
        tops = static_kj.values + dynamic_kj.values
        for xi, top, cg in zip(x, tops, carbon_g):
            ax.text(
                xi,
                top * 1.09,
                f"{cg:.0f}g",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                rotation=0,
            )

        ax.set_title(f"Grouped by {group_by} = {int(g)}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Energy (kJ)", fontsize=12, fontweight="bold", color="navy")
        
        # Set y-axis limits with proper scaling
        ax.set_ylim(y_min, y_max)
        
        ax.set_xticks(x)
        
        # Only show x-axis labels on bottom subplot with better formatting
        if idx == len(axes) - 1:
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            ax.margins(x=0.01)  # Reduce margins to compress x-axis
        else:
            ax.set_xticklabels([])
        
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add right y-axis for wall time
        ax_right = ax.twinx()
        ax_right.plot(x, subset["time_s"] / 60.0, color="red", marker="o", linewidth=2, markersize=5, label="Wall Time", alpha=0.4)
        ax_right.set_ylabel("Wall Time (min)", fontsize=12, fontweight="bold", color="darkred")
        # Set right y-axis limits with proper scaling
        ax_right.set_ylim(time_min, time_max)
        # Set ticks dynamically based on time_max
        max_ticks = 6
        tick_interval = time_max / max_ticks
        ax_right.set_yticks(np.arange(0, time_max + tick_interval/2, tick_interval))
        ax_right.tick_params(axis="y", labelsize=11, labelcolor="red")

    plt.tight_layout()
    outfile = results_dir / f"energy_breakdown_grouped_by_{group_by}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved grouped energy breakdown (by {group_by}) to {outfile}\n")


def plot_latency_throughput(df: pd.DataFrame, results_dir: Path) -> None:
    print("="*80)
    print("FIGURE: Latency vs Throughput Trade-off")
    print("="*80)
    print("Description: Log-log scatter plot showing latency (wall_time_sec) vs")
    print("throughput (Mvis/s). Marker size represents workload (Mvis); color")
    print("represents image size. Pareto front (black line) highlights optimal configs.")
    print("Interpretation: Low latency + high throughput is ideal (upper left).")
    print("Pareto points balance speed vs compute intensity. Larger workloads")
    print("tend toward higher latency but better throughput efficiency.")
    print()
    
    fig, ax = plt.subplots(figsize=(10, 7))

    im_sizes = sorted(df["im_size"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(im_sizes)))
    color_map = dict(zip(im_sizes, colors))

    # Marker by n_times bucket (small/med/large)
    def times_bucket(n):
        if n <= 4:
            return "small"
        if n <= 32:
            return "medium"
        return "large"

    markers = {"small": "o", "medium": "s", "large": "^"}

    for _, row in df.iterrows():
        bucket = times_bucket(row["n_times"])
        ax.scatter(
            row["time_s"],
            row["throughput_mvis_s"],
            color=color_map[row["im_size"]],
            marker=markers[bucket],
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.7,
        )

    # Pareto front: minimize latency, maximize throughput
    ordered = df.sort_values(["time_s"])
    best_throughput = -np.inf
    front_indices: List[int] = []
    for idx, row in ordered.iterrows():
        tp = row["throughput_mvis_s"]
        if tp > best_throughput:
            front_indices.append(idx)
            best_throughput = tp
    latency_throughput_front = df.loc[front_indices].sort_values("time_s")
    ax.plot(
        latency_throughput_front["time_s"],
        latency_throughput_front["throughput_mvis_s"],
        color="black",
        linewidth=2.5,
        label="Pareto front",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latency (Wall Time, seconds)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Throughput (Mvis/s)", fontsize=14, fontweight="bold")
    ax.set_title("Latency vs Throughput Trade-off", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)

    color_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=f"im={im}",
                   markerfacecolor=color_map[im], markeredgecolor="black",
                   markeredgewidth=0.5, markersize=9)
        for im in im_sizes
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker=m, color="gray", label=lab,
                   markerfacecolor="gray", markeredgecolor="black",
                   markeredgewidth=0.5, linestyle="None", markersize=9)
        for lab, m in markers.items()
    ]
    legend1 = ax.legend(handles=color_handles, title="Image Size", loc="upper left", fontsize=10, title_fontsize=11)
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles + [plt.Line2D([0], [0], color="black", linewidth=2.5, label="Pareto front")],
              title="n_times bucket", loc="lower right", fontsize=10, title_fontsize=11)

    outfile = results_dir / "latency_throughput_tradeoff.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved latency vs throughput plot to {outfile}\n")


def plot_metric_grouped(df: pd.DataFrame, results_dir: Path, metric: str, group_by: str = "n_times") -> None:
    """Generic grouped bar plot for any metric.
    Each subplot shows one group value with bars for different configurations.
    Uses consistent y-axis range across all subplots for easy comparison.
    """
    # Metric configurations
    metric_configs = {
        "time_s": {"label": "Latency (seconds)", "format": "{:.1f}s", "name": "Latency"},
        "throughput_mvis_s": {"label": "Throughput (Mvis/s)", "format": "{:.2f}", "name": "Throughput"},
        # Plot energy in kJ even though source metric is kWh
        "energy_kwh": {"label": "Energy (kJ)", "format": "{:.0f}kJ", "name": "Energy"},
        "total_cost_$": {"label": "Total Cost ($)", "format": "${:.2f}", "name": "Cost"},
        "total_carbon_kg": {"label": "Total Carbon (kg CO2)", "format": "{:.3f}kg", "name": "Carbon"},
        "efficiency_mvis_kwh": {"label": "Efficiency (Mvis/kWh)", "format": "{:.1f}", "name": "Efficiency"},
    }
    
    if metric not in metric_configs:
        raise ValueError(f"Unknown metric: {metric}")
    
    config = metric_configs[metric]
    
    print("="*80)
    print(f"FIGURE: {config['name']} Grouped by {group_by.upper()}")
    print("="*80)
    print(f"Description: Multi-panel bar plot showing {config['label'].lower()} for all configurations,")
    print(f"grouped by {group_by}. Each bar represents a specific configuration.")
    print(f"Interpretation: Consistent y-axis enables direct comparison across {group_by} groups.")
    print(f"Patterns reveal how {config['name'].lower()} varies with workload parameters.")
    print()
    
    assert group_by in {"im_size", "n_times", "n_chans"}
    groups = sorted(df[group_by].unique())
    
    # Set other_param based on group_by
    if group_by == "im_size":
        label_template = lambda row: f"t{int(row.n_times)}×c{int(row.n_chans)}"
    elif group_by == "n_times":
        label_template = lambda row: f"im{row.im_size}\nc{int(row.n_chans)}"
    else:  # n_chans
        label_template = lambda row: f"im{row.im_size}\nt{int(row.n_times)}"

    # Compute global y-axis limits for consistent comparison
    y_min = 0.0
    y_max = df[metric].max() * 1.15  # Add 15% headroom

    # Create multi-row figure
    nrows = len(groups)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, max(4 * nrows, 5)), sharex=False)
    if nrows == 1:
        axes = [axes]

    # Color by image size
    im_sizes = sorted(df["im_size"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(im_sizes)))
    color_map = dict(zip(im_sizes, colors))

    for ax, g in zip(axes, groups):
        subset = df[df[group_by] == g].copy()
        # Sort for consistent display
        if group_by == "im_size":
            subset = subset.sort_values(["n_times", "n_chans"]).reset_index(drop=True)
        else:
            subset = subset.sort_values(["im_size", "n_times", "n_chans"]).reset_index(drop=True)

        x = np.arange(len(subset))
        labels = [label_template(r) for _, r in subset.iterrows()]
        bar_colors = [color_map[r.im_size] for _, r in subset.iterrows()]

        # Convert energy to kJ if plotting energy
        values = subset[metric]
        if metric == "energy_kwh":
            values = values * 3.6e3
        ax.bar(x, values, color=bar_colors, width=0.8, alpha=0.8, edgecolor="black", linewidth=0.7)

        # Add values on top of bars
        for xi, val in zip(x, values):
            ax.text(
                xi,
                val * 1.02,
                config["format"].format(val),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                rotation=0,
            )

        ax.set_title(f"Grouped by {group_by} = {int(g)}", fontsize=12, fontweight="bold")
        ax.set_ylabel(config["label"], fontsize=12, fontweight="bold")
        # Compute y_max in kJ when plotting energy
        effective_y_max = y_max
        if metric == "energy_kwh":
            effective_y_max = (df[metric].max() * 3.6e3) * 1.15
        ax.set_ylim(y_min, effective_y_max)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, ha="right", fontsize=8)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Create legend for image size colors (outside loop, visible for all)
    color_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=f"im={im}",
                  markerfacecolor=color_map[im], markeredgecolor="black",
                  markeredgewidth=0.5, markersize=10)
        for im in im_sizes
    ]
    fig.legend(handles=color_handles, title="Image Size", loc="upper center", 
               bbox_to_anchor=(0.5, 1.02), ncol=len(im_sizes), fontsize=11, title_fontsize=11, frameon=True)

    plt.tight_layout()
    metric_name = metric.replace("_", "-")
    outfile = results_dir / f"{metric_name}_grouped_by_{group_by}.png"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved {config['name'].lower()} plot (grouped by {group_by}) to {outfile}\n")


def plot_carbon_cost(df: pd.DataFrame, results_dir: Path) -> None:
    print("="*80)
    print("FIGURE: Carbon vs Cost Trade-off")
    print("="*80)
    print("Description: Scatter plot of total carbon (g CO2) vs total cost ($) per run.")
    print("Marker size represents workload (Mvis); color represents image size.")
    print("Black line highlights Pareto-optimal configurations (minimize both).")
    print("Interpretation: Small workloads cluster at low cost/carbon. Larger workloads")
    print("spread across the cost-carbon plane. Pareto points show best trade-offs.")
    print()
    
    fig, ax = plt.subplots(figsize=(7, 7))

    im_sizes = sorted(df["im_size"].unique())
    colors = plt.cm.plasma(np.linspace(0, 1, len(im_sizes)))
    color_map = dict(zip(im_sizes, colors))

    sizes = np.clip(df["mvis"], 30, 180)
    sc = ax.scatter(
        df["total_cost_$"],
        df["total_carbon_kg"] * 1000.0,  # grams for readability
        c=[color_map[v] for v in df["im_size"]],
        s=sizes,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.7,
    )

    # Pareto for carbon vs cost: minimize both dimensions
    ordered = df.sort_values(["total_cost_$", "total_carbon_kg"])
    best_carbon = np.inf
    front_indices: List[int] = []
    for idx, row in ordered.iterrows():
        if row["total_carbon_kg"] < best_carbon:
            front_indices.append(idx)
            best_carbon = row["total_carbon_kg"]
    carbon_cost_front = df.loc[front_indices].sort_values("total_cost_$")
    ax.plot(
        carbon_cost_front["total_cost_$"],
        carbon_cost_front["total_carbon_kg"] * 1000.0,
        color="black",
        linewidth=2.5,
        label="Pareto front",
    )

    ax.set_xlabel("Total Cost per Run ($)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Total Carbon per Run (g CO2)", fontsize=14, fontweight="bold")
    ax.set_title("Carbon vs Cost Trade-off", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)

    color_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"im={im}",
            markerfacecolor=color_map[im],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=9,
        )
        for im in im_sizes
    ]

    size_examples = [1, 100, 1000]
    size_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"{val} Mvis",
            markerfacecolor="gray",
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=np.clip(val, 30, 180) ** 0.5,
        )
        for val in size_examples
    ]

    legend1 = ax.legend(handles=color_handles, title="Image Size", loc="upper left", fontsize=10, title_fontsize=11)
    ax.add_artist(legend1)
    legend2 = ax.legend(
        handles=[
            plt.Line2D([0], [0], color="black", linewidth=2.5, label="Pareto front"),
        ],
        loc="lower right",
        title="Front",
        fontsize=10,
        title_fontsize=11,
    )
    ax.add_artist(legend2)
    ax.legend(handles=size_handles, title="Workload (Mvis)", loc="upper right", fontsize=10, title_fontsize=11)

    outfile = results_dir / "carbon_cost_scatter.png"
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"✓ Saved carbon vs cost scatter to {outfile}\n")


def plot_efficiency_throughput_dual_axis(df: pd.DataFrame, results_dir: Path) -> None:
    """Dual-axis plot: left y-axis shows Efficiency (Mvis/kWh) as bars,
    right y-axis shows Throughput (Mvis/h) as a line. Generates three figures:
    - For the largest image size, grouped by timesteps (n_times)
    - For the largest image size, grouped by channels (n_chans)
    - For all image sizes, grouped by image size (im_size)
    """
    print("="*80)
    print("FIGURE: Efficiency (Mvis/kWh) vs Throughput (Mvis/h) — Dual Axis")
    print("="*80)
    print("Description: Bars show efficiency (Mvis/kWh) on the left axis;")
    print("a line shows throughput (Mvis/h) on the right axis. Grouped views")
    print("by timesteps, channels, and image size for comparison.")
    print()

    # Prepare data
    largest_im = df["im_size"].max()
    df_largest = df[df["im_size"] == largest_im].copy()
    # Throughput per hour
    df_largest["mvis_per_hour"] = df_largest["throughput_mvis_s"] * 3600.0
    df_all = df.copy()
    df_all["mvis_per_hour"] = df_all["throughput_mvis_s"] * 3600.0

    def _plot_grouped(local_df: pd.DataFrame, group_by: str, title_suffix: str, outfile_name: str) -> None:
        assert group_by in {"im_size", "n_times", "n_chans"}
        groups = sorted(local_df[group_by].unique())

        # Labels
        if group_by == "im_size":
            label_template = lambda row: f"t{int(row.n_times)}×c{int(row.n_chans)}"
        elif group_by == "n_times":
            label_template = lambda row: f"c{int(row.n_chans)}"
        else:  # n_chans
            label_template = lambda row: f"t{int(row.n_times)}"

        # Color by image size
        im_sizes = sorted(local_df["im_size"].unique())
        colors = plt.cm.viridis(np.linspace(0, 1, len(im_sizes)))
        color_map = dict(zip(im_sizes, colors))

        # Figure sized to content
        nrows = len(groups)
        fig, axes = plt.subplots(nrows, 1, figsize=(12, max(3.5 * nrows, 4)), sharex=False)
        if nrows == 1:
            axes = [axes]

        # Global axis ranges
        left_min = 0.0
        left_max = local_df["efficiency_mvis_kwh"].max() * 1.20
        right_min = 0.0
        right_max = local_df["mvis_per_hour"].max() * 1.20

        for ax, g in zip(axes, groups):
            subset = local_df[local_df[group_by] == g].copy()
            if group_by == "im_size":
                subset = subset.sort_values(["n_times", "n_chans"])  # stable ordering
            else:
                subset = subset.sort_values(["im_size", "n_times", "n_chans"])  # stable

            x = np.arange(len(subset))
            labels = [label_template(r) for _, r in subset.iterrows()]
            bar_colors = [color_map[r.im_size] for _, r in subset.iterrows()]

            # Left axis: efficiency as bars
            ax.bar(x, subset["efficiency_mvis_kwh"], color=bar_colors, width=0.8, alpha=0.85,
                   edgecolor="black", linewidth=0.7)
            ax.set_ylabel("Efficiency (Mvis/kWh)", fontsize=12, fontweight="bold")
            ax.set_ylim(left_min, left_max)
            ax.set_xticks(x)
            # Only show labels on last row
            if ax is axes[-1]:
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            else:
                ax.set_xticklabels([])
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            # Right axis: throughput per hour as line
            ax_r = ax.twinx()
            ax_r.plot(x, subset["mvis_per_hour"], color="darkred", marker="o",
                      linewidth=2, markersize=5, alpha=0.7)
            ax_r.set_ylabel("Throughput (Mvis/h)", fontsize=12, fontweight="bold", color="darkred")
            ax_r.set_ylim(right_min, right_max)
            ax_r.tick_params(axis="y", labelsize=11, labelcolor="darkred")

            ax.set_title(f"Grouped by {group_by} = {int(g)}{title_suffix}", fontsize=12, fontweight="bold")

        # Legend for image sizes
        color_handles = [
            plt.Line2D([0], [0], marker="s", color="w", label=f"im={im}",
                        markerfacecolor=color_map[im], markeredgecolor="black",
                        markeredgewidth=0.5, markersize=10)
            for im in im_sizes
        ]
        plt.tight_layout()
        fig.legend(handles=color_handles, title="Image Size", loc="upper center",
                   bbox_to_anchor=(0.5, 1.02), ncol=len(im_sizes), fontsize=11, title_fontsize=11, frameon=True)
        outfile = results_dir / outfile_name
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        print(f"✓ Saved dual-axis efficiency/throughput plot to {outfile}")
        plt.close(fig)

    # Largest image size grouped by timesteps
    _plot_grouped(
        df_largest,
        group_by="n_times",
        title_suffix=f" (Largest im_size={largest_im})",
        outfile_name=f"efficiency_throughput_dual_largest_grouped_by_n_times.png",
    )

    # Largest image size grouped by channels
    _plot_grouped(
        df_largest,
        group_by="n_chans",
        title_suffix=f" (Largest im_size={largest_im})",
        outfile_name=f"efficiency_throughput_dual_largest_grouped_by_n_chans.png",
    )

    # All image sizes grouped by image size
    _plot_grouped(
        df_all,
        group_by="im_size",
        title_suffix="",
        outfile_name=f"efficiency_throughput_dual_grouped_by_im_size.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready benchmark figures")
    parser.add_argument("-l", "--lifetime", type=int, default=5, help="Lifetime in years (default: 5)")
    parser.add_argument("--location", type=str, default="WA", help="Location ID (default: WA)")
    args = parser.parse_args()

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    df = load_data(args.lifetime, args.location)
    print("\n" + "="*80)
    print(f"AstroCAMP Benchmark Visualization Suite - Paper-Ready Figures")
    print("="*80)
    print(f"Configuration: Location={args.location}, Lifetime={args.lifetime} years")
    print(f"Data: {len(df)} configurations across 4 image sizes × 4 timesteps × 4 channels")
    print("="*80)
    print()

    plot_pareto(df, results_dir)
    plot_iso_heatmaps(df, results_dir)
    plot_energy_breakdown(df, results_dir)
    plot_energy_breakdown_grouped(df, results_dir, group_by="im_size")
    plot_latency_throughput(df, results_dir)
    
    # Grouped bar plots for all key metrics - group by image size
    plot_metric_grouped(df, results_dir, "time_s", group_by="im_size")
    plot_metric_grouped(df, results_dir, "throughput_mvis_s", group_by="im_size")
    plot_metric_grouped(df, results_dir, "energy_kwh", group_by="im_size")
    plot_metric_grouped(df, results_dir, "total_cost_$", group_by="im_size")
    plot_metric_grouped(df, results_dir, "total_carbon_kg", group_by="im_size")
    plot_metric_grouped(df, results_dir, "efficiency_mvis_kwh", group_by="im_size")
    
    plot_carbon_cost(df, results_dir)
    # Dual-axis efficiency vs throughput plots
    plot_efficiency_throughput_dual_axis(df, results_dir)
    
    print("="*80)
    print("✓ All figures generated and saved to:", results_dir)
    print("="*80)
    print()


if __name__ == "__main__":
    main()
