#!/usr/bin/env python3
"""
Bar plots for largest image size across all (n_times, n_chans) combinations,
plus line plots for specified metric pairs:
- wall_time_sec vs idg_h_sec
- tot_sys_j vs idg_h_jou
- (n_vis / wall_time_sec) vs idg_grid_mvs
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="Bar + line plots grouped by image size")
parser.add_argument("--location", type=str, default="WA")
parser.add_argument("--dpi", type=int, default=300)
args = parser.parse_args()

results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

# Load benchmarks
benchmarks = pd.read_csv(
    BASE_DIR / "benchmarks.csv",
    header=None,
    names=[
        "im_size","n_times","n_chans","wall_time","wall_time_sec","n_rows","n_vis","n_idg",
        "idg_h_sec","idg_h_watt","idg_h_jou","idg_d_sec","idg_d_watt","idg_d_jou","idg_grid_mvs",
        "cpu_j","gpu0_j","gpu1_j","gpu2_j","gpu3_j","tot_gpu_j","tot_sys_j","tot_pdu_j",
    ],
)

# Derive helper columns
benchmarks["throughput_mvis_s"] = benchmarks["n_vis"] / benchmarks["wall_time_sec"]
benchmarks["combo_label"] = (
    "t" + benchmarks["n_times"].astype(int).astype(str) + "-c" + benchmarks["n_chans"].astype(int).astype(str)
)

# Prepare grouped view by image size for bar/line plots
grouped_df = benchmarks.copy().sort_values(["im_size","n_times","n_chans"]).reset_index(drop=True)
im_sizes = grouped_df["im_size"].unique().tolist()

# Keep largest-only dataframe for components plot later
largest_im = int(sorted(benchmarks["im_size"].unique())[-1])
largest_df = benchmarks[benchmarks["im_size"] == largest_im].copy().sort_values(["n_times","n_chans"]).reset_index(drop=True)

# 1) Bar plots for largest image size for all (n_times, n_chans)
# We'll produce a compact figure with 3 bars: energy_kJ, throughput, wall_time_min
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
labels = ("img" + grouped_df["im_size"].astype(int).astype(str) + "|" + grouped_df["combo_label"]).tolist()
x = np.arange(len(grouped_df))
bar_width = 0.8

# Energy (kJ) = static + dynamic
idle_cpu_watt = 277.75/4
idle_gpu_watt = 65.44
grouped_df["time_h"] = grouped_df["wall_time_sec"] / 3600.0
grouped_df["energy_static_kwh"] = (idle_cpu_watt + idle_gpu_watt) * grouped_df["time_h"] / 1000.0
grouped_df["energy_dynamic_kwh"] = (grouped_df["gpu0_j"] + grouped_df["cpu_j"]) / 3.6e6
grouped_df["energy_kj"] = (grouped_df["energy_static_kwh"] + grouped_df["energy_dynamic_kwh"]) * 3.6e3

axes[0].bar(x, grouped_df["energy_kj"], width=bar_width, color="#1f77b4", edgecolor="black", linewidth=0.6)
axes[0].set_ylabel("Energy (kJ)", fontsize=12, fontweight="bold")
axes[0].set_title("Energy per (image|n_times, n_chans)", fontsize=13, fontweight="bold")
axes[0].grid(axis="y", alpha=0.3, linestyle="--")

axes[1].bar(x, grouped_df["throughput_mvis_s"], width=bar_width, color="#2ca02c", edgecolor="black", linewidth=0.6)
axes[1].set_ylabel("Throughput (Mvis/s)", fontsize=12, fontweight="bold")
axes[1].set_title("Throughput per (image|n_times, n_chans)", fontsize=13, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3, linestyle="--")

axes[2].bar(x, grouped_df["wall_time_sec"] / 60.0, width=bar_width, color="#d62728", edgecolor="black", linewidth=0.6)
axes[2].set_ylabel("Wall Time (min)", fontsize=12, fontweight="bold")
axes[2].set_title("Wall Time per (image|n_times, n_chans)", fontsize=13, fontweight="bold")
axes[2].grid(axis="y", alpha=0.3, linestyle="--")

def add_group_headers(ax):
    # Draw separators and group headers for each image size
    idx = 0
    for im in im_sizes:
        count = int((grouped_df["im_size"] == im).sum())
        if count == 0:
            continue
        start = idx
        end = idx + count - 1
        # Add a light span behind each group
        ax.axvspan(start - 0.5, end + 0.5, color="#f5f5f5", alpha=0.5, zorder=0)
        # Add a vertical separator after group (except last)
        if im != im_sizes[-1]:
            ax.axvline(end + 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        # Add group label centered near top
        ymax = ax.get_ylim()[1]
        ax.text((start + end) / 2.0, ymax * 0.98, f"Image {int(im)}", ha="center", va="top", fontsize=10, fontweight="bold")
        idx += count

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    add_group_headers(ax)

plt.tight_layout()
outfile_bars = results_dir / f"bars_grouped_by_image.png"
plt.savefig(outfile_bars, dpi=args.dpi, bbox_inches="tight")
print(f"✓ Saved bar plots to {outfile_bars}")

# 2) Line plots for requested pairs on the largest image only
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
# Use same grouped ordering
x2 = np.arange(len(grouped_df))

# Pair 1: wall_time_sec vs idg_h_sec
axes2[0].plot(x2, grouped_df["wall_time_sec"], color="#d62728", marker="o", linewidth=2, label="wall_time_sec")
axes2[0].plot(x2, grouped_df["idg_h_sec"], color="#1f77b4", marker="s", linewidth=2, label="idg_h_sec")
axes2[0].set_ylabel("Seconds", fontsize=12, fontweight="bold")
axes2[0].set_title("wall_time_sec vs idg_h_sec (grouped by image)", fontsize=13, fontweight="bold")
axes2[0].grid(True, alpha=0.3, linestyle="--")
axes2[0].legend(fontsize=10)

# Pair 2: tot_sys_j vs idg_h_jou
axes2[1].plot(x2, grouped_df["tot_sys_j"], color="#9467bd", marker="o", linewidth=2, label="tot_sys_j")
axes2[1].plot(x2, grouped_df["idg_h_jou"], color="#ff7f0e", marker="s", linewidth=2, label="idg_h_jou")
axes2[1].set_ylabel("Joules", fontsize=12, fontweight="bold")
axes2[1].set_title("tot_sys_j vs idg_h_jou (grouped by image)", fontsize=13, fontweight="bold")
axes2[1].grid(True, alpha=0.3, linestyle="--")
axes2[1].legend(fontsize=10)

# Pair 3: (n_vis / wall_time_sec) vs idg_grid_mvs
axes2[2].plot(x2, grouped_df["throughput_mvis_s"], color="#2ca02c", marker="o", linewidth=2, label="n_vis/wall_time_sec (Mvis/s)")
axes2[2].plot(x2, grouped_df["idg_grid_mvs"], color="#8c564b", marker="s", linewidth=2, label="idg_grid_mvs")
axes2[2].set_ylabel("Value", fontsize=12, fontweight="bold")
axes2[2].set_title("throughput vs idg_grid_mvs (grouped by image)", fontsize=13, fontweight="bold")
axes2[2].grid(True, alpha=0.3, linestyle="--")
axes2[2].legend(fontsize=10)

# X labels as combination identifiers
axes2[2].set_xticks(x2)
axes2[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

# Add the same group headers/separators on line plots
for ax in axes2:
    add_group_headers(ax)

plt.tight_layout()
outfile_lines = results_dir / f"lines_grouped_by_image.png"
plt.savefig(outfile_lines, dpi=args.dpi, bbox_inches="tight")
print(f"✓ Saved line plots to {outfile_lines}")

# 3) Stacked bar plot showing energy hierarchy:
# PDU total (outer) > System total > breakdown into GPU + IDG host + IDG device
# Half width for two-column paper format, slightly increased height
fig3, ax3 = plt.subplots(figsize=(7, 4))

x = np.arange(len(largest_df))
labels = largest_df["combo_label"].tolist()
bar_width = 0.7

# Calculate component breakdown
# System total should roughly equal cpu_j + gpu0_j + idg_h_jou + idg_d_jou (with some overhead)
# PDU total should be slightly more than System total (measurement differences)

# Convert to kJ (kilojoules) = J / 1000
cpu_kj = largest_df["cpu_j"].values / 1000
gpu_kj = largest_df["gpu0_j"].values / 1000
idg_h_kj = largest_df["idg_h_jou"].values / 1000
idg_d_kj = largest_df["idg_d_jou"].values / 1000
sys_kj = largest_df["tot_sys_j"].values / 1000
pdu_kj = largest_df["tot_pdu_j"].values / 1000

# Calculate residuals to show hierarchy
# System = CPU + GPU + IDG_host + IDG_device + system_overhead
system_overhead = sys_kj - (cpu_kj + gpu_kj + idg_h_kj + idg_d_kj)
# PDU = System + pdu_overhead
pdu_overhead = pdu_kj - sys_kj

# Stack in order: CPU -> GPU -> IDG device -> IDG host -> system overhead -> PDU overhead
ax3.bar(x, cpu_kj, width=bar_width, label="CPU (cpu_j)", 
        color="#8c564b", edgecolor="black", linewidth=0.6, alpha=0.9)
ax3.bar(x, gpu_kj, bottom=cpu_kj, width=bar_width, label="GPU (gpu0_j)", 
        color="#9467bd", edgecolor="black", linewidth=0.6, alpha=0.9)
ax3.bar(x, idg_d_kj, bottom=cpu_kj + gpu_kj, width=bar_width, label="IDG device (idg_d_jou)", 
        color="#d62728", edgecolor="black", linewidth=0.6, alpha=0.9)
ax3.bar(x, idg_h_kj, bottom=cpu_kj + gpu_kj + idg_d_kj, width=bar_width, label="IDG host (idg_h_jou)", 
        color="#2ca02c", edgecolor="black", linewidth=0.6, alpha=0.9)
ax3.bar(x, system_overhead, bottom=cpu_kj + gpu_kj + idg_d_kj + idg_h_kj, width=bar_width, 
        label="System overhead", color="#ff7f0e", edgecolor="black", linewidth=0.6, alpha=0.75)
ax3.bar(x, pdu_overhead, bottom=sys_kj, width=bar_width, 
        label="PDU overhead (vs System)", color="#1f77b4", edgecolor="black", linewidth=0.6, alpha=0.75)

# Add reference lines for System total and PDU total
for i, (sys_val, pdu_val) in enumerate(zip(sys_kj, pdu_kj)):
    ax3.hlines(sys_val, i - bar_width/2, i + bar_width/2, colors='orange', linewidth=2, 
               linestyles='--', alpha=0.8, zorder=10)
    ax3.hlines(pdu_val, i - bar_width/2, i + bar_width/2, colors='blue', linewidth=2, 
               linestyles='--', alpha=0.8, zorder=10)

# Add custom legend entries for reference lines
from matplotlib.lines import Line2D
legend_elements = ax3.get_legend_handles_labels()
legend_elements[0].extend([
    Line2D([0], [0], color='orange', linewidth=2, linestyle='--', label='System total (tot_sys_j)'),
    Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='PDU total (tot_pdu_j)')
])
legend_elements[1].extend(['System total (tot_sys_j)', 'PDU total (tot_pdu_j)'])

ax3.set_ylabel("Energy (kJ)", fontsize=11, fontweight="bold")
ax3.set_title(f"Energy (bars) & Time (lines) - Image {largest_im}\nComponents → System → PDU", 
              fontsize=10, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=90, ha="center", fontsize=11)
ax3.legend(*legend_elements, ncol=1, fontsize=8, loc='upper left')
ax3.grid(axis="y", alpha=0.3, linestyle="--")
# Use linear scale instead of log
ax3.set_ylim(bottom=0)

# Add right y-axis with corresponding *_sec metrics for comparison
ax3_right = ax3.twinx()
sec_series = [
    ("wall_time_sec", "Wall time (s)", "#7f7f7f", "o", 0.0, "-", 0.3),
    ("idg_h_sec", "IDG host (s)", "#1f77b4", "^", -0.12, "--", 0.85),
    ("idg_d_sec", "IDG device (s)", "#d62728", "s", 0.12, ":", 0.85),
]
for col, label, color, marker, x_offset, linestyle, alpha_val in sec_series:
    if col in largest_df.columns:
        x_shifted = x + x_offset
        ax3_right.plot(
            x_shifted,
            largest_df[col].values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2,
            markersize=6,
            alpha=alpha_val,
            markeredgecolor="black",
            markeredgewidth=0.6,
            label=label,
            zorder=3,
        )
ax3_right.set_ylabel("Time (s)", fontsize=10, fontweight="bold", color="darkred")
ax3_right.tick_params(axis="y", labelsize=8, labelcolor="darkred")
ax3_right.set_yscale("log")

# Combine legends: bars (left) + lines (right)
lines_handles, lines_labels = ax3_right.get_legend_handles_labels()
bars_handles, bars_labels = ax3.get_legend_handles_labels()
ax3.legend(bars_handles + lines_handles, bars_labels + lines_labels, ncol=1, fontsize=7, loc='upper left')

plt.tight_layout()
outfile_components = results_dir / f"largest_{largest_im}_components_joules.png"
plt.savefig(outfile_components, dpi=args.dpi, bbox_inches="tight")
print(f"\n{'='*80}")
print(f"FIGURE: Energy Component Hierarchy (Image size {largest_im})")
print(f"{'='*80}")
print("Description: Stacked bar chart showing energy measurement hierarchy across")
print("all (n_times, n_chans) combinations for the largest image size.")
print("")
print("Stacked components (bottom to top):")
print("  • CPU (brown) - CPU energy consumption (cpu_j)")
print("  • GPU (purple) - GPU0 energy consumption (gpu0_j)")
print("  • IDG device (red) - IDG device computation energy (idg_d_jou)")
print("  • IDG host (green) - IDG host computation energy (idg_h_jou)")
print("  • System overhead (orange) - Additional system-level overhead")
print("  • PDU overhead (blue) - PDU measurement vs system total difference")
print("")
print("Reference lines:")
print("  • Orange dashed line: System total (tot_sys_j) = sum of measured components")
print("  • Blue dashed line: PDU total (tot_pdu_j) = top of stack")
print("")
print("Right y-axis (log scale): Execution times in seconds")
print("  • Wall time (gray solid line, circles) - Total execution time")
print("  • IDG host time (blue dashed, triangles) - Host computation time")
print("  • IDG device time (red dotted, squares) - Device computation time")
print("  Note: Slight x-offsets applied to separate overlapping time series.")
print("")
print("Interpretation: The stacked bars demonstrate energy accounting hierarchy.")
print("PDU measurements (blue top line) exceed system measurements (orange line)")
print("due to power distribution overhead. System total comprises CPU, GPU, and IDG")
print("components plus system-level overhead. IDG host and device times nearly overlap,")
print("indicating balanced host-device workload. Larger n_times and n_chans increase")
print("both energy consumption and execution time proportionally.")
print(f"\n✓ Saved to {outfile_components}\n")
