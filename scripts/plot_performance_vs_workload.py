#!/usr/bin/env python3
"""
Plot Performance vs Workload Size
Generates Figure A: Throughput and energy efficiency vs work
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import argparse
from pathlib import Path
from results_dataframe import generate_results_dataframe

# Optional hover tooltips (avoids clutter vs static annotations)
try:
    import mplcursors
except ImportError:  # pragma: no cover - optional dependency
    mplcursors = None

# Create results directory if it doesn't exist
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = BASE_DIR.parent / "results"
results_dir = RESULTS_DIR
results_dir.mkdir(exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot performance vs workload size.')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years (default: 5)')
parser.add_argument('-o', '--output', type=str, default='performance_vs_workload.png', 
                    help='Output filename (default: performance_vs_workload.png)')
parser.add_argument('--dpi', type=int, default=300, help='DPI for output image (default: 300)')
parser.add_argument('--annotate-configs', action='store_true',
                    help='Annotate points with n_times and n_chans labels to make configurations explicit')
parser.add_argument('--hover-labels', action='store_true',
                    help='Show hover tooltips (n_times, n_chans, throughput, efficiency) instead of static labels; uses mplcursors')
args = parser.parse_args()

# Ensure output path is in results directory
output_path = results_dir / args.output

print(f"Using lifetime of {args.lifetime} years for all machines.")

# Generate results DataFrame using helper function
results_df = generate_results_dataframe(
    benchmarks_csv_path=DATA_DIR / 'benchmarks.csv',
    machines_csv_path=DATA_DIR / 'machines.csv',
    locations_csv_path=DATA_DIR / 'locations.csv',
    lifetime_years=args.lifetime,
    location_ids=['WA']
)

# Filter to just WA location for consistency
results_df = results_df[results_df['Location'] == 'WA'].copy()

# Calculate Mvis/s (throughput) from Mvis and Time (s)
results_df['Mvis/s'] = results_df['Mvis'] / (results_df['Time (s)'] / 3600)

# Add regime classification (time-heavy vs channel-heavy)
results_df['regime'] = results_df.apply(
    lambda row: 'Time-heavy' if row['Timesteps'] > row['Channels'] 
    else 'Channel-heavy' if row['Channels'] > row['Timesteps']
    else 'Balanced', axis=1
)

print(f"\nGenerated {len(results_df)} data points")
print(f"Image sizes: {sorted(results_df['Image Size'].unique())}")
print(f"Mvis range: {results_df['Mvis'].min():.2f} - {results_df['Mvis'].max():.2f}")
print(f"Throughput range: {results_df['Mvis/s'].min():.4f} - {results_df['Mvis/s'].max():.4f} Mvis/s")
print(f"Energy efficiency range: {results_df['Mvis/kWh'].min():.2f} - {results_df['Mvis/kWh'].max():.2f} Mvis/kWh")

# Create the figure with three subplots stacked vertically
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=False)
fig.suptitle('Performance vs Workload Size', fontsize=16, fontweight='bold')

# Collect scatter artists for optional hover tooltips
scatter_payload = []

# Color mapping for image sizes
im_sizes = sorted(results_df['Image Size'].unique())
colors = plt.cm.viridis(np.linspace(0, 1, len(im_sizes)))
color_map = dict(zip(im_sizes, colors))

# Marker mapping for regimes
regime_markers = {
    'Time-heavy': '^',      # triangle up
    'Channel-heavy': 'v',   # triangle down
    'Balanced': 'o'         # circle
}

# Plot 1: Throughput (Mvis/s) vs Mvis
for im_size in im_sizes:
    for regime in results_df['regime'].unique():
        mask = (results_df['Image Size'] == im_size) & (results_df['regime'] == regime)
        subset = results_df[mask]
        if not subset.empty:
            sc1 = ax1.scatter(subset['Mvis'], subset['Mvis/s'], 
                              c=[color_map[im_size]], 
                              marker=regime_markers[regime],
                              s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                              label=f'{im_size} - {regime}' if regime == 'Time-heavy' else '')

            scatter_payload.append((sc1, subset[['Timesteps','Channels','Mvis','Mvis/s','Mvis/kWh']].reset_index(drop=True)))

            # Optional point annotation for n_times/n_chans
            if args.annotate_configs:
                for _, row in subset.iterrows():
                    ax1.text(row['Mvis'], row['Mvis/s'], 
                             f"t{int(row['Timesteps'])},c{int(row['Channels'])}",
                             fontsize=7, ha='center', va='center', color='black',
                             alpha=0.8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5, linewidth=0))

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylabel('Throughput (Mvis/s)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, which='both', linestyle='--')
ax1.set_title('(a) Throughput vs Workload', fontsize=13)

# Plot 2: Energy Efficiency (Mvis/kWh) vs Mvis
for im_size in im_sizes:
    for regime in results_df['regime'].unique():
        mask = (results_df['Image Size'] == im_size) & (results_df['regime'] == regime)
        subset = results_df[mask]
        if not subset.empty:
            sc2 = ax2.scatter(subset['Mvis'], subset['Mvis/kWh'], 
                              c=[color_map[im_size]], 
                              marker=regime_markers[regime],
                              s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

            scatter_payload.append((sc2, subset[['Timesteps','Channels','Mvis','Mvis/s','Mvis/kWh']].reset_index(drop=True)))

            # Optional point annotation for n_times/n_chans
            if args.annotate_configs:
                for _, row in subset.iterrows():
                    ax2.text(row['Mvis'], row['Mvis/kWh'], 
                             f"t{int(row['Timesteps'])},c{int(row['Channels'])}",
                             fontsize=7, ha='center', va='center', color='black',
                             alpha=0.8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5, linewidth=0))

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Workload (Mvis)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Energy Efficiency (Mvis/kWh)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both', linestyle='--')
ax2.set_title('(b) Energy Efficiency vs Workload', fontsize=13)

# Plot 3: Coverage of (n_times, n_chans) combinations
for im_size in im_sizes:
    for regime in results_df['regime'].unique():
        mask = (results_df['Image Size'] == im_size) & (results_df['regime'] == regime)
        subset = results_df[mask]
        if not subset.empty:
            sc3 = ax3.scatter(subset['Timesteps'], subset['Channels'],
                              c=[color_map[im_size]],
                              marker=regime_markers[regime],
                              s=80, alpha=0.8, edgecolors='black', linewidth=0.5,
                              label=f'{im_size} - {regime}' if regime == 'Time-heavy' else '')

            scatter_payload.append((sc3, subset[['Timesteps','Channels','Mvis','Mvis/s','Mvis/kWh']].reset_index(drop=True)))

            # Optional point annotation for n_times/n_chans (redundant here but explicit)
            if args.annotate_configs:
                for _, row in subset.iterrows():
                    ax3.text(row['Timesteps'], row['Channels'], 
                             f"t{int(row['Timesteps'])},c{int(row['Channels'])}",
                             fontsize=7, ha='center', va='center', color='black',
                             alpha=0.8, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5, linewidth=0))

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlabel('Timesteps (n_times)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Channels (n_chans)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both', linestyle='--')
ax3.set_title('(c) Coverage of n_times vs n_chans', fontsize=13)

# Build a legend for the coverage plot (reuse handles)
color_patches_cov = [mpatches.Patch(color=color_map[im_size], label=f'{im_size}') 
                     for im_size in im_sizes]
regime_lines_cov = [Line2D([0], [0], marker=regime_markers[regime], color='gray',
                          linestyle='None', markersize=8, label=regime,
                          markeredgecolor='black', markeredgewidth=0.5)
                   for regime in sorted(regime_markers.keys())]

legend_cov1 = ax3.legend(handles=color_patches_cov, title='Image Size',
                         loc='upper left', fontsize=9, title_fontsize=10,
                         framealpha=0.9)
ax3.add_artist(legend_cov1)
legend_cov2 = ax3.legend(handles=regime_lines_cov, title='Regime (n_times vs n_chans)',
                         loc='lower right', fontsize=9, title_fontsize=10,
                         framealpha=0.9)

# Create custom legend
# Legend for image sizes (colors)
color_patches = [mpatches.Patch(color=color_map[im_size], label=f'{im_size}') 
                for im_size in im_sizes]

# Legend for regimes (markers)
regime_lines = [Line2D([0], [0], marker=regime_markers[regime], color='gray', 
                      linestyle='None', markersize=8, label=regime,
                      markeredgecolor='black', markeredgewidth=0.5)
               for regime in sorted(regime_markers.keys())]

# Add legends
legend1 = ax1.legend(handles=color_patches, title='Image Size', 
                     loc='upper left', fontsize=9, title_fontsize=10, 
                     framealpha=0.9)
ax1.add_artist(legend1)
legend2 = ax1.legend(handles=regime_lines, title='Regime (n_times vs n_chans)', 
                     loc='lower right', fontsize=9, title_fontsize=10,
                     framealpha=0.9)

plt.tight_layout()

# Optional hover tooltips to avoid clutter from static annotations
if args.hover_labels:
    if mplcursors is None:
        print("mplcursors not installed; install it (pip install mplcursors) to enable hover labels.")
    elif scatter_payload:
        artist_to_df = {artist: df for artist, df in scatter_payload}
        cursor = mplcursors.cursor([artist for artist, _ in scatter_payload], hover=True)

        @cursor.connect("add")
        def _(sel):
            df = artist_to_df.get(sel.artist)
            if df is not None and sel.index < len(df):
                row = df.iloc[sel.index]
                sel.annotation.set_text(
                    f"n_times={int(row['Timesteps'])}, n_chans={int(row['Channels'])}\n"
                    f"Mvis={row['Mvis']:.2f}\n"
                    f"Throughput={row['Mvis/s']:.4f} Mvis/s\n"
                    f"Efficiency={row['Mvis/kWh']:.1f} Mvis/kWh"
                )
                sel.annotation.get_bbox_patch().set(alpha=0.85)

# Save the figure
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Print some insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Find sweet spots (high throughput AND high efficiency)
median_throughput = results_df['Mvis/s'].median()
median_efficiency = results_df['Mvis/kWh'].median()

sweet_spots = results_df[
    (results_df['Mvis/s'] > median_throughput) & 
    (results_df['Mvis/kWh'] > median_efficiency)
].sort_values('Mvis/s', ascending=False)

print(f"\nSweet spots (above median throughput {median_throughput:.4f} Mvis/s")
print(f"             AND above median efficiency {median_efficiency:.2f} Mvis/kWh):")
print("-"*80)
if not sweet_spots.empty:
    for idx, row in sweet_spots.head(10).iterrows():
        print(f"  Image: {row['Image Size']:5.0f}, Times: {row['Timesteps']:3.0f}, "
              f"Chans: {row['Channels']:3.0f}, Mvis: {row['Mvis']:8.2f}, "
              f"Throughput: {row['Mvis/s']:.4f} Mvis/s, "
              f"Efficiency: {row['Mvis/kWh']:.2f} Mvis/kWh")
else:
    print("  No clear sweet spots found with current thresholds")

# Efficiency trends
print("\n" + "="*80)
print("EFFICIENCY TRENDS BY IMAGE SIZE")
print("="*80)
for im_size in im_sizes:
    subset = results_df[results_df['Image Size'] == im_size]
    avg_throughput = subset['Mvis/s'].mean()
    avg_efficiency = subset['Mvis/kWh'].mean()
    print(f"  Image Size {im_size:5.0f}: Avg Throughput = {avg_throughput:.4f} Mvis/s, "
          f"Avg Efficiency = {avg_efficiency:.2f} Mvis/kWh")

# Scaling behavior
print("\n" + "="*80)
print("SCALING BEHAVIOR")
print("="*80)
small_workloads = results_df[results_df['Mvis'] < results_df['Mvis'].quantile(0.33)]
large_workloads = results_df[results_df['Mvis'] > results_df['Mvis'].quantile(0.67)]

print(f"Small workloads (Mvis < {results_df['Mvis'].quantile(0.33):.2f}):")
print(f"  Avg Throughput: {small_workloads['Mvis/s'].mean():.4f} Mvis/s")
print(f"  Avg Efficiency: {small_workloads['Mvis/kWh'].mean():.2f} Mvis/kWh")

print(f"\nLarge workloads (Mvis > {results_df['Mvis'].quantile(0.67):.2f}):")
print(f"  Avg Throughput: {large_workloads['Mvis/s'].mean():.4f} Mvis/s")
print(f"  Avg Efficiency: {large_workloads['Mvis/kWh'].mean():.2f} Mvis/kWh")

throughput_ratio = large_workloads['Mvis/s'].mean() / small_workloads['Mvis/s'].mean()
efficiency_ratio = large_workloads['Mvis/kWh'].mean() / small_workloads['Mvis/kWh'].mean()

print(f"\nScaling from small to large workloads:")
print(f"  Throughput ratio: {throughput_ratio:.2f}x")
print(f"  Efficiency ratio: {efficiency_ratio:.2f}x")

plt.show()
