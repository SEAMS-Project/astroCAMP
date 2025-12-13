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

# Optional hover tooltips (avoids clutter vs static annotations)
try:
    import mplcursors
except ImportError:  # pragma: no cover - optional dependency
    mplcursors = None

# Create results directory if it doesn't exist
results_dir = Path('results')
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

# Parameters (matching cea.py)
Lifetime = args.lifetime * 365 * 24  # Lifetime in hours
idle_cpu_watt = 277.75 / 4  # Idle CPU power consumption in watts
idle_gpu_watt = 65.44       # Idle GPU power consumption in watts
location_ids = ['WA']

# Read benchmarks
benchmarks_df = pd.read_csv("benchmarks.csv",
                 header=None,
                 names=["im_size", "n_times", "n_chans", "wall_time", "wall_time_sec", "n_rows", "n_vis",
                        "n_idg",
                        "idg_h_sec", "idg_h_watt", "idg_h_jou",
                        "idg_d_sec", "idg_d_watt", "idg_d_jou",
                        "idg_grid_mvs",
                        "cpu_j",
                        "gpu0_j", "gpu1_j", "gpu2_j", "gpu3_j",
                        "tot_gpu_j", "tot_sys_j", "tot_pdu_j"])

benchmarks_df['machine'] = 'R675 V3 + 4xH100 96GB'
benchmarks_df['benchmark'] = benchmarks_df['im_size'].astype(str) + '_' + benchmarks_df['n_times'].astype(str) + '_' + benchmarks_df['n_chans'].astype(str)
benchmarks_df['time'] = benchmarks_df['wall_time_sec']
benchmarks_df['mvis'] = benchmarks_df['n_vis'] / 1e6

# Read machine and location data
machines_df = pd.read_csv('machines.csv').set_index('machine')
locations_df = pd.read_csv('locations.csv').set_index('id').reset_index()
locations_df = locations_df[locations_df['id'].isin(location_ids)]

# Calculate metrics for each benchmark
results = []
for _, benchmark in benchmarks_df.iterrows():
    benchmark_name = benchmark['benchmark']
    machine_name = benchmark['machine']
    time = benchmark['time'] / 3600  # from seconds to hours
    energy_static = (idle_cpu_watt + idle_gpu_watt) * time / 1000  # Static energy in kWh
    energy_dynamic = (benchmark['gpu0_j'] + benchmark['cpu_j']) / 3.6e6  # Dynamic energy in kWh
    energy = energy_dynamic + energy_static  # Total energy in kWh
    
    machine_cost = machines_df.loc[machine_name, 'cost']  # in $
    machine_embodied = machines_df.loc[machine_name, 'embodied']  # in kg CO2
    
    for _, location in locations_df.iterrows():
        location_id = location['id']
        location_name = location['location']
        ci = location['ci']  # Carbon intensity in kg CO2/kWh
        ep = location['ep']  # Electricity price in $/kWh
        
        operational_energy_cost = energy * ep  # in $
        operational_carbon = energy * ci  # in kg CO2
        capital_cost = machine_cost * (time / Lifetime)
        capital_carbon = machine_embodied * (time / Lifetime)
        mvis = benchmark['mvis']
        
        results.append({
            'Image Size': benchmark['im_size'],
            'Timesteps': benchmark['n_times'],
            'Channels': benchmark['n_chans'],
            'Mvis': mvis,
            'Time (s)': time * 3600,
            'Energy (kWh)': energy,
            'Mvis/s': mvis / (time * 3600),
            'Mvis/kWh': mvis / energy,
        })

results_df = pd.DataFrame(results)

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
        print(f"  Image: {row['Image Size']:5d}, Times: {row['Timesteps']:3d}, "
              f"Chans: {row['Channels']:3d}, Mvis: {row['Mvis']:8.2f}, "
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
    print(f"  Image Size {im_size:5d}: Avg Throughput = {avg_throughput:.4f} Mvis/s, "
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


# Example output using 5 years lifetime:
# ================================================================================
# KEY INSIGHTS
# ================================================================================

# Sweet spots (above median throughput 0.3969 Mvis/s
#              AND above median efficiency 7829.74 Mvis/kWh):
# --------------------------------------------------------------------------------
#   Image: 16384, Times: 128, Chans: 128, Mvis:  2143.29, Throughput: 1.9543 Mvis/s, Efficiency: 40542.43 Mvis/kWh
#   Image: 32768, Times: 128, Chans: 128, Mvis:  2143.29, Throughput: 1.9016 Mvis/s, Efficiency: 39165.28 Mvis/kWh
#   Image: 16384, Times:  64, Chans: 128, Mvis:  1071.64, Throughput: 1.8965 Mvis/s, Efficiency: 39014.99 Mvis/kWh
#   Image:  8192, Times: 128, Chans: 128, Mvis:  2143.29, Throughput: 1.8782 Mvis/s, Efficiency: 38660.05 Mvis/kWh
#   Image:  4096, Times: 128, Chans: 128, Mvis:  2143.29, Throughput: 1.8253 Mvis/s, Efficiency: 37622.80 Mvis/kWh
#   Image:  8192, Times:  64, Chans: 128, Mvis:  1071.64, Throughput: 1.8217 Mvis/s, Efficiency: 37300.97 Mvis/kWh
#   Image:  4096, Times:  64, Chans: 128, Mvis:  1071.64, Throughput: 1.8079 Mvis/s, Efficiency: 37241.20 Mvis/kWh
#   Image: 32768, Times:  64, Chans: 128, Mvis:  1071.64, Throughput: 1.7523 Mvis/s, Efficiency: 36030.97 Mvis/kWh
#   Image: 16384, Times: 128, Chans:  64, Mvis:  1071.64, Throughput: 1.6994 Mvis/s, Efficiency: 35060.25 Mvis/kWh
#   Image:  8192, Times: 128, Chans:  64, Mvis:  1071.64, Throughput: 1.6749 Mvis/s, Efficiency: 34380.95 Mvis/kWh

# ================================================================================
# EFFICIENCY TRENDS BY IMAGE SIZE
# ================================================================================
#   Image Size  4096: Avg Throughput = 0.7030 Mvis/s, Avg Efficiency = 14369.33 Mvis/kWh
#   Image Size  8192: Avg Throughput = 0.7132 Mvis/s, Avg Efficiency = 14501.42 Mvis/kWh
#   Image Size 16384: Avg Throughput = 0.7059 Mvis/s, Avg Efficiency = 14372.99 Mvis/kWh
#   Image Size 32768: Avg Throughput = 0.6037 Mvis/s, Avg Efficiency = 12250.76 Mvis/kWh

# ================================================================================
# SCALING BEHAVIOR
# ================================================================================
# Small workloads (Mvis < 8.37):
#   Avg Throughput: 0.0159 Mvis/s
#   Avg Efficiency: 308.35 Mvis/kWh

# Large workloads (Mvis > 133.96):
#   Avg Throughput: 1.7263 Mvis/s
#   Avg Efficiency: 35482.42 Mvis/kWh

# Scaling from small to large workloads:
#   Throughput ratio: 108.85x
#   Efficiency ratio: 115.07x