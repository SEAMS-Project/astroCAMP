#!/usr/bin/env python3
"""
Plot Dual-Axis Heatmaps for Largest Two Image Sizes
Generates heatmaps with two y-axes showing different metrics side-by-side
for latency, speedup, throughput, and energy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
from pathlib import Path

# Create results directory if it doesn't exist
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot dual-axis heatmaps for largest two image sizes.')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years (default: 5)')
parser.add_argument('--dpi', type=int, default=300, help='DPI for output image (default: 300)')
parser.add_argument('--annotate', action='store_true', help='Annotate cells with values')
args = parser.parse_args()

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
benchmarks_df['time'] = benchmarks_df['wall_time_sec']
benchmarks_df['mvis'] = benchmarks_df['n_vis'] / 1e6

# Read machine and location data
machines_df = pd.read_csv('machines.csv').set_index('machine')
locations_df = pd.read_csv('locations.csv').set_index('id').reset_index()
locations_df = locations_df[locations_df['id'].isin(location_ids)]

# Calculate metrics for each benchmark
results = []
for _, benchmark in benchmarks_df.iterrows():
    machine_name = benchmark['machine']
    time = benchmark['time'] / 3600  # from seconds to hours
    energy_static = (idle_cpu_watt + idle_gpu_watt) * time / 1000  # Static energy in kWh
    energy_dynamic = (benchmark['gpu0_j'] + benchmark['cpu_j']) / 3.6e6  # Dynamic energy in kWh
    energy = energy_dynamic + energy_static  # Total energy in kWh
    energy_kj = energy * 3.6e3  # Convert to kJ
    
    machine_cost = machines_df.loc[machine_name, 'cost']  # in $
    machine_embodied = machines_df.loc[machine_name, 'embodied']  # in kg CO2
    
    for _, location in locations_df.iterrows():
        location_id = location['id']
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
            'Latency (s)': time * 3600,
            'Energy (kJ)': energy_kj,
            'Throughput (Mvis/s)': mvis / (time * 3600),
            'Energy Efficiency (Mvis/kJ)': mvis / energy_kj,
        })

results_df = pd.DataFrame(results)

# Get unique values for axes
im_sizes = sorted(results_df['Image Size'].unique())
n_times_vals = sorted(results_df['Timesteps'].unique())
n_chans_vals = sorted(results_df['Channels'].unique())

# Select largest two image sizes
largest_two = im_sizes[-2:]

print(f"\nData summary:")
print(f"  Image sizes: {im_sizes}")
print(f"  Largest two: {largest_two}")
print(f"  Timesteps: {n_times_vals}")
print(f"  Channels: {n_chans_vals}")

# Calculate baseline (n_times=1, n_chans=1) for speedup
baseline_times = {}
for im_size in largest_two:
    baseline = results_df[
        (results_df['Image Size'] == im_size) & 
        (results_df['Timesteps'] == 1) & 
        (results_df['Channels'] == 1)
    ]
    if not baseline.empty:
        baseline_times[im_size] = baseline.iloc[0]['Latency (s)']
    else:
        baseline_times[im_size] = None

# Add speedup metric
def calculate_speedup(row):
    if baseline_times.get(row['Image Size']) is not None:
        return baseline_times[row['Image Size']] / row['Latency (s)']
    return 1.0

results_df['Speedup'] = results_df.apply(calculate_speedup, axis=1)

# Define metrics to plot
metrics = [
    {'name': 'Latency (s)', 'title': 'Latency (seconds)', 'cmap': 'RdYlGn_r', 'log': True},
    {'name': 'Speedup', 'title': 'Speedup (× vs baseline)', 'cmap': 'RdYlGn', 'log': False},
    {'name': 'Throughput (Mvis/s)', 'title': 'Throughput (Mvis/s)', 'cmap': 'RdYlGn', 'log': True},
    {'name': 'Energy (kJ)', 'title': 'Energy (kJ)', 'cmap': 'RdYlGn_r', 'log': True},
]

# Create figure with 4 rows (one per metric) and 2 columns (one per image size)
fig, axes = plt.subplots(4, 2, figsize=(10, 16))
fig.suptitle(f'Performance Metrics: Largest Two Image Sizes\n(n_times × n_chans)', 
             fontsize=14, fontweight='bold', y=0.995)

for row_idx, metric in enumerate(metrics):
    metric_name = metric['name']
    metric_title = metric['title']
    cmap = metric['cmap']
    use_log = metric['log']
    
    print(f"\n{'='*80}")
    print(f"GENERATING HEATMAPS FOR: {metric_name}")
    print(f"{'='*80}")
    
    # Find global min/max for consistent color scale across both image sizes
    subset_all = results_df[results_df['Image Size'].isin(largest_two)]
    vmin = subset_all[metric_name].min()
    vmax = subset_all[metric_name].max()
    
    if use_log and vmin > 0:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        print(f"  Using log scale: {vmin:.4f} to {vmax:.4f}")
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        print(f"  Using linear scale: {vmin:.4f} to {vmax:.4f}")
    
    for col_idx, im_size in enumerate(largest_two):
        ax = axes[row_idx, col_idx]
        
        # Filter data for this image size
        subset = results_df[results_df['Image Size'] == im_size]
        
        # Create pivot table for heatmap
        heatmap_data = subset.pivot_table(
            values=metric_name,
            index='Timesteps',
            columns='Channels',
            aggfunc='mean'
        )
        
        # Create heatmap
        im = ax.imshow(heatmap_data.values, cmap=cmap, aspect='auto', 
                       origin='lower', norm=norm, interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(n_chans_vals)))
        ax.set_yticks(range(len(n_times_vals)))
        ax.set_xticklabels(n_chans_vals, fontsize=8)
        ax.set_yticklabels(n_times_vals, fontsize=8)
        
        # Labels only on outer edges
        if col_idx == 0:
            ax.set_ylabel('Timesteps (n_times)', fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel('')
            
        if row_idx == 3:  # Bottom row
            ax.set_xlabel('Channels (n_chans)', fontsize=10, fontweight='bold')
        else:
            ax.set_xlabel('')
        
        # Title with image size
        if row_idx == 0:
            ax.set_title(f'Image Size: {im_size}', fontsize=11, fontweight='bold')
        
        # Add metric name on left side
        if col_idx == 0:
            ax.text(-0.35, 0.5, metric_title, transform=ax.transAxes,
                   fontsize=10, fontweight='bold', va='center', ha='right',
                   rotation=90)
        
        # Add grid
        ax.set_xticks(np.arange(len(n_chans_vals)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(n_times_vals)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        
        # Annotate cells with values if requested
        if args.annotate:
            for i, n_times in enumerate(n_times_vals):
                for j, n_chans in enumerate(n_chans_vals):
                    if n_times in heatmap_data.index and n_chans in heatmap_data.columns:
                        value = heatmap_data.loc[n_times, n_chans]
                        
                        # Format the value
                        if value >= 100:
                            value_text = f'{value:.0f}'
                        elif value >= 10:
                            value_text = f'{value:.1f}'
                        elif value >= 1:
                            value_text = f'{value:.2f}'
                        else:
                            value_text = f'{value:.3f}'
                        
                        # Use black text for readability
                        ax.text(j, i, value_text, ha='center', va='center', 
                               color='black', fontsize=7, fontweight='bold')
        
        # Add colorbar for each row (shared across columns)
        if col_idx == 1:  # Right column
            cbar = plt.colorbar(im, ax=axes[row_idx, :], orientation='vertical',
                               pad=0.02, aspect=20, shrink=0.9, fraction=0.05)
            cbar.set_label(metric_title, fontsize=9, fontweight='bold')
            cbar.ax.tick_params(labelsize=8)

plt.tight_layout(rect=[0, 0, 0.96, 0.99])

# Save figure
output_path = results_dir / 'dual_axis_heatmaps_largest_two.png'
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"\n{'='*80}")
print(f"Figure saved to: {output_path}")
print(f"{'='*80}")

# Print analysis
print(f"\n{'='*80}")
print("METRIC ANALYSIS - LARGEST TWO IMAGE SIZES")
print(f"{'='*80}")

for metric in metrics:
    metric_name = metric['name']
    metric_title = metric['title']
    
    print(f"\n{metric_title}:")
    print("-" * 80)
    
    for im_size in largest_two:
        subset = results_df[results_df['Image Size'] == im_size]
        
        best = subset.loc[subset[metric_name].idxmax()]
        worst = subset.loc[subset[metric_name].idxmin()]
        
        # For latency and energy, lower is better, so swap labels
        if 'Latency' in metric_name or 'Energy' in metric_name:
            best, worst = worst, best
        
        print(f"\n  Image Size {im_size}:")
        print(f"    Best:  n_times={int(best['Timesteps']):3d}, n_chans={int(best['Channels']):3d}, "
              f"{metric_name}={best[metric_name]:.4f}")
        print(f"    Worst: n_times={int(worst['Timesteps']):3d}, n_chans={int(worst['Channels']):3d}, "
              f"{metric_name}={worst[metric_name]:.4f}")
        
        ratio = best[metric_name] / worst[metric_name] if worst[metric_name] != 0 else float('inf')
        print(f"    Ratio: {ratio:.2f}×")

print(f"\n{'='*80}")
print("INTERPRETATION GUIDE")
print(f"{'='*80}")
print("• Latency (red=bad, green=good): Lower latency is better (faster execution)")
print("• Speedup (red=bad, green=good): Higher speedup relative to baseline is better")
print("• Throughput (red=bad, green=good): Higher throughput is better (more work/time)")
print("• Energy (red=bad, green=good): Lower energy consumption is better (more efficient)")
print("\n• Each row shows a different metric")
print("• Each column shows a different image size (largest two)")
print("• Color scale is consistent across both image sizes for each metric")
print(f"{'='*80}\n")
