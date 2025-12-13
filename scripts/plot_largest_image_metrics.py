#!/usr/bin/env python3
"""
Plot All Metrics for Largest Image Size
Generates a 2x2 grid of heatmaps (carbon, cost, energy, throughput) for the largest image size
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import argparse
from pathlib import Path

# Create results directory if it doesn't exist
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot all metrics for largest image size.')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years (default: 5)')
parser.add_argument('-o', '--output', type=str, default='largest_image_metrics.png', 
                    help='Output filename (default: largest_image_metrics.png)')
parser.add_argument('--dpi', type=int, default=300, help='DPI for output image (default: 300)')
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
            'Time (s)': time * 3600,
            'Energy (kWh)': energy,
            'Carbon (kgCO2)': operational_carbon + capital_carbon,
            'Cost ($)': operational_energy_cost + capital_cost,
            'Mvis/s': mvis / (time * 3600),
            'Mvis/kWh': mvis / energy,
            'Mvis/kgCO2': mvis / (operational_carbon + capital_carbon),
            'Mvis/$': mvis / (operational_energy_cost + capital_cost),
        })

results_df = pd.DataFrame(results)

# Define metric labels and column names
metric_info = {
    'energy': {
        'column': 'Mvis/kWh',
        'title': 'Energy Efficiency\n(Mvis/kWh)',
        'short': 'Energy Eff.'
    },
    'carbon': {
        'column': 'Mvis/kgCO2',
        'title': 'Carbon Efficiency\n(Mvis/kgCO2)',
        'short': 'Carbon Eff.'
    },
    'cost': {
        'column': 'Mvis/$',
        'title': 'Cost Efficiency\n(Mvis/$)',
        'short': 'Cost Eff.'
    },
    'throughput': {
        'column': 'Mvis/s',
        'title': 'Throughput\n(Mvis/s)',
        'short': 'Throughput'
    }
}

# Get unique values for axes
im_sizes = sorted(results_df['Image Size'].unique())
n_times_vals = sorted(results_df['Timesteps'].unique())
n_chans_vals = sorted(results_df['Channels'].unique())

# Get largest image size
largest_im_size = max(im_sizes)
print(f"\nLargest image size: {largest_im_size}")
print(f"Timesteps: {n_times_vals}")
print(f"Channels: {n_chans_vals}")

# Filter data for largest image size
largest_subset = results_df[results_df['Image Size'] == largest_im_size]

# Create figure with 2x2 grid for all metrics
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle(f'All Metrics for Image Size {largest_im_size}', 
             fontsize=14, fontweight='bold', y=0.995)

axes = axes.flatten()
metrics_list = ['energy', 'carbon', 'cost', 'throughput']
colorbars = []

for plot_idx, current_metric in enumerate(metrics_list):
    ax = axes[plot_idx]
    metric_col = metric_info[current_metric]['column']
    metric_title = metric_info[current_metric]['title']
    
    # Create pivot table for heatmap
    heatmap_data = largest_subset.pivot_table(
        values=metric_col,
        index='Timesteps',
        columns='Channels',
        aggfunc='mean'
    )
    
    # Find min/max for this metric
    vmin = heatmap_data.min().min()
    vmax = heatmap_data.max().max()
    
    print(f"\n{metric_col} range: {vmin:.4f} to {vmax:.4f}")
    
    # Determine if we should use log scale
    use_log = (vmax / vmin) > 10
    if use_log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        print(f"Using log scale for color mapping (ratio: {vmax/vmin:.2f})")
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        print(f"Using linear scale for color mapping")
    
    # Create heatmap
    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', 
                   origin='lower', norm=norm, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(n_chans_vals)))
    ax.set_yticks(range(len(n_times_vals)))
    ax.set_xticklabels(n_chans_vals, fontsize=9)
    ax.set_yticklabels(n_times_vals, fontsize=9)
    
    ax.set_xlabel('Channels', fontsize=10, fontweight='bold')
    ax.set_ylabel('Time Steps', fontsize=10, fontweight='bold')
    ax.set_title(metric_title, fontsize=11, fontweight='bold')
    
    # Add grid
    ax.set_xticks(np.arange(len(n_chans_vals)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(n_times_vals)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.8)
    ax.tick_params(which='minor', size=0)
    
    # Annotate cells with values
    for i, n_times in enumerate(n_times_vals):
        for j, n_chans in enumerate(n_chans_vals):
            if n_times in heatmap_data.index and n_chans in heatmap_data.columns:
                value = heatmap_data.loc[n_times, n_chans]
                
                # Use black text for all cells
                text_color = 'black'
                
                # Format the main value
                if value >= 100:
                    value_text = f'{value:.0f}'
                elif value >= 10:
                    value_text = f'{value:.1f}'
                elif value >= 1:
                    value_text = f'{value:.2f}'
                else:
                    value_text = f'{value:.3f}'
                
                ax.text(j, i, value_text, ha='center', va='center', 
                       color=text_color, fontsize=9, fontweight='bold')
    
    # Add individual colorbar for each subplot
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
    cbar.set_label(metric_info[current_metric]['short'], fontsize=8, fontweight='bold')
    cbar.ax.tick_params(labelsize=7)

# Overall layout
plt.tight_layout()

# Save figure
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Print summary statistics for the largest image size
print("\n" + "="*80)
print(f"SUMMARY STATISTICS FOR IMAGE SIZE {largest_im_size}")
print("="*80)

for current_metric in metrics_list:
    metric_col = metric_info[current_metric]['column']
    print(f"\n{metric_col}:")
    print(f"  Min: {largest_subset[metric_col].min():.4f}")
    print(f"  Max: {largest_subset[metric_col].max():.4f}")
    print(f"  Mean: {largest_subset[metric_col].mean():.4f}")
    print(f"  Median: {largest_subset[metric_col].median():.4f}")
    
    # Find best and worst for each metric
    best_config = largest_subset.loc[largest_subset[metric_col].idxmax()]
    worst_config = largest_subset.loc[largest_subset[metric_col].idxmin()]
    
    print(f"  Best: n_times={int(best_config['Timesteps'])}, n_chans={int(best_config['Channels'])}, value={best_config[metric_col]:.4f}")
    print(f"  Worst: n_times={int(worst_config['Timesteps'])}, n_chans={int(worst_config['Channels'])}, value={worst_config[metric_col]:.4f}")
    print(f"  Ratio (best/worst): {best_config[metric_col] / worst_config[metric_col]:.2f}x")

print("\n" + "="*80)
