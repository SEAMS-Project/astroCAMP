#!/usr/bin/env python3
"""
Plot Regime Maps: Times vs Channels
Generates Figure B: Efficiency heatmaps per image size
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
parser = argparse.ArgumentParser(description='Plot regime heatmaps (times vs channels).')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years (default: 5)')
parser.add_argument('-m', '--metric', type=str, default='all', 
                    choices=['energy', 'carbon', 'cost', 'throughput', 'all'],
                    help='Primary metric to display (default: all - generates all metrics)')
parser.add_argument('-o', '--output-prefix', type=str, default='regime_heatmaps', 
                    help='Output filename prefix (default: regime_heatmaps)')
parser.add_argument('--dpi', type=int, default=300, help='DPI for output image (default: 300)')
parser.add_argument('--annotate-throughput', action='store_true',
                    help='Annotate cells with throughput values')
args = parser.parse_args()

print(f"Using lifetime of {args.lifetime} years for all machines.")

# Determine which metrics to generate
if args.metric == 'all':
    metrics_to_generate = ['energy', 'carbon', 'cost', 'throughput']
    print(f"Generating heatmaps for all metrics: {metrics_to_generate}")
else:
    metrics_to_generate = [args.metric]
    print(f"Primary metric: {args.metric}")

# Parameters (matching cea.py)
Lifetime = args.lifetime * 365 * 24  # Lifetime in hours
idle_cpu_watt = 277.75 / 4  # Idle CPU power consumption in watts
idle_gpu_watt = 65.44       # Idle GPU power consumption in watts
location_ids = ['WA', 'SA']

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
        'title': 'Energy Efficiency (Mvis/kWh)',
        'short': 'Energy Eff.'
    },
    'carbon': {
        'column': 'Mvis/kgCO2',
        'title': 'Carbon Efficiency (Mvis/kgCO2)',
        'short': 'Carbon Eff.'
    },
    'cost': {
        'column': 'Mvis/$',
        'title': 'Cost Efficiency (Mvis/$)',
        'short': 'Cost Eff.'
    },
    'throughput': {
        'column': 'Mvis/s',
        'title': 'Throughput (Mvis/s)',
        'short': 'Throughput'
    }
}

# Get unique values for axes
im_sizes = sorted(results_df['Image Size'].unique())
n_times_vals = sorted(results_df['Timesteps'].unique())
n_chans_vals = sorted(results_df['Channels'].unique())

print(f"\nData summary:")
print(f"  Image sizes: {im_sizes}")
print(f"  Timesteps: {n_times_vals}")
print(f"  Channels: {n_chans_vals}")
print(f"  Total configurations: {len(results_df)}")

# Loop through each metric to generate
for current_metric in metrics_to_generate:
    print(f"\n{'='*80}")
    print(f"GENERATING HEATMAP FOR: {current_metric.upper()}")
    print(f"{'='*80}")
    
    metric_col = metric_info[current_metric]['column']
    metric_title = metric_info[current_metric]['title']
    metric_short = metric_info[current_metric]['short']

    # Create figure with 2x2 grid (optimized for single-column LaTeX figure)
    fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
    fig.suptitle(f'Regime Maps: {metric_title}\nTime Steps vs Channels by Image Size', 
                 fontsize=12, fontweight='bold', y=0.98)

    axes = axes.flatten()

    # Find global min/max for consistent color scale
    vmin = results_df[metric_col].min()
    vmax = results_df[metric_col].max()

    print(f"\n{metric_col} range: {vmin:.4f} to {vmax:.4f}")

    # Determine if we should use log scale
    use_log = (vmax / vmin) > 10
    if use_log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
        print(f"Using log scale for color mapping (ratio: {vmax/vmin:.2f})")
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        print(f"Using linear scale for color mapping")

    # Create heatmap for each image size
    for idx, im_size in enumerate(im_sizes):
        ax = axes[idx]
        
        # Filter data for this image size
        subset = results_df[results_df['Image Size'] == im_size]
        
        # Create pivot table for heatmap
        heatmap_data = subset.pivot_table(
            values=metric_col,
            index='Timesteps',
            columns='Channels',
            aggfunc='mean'
        )
        
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
        ax.set_title(f'Image: {im_size}', fontsize=11, fontweight='bold')
        
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
                    
                    # Get throughput for annotation if requested
                    throughput_val = subset[
                        (subset['Timesteps'] == n_times) & 
                        (subset['Channels'] == n_chans)
                    ]['Mvis/s'].values[0] if not subset[
                        (subset['Timesteps'] == n_times) & 
                        (subset['Channels'] == n_chans)
                    ].empty else 0
                    
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
                    
                    if args.annotate_throughput:
                        # Annotate with both main metric and throughput
                        text = f'{value_text}\n({throughput_val:.3f})'
                        ax.text(j, i, text, ha='center', va='center', 
                               color=text_color, fontsize=6, fontweight='bold')
                    else:
                        # Just the main metric
                        ax.text(j, i, value_text, ha='center', va='center', 
                               color=text_color, fontsize=9, fontweight='bold')

    # Add colorbar below the figures
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # Leave space at bottom for colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', 
                        pad=0.12, aspect=30, shrink=0.9, fraction=0.05)
    cbar.set_label(metric_title, fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)

    # Save figure
    output_filename = f"{args.output_prefix}_{current_metric}.png"
    output_path = results_dir / output_filename
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close(fig)

    # Analyze trends
    print("\n" + "="*80)
    print(f"REGIME ANALYSIS - {metric_title}")
    print("="*80)

    for im_size in im_sizes:
        print(f"\nImage Size: {im_size}")
        print("-" * 80)
        subset = results_df[results_df['Image Size'] == im_size]
        
        # Compare time-heavy vs channel-heavy
        # Fix n_chans=1, vary n_times
        time_heavy = subset[subset['Channels'] == 1].sort_values('Timesteps')
        print(f"\n  Time-heavy (n_chans=1, varying n_times):")
        if not time_heavy.empty:
            for _, row in time_heavy.iterrows():
                print(f"    n_times={int(row['Timesteps']):3d}: {metric_col}={row[metric_col]:8.3f}, "
                      f"Mvis/s={row['Mvis/s']:.4f}")
            
            # Calculate gradient
            if len(time_heavy) > 1:
                first = time_heavy.iloc[0][metric_col]
                last = time_heavy.iloc[-1][metric_col]
                change = ((last - first) / first) * 100
                print(f"    → Change: {change:+.1f}% from n_times={int(time_heavy.iloc[0]['Timesteps'])} to {int(time_heavy.iloc[-1]['Timesteps'])}")
        
        # Fix n_times=1, vary n_chans
        channel_heavy = subset[subset['Timesteps'] == 1].sort_values('Channels')
        print(f"\n  Channel-heavy (n_times=1, varying n_chans):")
        if not channel_heavy.empty:
            for _, row in channel_heavy.iterrows():
                print(f"    n_chans={int(row['Channels']):3d}: {metric_col}={row[metric_col]:8.3f}, "
                      f"Mvis/s={row['Mvis/s']:.4f}")
            
            # Calculate gradient
            if len(channel_heavy) > 1:
                first = channel_heavy.iloc[0][metric_col]
                last = channel_heavy.iloc[-1][metric_col]
                change = ((last - first) / first) * 100
                print(f"    → Change: {change:+.1f}% from n_chans={int(channel_heavy.iloc[0]['Channels'])} to {int(channel_heavy.iloc[-1]['Channels'])}")
        
        # Find best configuration for this image size
        best_config = subset.loc[subset[metric_col].idxmax()]
        print(f"\n  Best configuration (max {metric_col}):")
        print(f"    n_times={int(best_config['Timesteps']):3d}, n_chans={int(best_config['Channels']):3d}")
        print(f"    {metric_col}={best_config[metric_col]:.3f}, Mvis/s={best_config['Mvis/s']:.4f}, Mvis={best_config['Mvis']:.2f}")
        
        # Find worst configuration for this image size
        worst_config = subset.loc[subset[metric_col].idxmin()]
        print(f"\n  Worst configuration (min {metric_col}):")
        print(f"    n_times={int(worst_config['Timesteps']):3d}, n_chans={int(worst_config['Channels']):3d}")
        print(f"    {metric_col}={worst_config[metric_col]:.3f}, Mvis/s={worst_config['Mvis/s']:.4f}, Mvis={worst_config['Mvis']:.2f}")
        
        ratio = best_config[metric_col] / worst_config[metric_col]
        print(f"\n  Best/Worst ratio: {ratio:.2f}x")

    # Overall comparison: increasing times vs increasing channels
    print("\n" + "="*80)
    print("OVERALL TRENDS: TIME vs CHANNEL SCALING")
    print("="*80)

    # Average efficiency when increasing time steps (averaged over all image sizes and channel configs)
    print("\nEffect of increasing n_times (averaged across all image sizes and n_chans):")
    for n_times in n_times_vals:
        time_subset = results_df[results_df['Timesteps'] == n_times]
        avg_metric = time_subset[metric_col].mean()
        avg_throughput = time_subset['Mvis/s'].mean()
        print(f"  n_times={int(n_times):3d}: avg {metric_col}={avg_metric:8.3f}, avg Mvis/s={avg_throughput:.4f}")

    print("\nEffect of increasing n_chans (averaged across all image sizes and n_times):")
    for n_chans in n_chans_vals:
        chan_subset = results_df[results_df['Channels'] == n_chans]
        avg_metric = chan_subset[metric_col].mean()
        avg_throughput = chan_subset['Mvis/s'].mean()
        print(f"  n_chans={int(n_chans):3d}: avg {metric_col}={avg_metric:8.3f}, avg Mvis/s={avg_throughput:.4f}")

    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print(f"• Green (high values): More efficient configurations - better {metric_short}")
    print(f"• Red (low values): Less efficient configurations - worse {metric_short}")
    print("• Horizontal gradients: Effect of increasing channels (spectral resolution)")
    print("• Vertical gradients: Effect of increasing time steps (temporal resolution)")
    print("• Diagonal trends: Combined scaling behavior")

print("\n" + "="*80)
print(f"SUMMARY: Generated {len(metrics_to_generate)} heatmap(s)")
print("="*80)
for metric in metrics_to_generate:
    print(f"  - {args.output_prefix}_{metric}.png")
print("="*80)
