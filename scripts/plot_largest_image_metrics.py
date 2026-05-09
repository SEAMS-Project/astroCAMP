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
from results_dataframe import generate_results_dataframe

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

# Generate results DataFrame using helper function
results_df = generate_results_dataframe(
    benchmarks_csv_path='benchmarks.csv',
    machines_csv_path='machines.csv',
    locations_csv_path='locations.csv',
    lifetime_years=args.lifetime,
    location_ids=['WA']
)

# Filter to just WA location
results_df = results_df[results_df['Location'] == 'WA'].copy()

# Calculate Mvis/s (throughput) if not present
if 'Mvis/s' not in results_df.columns:
    results_df['Mvis/s'] = results_df['Mvis'] / (results_df['Time (s)'] / 3600)

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
fig.suptitle(f'All Metrics for Image Size {int(largest_im_size)}', 
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
    
    if heatmap_data.empty:
        continue
    
    # Find global min/max for consistent scale
    vmin = heatmap_data.min().min()
    vmax = heatmap_data.max().max()
    
    # Determine if we should use log scale
    use_log = (vmax / vmin) > 10
    if use_log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Create heatmap
    im = ax.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', 
                   origin='lower', norm=norm, interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(n_chans_vals)))
    ax.set_yticks(range(len(n_times_vals)))
    ax.set_xticklabels(n_chans_vals, fontsize=10)
    ax.set_yticklabels(n_times_vals, fontsize=10)
    
    ax.set_xlabel('Channels', fontsize=11, fontweight='bold')
    ax.set_ylabel('Timesteps', fontsize=11, fontweight='bold')
    ax.set_title(metric_info[current_metric]['title'], fontsize=12, fontweight='bold')
    
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
                
                # Format the value
                if value >= 100:
                    value_text = f'{value:.0f}'
                elif value >= 10:
                    value_text = f'{value:.1f}'
                elif value >= 1:
                    value_text = f'{value:.2f}'
                else:
                    value_text = f'{value:.3f}'
                
                ax.text(j, i, value_text, ha='center', va='center', 
                       color='black', fontsize=10, fontweight='bold')
    
    # Add colorbar for this subplot
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_info[current_metric]['short'], fontsize=9, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)

plt.tight_layout()

# Save figure
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Print analysis
print("\n" + "="*80)
print(f"ANALYSIS FOR LARGEST IMAGE SIZE ({int(largest_im_size)})")
print("="*80)

for current_metric in metrics_list:
    metric_col = metric_info[current_metric]['column']
    print(f"\n{metric_info[current_metric]['title']}:")
    print(f"  Range: {largest_subset[metric_col].min():.4f} to {largest_subset[metric_col].max():.4f}")
    print(f"  Mean: {largest_subset[metric_col].mean():.4f}")
    
    # Find best and worst configurations
    best_idx = largest_subset[metric_col].idxmax()
    worst_idx = largest_subset[metric_col].idxmin()
    best_row = largest_subset.loc[best_idx]
    worst_row = largest_subset.loc[worst_idx]
    
    print(f"  Best: t={int(best_row['Timesteps'])} c={int(best_row['Channels'])}, value={best_row[metric_col]:.4f}")
    print(f"  Worst: t={int(worst_row['Timesteps'])} c={int(worst_row['Channels'])}, value={worst_row[metric_col]:.4f}")
    print(f"  Ratio (Best/Worst): {best_row[metric_col] / worst_row[metric_col]:.2f}x")
