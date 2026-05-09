#!/usr/bin/env python3
"""
Plot Largest Image Metrics with Location Comparison
Shows all 4 metrics (energy, carbon, cost, throughput) for largest workload
with side-by-side comparison of WA vs SA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
from pathlib import Path
from results_dataframe import generate_results_dataframe

# Create results directory if it doesn't exist
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Parse arguments
parser = argparse.ArgumentParser(description='Plot largest image metrics with location comparison.')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years (default: 5)')
parser.add_argument('-o', '--output', type=str, default='largest_image_metrics_comparison.png',
                    help='Output filename')
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
    location_ids=['WA', 'SA']
)

# Calculate Mvis/s (throughput) if not present
if 'Mvis/s' not in results_df.columns:
    results_df['Mvis/s'] = results_df['Mvis'] / (results_df['Time (s)'] / 3600)

# Filter to largest image size
largest_size = results_df['Image Size'].max()
largest_df = results_df[results_df['Image Size'] == largest_size].copy()
largest_df['Config'] = (largest_df['Timesteps'].astype(int).astype(str) + '×' + 
                        largest_df['Channels'].astype(int).astype(str))

print(f"Analyzing largest image size: {int(largest_size)}")
print(f"Found {len(largest_df)} configurations ({len(largest_df)//2} per location)")

# Create figure: 4 metrics × 2 locations (2 rows × 4 columns)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f'Efficiency Metrics Comparison: Largest Image ({int(largest_size)}×{int(largest_size)})\nWA vs SA Location', 
             fontsize=14, fontweight='bold')

metrics = [
    ('Mvis/s', 'Throughput', 'viridis'),
    ('Mvis/kWh', 'Energy Efficiency', 'RdYlGn'),
    ('Mvis/kgCO2', 'Carbon Efficiency', 'RdYlGn'),
    ('Mvis/$', 'Cost Efficiency', 'RdYlGn')
]

locations = ['WA', 'SA']
location_names = {'WA': 'Western Australia', 'SA': 'South Africa'}

for loc_idx, location_id in enumerate(locations):
    loc_data = largest_df[largest_df['Location'] == location_id].sort_values('Config')
    
    for metric_idx, (metric_col, metric_title, cmap) in enumerate(metrics):
        ax = axes[loc_idx, metric_idx]
        
        # Create 2D grid for heatmap (times × channels)
        times = sorted(loc_data['Timesteps'].unique())
        chans = sorted(loc_data['Channels'].unique())
        
        heatmap = np.zeros((len(times), len(chans)))
        for i, t in enumerate(times):
            for j, c in enumerate(chans):
                val = loc_data[(loc_data['Timesteps'] == t) & (loc_data['Channels'] == c)][metric_col]
                heatmap[i, j] = val.values[0] if len(val) > 0 else 0
        
        # Determine normalization
        valid_vals = heatmap[heatmap > 0]
        if len(valid_vals) > 0:
            vmin, vmax = valid_vals.min(), valid_vals.max()
            if vmax / vmin > 10:
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        
        # Plot heatmap
        im = ax.imshow(heatmap, cmap=cmap, norm=norm, aspect='auto', origin='lower')
        
        # Labels
        ax.set_xticks(range(len(chans)))
        ax.set_yticks(range(len(times)))
        ax.set_xticklabels([int(c) for c in chans], fontsize=8)
        ax.set_yticklabels([int(t) for t in times], fontsize=8)
        
        if loc_idx == 1:
            ax.set_xlabel('Channels', fontsize=9, fontweight='bold')
        if metric_idx == 0:
            ax.set_ylabel(f'{location_names[location_id]}\nTimesteps', fontsize=9, fontweight='bold')
        
        if loc_idx == 0:
            ax.set_title(f'{metric_title}', fontsize=10, fontweight='bold')
        
        # Annotations
        for i in range(len(times)):
            for j in range(len(chans)):
                val = heatmap[i, j]
                if val > 0:
                    if val >= 100:
                        text = f'{val:.0f}'
                    elif val >= 10:
                        text = f'{val:.1f}'
                    else:
                        text = f'{val:.2f}'
                    ax.text(j, i, text, ha='center', va='center', 
                           color='black', fontsize=7, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"\nComparison figure saved to: {output_path}")

# Analysis
print("\n" + "="*80)
print("LARGEST IMAGE EFFICIENCY ANALYSIS")
print("="*80)

for location_id in locations:
    loc_data = largest_df[largest_df['Location'] == location_id]
    print(f"\n{location_names[location_id]} ({location_id}):")
    
    for metric_col, metric_title, _ in metrics:
        min_val = loc_data[metric_col].min()
        max_val = loc_data[metric_col].max()
        best_config = loc_data.loc[loc_data[metric_col].idxmax(), 'Config']
        worst_config = loc_data.loc[loc_data[metric_col].idxmin(), 'Config']
        
        print(f"\n  {metric_title}:")
        print(f"    Range: {min_val:.2e} to {max_val:.2e}")
        print(f"    Ratio (best/worst): {max_val/min_val:.1f}x")
        print(f"    Best: {best_config} ({max_val:.2e})")
        print(f"    Worst: {worst_config} ({min_val:.2e})")

print("\n" + "="*80)
