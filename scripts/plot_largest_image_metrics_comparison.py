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

# Create results directory if it doesn't exist
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Parse arguments
parser = argparse.ArgumentParser(description='Plot largest image metrics with location comparison.')
parser.add_argument('-o', '--output', type=str, default='largest_image_metrics_comparison.png',
                    help='Output filename')
args = parser.parse_args()

# Ensure output path is in results directory
output_path = results_dir / args.output

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
benchmarks_df['time'] = benchmarks_df['wall_time_sec'] / 3600  # Convert to hours
benchmarks_df['mvis'] = benchmarks_df['n_vis'] / 1e6

# Read machine and location data
machines_df = pd.read_csv('machines.csv').set_index('machine')
locations_df = pd.read_csv('locations.csv').set_index('id')

# Calculate metrics
results = []
for _, benchmark in benchmarks_df.iterrows():
    machine_name = benchmark['machine']
    time = benchmark['time']
    energy_dynamic = (benchmark['gpu0_j'] + benchmark['cpu_j']) / 3.6e6  # kWh
    energy_static = ((277.75 / 4 + 65.44) * time) / 1000  # kWh
    energy = energy_dynamic + energy_static
    
    machine_cost = machines_df.loc[machine_name, 'cost']
    machine_embodied = machines_df.loc[machine_name, 'embodied']
    
    mvis = benchmark['mvis']
    
    for location_id in ['WA', 'SA']:
        if location_id not in locations_df.index:
            continue
            
        location = locations_df.loc[location_id]
        ci = location['ci']
        ep = location['ep']
        
        operational_carbon = energy * ci
        capital_carbon = machine_embodied * time / (5 * 365 * 24)  # 5-year amortization
        
        operational_cost = energy * ep
        capital_cost = machine_cost * time / (5 * 365 * 24)
        
        results.append({
            'Image Size': benchmark['im_size'],
            'Timesteps': benchmark['n_times'],
            'Channels': benchmark['n_chans'],
            'Location': location_id,
            'Mvis': mvis,
            'Throughput (Mvis/s)': mvis / (time * 3600) if time > 0 else 0,
            'Energy Eff (Mvis/kWh)': mvis / energy if energy > 0 else 0,
            'Carbon Eff (Mvis/kgCO2)': mvis / (operational_carbon + capital_carbon) if (operational_carbon + capital_carbon) > 0 else 0,
            'Cost Eff (Mvis/$)': mvis / (operational_cost + capital_cost) if (operational_cost + capital_cost) > 0 else 0,
        })

results_df = pd.DataFrame(results)

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
    ('Throughput (Mvis/s)', 'Throughput', 'viridis'),
    ('Energy Eff (Mvis/kWh)', 'Energy Efficiency', 'RdYlGn'),
    ('Carbon Eff (Mvis/kgCO2)', 'Carbon Efficiency', 'RdYlGn'),
    ('Cost Eff (Mvis/$)', 'Cost Efficiency', 'RdYlGn')
]

locations = ['WA', 'SA']
location_names = {'WA': 'Western Australia', 'SA': 'South Africa'}
location_colors = {'WA': '#1f77b4', 'SA': '#ff7f0e'}

for loc_idx, location_id in enumerate(locations):
    loc_data = largest_df[largest_df['Location'] == location_id].sort_values('Config')
    
    for metric_idx, (metric_col, metric_title, cmap) in enumerate(metrics):
        ax = axes[loc_idx, metric_idx]
        
        # Create 2D grid for heatmap (4x4 grid of timesteps × channels)
        times = sorted(loc_data['Timesteps'].unique())
        chans = sorted(loc_data['Channels'].unique())
        
        heatmap = np.zeros((len(times), len(chans)))
        for i, t in enumerate(times):
            for j, c in enumerate(chans):
                val = loc_data[(loc_data['Timesteps'] == t) & (loc_data['Channels'] == c)][metric_col]
                heatmap[i, j] = val.values[0] if len(val) > 0 else 0
        
        # Determine normalization
        vmin, vmax = heatmap[heatmap > 0].min(), heatmap[heatmap > 0].max()
        if vmax / vmin > 10:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        
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
plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
