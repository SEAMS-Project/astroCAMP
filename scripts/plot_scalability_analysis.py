#!/usr/bin/env python3
"""
Scalability Analysis Script
Evaluates all configurations (n_times, n_chans) across all image sizes
for each metric, with grouped x-axis labels and dual-axis visualization.

Uses results DataFrame from cea.py for consistency with carbon and economic analysis.

Metrics analyzed:
- Energy efficiency (Mvis/kWh) vs Power (W)
- Carbon efficiency (Mvis/kgCO2) vs Carbon footprint (g CO2)
- Cost efficiency (Mvis/$) vs Total cost ($)
- Throughput (Mvis/s) vs Wall time (s)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
from pathlib import Path
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Scalability analysis with dual-axis plots.')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years (default: 5)')
args = parser.parse_args()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = BASE_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Load benchmarks
benchmarks_df = pd.read_csv(
    DATA_DIR / "benchmarks.csv",
    header=None,
    names=[
        "im_size", "n_times", "n_chans", "wall_time", "wall_time_sec", "n_rows", "n_vis",
        "n_idg",
        "idg_h_sec", "idg_h_watt", "idg_h_jou",
        "idg_d_sec", "idg_d_watt", "idg_d_jou",
        "idg_grid_mvs",
        "cpu_j", "cpu_bsl_j", "cpu_bsl_std_j",
        "gpu0_j", "gpu1_j", "gpu2_j", "gpu3_j",
        "gpu_j", "gpu_bsl_j", "gpu_bsl_std_j",
        "tot_sys_j",
        "tot_pdu_j", "pdu_bsl_j", "pdu_bsl_std_j",
        "abs_cpu_j", "abs_gpu_j", "abs_pdu_j"
    ],
)

# Generate results DataFrame using cea.py logic (single location: WA)
Lifetime = args.lifetime * 365 * 24  # Lifetime in hours
location_ids = ['WA']

# Load locations and machines
machines_df = pd.read_csv(DATA_DIR / 'machines.csv').set_index('machine')
locations_df = pd.read_csv(DATA_DIR / 'locations.csv').set_index('id').reset_index()
locations_df = locations_df[locations_df['id'].isin(location_ids)]

# Calculate idle power from baseline
idle_pdu_watt = benchmarks_df['pdu_bsl_j'].mean()

benchmarks_df['machine'] = 'R675 V3 + 4xH100 96GB'
benchmarks_df['mvis'] = benchmarks_df['n_vis'] / 1e6

# Create results list following cea.py structure
results = []

for _, benchmark in benchmarks_df.iterrows():
    benchmark_name = f"{benchmark['im_size']}_{benchmark['n_times']}_{benchmark['n_chans']}"
    machine_name = benchmark['machine']
    time = benchmark['wall_time_sec'] / 3600  # Convert to hours
    energy_dynamic = benchmark['tot_sys_j'] / 3.6e6  # Dynamic energy in kWh
    energy_static = idle_pdu_watt / 4 * time / 1000  # Static energy in kWh
    energy = energy_dynamic + energy_static  # Total energy in kWh
    
    # Get machine parameters
    machine_cost = machines_df.loc[machine_name, 'cost']
    machine_embodied = machines_df.loc[machine_name, 'embodied']
    
    for _, location in locations_df.iterrows():
        location_id = location['id']
        location_name = location['location']
        ci = location['ci']  # Carbon intensity in kg CO2/kWh
        ep = location['ep']  # Electricity price in $/kWh
        
        # Calculate operational and capital expenditures
        operational_energy_cost = energy * ep
        operational_carbon = energy * ci
        capital_cost = machine_cost * (time / Lifetime)
        capital_carbon = machine_embodied * (time / Lifetime)
        mvis = benchmark['mvis']
        
        results.append({
            'Image Size': benchmark['im_size'],
            'Timesteps': benchmark['n_times'],
            'Channels': benchmark['n_chans'],
            'Machine': machine_name,
            'Location': location_id,
            'Mvis': mvis,
            'Time (s)': time * 3600,
            'Dynamic Energy (Wh)': energy_dynamic * 1e3,
            'Static Energy (Wh)': energy_static * 1e3,
            'Energy (Wh)': energy * 1e3,
            'Power (W)': energy * 1e3 / time,
            'Operational Carbon (g CO2)': operational_carbon * 1e3,
            'Embodied Carbon (g CO2)': capital_carbon * 1e3,
            'Total Carbon (g CO2)': (operational_carbon + capital_carbon) * 1e3,
            'Operational Cost ($)': operational_energy_cost,
            'Capital Cost ($)': capital_cost,
            'Total Cost ($)': operational_energy_cost + capital_cost,
            'Mvis/h': mvis / time,
            'Mvis/kWh': mvis / energy,
            'Mvis/kgCO2': mvis / (operational_carbon + capital_carbon),
            'Mvis/$': mvis / (operational_energy_cost + capital_cost),
        })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Create config labels for visualization
results_df['config'] = (
    "t" + results_df["Timesteps"].astype(int).astype(str) +
    "-c" + results_df["Channels"].astype(int).astype(str)
)

# Get unique image sizes and configurations
im_sizes = sorted(results_df["Image Size"].unique())
configs = sorted(
    results_df.groupby("config").size().index,
    key=lambda x: (int(x.split('-')[0][1:]), int(x.split('-')[1][1:]))
)

print("\n" + "=" * 100)
print("SCALABILITY ANALYSIS - DUAL-AXIS VISUALIZATION (from results DataFrame)")
print("=" * 100)
print(f"Using lifetime: {args.lifetime} years")
print(f"Image sizes: {im_sizes}")
print(f"Configurations: {configs}")
print(f"Total data points: {len(results_df)}")
print()

# Color scheme for configurations
n_configs = len(configs)
colors = plt.cm.tab20(np.linspace(0, 1, n_configs))
config_colors = {config: colors[i] for i, config in enumerate(configs)}

def create_dual_axis_plot(metric_left, label_left, metric_right, label_right, title, filename):
    """Create dual-axis scalability plot with grouping by image size."""
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Prepare data grouped by image size and config
    x_pos = 0
    x_labels = []
    x_ticks = []
    group_boundaries = []
    
    for im_idx, im_size in enumerate(im_sizes):
        im_data = results_df[results_df["Image Size"] == im_size].copy()
        group_start = x_pos
        
        for config in configs:
            config_data = im_data[im_data["config"] == config]
            if len(config_data) == 0:
                continue
                
            # Average metrics for this config and image size
            y_left = config_data[metric_left].mean()
            y_right = config_data[metric_right].mean()
            
            # Left axis bar
            color = config_colors[config]
            ax1.bar(x_pos, y_left, width=0.7, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            x_labels.append(config)
            x_ticks.append(x_pos)
            x_pos += 1
        
        group_boundaries.append((group_start, x_pos - 0.5))
    
    # Configure left y-axis
    ax1.set_xlabel('Configuration (Image Size | n_times, n_chans)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(label_left, fontsize=12, fontweight='bold', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=11)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=90, fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    # Create right y-axis with line plot
    ax2 = ax1.twinx()
    
    x_line = []
    y_line = []
    for im_idx, im_size in enumerate(im_sizes):
        im_data = results_df[results_df["Image Size"] == im_size].copy()
        
        for config_idx, config in enumerate(configs):
            config_data = im_data[im_data["config"] == config]
            if len(config_data) == 0:
                continue
                
            y_right = config_data[metric_right].mean()
            x_line.append(x_ticks[len(x_line)] if len(x_line) < len(x_ticks) else 0)
            y_line.append(y_right)
    
    # Plot line on secondary axis
    if len(x_line) > 0:
        ax2.plot(x_line, y_line, color='#d62728', marker='o', linewidth=2.5, 
                markersize=6, label=label_right, zorder=10, markeredgecolor='black', markeredgewidth=0.5)
    
    ax2.set_ylabel(label_right, fontsize=12, fontweight='bold', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728', labelsize=11)
    
    # Add vertical separators for image size groups
    for group_start, group_end in group_boundaries[:-1]:
        ax1.axvline(group_end + 0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    
    # Add image size labels at group centers
    for im_idx, (group_start, group_end) in enumerate(group_boundaries):
        group_center = (group_start + group_end) / 2
        ax1.text(group_center, ax1.get_ylim()[1] * 0.95, f"Image {im_sizes[im_idx]}",
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    outfile = RESULTS_DIR / filename
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {outfile}")
    plt.close()


# Define metric pairs for dual-axis visualization
# Format: (metric_left, label_left, metric_right, label_right, title, filename)
metric_pairs = [
    (
        'Mvis/kWh', 'Energy Efficiency (Mvis/kWh)',
        'Power (W)', 'Power Consumption (W)',
        'Scalability: Energy Efficiency vs Power Consumption',
        'scalability_energy_vs_power.png'
    ),
    (
        'Mvis/kgCO2', 'Carbon Efficiency (Mvis/kgCO2)',
        'Total Carbon (g CO2)', 'Carbon Footprint (g CO2)',
        'Scalability: Carbon Efficiency vs Carbon Footprint',
        'scalability_carbon_eff_vs_footprint.png'
    ),
    (
        'Mvis/$', 'Cost Efficiency (Mvis/$)',
        'Total Cost ($)', 'Total Cost ($)',
        'Scalability: Cost Efficiency vs Total Cost',
        'scalability_cost_eff_vs_total.png'
    ),
    (
        'Mvis/h', 'Throughput (Mvis/h)',
        'Time (s)', 'Execution Time (s)',
        'Scalability: Throughput vs Execution Time',
        'scalability_throughput_vs_time.png'
    ),
    (
        'Mvis/kWh', 'Energy Efficiency (Mvis/kWh)',
        'Energy (Wh)', 'Total Energy (Wh)',
        'Scalability: Energy Efficiency vs Total Energy',
        'scalability_energy_eff_vs_total.png'
    ),
    (
        'Mvis/kgCO2', 'Carbon Efficiency (Mvis/kgCO2)',
        'Energy (Wh)', 'Energy Consumption (Wh)',
        'Scalability: Carbon Efficiency vs Energy',
        'scalability_carbon_eff_vs_energy.png'
    ),
]

print("\nGenerating dual-axis scalability plots...")
for metric_left, label_left, metric_right, label_right, title, filename in metric_pairs:
    try:
        create_dual_axis_plot(metric_left, label_left, metric_right, label_right, title, filename)
    except Exception as e:
        print(f"✗ Error in {filename}: {str(e)}")

# Create summary statistics table
print("\n" + "=" * 100)
print("SCALABILITY SUMMARY - BEST COMBINATIONS")
print("=" * 100)

summary_data = []
for config in configs:
    config_data = results_df[results_df["config"] == config]
    
    summary_data.append({
        'Config': config,
        'Avg Energy Eff': config_data['Mvis/kWh'].mean(),
        'Max Energy Eff': config_data['Mvis/kWh'].max(),
        'Avg Power (W)': config_data['Power (W)'].mean(),
        'Avg Carbon Eff': config_data['Mvis/kgCO2'].mean(),
        'Max Carbon Eff': config_data['Mvis/kgCO2'].max(),
        'Avg Carbon (g CO2)': config_data['Total Carbon (g CO2)'].mean(),
        'Avg Cost Eff': config_data['Mvis/$'].mean(),
        'Max Cost Eff': config_data['Mvis/$'].max(),
        'Avg Cost ($)': config_data['Total Cost ($)'].mean(),
        'Avg Throughput': config_data['Mvis/h'].mean(),
        'Max Throughput': config_data['Mvis/h'].max(),
        'Avg Time (s)': config_data['Time (s)'].mean(),
    })

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# Identify best combinations
print("\n" + "=" * 100)
print("BEST CONFIGURATIONS BY METRIC")
print("=" * 100)

best_configs = {
    'Energy Efficiency (avg)': summary_df.loc[summary_df['Avg Energy Eff'].idxmax(), 'Config'],
    'Energy Efficiency (max)': summary_df.loc[summary_df['Max Energy Eff'].idxmax(), 'Config'],
    'Carbon Efficiency (avg)': summary_df.loc[summary_df['Avg Carbon Eff'].idxmax(), 'Config'],
    'Carbon Efficiency (max)': summary_df.loc[summary_df['Max Carbon Eff'].idxmax(), 'Config'],
    'Cost Efficiency (avg)': summary_df.loc[summary_df['Avg Cost Eff'].idxmax(), 'Config'],
    'Cost Efficiency (max)': summary_df.loc[summary_df['Max Cost Eff'].idxmax(), 'Config'],
    'Throughput (avg)': summary_df.loc[summary_df['Avg Throughput'].idxmax(), 'Config'],
    'Throughput (max)': summary_df.loc[summary_df['Max Throughput'].idxmax(), 'Config'],
    'Lowest Power': summary_df.loc[summary_df['Avg Power (W)'].idxmin(), 'Config'],
    'Lowest Cost': summary_df.loc[summary_df['Avg Cost ($)'].idxmin(), 'Config'],
    'Fastest Execution': summary_df.loc[summary_df['Avg Time (s)'].idxmin(), 'Config'],
}

for metric, config in best_configs.items():
    print(f"{metric:30s}: {config}")

print("\n" + "=" * 100)
print("DUAL-AXIS RECOMMENDATIONS")
print("=" * 100)
print("""
Best Left-Right Axis Combinations for Scalability Analysis:

1. Energy Efficiency (Mvis/kWh) [LEFT] vs Power Consumption (W) [RIGHT]
   → Shows trade-off: high efficiency with increasing power at scale
   → Key insight: efficiency gains despite rising power requirements

2. Carbon Efficiency (Mvis/kgCO2) [LEFT] vs Carbon Footprint (kgCO2) [RIGHT]
   → Shows carbon amortization: efficiency increases while footprint grows
   → Key insight: larger jobs spread baseline carbon over more work

3. Cost Efficiency (Mvis/$) [LEFT] vs Total Cost ($) [RIGHT]
   → Shows cost scaling: efficiency improves but absolute cost rises
   → Key insight: larger jobs are more cost-efficient on per-unit basis

4. Throughput (Mvis/s) [LEFT] vs Execution Time (s) [RIGHT]
   → Shows inverse relationship: throughput improves with time investment
   → Key insight: larger batches process more work in longer runtime

5. Energy Efficiency [LEFT] vs Total Energy (kJ) [RIGHT]
   → Shows absolute vs relative metrics: efficiency vs resource consumption
   → Key insight: optimal efficiency configurations still consume significant energy

6. Carbon Efficiency [LEFT] vs Energy Consumption (kWh) [RIGHT]
   → Shows carbon impact: efficiency per unit energy consumed
   → Key insight: energy consumption drives carbon efficiency gains
""")

print("=" * 100)
print("✓ Scalability analysis complete!")
print(f"  Generated {len(metric_pairs)} dual-axis plots")
print(f"  All figures saved to: {RESULTS_DIR}")
print("=" * 100)
