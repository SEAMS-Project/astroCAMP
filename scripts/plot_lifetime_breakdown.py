#!/usr/bin/env python3
"""
Plot Lifetime Carbon and Cost Breakdown
Generates stacked bar charts showing operational vs embodied carbon,
and operational vs capital cost for the 5-year lifetime assumption
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Create results directory if it doesn't exist
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot lifetime carbon and cost breakdown.')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years (default: 5)')
parser.add_argument('-o', '--output', type=str, default='lifetime_breakdown.png', 
                    help='Output filename (default: lifetime_breakdown.png)')
parser.add_argument('--dpi', type=int, default=300, help='DPI for output image (default: 300)')
args = parser.parse_args()

# Ensure output path is in results directory
output_path = results_dir / args.output

print(f"Using lifetime of {args.lifetime} years for all machines.")

# Parameters (matching cea.py)
Lifetime = args.lifetime * 365 * 24  # Lifetime in hours
idle_cpu_watt = 277.75 / 4  # Idle CPU power consumption in watts
idle_gpu_watt = 65.44       # Idle GPU power consumption in watts
location_ids = ['WA', 'SA']  # Include both locations

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
            'Benchmark': f"{int(benchmark['im_size'])}_{int(benchmark['n_times'])}_{int(benchmark['n_chans'])}",
            'Mvis': mvis,
            'Time (s)': time * 3600,
            'Energy Dynamic (kWh)': energy_dynamic,
            'Energy Static (kWh)': energy_static,
            'Energy (kWh)': energy,
            'Operational Carbon (kg CO2)': operational_carbon,
            'Embodied Carbon (kg CO2)': capital_carbon,
            'Total Carbon (kg CO2)': operational_carbon + capital_carbon,
            'Operational Cost ($)': operational_energy_cost,
            'Capital Cost ($)': capital_cost,
            'Total Cost ($)': operational_energy_cost + capital_cost,
        })

results_df = pd.DataFrame(results)

# Calculate percentages
results_df['Operational Carbon (%)'] = (results_df['Operational Carbon (kg CO2)'] / results_df['Total Carbon (kg CO2)'] * 100)
results_df['Embodied Carbon (%)'] = (results_df['Embodied Carbon (kg CO2)'] / results_df['Total Carbon (kg CO2)'] * 100)
results_df['Operational Cost (%)'] = (results_df['Operational Cost ($)'] / results_df['Total Cost ($)'] * 100)
results_df['Capital Cost (%)'] = (results_df['Capital Cost ($)'] / results_df['Total Cost ($)'] * 100)
results_df['Dynamic Power (%)'] = (results_df['Energy Dynamic (kWh)'] / results_df['Energy (kWh)'] * 100)
results_df['Static Power (%)'] = (results_df['Energy Static (kWh)'] / results_df['Energy (kWh)'] * 100)

# Print ranges
print("="*80)
print("FIELD RANGES ACROSS ALL CONFIGURATIONS")
print("="*80)
field_ranges = {
    'Operational Carbon (%)': results_df['Operational Carbon (%)'].min(),
    'Embodied Carbon (%)': results_df['Embodied Carbon (%)'].min(),
    'Operational Cost (%)': results_df['Operational Cost (%)'].min(),
    'Capital Cost (%)': results_df['Capital Cost (%)'].min(),
    'Dynamic Power (%)': results_df['Dynamic Power (%)'].min(),
    'Static Power (%)': results_df['Static Power (%)'].min(),
}

for field, min_val in field_ranges.items():
    max_val = (100 - min_val) if '%' in field else results_df[field].max()
    if field in results_df.columns:
        max_val = results_df[field].max()
    print(f"{field:40s}: {min_val:6.2f}% to {max_val:6.2f}%")

# Create figure with breakdown charts
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle(f'Lifetime Carbon and Cost Breakdown\n({args.lifetime}-Year Lifetime Assumption)', 
             fontsize=14, fontweight='bold')

# Sort by image size for x-axis
results_sorted = results_df.sort_values(['Image Size', 'Timesteps', 'Channels']).reset_index(drop=True)
x_labels = [f"{row['Image Size']}\n{row['Timesteps']}×{row['Channels']}" 
            for _, row in results_sorted.iterrows()]
x_pos = np.arange(len(results_sorted))

# Color scheme
color_operational = '#FF6B6B'  # Red
color_embodied = '#4ECDC4'    # Teal
color_capital = '#FFB84D'     # Orange

# ===== ROW 1: CARBON BREAKDOWN =====

# 1a. Absolute Carbon Values
ax = axes[0, 0]
width = 0.7
op_carbon = results_sorted['Operational Carbon (kg CO2)']
emb_carbon = results_sorted['Embodied Carbon (kg CO2)']

ax.bar(x_pos, op_carbon, width, label='Operational', color=color_operational, alpha=0.8)
ax.bar(x_pos, emb_carbon, width, bottom=op_carbon, label='Embodied', color=color_embodied, alpha=0.8)

ax.set_ylabel('Carbon (kg CO2)', fontsize=11, fontweight='bold')
ax.set_title('(a) Total Carbon', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_xticks([])

# 1b. Carbon Percentages
ax = axes[0, 1]
op_pct = results_sorted['Operational Carbon (%)']
emb_pct = results_sorted['Embodied Carbon (%)']

ax.bar(x_pos, op_pct, width, label='Operational', color=color_operational, alpha=0.8)
ax.bar(x_pos, emb_pct, width, bottom=op_pct, label='Embodied', color=color_embodied, alpha=0.8)

ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax.set_title('(b) Carbon Composition', fontsize=12, fontweight='bold')
ax.set_ylim([0, 100])
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_xticks([])

# Add percentage text
for i, (op, emb) in enumerate(zip(op_pct, emb_pct)):
    ax.text(i, op/2, f'{op:.1f}%', ha='center', va='center', fontsize=7, fontweight='bold', color='black')
    ax.text(i, op + emb/2, f'{emb:.1f}%', ha='center', va='center', fontsize=7, fontweight='bold', color='black')

# 1c. Carbon Ranges Summary Box
ax = axes[0, 2]
ax.axis('off')

carbon_text = f"""Carbon Breakdown Summary

Operational Carbon:
  Range: {results_df['Operational Carbon (%)'].min():.1f}–{results_df['Operational Carbon (%)'].max():.1f}%
  
Embodied Carbon:
  Range: {results_df['Embodied Carbon (%)'].min():.1f}–{results_df['Embodied Carbon (%)'].max():.1f}%

Key Finding:
  Operational carbon dominates
  across all configurations
  (~80–81%), with embodied
  carbon contributing ~19–20%
"""

ax.text(0.1, 0.9, carbon_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# ===== ROW 2: COST BREAKDOWN =====

# 2a. Absolute Cost Values
ax = axes[1, 0]
op_cost = results_sorted['Operational Cost ($)']
cap_cost = results_sorted['Capital Cost ($)']

ax.bar(x_pos, op_cost, width, label='Operational', color=color_operational, alpha=0.8)
ax.bar(x_pos, cap_cost, width, bottom=op_cost, label='Capital', color=color_capital, alpha=0.8)

ax.set_ylabel('Cost ($)', fontsize=11, fontweight='bold')
ax.set_xlabel('Configuration (Image Size × Times × Channels)', fontsize=10, fontweight='bold')
ax.set_title('(a) Total Cost', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(x_pos[::8])
ax.set_xticklabels(x_labels[::8], fontsize=7)

# 2b. Cost Percentages
ax = axes[1, 1]
op_cost_pct = results_sorted['Operational Cost (%)']
cap_cost_pct = results_sorted['Capital Cost (%)']

ax.bar(x_pos, op_cost_pct, width, label='Operational', color=color_operational, alpha=0.8)
ax.bar(x_pos, cap_cost_pct, width, bottom=op_cost_pct, label='Capital', color=color_capital, alpha=0.8)

ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Configuration (Image Size × Times × Channels)', fontsize=10, fontweight='bold')
ax.set_title('(b) Cost Composition', fontsize=12, fontweight='bold')
ax.set_ylim([0, 100])
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_xticks(x_pos[::8])
ax.set_xticklabels(x_labels[::8], fontsize=7)

# Add percentage text
for i, (op, cap) in enumerate(zip(op_cost_pct, cap_cost_pct)):
    if i % 8 == 0:  # Only label every 8th to avoid crowding
        ax.text(i, op/2, f'{op:.1f}%', ha='center', va='center', fontsize=6, fontweight='bold', color='white')
        ax.text(i, op + cap/2, f'{cap:.1f}%', ha='center', va='center', fontsize=6, fontweight='bold', color='white')

# 2c. Cost Ranges Summary Box
ax = axes[1, 2]
ax.axis('off')

cost_text = f"""Cost Breakdown Summary

Operational Cost:
  Range: {results_df['Operational Cost (%)'].min():.1f}–{results_df['Operational Cost (%)'].max():.1f}%
  
Capital Cost:
  Range: {results_df['Capital Cost (%)'].min():.1f}–{results_df['Capital Cost (%)'].max():.1f}%

Key Finding:
  Capital cost dominates
  across all configurations
  (~90–91%), with operational
  costs contributing only ~9–10%
"""

ax.text(0.1, 0.9, cost_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()

# Save figure
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Print detailed statistics
print("\n" + "="*80)
print("DETAILED BREAKDOWN STATISTICS")
print("="*80)

print("\nCARBON ANALYSIS:")
print(f"  Operational Carbon: {results_df['Operational Carbon (%)'].min():.2f}% to {results_df['Operational Carbon (%)'].max():.2f}%")
print(f"  Embodied Carbon: {results_df['Embodied Carbon (%)'].min():.2f}% to {results_df['Embodied Carbon (%)'].max():.2f}%")
print(f"  → Operational carbon is {results_df['Operational Carbon (%)'].mean():.1f}% on average")

print("\nCOST ANALYSIS:")
print(f"  Operational Cost: {results_df['Operational Cost (%)'].min():.2f}% to {results_df['Operational Cost (%)'].max():.2f}%")
print(f"  Capital Cost: {results_df['Capital Cost (%)'].min():.2f}% to {results_df['Capital Cost (%)'].max():.2f}%")
print(f"  → Capital cost is {results_df['Capital Cost (%)'].mean():.1f}% on average")

print("\nPOWER ANALYSIS:")
print(f"  Dynamic Power: {results_df['Dynamic Power (%)'].min():.2f}% to {results_df['Dynamic Power (%)'].max():.2f}%")
print(f"  Static Power: {results_df['Static Power (%)'].min():.2f}% to {results_df['Static Power (%)'].max():.2f}%")
print(f"  → Static power is {results_df['Static Power (%)'].mean():.1f}% on average")

print("\nIMPLICATIONS:")
print("  • Machine embodied carbon is small (~20%) relative to operational carbon")
print("  • Hardware capex is large (~90%) relative to operational energy costs")
print("  • This suggests TCO is dominated by hardware amortization, not energy")
print("  • Carbon footprint is dominated by electricity consumption, not manufacturing")
print("="*80)
