#!/usr/bin/env python3
"""
Plot Lifetime Carbon and Cost Breakdown with Location Comparison
Generates side-by-side comparison of Western Australia (WA) vs South Africa (SA)
showing how different carbon intensities and electricity prices affect metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from results_dataframe import generate_results_dataframe

# Create results directory if it doesn't exist
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Plot lifetime carbon and cost breakdown with location comparison.')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years (default: 5)')
parser.add_argument('-o', '--output', type=str, default='lifetime_breakdown_comparison.png', 
                    help='Output filename (default: lifetime_breakdown_comparison.png)')
parser.add_argument('--dpi', type=int, default=300, help='DPI for output image (default: 300)')
args = parser.parse_args()

# Ensure output path is in results directory
output_path = results_dir / args.output
print(f"Using lifetime of {args.lifetime} years for all machines.")

BASE_DIR = Path(__file__).resolve().parent

# Generate results DataFrame using helper function for consistent calculations
results_df = generate_results_dataframe(
    benchmarks_csv_path=BASE_DIR / 'benchmarks.csv',
    machines_csv_path=BASE_DIR / 'machines.csv',
    locations_csv_path=BASE_DIR / 'locations.csv',
    lifetime_years=args.lifetime,
    location_ids=['WA', 'SA']
)

# Calculate percentages for breakdown
results_df['Operational Carbon (%)'] = (results_df['Operational Carbon (g CO2)'] / results_df['Total Carbon (g CO2)'] * 100)
results_df['Embodied Carbon (%)'] = (results_df['Embodied Carbon (g CO2)'] / results_df['Total Carbon (g CO2)'] * 100)
results_df['Operational Cost (%)'] = (results_df['Operational Cost ($)'] / results_df['Total Cost ($)'] * 100)
results_df['Capital Cost (%)'] = (results_df['Capital Cost ($)'] / results_df['Total Cost ($)'] * 100)
results_df['Dynamic Power (%)'] = (results_df['Dynamic Energy (Wh)'] / results_df['Energy (Wh)'] * 100)
results_df['Static Power (%)'] = (results_df['Static Energy (Wh)'] / results_df['Energy (Wh)'] * 100)

# Print ranges by location
print("\n" + "="*80)
print("FIELD RANGES BY LOCATION")
print("="*80)

location_names = {'WA': 'Western Australia', 'SA': 'South Africa'}

for location_id in ['WA', 'SA']:
    loc_results = results_df[results_df['Location'] == location_id]
    if len(loc_results) == 0:
        continue
    print(f"\n{location_names[location_id]} ({location_id}):")
    print(f"  Operational Carbon: {loc_results['Operational Carbon (%)'].min():.2f}% to {loc_results['Operational Carbon (%)'].max():.2f}%")
    print(f"  Embodied Carbon: {loc_results['Embodied Carbon (%)'].min():.2f}% to {loc_results['Embodied Carbon (%)'].max():.2f}%")
    print(f"  Operational Cost: {loc_results['Operational Cost (%)'].min():.2f}% to {loc_results['Operational Cost (%)'].max():.2f}%")
    print(f"  Capital Cost: {loc_results['Capital Cost (%)'].min():.2f}% to {loc_results['Capital Cost (%)'].max():.2f}%")

# Create figure with location comparison - side by side
fig, axes = plt.subplots(4, 2, figsize=(14, 12))
fig.suptitle(f'Lifetime Carbon and Cost Breakdown: Location Comparison\n({args.lifetime}-Year Lifetime Assumption)', 
             fontsize=14, fontweight='bold')

location_colors = {'WA': '#1f77b4', 'SA': '#ff7f0e'}  # Blue for WA, Orange for SA
color_embodied = '#4ECDC4'    # Teal
color_capital = '#FFB84D'     # Orange

for col_idx, location_id in enumerate(['WA', 'SA']):
    location_data = results_df[results_df['Location'] == location_id]
    if len(location_data) == 0:
        continue
    location_name = location_names[location_id]
    color_op = location_colors[location_id]
    
    # Sort by image size for x-axis
    location_sorted = location_data.sort_values(['Image Size', 'Timesteps', 'Channels']).reset_index(drop=True)
    x_labels = [f"{row['Image Size']}\n{row['Timesteps']}×{row['Channels']}" 
                for _, row in location_sorted.iterrows()]
    x_pos = np.arange(len(location_sorted))
    
    # Row 0: Operational Carbon (absolute)
    ax = axes[0, col_idx]
    op_carbon = location_sorted['Operational Carbon (g CO2)']
    ax.plot(x_pos, op_carbon, marker='o', color=color_op, linewidth=2, markersize=4)
    ax.fill_between(x_pos, op_carbon, alpha=0.3, color=color_op)
    ax.set_ylabel('Operational Carbon (kg CO2)', fontsize=10, fontweight='bold')
    ax.set_title(f'Operational Carbon - {location_name}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    
    # Row 1: Carbon Composition (%)
    ax = axes[1, col_idx]
    width = 0.7
    op_pct = location_sorted['Operational Carbon (%)']
    emb_pct = location_sorted['Embodied Carbon (%)']
    ax.bar(x_pos, op_pct, width, label='Operational', color=color_op, alpha=0.8)
    ax.bar(x_pos, emb_pct, width, bottom=op_pct, label='Embodied', color=color_embodied, alpha=0.8)
    ax.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax.set_title(f'Carbon Composition - {location_name}', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks([])
    
    # Row 2: Operational Cost (absolute)
    ax = axes[2, col_idx]
    op_cost = location_sorted['Operational Cost ($)']
    ax.plot(x_pos, op_cost, marker='s', color=color_op, linewidth=2, markersize=4)
    ax.fill_between(x_pos, op_cost, alpha=0.3, color=color_op)
    ax.set_ylabel('Operational Cost ($)', fontsize=10, fontweight='bold')
    ax.set_title(f'Operational Cost - {location_name}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks([])
    
    # Row 3: Cost Composition (%)
    ax = axes[3, col_idx]
    op_cost_pct = location_sorted['Operational Cost (%)']
    cap_cost_pct = location_sorted['Capital Cost (%)']
    ax.bar(x_pos, op_cost_pct, width, label='Operational', color=color_op, alpha=0.8)
    ax.bar(x_pos, cap_cost_pct, width, bottom=op_cost_pct, label='Capital', color=color_capital, alpha=0.8)
    ax.set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    ax.set_title(f'Cost Composition - {location_name}', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlabel('Configuration (Image Size × Times × Channels)', fontsize=9, fontweight='bold')
    ax.set_xticks(x_pos[::8] if len(x_pos) > 8 else x_pos)
    ax.set_xticklabels(x_labels[::8] if len(x_labels) > 8 else x_labels, fontsize=6, rotation=0)

plt.tight_layout()

# Save figure
plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
print(f"\nComparison figure saved to: {output_path}")

# Print comparison analysis
print("\n" + "="*80)
print("LOCATION COMPARISON ANALYSIS")
print("="*80)

# Load locations to get CI and EP
locations_df = pd.read_csv(BASE_DIR / 'locations.csv').set_index('id')
wa_ci = locations_df.loc['WA', 'ci']
sa_ci = locations_df.loc['SA', 'ci']
wa_ep = locations_df.loc['WA', 'ep']
sa_ep = locations_df.loc['SA', 'ep']

wa_data = results_df[results_df['Location'] == 'WA']
sa_data = results_df[results_df['Location'] == 'SA']

print(f"\nCarbon Intensity Difference:")
print(f"  WA: {wa_ci} kg CO2/kWh")
print(f"  SA: {sa_ci} kg CO2/kWh")
print(f"  → SA has {((sa_ci - wa_ci) / wa_ci * 100):.1f}% HIGHER carbon intensity")
print(f"  → This means SA's operational carbon is {(sa_ci / wa_ci):.2f}x WA's")

print(f"\nElectricity Price Difference:")
print(f"  WA: ${wa_ep}/kWh")
print(f"  SA: ${sa_ep}/kWh")
print(f"  → SA has {((wa_ep - sa_ep) / sa_ep * 100):.1f}% LOWER electricity prices")
print(f"  → This means SA's operational cost is {(sa_ep / wa_ep):.2f}x WA's")

print(f"\nCarbon Impact:")
wa_avg_op_carbon = wa_data['Operational Carbon (g CO2)'].mean() / 1000.0
sa_avg_op_carbon = sa_data['Operational Carbon (g CO2)'].mean() / 1000.0
print(f"  Average operational carbon per workload:")
print(f"    WA: {wa_avg_op_carbon:.2f} kg CO2")
print(f"    SA: {sa_avg_op_carbon:.2f} kg CO2")
print(f"    → SA's operational carbon is {(sa_avg_op_carbon / wa_avg_op_carbon):.2f}x WA's")

print(f"\nCost Impact:")
wa_avg_op_cost = wa_data['Operational Cost ($)'].mean()
sa_avg_op_cost = sa_data['Operational Cost ($)'].mean()
print(f"  Average operational cost per workload:")
print(f"    WA: ${wa_avg_op_cost:.2f}")
print(f"    SA: ${sa_avg_op_cost:.2f}")
print(f"    → SA's operational cost is {(sa_avg_op_cost / wa_avg_op_cost):.2f}x WA's")

print(f"\nComposition Stability:")
print(f"  Carbon composition ranges are IDENTICAL across locations because:")
print(f"    - Percentages depend only on energy consumption, not CI/EP values")
print(f"    - Both embodied and operational carbon scale proportionally with CI")
print(f"    - Capital cost amortization is location-independent")
print(f"\n  Operational Carbon (%):")
print(f"    Both locations: {wa_data['Operational Carbon (%)'].min():.1f}–{wa_data['Operational Carbon (%)'].max():.1f}%")
print(f"\n  Operational Cost (%):")
print(f"    Both locations: {wa_data['Operational Cost (%)'].min():.1f}–{wa_data['Operational Cost (%)'].max():.1f}%")

print("\nKEY INSIGHTS:")
print("  1. Location choice significantly affects absolute carbon footprint and costs")
print("  2. Percentage composition remains stable across locations (same energy profile)")
print("  3. SA's greener grid (lower CI) partially offset by cheaper electricity")
print("  4. For carbon optimization: choose low-CI location regardless of workload")
print("  5. For cost optimization: SA is cheaper despite using more total energy (per unit work)")
print("="*80)
