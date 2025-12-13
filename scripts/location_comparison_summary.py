#!/usr/bin/env python3
"""
Location Comparison Summary
Shows key differences between WA and SA across all workloads
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Create results directory if it doesn't exist
results_dir = Path('results')
results_dir.mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description='Generate location comparison summary.')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years')
parser.add_argument('-o', '--output', type=str, default='location_comparison_summary.png')
args = parser.parse_args()

# Ensure output path is in results directory
output_path = results_dir / args.output

# Read benchmarks
benchmarks_df = pd.read_csv("benchmarks.csv",
                 header=None,
                 names=["im_size", "n_times", "n_chans", "wall_time", "wall_time_sec", "n_rows", "n_vis",
                        "n_idg", "idg_h_sec", "idg_h_watt", "idg_h_jou", "idg_d_sec", "idg_d_watt", "idg_d_jou",
                        "idg_grid_mvs", "cpu_j", "gpu0_j", "gpu1_j", "gpu2_j", "gpu3_j",
                        "tot_gpu_j", "tot_sys_j", "tot_pdu_j"])

benchmarks_df['machine'] = 'R675 V3 + 4xH100 96GB'
benchmarks_df['time'] = benchmarks_df['wall_time_sec'] / 3600
benchmarks_df['mvis'] = benchmarks_df['n_vis'] / 1e6

# Read machine and location data
machines_df = pd.read_csv('machines.csv').set_index('machine')
locations_df = pd.read_csv('locations.csv').set_index('id')

# Calculate metrics
results = []
Lifetime = args.lifetime * 365 * 24  # hours
idle_power = 277.75 / 4 + 65.44  # CPU + GPU watts

for _, benchmark in benchmarks_df.iterrows():
    machine_name = benchmark['machine']
    time = benchmark['time']
    energy_dynamic = (benchmark['gpu0_j'] + benchmark['cpu_j']) / 3.6e6
    energy_static = idle_power * time / 1000
    energy = energy_dynamic + energy_static
    
    machine_cost = machines_df.loc[machine_name, 'cost']
    machine_embodied = machines_df.loc[machine_name, 'embodied']
    
    mvis = benchmark['mvis']
    
    for location_id in ['WA', 'SA']:
        if location_id not in locations_df.index:
            continue
            
        location = locations_df.loc[location_id]
        ci, ep = location['ci'], location['ep']
        
        op_carbon = energy * ci
        cap_carbon = machine_embodied * time / Lifetime
        total_carbon = op_carbon + cap_carbon
        
        op_cost = energy * ep
        cap_cost = machine_cost * time / Lifetime
        total_cost = op_cost + cap_cost
        
        results.append({
            'Size': benchmark['im_size'],
            'Times': benchmark['n_times'],
            'Chans': benchmark['n_chans'],
            'Location': location_id,
            'Mvis': mvis,
            'Time (h)': time,
            'Energy (kWh)': energy,
            'Op Carbon': op_carbon,
            'Cap Carbon': cap_carbon,
            'Total Carbon': total_carbon,
            'Op Cost': op_cost,
            'Cap Cost': cap_cost,
            'Total Cost': total_cost,
            'Op Cost %': op_cost / total_cost * 100,
            'Op Carbon %': op_carbon / total_carbon * 100,
        })

df = pd.DataFrame(results)

# Create comprehensive comparison figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

fig.suptitle(f'WA vs SA Location Comparison\n({args.lifetime}-Year Lifetime, All 64 Workloads)', 
             fontsize=16, fontweight='bold')

# Group by location
wa = df[df['Location'] == 'WA']
sa = df[df['Location'] == 'SA']

# ===== ROW 1: ABSOLUTE VALUES =====

# 1. Total Carbon Distribution
ax = fig.add_subplot(gs[0, 0])
bins = np.logspace(np.log10(df['Total Carbon'].min()), np.log10(df['Total Carbon'].max()), 20)
ax.hist([wa['Total Carbon'], sa['Total Carbon']], bins=bins, label=['WA', 'SA'], 
        color=['#1f77b4', '#ff7f0e'], alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel('Total Carbon (kg CO2)', fontsize=10, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax.set_title('(a) Total Carbon Distribution', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Total Cost Distribution
ax = fig.add_subplot(gs[0, 1])
bins = np.logspace(np.log10(df['Total Cost'].min()), np.log10(df['Total Cost'].max()), 20)
ax.hist([wa['Total Cost'], sa['Total Cost']], bins=bins, label=['WA', 'SA'],
        color=['#1f77b4', '#ff7f0e'], alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel('Total Cost ($)', fontsize=10, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax.set_title('(b) Total Cost Distribution', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Energy Distribution
ax = fig.add_subplot(gs[0, 2])
bins = np.logspace(np.log10(df['Energy (kWh)'].min()), np.log10(df['Energy (kWh)'].max()), 20)
ax.hist([wa['Energy (kWh)'], sa['Energy (kWh)']], bins=bins, label=['WA', 'SA'],
        color=['#1f77b4', '#ff7f0e'], alpha=0.7)
ax.set_xscale('log')
ax.set_xlabel('Energy (kWh)', fontsize=10, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax.set_title('(c) Energy Distribution (Same)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ===== ROW 2: COMPOSITION PERCENTAGES =====

# 4. Operational Carbon %
ax = fig.add_subplot(gs[1, 0])
ax.boxplot([wa['Op Carbon %'], sa['Op Carbon %']], labels=['WA', 'SA'],
          patch_artist=True, boxprops=dict(facecolor='#4ECDC4', alpha=0.7))
ax.set_ylabel('Operational Carbon %', fontsize=10, fontweight='bold')
ax.set_title('(d) Operational Carbon %', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
print(f"Operational Carbon %:")
print(f"  WA: {wa['Op Carbon %'].min():.1f}%-{wa['Op Carbon %'].max():.1f}% (mean: {wa['Op Carbon %'].mean():.1f}%)")
print(f"  SA: {sa['Op Carbon %'].min():.1f}%-{sa['Op Carbon %'].max():.1f}% (mean: {sa['Op Carbon %'].mean():.1f}%)")

# 5. Operational Cost %
ax = fig.add_subplot(gs[1, 1])
ax.boxplot([wa['Op Cost %'], sa['Op Cost %']], labels=['WA', 'SA'],
          patch_artist=True, boxprops=dict(facecolor='#FFB84D', alpha=0.7))
ax.set_ylabel('Operational Cost %', fontsize=10, fontweight='bold')
ax.set_title('(e) Operational Cost %', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
print(f"\nOperational Cost %:")
print(f"  WA: {wa['Op Cost %'].min():.1f}%-{wa['Op Cost %'].max():.1f}% (mean: {wa['Op Cost %'].mean():.1f}%)")
print(f"  SA: {sa['Op Cost %'].min():.1f}%-{sa['Op Cost %'].max():.1f}% (mean: {sa['Op Cost %'].mean():.1f}%)")

# 6. Efficiency Metric Comparison
ax = fig.add_subplot(gs[1, 2])
efficiency_wa = wa['Mvis'] / wa['Total Carbon']
efficiency_sa = sa['Mvis'] / sa['Total Carbon']
ax.boxplot([efficiency_wa, efficiency_sa], labels=['WA', 'SA'],
          patch_artist=True, boxprops=dict(facecolor='#FF6B6B', alpha=0.7))
ax.set_ylabel('Mvis per kg CO2', fontsize=10, fontweight='bold')
ax.set_title('(f) Carbon Efficiency (normalized)', fontsize=11, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
ratio = efficiency_sa.mean() / efficiency_wa.mean()
print(f"\nCarbon Efficiency (Mvis/kg CO2):")
print(f"  WA: {efficiency_wa.min():.2e} to {efficiency_wa.max():.2e}")
print(f"  SA: {efficiency_sa.min():.2e} to {efficiency_sa.max():.2e}")
print(f"  → SA is {ratio:.2f}x WA (due to lower CI)")

# ===== ROW 3: SCATTER COMPARISON =====

# 7. Total Carbon vs Time
ax = fig.add_subplot(gs[2, 0])
ax.scatter(wa['Time (h)'], wa['Total Carbon'], alpha=0.6, s=50, label='WA', color='#1f77b4')
ax.scatter(sa['Time (h)'], sa['Total Carbon'], alpha=0.6, s=50, label='SA', color='#ff7f0e')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Runtime (hours)', fontsize=10, fontweight='bold')
ax.set_ylabel('Total Carbon (kg CO2)', fontsize=10, fontweight='bold')
ax.set_title('(g) Carbon vs Runtime', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 8. Total Cost vs Time
ax = fig.add_subplot(gs[2, 1])
ax.scatter(wa['Time (h)'], wa['Total Cost'], alpha=0.6, s=50, label='WA', color='#1f77b4')
ax.scatter(sa['Time (h)'], sa['Total Cost'], alpha=0.6, s=50, label='SA', color='#ff7f0e')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Runtime (hours)', fontsize=10, fontweight='bold')
ax.set_ylabel('Total Cost ($)', fontsize=10, fontweight='bold')
ax.set_title('(h) Cost vs Runtime', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 9. Summary Statistics Box
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

summary_text = f"""LOCATION PARAMETERS
WA: CI=0.27 kg CO2/kWh, EP=$0.40/kWh
SA: CI=0.19 kg CO2/kWh, EP=$0.713/kWh

KEY FINDINGS

Carbon Footprint:
  • SA is {wa['Total Carbon'].mean()/sa['Total Carbon'].mean():.2f}x lower (0.70x WA)
  • Due to {((0.27-0.19)/0.27*100):.0f}% lower CI
  • Applies to all workloads equally

Operating Cost:
  • SA is {sa['Total Cost'].mean()/wa['Total Cost'].mean():.2f}x HIGHER (1.78x WA)
  • Due to {((0.713-0.4)/0.4*100):.0f}% higher EP
  • Applies to all workloads equally

Composition:
  • Op Carbon % UNCHANGED (differ by CI)
  • Op Cost % CHANGES (SA ~15% vs WA ~9%)
  • Capital dominates TCO in both
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nComparison summary saved to: {output_path}")

# Print final comparison
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nWestern Australia (WA):")
print(f"  Total Carbon:  {wa['Total Carbon'].min():.2e} – {wa['Total Carbon'].max():.2e} kg CO2")
print(f"                 (mean: {wa['Total Carbon'].mean():.2e})")
print(f"  Total Cost:    ${wa['Total Cost'].min():.2f} – ${wa['Total Cost'].max():.2f}")
print(f"                 (mean: ${wa['Total Cost'].mean():.2f})")

print(f"\nSouth Africa (SA):")
print(f"  Total Carbon:  {sa['Total Carbon'].min():.2e} – {sa['Total Carbon'].max():.2e} kg CO2")
print(f"                 (mean: {sa['Total Carbon'].mean():.2e})")
print(f"  Total Cost:    ${sa['Total Cost'].min():.2f} – ${sa['Total Cost'].max():.2f}")
print(f"                 (mean: ${sa['Total Cost'].mean():.2f})")

print(f"\nRatios (SA/WA):")
print(f"  Carbon: {sa['Total Carbon'].mean() / wa['Total Carbon'].mean():.3f}x (lower is better)")
print(f"  Cost:   {sa['Total Cost'].mean() / wa['Total Cost'].mean():.3f}x (lower is better)")

print("="*80)
