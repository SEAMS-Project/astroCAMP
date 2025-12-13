#!/usr/bin/env python3
"""
Generate a single figure showing the key takeaway:
Universal efficiency advantage for large, balanced workloads across carbon, cost, and throughput metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load benchmarks
data_path = Path(__file__).parent / "benchmarks.csv"
locations_path = Path(__file__).parent / "locations.csv"
benchmarks = pd.read_csv(
    data_path,
    header=None,
    names=["im_size", "n_times", "n_chans", "wall_time", "wall_time_sec", "n_rows", "n_vis",
           "n_idg",
           "idg_h_sec", "idg_h_watt", "idg_h_jou",
           "idg_d_sec", "idg_d_watt", "idg_d_jou",
           "idg_grid_mvs",
           "cpu_j",
           "gpu0_j", "gpu1_j", "gpu2_j", "gpu3_j",
           "tot_gpu_j", "tot_sys_j", "tot_pdu_j"]
)

# Calculate metrics for all configurations
benchmarks['mvis'] = benchmarks['n_vis'] / 1e6

# Energy and cost calculations
benchmarks['tot_sys_kj'] = benchmarks['tot_sys_j'] / 1000
benchmarks['cpu_kj'] = benchmarks['cpu_j'] / 1000
benchmarks['tot_gpu_kj'] = benchmarks['tot_gpu_j'] / 1000
benchmarks['idg_h_kj'] = benchmarks['idg_h_jou'] / 1000
benchmarks['idg_d_kj'] = benchmarks['idg_d_jou'] / 1000
benchmarks['pdu_kj'] = benchmarks['tot_pdu_j'] / 1000

benchmarks['carbon_kgco2'] = benchmarks['tot_sys_j'] / 1000 / 612  # 612 J/kgCO2
benchmarks['cost_usd'] = benchmarks['pdu_kj'] * 0.15 / 3600 + 0.0001  # ~$0.15/kWh
benchmarks['carbon_efficiency'] = benchmarks['mvis'] / benchmarks['carbon_kgco2']
benchmarks['throughput'] = benchmarks['mvis'] / benchmarks['wall_time_sec']
benchmarks['cost_efficiency'] = benchmarks['mvis'] / benchmarks['cost_usd']

# Filter to largest image only
largest_im = benchmarks['im_size'].max()
data = benchmarks[benchmarks['im_size'] == largest_im].copy()

# Load location parameters (South Africa, Western Australia)
locations_df = pd.read_csv(locations_path)
locations_df = locations_df[locations_df['id'].isin(['SA', 'WA'])]
ci_map = dict(zip(locations_df['id'], locations_df['ci']))  # kgCO2/kWh
ep_map = dict(zip(locations_df['id'], locations_df['ep']))  # $/kWh

# Group by (n_times, n_chans) and calculate mean metrics
grouped = data.groupby(['n_times', 'n_chans']).agg({
    'carbon_efficiency': 'mean',
    'throughput': 'mean',
    'cost_efficiency': 'mean',
    'wall_time_sec': 'mean',
    'mvis': 'mean',
    'pdu_kj': 'mean'
}).reset_index()

# Convert to integers
grouped['n_times'] = grouped['n_times'].astype(int)
grouped['n_chans'] = grouped['n_chans'].astype(int)

# Normalize metrics for visualization (0-1 scale for fair comparison)
carbon_min, carbon_max = grouped['carbon_efficiency'].min(), grouped['carbon_efficiency'].max()
throughput_min, throughput_max = grouped['throughput'].min(), grouped['throughput'].max()
cost_min, cost_max = grouped['cost_efficiency'].min(), grouped['cost_efficiency'].max()

grouped['carbon_norm'] = (grouped['carbon_efficiency'] - carbon_min) / (carbon_max - carbon_min)
grouped['throughput_norm'] = (grouped['throughput'] - throughput_min) / (throughput_max - throughput_min)
grouped['cost_norm'] = (grouped['cost_efficiency'] - cost_min) / (cost_max - cost_min)

# Calculate composite score (average of normalized metrics)
grouped['composite'] = (grouped['carbon_norm'] + grouped['throughput_norm'] + grouped['cost_norm']) / 3

# Location-specific carbon and cost efficiencies
energy_kwh = grouped['pdu_kj'] / 3600.0  # kWh
grouped['carbon_eff_sa'] = grouped['mvis'] / (energy_kwh * ci_map.get('SA', 1.0))
grouped['carbon_eff_wa'] = grouped['mvis'] / (energy_kwh * ci_map.get('WA', 1.0))
grouped['cost_eff_sa'] = grouped['mvis'] / (energy_kwh * ep_map.get('SA', 1.0))
grouped['cost_eff_wa'] = grouped['mvis'] / (energy_kwh * ep_map.get('WA', 1.0))

print("\n" + "="*80)
print(f"KEY TAKEAWAY VISUALIZATION - Image {largest_im}×{largest_im}")
print("="*80)
print("\nComposite Efficiency Score (0=worst, 1=best):")
print(grouped[['n_times', 'n_chans', 'composite', 'carbon_efficiency', 'throughput', 'cost_efficiency']].to_string(index=False))

# Create figure with four subplots
fig = plt.figure(figsize=(16, 10.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 0.85], hspace=0.35, wspace=0.20)

# Color map for bars
def get_color(value):
    if value < 0.25:
        return '#d73027'  # Red
    elif value < 0.5:
        return '#fee08b'  # Yellow
    elif value < 0.75:
        return '#91bfdb'  # Light blue
    else:
        return '#1a9850'  # Green

# Shared y-limits for consistency
carbon_ylim = (0, max(grouped['carbon_eff_sa'].max(), grouped['carbon_eff_wa'].max()) * 1.1)
throughput_ylim = (0, throughput_max * 1.1)
cost_ylim = (0, max(grouped['cost_eff_sa'].max(), grouped['cost_eff_wa'].max()) * 1.1)
carbon_offset = carbon_ylim[1] * 0.012
throughput_offset = throughput_ylim[1] * 0.012
cost_offset = cost_ylim[1] * 0.012

# Plot 1: Carbon Efficiency
ax1 = fig.add_subplot(gs[0, 0])
configs = [f"{int(r['n_times'])},{int(r['n_chans'])}" for _, r in grouped.iterrows()]
x = np.arange(len(grouped))
width = 0.4
bars1_sa = ax1.bar(x - width/2, grouped['carbon_eff_sa'], width, label='South Africa', color='#1a9850', edgecolor='black', linewidth=0.7)
bars1_wa = ax1.bar(x + width/2, grouped['carbon_eff_wa'], width, label='Western Australia', color='#91bfdb', edgecolor='black', linewidth=0.7)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=60, ha='right', fontsize=12)
ax1.set_ylabel('Mvis/kgCO₂', fontsize=12, fontweight='bold')
ax1.set_title('Carbon Efficiency by Location', fontsize=13, fontweight='bold', pad=6)
ax1.set_ylim(carbon_ylim)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.tick_params(axis='y', labelsize=11)
ax1.legend(fontsize=10, frameon=True)
# Add value labels on bars (exact, rotated for readability)
for bars in (bars1_sa, bars1_wa):
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + carbon_offset,
             f'{height:,.0f}', ha='center', va='bottom', fontsize=9, rotation=90)

# Plot 2: Throughput
ax2 = fig.add_subplot(gs[0, 1])
colors_throughput = [get_color(x) for x in grouped['throughput_norm']]
bars2 = ax2.bar(range(len(grouped)), grouped['throughput'], color=colors_throughput, edgecolor='black', linewidth=0.8)
ax2.set_xticks(range(len(grouped)))
ax2.set_xticklabels(configs, rotation=60, ha='right', fontsize=12)
ax2.set_ylabel('Mvis/s', fontsize=12, fontweight='bold')
ax2.set_title('Throughput', fontsize=13, fontweight='bold', pad=6)
ax2.set_ylim(throughput_ylim)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.tick_params(axis='y', labelsize=11)
# Add value labels on bars (exact, rotated)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + throughput_offset,
             f'{height:.2f}', ha='center', va='bottom', fontsize=9, rotation=90)

# Plot 3: Cost Efficiency
ax3 = fig.add_subplot(gs[1, 0])
bars3_sa = ax3.bar(x - width/2, grouped['cost_eff_sa'], width, label='South Africa', color='#1a9850', edgecolor='black', linewidth=0.7)
bars3_wa = ax3.bar(x + width/2, grouped['cost_eff_wa'], width, label='Western Australia', color='#91bfdb', edgecolor='black', linewidth=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(configs, rotation=60, ha='right', fontsize=12)
ax3.set_ylabel('Mvis/$', fontsize=12, fontweight='bold')
ax3.set_title('Cost Efficiency by Location', fontsize=13, fontweight='bold', pad=6)
ax3.set_ylim(cost_ylim)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.tick_params(axis='y', labelsize=11)
ax3.legend(fontsize=10, frameon=True)
# Add value labels on bars (exact, rotated)
for bars in (bars3_sa, bars3_wa):
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + cost_offset,
             f'{height:,.0f}', ha='center', va='bottom', fontsize=9, rotation=90)

# Plot 4: Composite Score (Radar-style bar)
ax4 = fig.add_subplot(gs[1, 1])
colors_composite = [get_color(x) for x in grouped['composite']]
bars4 = ax4.barh(range(len(grouped)), grouped['composite'], color=colors_composite, edgecolor='black', linewidth=0.8)
ax4.set_yticks(range(len(grouped)))
ax4.set_yticklabels(configs, fontsize=12)
ax4.set_xlabel('Overall Efficiency Score (0=worst, 1=best)', fontsize=12, fontweight='bold')
ax4.set_title('Composite Score (Carbon + Throughput + Cost)', fontsize=13, fontweight='bold', pad=6)
ax4.set_xlim(0, 1.0)
ax4.grid(axis='x', alpha=0.3, linestyle='--')
ax4.tick_params(axis='x', labelsize=11)
# Add value labels on bars
for i, bar in enumerate(bars4):
    width = bar.get_width()
    ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
             f'{width:.2f}', ha='left', va='center', fontsize=9, fontweight='bold')

# Add overall title and annotation
fig.suptitle(f'Key Takeaway: Large Balanced Workloads Maximize Efficiency | Image Size {largest_im}×{largest_im} | Config: (n_times, n_chans)',
             fontsize=15, fontweight='bold', y=0.965)

# Add color legend at bottom
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d73027', edgecolor='black', label='Red (0-25%): Poor efficiency'),
    Patch(facecolor='#fee08b', edgecolor='black', label='Yellow (25-50%): Moderate'),
    Patch(facecolor='#91bfdb', edgecolor='black', label='Light Blue (50-75%): Good'),
    Patch(facecolor='#1a9850', edgecolor='black', label='Green (75-100%): Excellent'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
          bbox_to_anchor=(0.5, -0.01), frameon=True, framealpha=0.95)

plt.tight_layout(rect=[0, 0.08, 1, 0.965])

# Save figure
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "key_takeaway_efficiency.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Figure saved: {output_path}")

# Print summary statistics
print("\n" + "="*80)
print("EFFICIENCY GAIN ANALYSIS")
print("="*80)

# Best vs Worst
best_idx = grouped['composite'].idxmax()
worst_idx = grouped['composite'].idxmin()
best_row = grouped.loc[best_idx]
worst_row = grouped.loc[worst_idx]

print(f"\nBest Configuration: n_times={int(best_row['n_times'])}, n_chans={int(best_row['n_chans'])}")
print(f"  → Carbon Efficiency: {best_row['carbon_efficiency']:,.0f} Mvis/kgCO2")
print(f"  → Throughput: {best_row['throughput']:.2f} Mvis/s")
print(f"  → Cost Efficiency: {best_row['cost_efficiency']:,.0f} Mvis/$")
print(f"  → Composite Score: {best_row['composite']:.3f}/1.0")

print(f"\nWorst Configuration: n_times={int(worst_row['n_times'])}, n_chans={int(worst_row['n_chans'])}")
print(f"  → Carbon Efficiency: {worst_row['carbon_efficiency']:,.0f} Mvis/kgCO2")
print(f"  → Throughput: {worst_row['throughput']:.2f} Mvis/s")
print(f"  → Cost Efficiency: {worst_row['cost_efficiency']:,.0f} Mvis/$")
print(f"  → Composite Score: {worst_row['composite']:.3f}/1.0")

# Calculate gain factors
carbon_gain = best_row['carbon_efficiency'] / worst_row['carbon_efficiency']
throughput_gain = best_row['throughput'] / worst_row['throughput']
cost_gain = best_row['cost_efficiency'] / worst_row['cost_efficiency']
composite_gain = best_row['composite'] / worst_row['composite']

print(f"\nEfficiency Gain (Best / Worst):")
print(f"  → Carbon: {carbon_gain:.0f}× improvement")
print(f"  → Throughput: {throughput_gain:.0f}× improvement")
print(f"  → Cost: {cost_gain:.0f}× improvement")
print(f"  → Composite: {composite_gain:.2f}× improvement")

# Channel vs Time scaling
config_1_1 = grouped[(grouped['n_times'] == 1) & (grouped['n_chans'] == 1)].iloc[0]
config_128_1 = grouped[(grouped['n_times'] == 128) & (grouped['n_chans'] == 1)].iloc[0]
config_1_128 = grouped[(grouped['n_times'] == 1) & (grouped['n_chans'] == 128)].iloc[0]

time_gain = config_128_1['carbon_efficiency'] / config_1_1['carbon_efficiency']
channel_gain = config_1_128['carbon_efficiency'] / config_1_1['carbon_efficiency']

print(f"\nScaling Analysis (Carbon Efficiency):")
print(f"  → Time scaling (1→128 times, 1 channel): {time_gain:.0f}× improvement")
print(f"  → Channel scaling (1 time, 1→128 channels): {channel_gain:.0f}× improvement")
print(f"  → Channel dominance: Channels provide {channel_gain/time_gain:.1f}× more benefit than times")

print("\n" + "="*80)
