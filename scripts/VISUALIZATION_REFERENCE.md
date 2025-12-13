# AstroCAMP Visualization Suite - Complete Reference

## Overview
This document provides a comprehensive guide to all visualization scripts and generated figures for the astroCAMP benchmark analysis project. All figures are optimized for conference/journal publication with large, readable fonts.

---

## New: Paper-Ready Visualization Suite (plot_paper_views.py)

A unified script generating **12 publication-ready figures** with descriptions and interpretations printed during execution. The suite includes traditional analysis plots (Pareto fronts, scatter plots, energy breakdowns) and a comprehensive set of **grouped bar plots** for all key metrics.

### 1. Pareto Front: Throughput vs Energy
**Output**: `pareto_throughput_energy.png`

**Description**: Log-log scatter plot showing the energy-throughput trade-off across all configurations. Color represents image size (viridis colormap); marker shape represents n_times bucket (○=small, □=medium, △=large). Pareto front traced in black.

**Figure Interpretation**:
- Points on the Pareto front represent non-dominated (optimal) configurations
- Larger image sizes and more timesteps generally shift toward better (lower energy, higher throughput) corners
- Clustered points at low throughput represent small workloads with inherent inefficiency
- Sweet spots: large image sizes (16384+) with balanced/channel-heavy configurations

**Font Sizes**: Title 14pt, axes 13pt, ticks 11pt, legend 10pt

---

### 2. Energy Breakdown: Static vs Dynamic
**Output**: `energy_breakdown_best_configs.png`

**Description**: Stacked bar chart showing static (idle) vs dynamic (active computation) energy for the best-throughput configuration per (image_size, n_times) pair. Static energy (light blue) stacked below dynamic (dark blue).

**Figure Interpretation**:
- Dynamic energy dominates for large workloads (>1 kWh), reflecting GPU compute intensity
- Static energy is non-negligible for small workloads (<0.1 kWh), suggesting inefficiency of tiny jobs
- Energy roughly scales with workload complexity (n_times × n_chans × im_size)
- Larger image sizes show steeper energy growth, but also better throughput-per-joule

**Font Sizes**: Title 14pt, axes 13pt, ticks 11pt, labels 10pt, legend 11pt

---

### 3. Energy Breakdown Grouped by n_times
**Output**: `energy_breakdown_grouped_by_n_times.png`

**Description**: Multi-panel (stacked) visualization grouped by timesteps. Each panel shows all configurations with that n_times value as stacked bars (static + dynamic). Carbon (g CO2) labeled above each bar. **Consistent y-axis across all panels enables direct visual comparison.**

**Figure Interpretation**:
- Larger n_times values show progressively higher energy and carbon footprints
- Within each group, image size and channels primarily determine energy
- The 15% headroom on y-axis accommodates carbon labels without clutter
- Grouped view reveals n_times as a primary scaling lever for both metrics

**Font Sizes**: Titles 12pt, axes 11pt, ticks 10pt, annotations 9pt bold, legend 10pt

---

### 4. Carbon vs Cost Trade-off
**Output**: `carbon_cost_scatter.png`

**Description**: Scatter plot of total carbon (g CO2) vs total cost ($) per run. Marker size represents workload (Mvis, clipped 30–180 points); color represents image size (plasma colormap). Pareto front traced in black.

**Figure Interpretation**:
- Small workloads (small marker size) cluster at low cost/carbon corners (< $1, < 100g CO2)
- Larger workloads spread across the plane, with cost and carbon scaling together
- Pareto front roughly diagonal: minimizing cost and carbon simultaneously is difficult
- Environmental and economic goals align: lower-cost configs tend to have lower carbon footprints
- Large image sizes (hot colors) dominate the Pareto front in the high-workload regime

**Font Sizes**: Title 14pt, axes 13pt, ticks 11pt, legend 10–11pt

---

### 5. Latency vs Throughput Trade-off
**Output**: `latency_throughput_tradeoff.png`

**Description**: Log-log scatter plot showing latency (wall_time_sec) vs throughput (Mvis/s). Color represents image size (viridis colormap); marker shape represents n_times bucket (○=small, □=medium, △=large). Pareto front traced in black minimizes latency while maximizing throughput.

**Figure Interpretation**:
- Ideal configurations appear in the upper-left quadrant (low latency, high throughput)
- Pareto front points balance speed vs compute intensity
- Larger workloads tend toward higher latency but better throughput efficiency
- Small n_times configurations cluster at low throughput regardless of latency
- Sweet spots: large image sizes with balanced/large n_times values

**Font Sizes**: Title 14pt, axes 14pt, ticks 12pt, legend 10–11pt

---

## Grouped Bar Plot Suite (Multi-Metric Analysis)

The following six figures use identical format and styling, enabling direct cross-metric comparison. Each figure shows multi-panel bar plots grouped by `n_times`, with consistent y-axis scaling across panels for visual comparison.

### Common Features Across All Grouped Bar Plots:
- **Layout**: Multi-panel vertical stack (one panel per n_times value: 1, 2, 8, 128)
- **X-axis**: Configurations labeled as "imXXXX\ncY" (image size + n_chans value)
- **Y-axis**: Consistent range across all panels (0 to max×1.15) for easy comparison
- **Color**: Image size encoded via viridis colormap (purple=4096, yellow=32768)
- **Annotations**: Metric values labeled above each bar
- **Legend**: Image size color key in upper-left of each panel
- **Font Sizes**: Titles 12pt, axes 12pt, ticks 11pt, labels 9pt bold

**How to Interpret These Figures**:
1. **Vertical comparison** (within panel): Shows how metric varies with n_chans for fixed n_times
2. **Horizontal comparison** (across panels): Shows how metric scales with increasing n_times
3. **Color patterns**: Reveals image size impact (yellow bars typically higher/lower depending on metric)
4. **Bar height trends**: Identifies parameter sensitivity and sweet spots

---

### 7. Latency Grouped by n_times
**Output**: `time-s_grouped_by_n_times.png`

**Caption**: *Latency (seconds) for all 64 benchmark configurations grouped by number of timesteps. Each panel shows configurations with fixed n_times, varying image size and channel count. Consistent y-axis enables direct comparison across timestep groups.*

**Metric Range**: 1.3s (smallest workload: im4096, n_times=1, n_chans=1) to 134.2s (largest: im32768, n_times=128, n_chans=128)

**Interpretation**:
- Latency increases monotonically with n_times (panels progress from ~5s max to ~140s max)
- Within each panel, larger image sizes (yellow/green) show higher latency
- n_chans has moderate impact: 128 channels take ~2-4× longer than 1 channel at fixed n_times
- Small n_times values (1, 2) cluster at low latency regardless of other parameters
- **Sweet spot for low latency**: Small images (4096) with small n_times/n_chans
- **Use case**: Identifies fastest configurations for latency-critical applications

---

### 8. Throughput Grouped by n_times
**Output**: `throughput-mvis-s_grouped_by_n_times.png`

**Caption**: *Throughput (Mvis/s) for all 64 benchmark configurations grouped by number of timesteps. Higher bars indicate better performance. Consistent y-axis reveals throughput scaling patterns across parameter space.*

**Metric Range**: 0.0016 Mvis/s (worst: im4096, n_times=1, n_chans=1) to 1.92 Mvis/s (best: im32768, n_times=128, n_chans=128) — **1200× dynamic range**

**Interpretation**:
- Throughput increases dramatically with n_times (note changing scale: 0.02 → 0.2 → 0.5 → 2.0 Mvis/s)
- Larger image sizes consistently outperform smaller ones (yellow bars tallest)
- Within each panel, balanced/large n_chans values boost throughput significantly
- n_times=128 panel shows all configurations achieving >0.2 Mvis/s
- **Sweet spot for throughput**: Large images (32768) with large n_times and n_chans (128×128)
- **Key finding**: 128×128 configuration achieves 1200× better throughput than 1×1

---

### 9. Energy Grouped by n_times
**Output**: `energy-kwh_grouped_by_n_times.png`

**Caption**: *Total energy consumption (kWh) for all configurations grouped by timesteps. Energy includes static (idle) and dynamic (compute) components. Consistent y-axis shows energy scaling across workload complexity.*

**Metric Range**: 0.001 kWh (tiny workloads) to 3.7 kWh (largest workloads) — **3700× dynamic range**

**Interpretation**:
- Energy scales superlinearly with n_times (panels: ~0.03 → 0.1 → 0.4 → 4.0 kWh max)
- Large image sizes dominate energy consumption (yellow bars consistently highest)
- n_chans has moderate impact within each panel (~2-3× variation)
- Small workloads (n_times=1,2) consume <0.1 kWh regardless of configuration
- **Energy efficiency paradox**: Larger workloads use more total energy but deliver much higher throughput
- **Sweet spot for absolute energy**: Small images with minimal n_times/n_chans
- **Note**: Compare with throughput plot to identify energy-efficient high-performance configs

---

### 10. Cost Grouped by n_times
**Output**: `total-cost-$_grouped_by_n_times.png`

**Caption**: *Total cost per run ($) including operational (electricity) and capital (hardware amortization) components. Grouped by timesteps with consistent y-axis for cost comparison across workload sizes.*

**Metric Range**: $0.02 (cheapest) to $7.56 (most expensive) — **378× dynamic range**

**Interpretation**:
- Cost scales linearly with runtime (directly proportional to latency)
- Operational cost (electricity) dominates for long-running jobs (9-16% of total)
- Capital cost (hardware amortization) dominates overall (~85-91% of total cost)
- Larger n_times values push costs higher (panels: ~$0.1 → $0.3 → $1.0 → $8.0 max)
- Image size shows moderate cost impact (yellow bars 2-3× higher than purple)
- **Sweet spot for cost**: Small, fast workloads (im4096, n_times≤8)
- **Use case**: Budget-constrained scenarios should favor lower n_times configurations

---

### 11. Carbon Grouped by n_times
**Output**: `total-carbon-kg_grouped_by_n_times.png`

**Caption**: *Total carbon footprint (kg CO2) per run including operational emissions (grid electricity) and embodied emissions (hardware manufacturing). Grouped by timesteps with consistent y-axis for environmental impact comparison.*

**Metric Range**: 0.0004 kg (cleanest) to 1.02 kg (highest emissions) — **2550× dynamic range**

**Interpretation**:
- Carbon footprint tracks energy consumption closely (operational carbon = energy × CI)
- Embodied carbon (hardware manufacturing) contributes 19-20% of total for short jobs
- Operational carbon (grid emissions) dominates for long-running workloads (80-81%)
- Larger n_times panels show progressively higher carbon (max: 0.008 → 0.03 → 0.12 → 1.0 kg)
- Large image sizes contribute most emissions (yellow bars highest)
- **Sweet spot for carbon**: Small, efficient workloads with low n_times
- **Location sensitivity**: Carbon results assume WA grid (CI=0.27 kg/kWh); cleaner grids reduce operational component
- **Use case**: Carbon-aware scheduling should prioritize low-n_times configurations

---

### 12. Efficiency Grouped by n_times
**Output**: `efficiency-mvis-kwh_grouped_by_n_times.png`

**Caption**: *Energy efficiency (Mvis/kWh) showing computational work per unit energy. Higher bars indicate better efficiency. Grouped by timesteps to reveal efficiency scaling patterns.*

**Metric Range**: 33 Mvis/kWh (least efficient) to 39,165 Mvis/kWh (most efficient) — **1186× dynamic range**

**Interpretation**:
- **Counter-intuitive finding**: Larger workloads are dramatically more energy-efficient
- Efficiency increases with n_times (panels: ~200 → 500 → 2000 → 40,000 Mvis/kWh max)
- Large image sizes (yellow) achieve highest efficiency at high n_times
- Small workloads suffer from static power overhead (idle GPU/CPU consumption wastes energy)
- n_chans shows moderate efficiency impact (~2× variation within panels)
- **Sweet spot for efficiency**: Large images (32768) with large n_times and n_chans (128×128)
- **Key finding**: The 128×128 configuration is 1186× more energy-efficient than 1×1
- **Implication**: For throughput-oriented workloads, running larger jobs is environmentally better
- **Use case**: Batch processing should consolidate small jobs into larger ones for efficiency

---

### Summary: Grouped Bar Plot Insights

**Cross-Metric Patterns**:
1. **Image size dominance**: Larger images (32768) consistently show extreme values (high latency/energy/cost/carbon, very high throughput/efficiency)
2. **n_times scaling**: Primary driver for all metrics; moving 1→128 increases latency/energy 100×, throughput/efficiency 50-100×
3. **n_chans sensitivity**: Moderate impact (2-4× variation) within fixed n_times groups
4. **Trade-offs**: Fast/cheap/clean configs sacrifice throughput; efficient/high-throughput configs require patience and resources

**Recommended Configurations by Use Case**:
- **Latency-critical**: im4096, n_times≤8, n_chans≤8 (< 5s runtime)
- **Cost-constrained**: im4096-8192, n_times≤8, any n_chans (< $0.50/run)
- **Carbon-aware**: Same as cost-constrained (< 0.1 kg CO2/run)
- **Throughput-maximizing**: im32768, n_times=128, n_chans=128 (1.92 Mvis/s, but $7.56 and 134s)
- **Efficiency-optimized**: im16384-32768, n_times≥8, n_chans≥8 (balance all metrics)

---

## Original Visualization Scripts

### 1. Performance vs Workload Analysis
**Script**: `plot_performance_vs_workload.py`
**Output**: `performance_vs_workload.png`
**Description**: Dual-axis scatter plot showing throughput and energy efficiency vs workload size

**Features**:
- X-axis: Workload (Mvis) on logarithmic scale, ranging from 0.13 to 2143 Mvis
- Y-axes: 
  - Left: Throughput (Mvis/s), range 0.0016 to 1.9 Mvis/s (1200x dynamic range)
  - Right: Energy Efficiency (Mvis/kWh), range 33 to 39,165 Mvis/kWh (1200x dynamic range)
- Encoding:
  - Color: Image size (4096, 8192, 16384, 32768)
  - Marker shape: Regime (time-heavy, channel-heavy, balanced)
  
**Key Insights**:
- 108.85× throughput scaling from small to large workloads
- 115.07× efficiency scaling
- Sweet spot: balanced workloads (equal timesteps and channels)
- Larger workloads consistently outperform smaller ones

---

### 2. Regime Heatmaps (4 Variants)
**Script**: `plot_regime_heatmaps.py`
**Outputs**: 
- `regime_heatmaps_energy.png` (Energy Efficiency)
- `regime_heatmaps_carbon.png` (Carbon Efficiency)
- `regime_heatmaps_cost.png` (Cost Efficiency)
- `regime_heatmaps_throughput.png` (Throughput)

**Description**: 2×2 grid heatmaps showing time vs channels for each image size

**Features**:
- 4 subplots (one per image size)
- X-axis: Channels (1, 2, 8, 128)
- Y-axis: Timesteps (1, 2, 8, 128)
- Color: Efficiency metric (RdYlGn colormap, log-normalized for wide ranges)
- Annotations: Cell values with black text
- Dynamic range varies by metric

**Color Interpretation**:
- Green: High efficiency (good)
- Yellow: Medium efficiency
- Red: Low efficiency (poor)

**File Sizes**: 247-282 KB each

---

### 3. Largest Image Metrics Grid
**Script**: `plot_largest_image_metrics.py`
**Output**: `largest_image_metrics.png`
**Description**: Single figure with 2×2 grid of all 4 metrics for the largest image only

**Features**:
- Image size: 32768×32768 (largest workload)
- 4 subplots (Throughput, Energy Eff, Carbon Eff, Cost Eff)
- Individual colorbars per subplot
- Log normalization for ~1200x efficiency range
- Annotations: Configuration performance values

**Size**: 334 KB

**Key Finding**: Best configuration (128×128) is 1201.97× more efficient than worst (1×1) across all metrics

---

### 4. Lifetime Breakdown Analysis
**Script**: `plot_lifetime_breakdown.py`
**Output**: `lifetime_breakdown.png`
**Description**: Stacked bar charts showing carbon and cost composition across all configurations

**Features**:
- 2 rows × 3 columns (6 panels total)
- Row 1: Carbon (absolute values, percentages, summary)
- Row 2: Cost (absolute values, percentages, summary)
- X-axis: All 64 configurations (labeled every 8th)
- Stacked bars showing operational vs capital breakdown

**Field Ranges**:
- Operational Carbon: 80.02%-81.23%
- Embodied Carbon: 18.77%-19.98%
- Operational Cost: 9.24%-9.91%
- Capital Cost: 90.09%-90.76%

**Size**: 477 KB

---

## Location Comparison Visualizations

### 5. Lifetime Breakdown Comparison (WA vs SA)
**Script**: `plot_lifetime_breakdown_comparison.py`
**Output**: `lifetime_breakdown_comparison.png`
**Description**: Side-by-side comparison of Western Australia vs South Africa

**Features**:
- 2 rows (WA, SA) × 4 columns (metrics)
- Metrics: Op Carbon (absolute), Carbon Composition (%), Op Cost (absolute), Cost Composition (%)
- Shows how CI and EP parameters affect breakdown

**Location Parameters**:
| | WA | SA |
|---|---|---|
| CI | 0.27 kg CO2/kWh | 0.19 kg CO2/kWh |
| EP | $0.40/kWh | $0.713/kWh |

**Key Findings**:
- SA carbon: 0.70× WA (30% reduction)
- SA cost: 1.78× WA (78% increase)
- Op Carbon %: WA 80.6% vs SA 74.5%
- Op Cost %: WA 9.6% vs SA 15.9%

**Size**: 608 KB

---

### 6. Largest Image Metrics Comparison (WA vs SA)
**Script**: `plot_largest_image_metrics_comparison.py`
**Output**: `largest_image_metrics_comparison.png`
**Description**: Side-by-side efficiency heatmaps for largest workload

**Features**:
- 2 rows (WA, SA) × 4 columns (metrics)
- Focus on largest image (32768×32768)
- Same heatmap structure as single-location version
- Shows how location affects absolute efficiency values

**Size**: 442 KB

---

### 7. Location Comparison Summary
**Script**: `location_comparison_summary.py`
**Output**: `location_comparison_summary.png`
**Description**: Comprehensive 3×3 grid comparing all aspects of WA vs SA

**Panels**:
- (a-c): Distributions (Carbon, Cost, Energy)
- (d-f): Composition metrics with boxplots
- (g-h): Scatter plots (Carbon vs Time, Cost vs Time)
- (i): Summary statistics box

**Key Statistics**:
- Carbon ratio (SA/WA): 0.762×
- Cost ratio (SA/WA): 1.073×
- Carbon efficiency improvement: 1.31×

**Size**: 628 KB

---

## Quick Reference Table

| Script | Output(s) | Count | Key Metric(s) | Focus | Font Size |
|--------|-----------|-------|---------------|-------|-----------|
| **plot_paper_views.py** | **12 figures** | **12** | **All metrics** | **Paper-ready suite** | **Large (11–14pt)** |
| ├─ Traditional plots | 5 figures | 5 | Pareto, scatter, breakdowns | Analysis & trade-offs | 11–14pt |
| └─ Grouped bar plots | 7 figures | 7 | Latency, throughput, energy, cost, carbon, efficiency | Metric comparison | 11–12pt |
| plot_performance_vs_workload.py | performance_vs_workload.png | 1 | Throughput vs Energy | Scaling behavior | Standard |
| plot_regime_heatmaps.py | regime_heatmaps_*.png | 4 | 4 efficiency metrics | Regime patterns | Standard |
| plot_largest_image_metrics.py | largest_image_metrics.png | 1 | All 4 metrics | Max workload | Standard |
| plot_lifetime_breakdown.py | lifetime_breakdown.png | 1 | Carbon & Cost % | Single location | Standard |
| plot_lifetime_breakdown_comparison.py | lifetime_breakdown_comparison.png | 1 | Carbon & Cost % | WA vs SA | Standard |
| plot_largest_image_metrics_comparison.py | largest_image_metrics_comparison.png | 1 | All 4 metrics | Max workload comparison | Standard |
| location_comparison_summary.py | location_comparison_summary.png | 1 | All comparisons | Comprehensive analysis | Standard |

**Total**: 8 scripts, 22+ figures, ~8 MB

### plot_paper_views.py Figure Breakdown:
1. `pareto_throughput_energy.png` - Throughput vs Energy Pareto front
2. `energy_breakdown_best_configs.png` - Static vs dynamic energy (best configs)
3. `energy_breakdown_grouped_by_n_times.png` - Grouped energy with carbon labels
4. `carbon_cost_scatter.png` - Carbon vs cost Pareto front
5. `latency_throughput_tradeoff.png` - Latency vs throughput scatter
6. `time-s_grouped_by_n_times.png` - Latency bars grouped by n_times
7. `throughput-mvis-s_grouped_by_n_times.png` - Throughput bars grouped by n_times
8. `energy-kwh_grouped_by_n_times.png` - Energy bars grouped by n_times
9. `total-cost-$_grouped_by_n_times.png` - Cost bars grouped by n_times
10. `total-carbon-kg_grouped_by_n_times.png` - Carbon bars grouped by n_times
11. `efficiency-mvis-kwh_grouped_by_n_times.png` - Efficiency bars grouped by n_times
12. Bonus: Can also generate `energy_breakdown_grouped_by_n_chans.png` and all metrics grouped by n_chans

---

## Data Pipeline

```
benchmarks.csv (64 rows)
├─ Columns: im_size, n_times, n_chans, wall_time_sec, n_vis, gpu_j, cpu_j
├─ Processed by all scripts
└─ Combined with machines.csv and locations.csv

machines.csv
├─ Columns: machine, cost, embodied (CO2)
└─ Used for capital amortization

locations.csv
├─ Columns: id, ci (carbon intensity), ep (electricity price)
├─ WA: 0.27 CI, 0.4 EP
└─ SA: 0.19 CI, 0.713 EP
```

---

## Metrics Computed

### Energy-Related
- **Energy (kWh)**: Dynamic + Static
  - Dynamic: GPU + CPU energy from measurements
  - Static: Idle power (343.19W) × runtime
- **Mvis/kWh**: Throughput normalized by energy (energy efficiency)

### Carbon-Related
- **Operational Carbon**: Energy (kWh) × CI (kg CO2/kWh)
- **Embodied Carbon**: Machine embodied CO2 × (runtime / lifetime)
- **Total Carbon**: Operational + Embodied
- **Mvis/kgCO2**: Throughput normalized by carbon (carbon efficiency)

### Cost-Related
- **Operational Cost**: Energy (kWh) × EP ($/kWh)
- **Capital Cost**: Machine cost × (runtime / lifetime)
- **Total Cost**: Operational + Capital
- **Mvis/$**: Throughput normalized by cost (cost efficiency)

### Throughput
- **Mvis/s**: n_vis / (1e6 × runtime_sec)
- **Mvis**: n_vis / 1e6 (normalized work)

---

## Configuration Space

All analyses cover 64 unique configurations:

| Dimension | Values | Count |
|-----------|--------|-------|
| Image Size | 4096, 8192, 16384, 32768 | 4 |
| Timesteps | 1, 2, 8, 128 | 4 |
| Channels | 1, 2, 8, 128 | 4 |
| **Total** | - | **64** |

### Classification
- **Time-heavy**: Timesteps >> Channels
- **Channel-heavy**: Channels >> Timesteps  
- **Balanced**: Timesteps ≈ Channels

---

## Usage Examples

### Generate all paper-ready figures (recommended for conferences):
```bash
cd /Users/nisa/code/astroCAMP/scripts
source ../astrocamp-env/bin/activate

# Paper-ready suite (12 publication-optimized figures)
python3 plot_paper_views.py -l 5 --location WA
```

**Output location**: All figures saved to `scripts/results/`

**Execution time**: ~10-15 seconds on modern hardware

**Expected console output**: Each figure prints a description and interpretation before generation, followed by a checkmark (✓) and file path confirmation.

### Generate all original visualizations:
```bash
# Single-location analysis
python3 plot_performance_vs_workload.py
python3 plot_regime_heatmaps.py
python3 plot_largest_image_metrics.py
python3 plot_lifetime_breakdown.py -l 5

# Location comparison
python3 plot_lifetime_breakdown_comparison.py -l 5
python3 plot_largest_image_metrics_comparison.py
python3 location_comparison_summary.py -l 5
```

### Custom parameters:
```bash
python3 plot_paper_views.py -l 10 --location WA      # 10-year lifetime
python3 plot_lifetime_breakdown.py -o custom.png      # Custom output name
```

---

## Font Specifications for Conference Papers

### Paper-Ready Suite (plot_paper_views.py)
Optimized for single-column and double-column conference layouts:

| Element | Size | Weight | Figure Type | Notes |
|---------|------|--------|-------------|-------|
| Figure title | 14pt | Bold | Scatter, Pareto | Main title |
| Figure title | 12pt | Bold | Grouped bars, heatmaps | Multi-panel titles |
| Axis labels | 13-14pt | Bold | All | X, Y labels |
| Subplot titles | 12pt | Bold | Multi-panel | Per-panel titles |
| Tick labels | 10–12pt | Regular | All | Axis values |
| Legend title | 11pt | Regular | All | Legend headers |
| Legend labels | 10–11pt | Regular | All | Component names |
| Bar annotations | 9pt | Bold | Grouped bars | Value labels on bars |
| Heatmap annotations | 9pt | Bold | Heatmaps | Cell values |

**Recommended print sizes**:
- Single-column figures (scatter, Pareto): 3.5"–4" wide
- Double-column figures (heatmaps, grouped bars): 6"–7" wide
- DPI: ≥ 300 for all figures
- Format: PNG with bbox_inches="tight" (no whitespace waste)

### Original Scripts
Standard matplotlib defaults (smaller fonts, suitable for web/interactive viewing)

---

## Known Limitations and Future Enhancements

### Current Limitations
1. **Single machine type**: Only R675 V3 + 4× H100 96GB analyzed
2. **Fixed locations**: WA and SA hardcoded; other locations require CSV updates
3. **5-year default**: Lifetime assumption fixed; parameterizable but not in UI
4. **Percentage-only composition**: Absolute values also shown but limited space

### Possible Enhancements
1. Add other machine configurations for comparison
2. Support more locations (Europe, Asia, etc.)
3. Interactive plots with hover information
4. Export data as CSV alongside visualizations
5. Confidence intervals for noisy measurements
6. Trend analysis (linear regression on workload size)

---

## Version History

**Current Version**: 2.0  
**Date**: December 2024  
**Status**: Paper-ready visualization suite complete; all original visualizations active

### Version 2.1 Changes (Current)
- Expanded `plot_paper_views.py` to generate **12 figures** (was 6, removed iso-performance heatmaps)
- Added **generic `plot_metric_grouped()` function** supporting 6 metrics:
  - `time_s` (latency)
  - `throughput_mvis_s`
  - `energy_kwh`
  - `total_cost_$`
  - `total_carbon_kg`
  - `efficiency_mvis_kwh`
- All grouped bar plots use **consistent multi-panel format** with:
  - Vertical stacking by group value (n_times or n_chans)
  - Consistent y-axis scaling across panels for visual comparison
  - Image size color encoding (viridis colormap)
  - Value labels on bars (format varies by metric)
- Comprehensive documentation with captions and interpretation guidance

### Version 2.0 Changes
- New `plot_paper_views.py` script with 6 publication-ready figures
- Conference-friendly font sizes (11–14pt for readability)
- Printed figure descriptions and interpretations during execution
- Consistent y-axis ranges across comparative plots
- Improved legend and annotation design
- Added latency vs throughput Pareto front

### Previous Phases
1. **Phase 1**: Performance vs workload visualization
2. **Phase 2**: Regime heatmaps (4 metrics × 4 image sizes)
3. **Phase 3**: Largest image metric grid
4. **Phase 4**: Lifetime carbon/cost breakdown (single location)
5. **Phase 5**: Location comparison
6. **Phase 6** (current): Paper-ready multi-metric suite

---

## Advanced Usage: Customizing Grouped Plots

The `plot_metric_grouped()` function in `plot_paper_views.py` supports flexible grouping:

```python
# Group by n_times (default)
plot_metric_grouped(df, results_dir, "energy_kwh", group_by="n_times")

# Group by n_chans instead
plot_metric_grouped(df, results_dir, "energy_kwh", group_by="n_chans")

# Supported metrics:
# - "time_s" (latency in seconds)
# - "throughput_mvis_s" (Mvis per second)
# - "energy_kwh" (total energy)
# - "total_cost_$" (operational + capital cost)
# - "total_carbon_kg" (operational + embodied carbon)
# - "efficiency_mvis_kwh" (Mvis per kWh)
```

To add new metrics, edit the `metric_configs` dictionary in the function:
```python
metric_configs = {
    "your_metric": {
        "label": "Your Metric (units)",
        "format": "{:.2f}",  # Python format string
        "name": "Short Name"
    }
}
```

---

## Figure Selection Guide for Papers

### For Conference Papers (Space-Limited):
**Minimum set** (3 figures):
1. `pareto_throughput_energy.png` - Shows trade-offs and Pareto front
2. One grouped bar plot: `throughput-mvis-s_grouped_by_n_times.png` - Performance comparison
3. `carbon_cost_scatter.png` - Environmental/economic trade-offs

**Extended set** (7 figures):
- Add all 6 grouped bar plots for comprehensive metric coverage

### For Journal Papers (More Space):
**Full suite** (12 figures):
- Include all figures for complete analysis
- Use grouped bar plots as main text figures
- Relegate scatter/Pareto plots to appendix or supplementary materials

### For Presentations/Posters:
**Visual impact set** (3-4 figures):
1. `pareto_throughput_energy.png` - Eye-catching Pareto front
2. `efficiency-mvis-kwh_grouped_by_n_times.png` - Clear bar comparison showing 1000× range
3. `throughput-mvis-s_grouped_by_n_times.png` - Performance comparison
4. Optional: `carbon_cost_scatter.png` - Sustainability story

---

## Contact & Questions
All scripts are self-contained Python files using standard scientific libraries (pandas, numpy, matplotlib). See individual script headers for detailed documentation.

**Script location**: `/Users/nisa/code/astroCAMP/scripts/plot_paper_views.py`
**Output location**: `/Users/nisa/code/astroCAMP/scripts/results/`
**Data sources**: `benchmarks.csv`, `machines.csv`, `locations.csv`

Generated as part of astroCAMP benchmark analysis project.

---

## Appendix: Metric Formulas

For reference, here are the exact formulas used to compute each metric:

### Energy Metrics
```
energy_static_kwh = (idle_cpu_watt + idle_gpu_watt) × time_h / 1000
  where idle_cpu_watt = 69.44W, idle_gpu_watt = 65.44W, time_h = runtime in hours

energy_dynamic_kwh = (gpu0_j + cpu_j) / 3.6e6
  where gpu0_j, cpu_j are measured joules from benchmarks

energy_kwh = energy_static_kwh + energy_dynamic_kwh
```

### Performance Metrics
```
mvis = n_vis / 1e6
throughput_mvis_s = mvis / time_s
efficiency_mvis_kwh = mvis / energy_kwh
```

### Carbon Metrics
```
operational_carbon_kg = energy_kwh × CI
  where CI = carbon intensity (0.27 kg CO2/kWh for WA)

embodied_carbon_kg = machine_embodied × (time_h / lifetime_hours)
  where machine_embodied = 1800 kg CO2, lifetime_hours = 5 years × 365 × 24

total_carbon_kg = operational_carbon_kg + embodied_carbon_kg
```

### Cost Metrics
```
operational_cost_$ = energy_kwh × EP
  where EP = electricity price ($0.40/kWh for WA)

capital_cost_$ = machine_cost × (time_h / lifetime_hours)
  where machine_cost = $47,000

total_cost_$ = operational_cost_$ + capital_cost_$
```

### Derived Ratios
```
operational_carbon_% = operational_carbon_kg / total_carbon_kg × 100
embodied_carbon_% = embodied_carbon_kg / total_carbon_kg × 100
operational_cost_% = operational_cost_$ / total_cost_$ × 100
capital_cost_% = capital_cost_$ / total_cost_$ × 100
```
