# Location Comparison Analysis: Western Australia vs South Africa

## Overview
This analysis compares how two geographic locations with different carbon intensities (CI) and electricity prices (EP) affect the lifetime carbon footprint and total cost of operating a radio astronomy telescope.

### Location Parameters
| Parameter | Western Australia (WA) | South Africa (SA) |
|-----------|------------------------|-------------------|
| **Carbon Intensity (CI)** | 0.27 kg CO2/kWh | 0.19 kg CO2/kWh |
| **Electricity Price (EP)** | $0.40/kWh | $0.713/kWh |
| **Grid Description** | Moderate carbon grid | Cleaner grid (30% lower) |
| **Electricity Cost** | Lower cost | Higher cost (78% more) |

---

## Key Findings

### 1. **Carbon Footprint Impact**
- **SA produces 30% LESS operational carbon** due to lower grid carbon intensity (0.70x WA)
- This advantage applies uniformly to **all workloads and configurations**
- Example: A typical workload produces ~3.97 kg CO2 in WA vs ~3.03 kg CO2 in SA

**Implication**: For carbon optimization, choose the location with the lowest carbon intensity, regardless of workload characteristics.

### 2. **Operating Cost Impact**
- **SA costs 78% MORE** due to higher electricity prices (1.78x WA)
- Despite having cleaner energy, SA's higher EP makes it more expensive operationally
- Example: A typical workload costs $0.05 in WA vs $0.09 in SA

**Implication**: Cost optimization favors WA despite WA's higher carbon intensity.

### 3. **Composition Changes**

#### Operational Carbon Percentage:
- **WA**: 80.0% - 81.2% (mean: 80.6%)
- **SA**: 73.8% - 75.3% (mean: 74.5%)
- **Change**: -6.1% absolute change
- **Reason**: Lower CI in SA increases embodied carbon as a proportion of total

#### Operational Cost Percentage:
- **WA**: 9.2% - 9.9% (mean: 9.6%)
- **SA**: 15.4% - 16.4% (mean: 15.9%)
- **Change**: +6.3% absolute change
- **Reason**: Higher EP in SA increases operational costs relative to capital costs

#### Capital Cost Dominance:
- **WA**: 90.1% - 90.8% of total cost
- **SA**: 83.6% - 84.7% of total cost
- **Key Finding**: Hardware amortization dominates TCO in both locations, but is more dominant in WA

---

## Generated Visualizations

### 1. **lifetime_breakdown_comparison.png**
- **Type**: Side-by-side comparison (2 rows × 3 columns)
- **Content**: 
  - Row 1: WA carbon composition, operational carbon, summary
  - Row 2: SA carbon composition, operational carbon, summary
- **Insight**: Shows that percentage composition differs between locations while energy use remains identical

### 2. **largest_image_metrics_comparison.png**
- **Type**: 2×4 grid (4 metrics × 2 locations)
- **Content**: Efficiency heatmaps for largest workload (32768×32768)
  - Metrics: Throughput, Energy Efficiency, Carbon Efficiency, Cost Efficiency
  - Configurations: All 16 time-channel combinations
- **Insight**: Best configuration (128×128) is 1200x more efficient than worst (1×1) for all metrics

### 3. **location_comparison_summary.png**
- **Type**: 3×3 grid with 9 different comparison views
- **Panels**:
  - (a-c) Distributions: Carbon, Cost, Energy
  - (d-f) Composition: Operational carbon %, Operational cost %, Efficiency metric
  - (g-i) Scatter plots: Carbon vs Time, Cost vs Time, Summary statistics
- **Insight**: Comprehensive view of how location parameters affect absolute and relative metrics

---

## Detailed Analysis

### Carbon Intensity Effect (0.27 → 0.19 kg CO2/kWh)
The 30% reduction in carbon intensity translates directly to:
- **Operational carbon**: 0.70x (proportional reduction)
- **Total carbon**: 0.76x (slightly higher due to increased embodied carbon %)
- **Carbon efficiency** (Mvis/kg CO2): 1.31x improvement

**Stability Across Workloads**: The ratio remains constant because it depends only on energy consumption, which is identical for both locations.

### Electricity Price Effect ($0.40 → $0.713/kWh)
The 78% price increase translates directly to:
- **Operational cost**: 1.78x (proportional increase)
- **Total cost**: 1.07x (smaller due to capital cost dominance)
- **Operating margin**: Decreases from 9.6% to 15.9% of TCO

**Implication**: Even though SA has cleaner energy, the higher electricity price makes it more expensive overall, demonstrating that both carbon and cost optimization matter.

---

## Composition Insights

### Why Percentages Change Differently

**Operational Carbon % decreases from 80.6% to 74.5%:**
- Operational carbon scales with CI (proportional)
- Embodied carbon is location-independent
- Lower CI means embodied carbon becomes relatively larger
- Formula: Op% = (Energy × CI) / (Energy × CI + Embodied)
- Lower CI → Higher denominator relative to numerator → Lower percentage

**Operational Cost % increases from 9.6% to 15.9%:**
- Operational cost scales with EP (proportional)
- Capital cost is location-independent
- Higher EP means operational cost becomes relatively larger
- Formula: OpCost% = (Energy × EP) / (Energy × EP + Capital)
- Higher EP → Higher numerator relative to denominator → Higher percentage

---

## Recommendations

### For Carbon Minimization:
1. **Choose SA** (South Africa) - achieves 30% lower operational carbon
2. This advantage is **workload-independent** - applies to all configurations equally
3. Embodied carbon becomes more significant in SA (26% vs 19%) but net carbon is still lower

### For Cost Minimization:
1. **Choose WA** (Western Australia) - despite higher carbon intensity
2. Lower electricity prices offset the cleaner grid advantage
3. Capital costs dominate TCO anyway (~90%), making location less critical for cost

### For Combined Optimization (Carbon + Cost):
1. **Consider the ratio**: WA has 1.3x higher carbon but 0.94x lower cost
2. **Workload characteristics don't matter** - both ratios apply uniformly
3. **Embodied factors dominate** - focus on hardware efficiency and utilization first

---

## Technical Notes

### Assumptions:
- 5-year equipment lifetime (amortization period)
- Same hardware in both locations (R675 V3 + 4× H100 96GB)
- Idle power: 277.75W CPU + 65.44W GPU = 343.19W combined
- Capital cost amortization: linear across 5 years

### Data Coverage:
- 64 unique workload configurations
- Timesteps: 1, 2, 8, 128
- Channels: 1, 2, 8, 128
- Image sizes: 4096, 8192, 16384, 32768 pixels
- All analyses cover the full range of workloads

---

## Generated Python Scripts

1. **plot_lifetime_breakdown_comparison.py**
   - Generates side-by-side comparison of carbon and cost composition
   - Shows how CI and EP parameters affect breakdown percentages
   - Includes summary statistics and field ranges

2. **plot_largest_image_metrics_comparison.py**
   - Compares efficiency metrics for largest workload
   - Shows throughput, energy, carbon, and cost efficiency
   - Highlights that relative efficiency gains are location-independent

3. **location_comparison_summary.py**
   - Comprehensive statistical analysis across all workloads
   - Generates 9-panel summary visualization
   - Prints detailed comparison statistics and ratios

---

## Conclusion

Location choice significantly affects both absolute carbon footprint and costs, but in opposite ways:
- **SA wins on carbon** (0.70x WA) due to cleaner grid
- **WA wins on cost** (0.94x SA) despite higher carbon intensity
- **Composition changes** but hardware dominance remains (~90% capital in TCO)
- **Workload characteristics are irrelevant** to location advantage - it applies equally to all configurations

The data suggests that for radio astronomy observatories, the choice between these locations should be based on institutional priorities: environmental impact favors SA, while cost-consciousness favors WA.
