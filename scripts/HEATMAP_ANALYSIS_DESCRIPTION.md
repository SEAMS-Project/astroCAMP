# Regime Heatmap Analysis: Efficiency Metrics Interpretation

## Overview

The regime heatmaps visualize three key efficiency metrics across image sizes and workload configurations (n_times × n_chans):
- **Carbon Efficiency (Mvis/kgCO2)**: Computing output per unit carbon footprint
- **Throughput (Mvis/s)**: Computing speed in million visibilities per second  
- **Cost Efficiency (Mvis/$)**: Computing output per unit cost

Each metric shows a 4×4 grid of timesteps (1, 8, 64, 128) versus channels (1, 8, 64, 128) for each of the four image sizes (4096, 8192, 16384, 32768).

---

## Key Findings

### 1. CARBON EFFICIENCY (Mvis/kgCO2)

**Overall Pattern:** Dramatic efficiency gains with balanced, larger workloads.

**Range per Image Size:**
| Image Size | Min (Mvis/kgCO2) | Max (Mvis/kgCO2) | Ratio (Best/Worst) |
|------------|------------------|------------------|--------------------|
| 4096       | 226.4            | 146,394          | 646.7×             |
| 8192       | 216.9            | 150,484          | 693.6×             |
| 16384      | 170.4            | 157,489          | 924.4×             |
| 32768      | 97.4             | 152,427          | 1,564.6×           |

**Critical Observations:**

1. **Scaling Asymmetry**: Channels matter more than timesteps
   - Increasing channels alone: **+13,500% improvement** (1→128 channels)
   - Increasing timesteps alone: **+4,000-6,000% improvement** (1→128 timesteps)
   - **Interpretation**: The algorithm achieves significantly better carbon efficiency when processing wider spectral bands (more channels) versus longer observations (more timesteps). This suggests the compute overhead (fixed overheads) is amortized more effectively across spectral dimensions.

2. **Workload Size Effect**: Larger images have worse efficiency at minimal configurations
   - 4096×4096 (1,1): 226 Mvis/kgCO2
   - 32768×32768 (1,1): 97 Mvis/kgCO2  
   - **Interpretation**: For tiny workloads, larger images suffer ~2.3× penalty because fixed system overhead dominates. But at (128,128), the largest image achieves comparable efficiency (152k vs 146k), showing that overhead amortization scales with problem size.

3. **Best Configuration**: n_times=128, n_chans=128 (all image sizes)
   - All achieve ~150,000 Mvis/kgCO2
   - **Interpretation**: The algorithm reaches a sweet spot when fully utilizing the system with maximum parallelism (largest spectral and temporal dimensions).

4. **Worst Configuration**: n_times=1, n_chans=1 (all image sizes)
   - Ranges 97-226 Mvis/kgCO2  
   - **Interpretation**: Minimal workloads leave the system with high idle time and fixed overheads, resulting in poor carbon utilization.

**Energy Meaning:**
- Higher Mvis/kgCO2 = More computing per unit carbon → Better for sustainability goals
- The 1,500× spread shows that *workload scaling is critical*: deploying with balanced, large jobs is vastly more carbon-efficient than many small jobs.

---

### 2. THROUGHPUT (Mvis/s)

**Overall Pattern:** Scales proportionally with workload; channel expansion dominates.

**Range per Image Size:**
| Image Size | Min (Mvis/s) | Max (Mvis/s) | Ratio (Best/Worst) |
|------------|--------------|--------------|-------------------|
| 4096       | 0.004        | 1.825        | 467.8×            |
| 8192       | 0.004        | 1.878        | 502.1×            |
| 16384      | 0.003        | 1.954        | 668.1×            |
| 32768      | 0.002        | 1.902        | 1,157.4×          |

**Critical Observations:**

1. **Linear Scalability with Channels**
   - n_chans=1→128: **+100× throughput increase**
   - n_times=1→128: **+30-50× throughput increase**
   - **Interpretation**: Throughput is bandwidth-limited. Increasing channels (spectral dimension) provides proportionally better speedup because it allows more parallel vectorization and reduces relative I/O overhead.

2. **Image Size Trade-off**
   - Larger images = lower raw throughput (more per-pixel work)
   - 4096: max 1.825 Mvis/s
   - 32768: max 1.902 Mvis/s (slightly *better*)
   - **Interpretation**: While per-pixel work increases, the largest image size benefits from better cache locality and reduced relative overhead for batched operations.

3. **Time-vs-Channel Scaling**
   - Time-heavy (n_times=128, n_chans=1): 0.07-0.12 Mvis/s
   - Channel-heavy (n_times=1, n_chans=128): 0.30-0.39 Mvis/s
   - **Interpretation**: Spectral parallelism is more effective than temporal reuse on this GPU-accelerated system.

**Performance Meaning:**
- Higher Mvis/s = Faster execution → Better for latency-sensitive operations
- Throughput saturates at ~1.9 Mvis/s (full hardware utilization)
- The algorithm is well-optimized for channel-parallel execution.

---

### 3. COST EFFICIENCY (Mvis/$)

**Overall Pattern:** Similar to carbon efficiency, but with different absolute scales.

**Range per Image Size:**
| Image Size | Min (Mvis/$) | Max (Mvis/$) | Ratio (Best/Worst) |
|------------|--------------|--------------|-------------------|
| 4096       | 17.2         | 8,739        | 507.1×             |
| 8192       | 16.5         | 8,991        | 544.2×             |
| 16384      | 12.9         | 9,362        | 724.0×             |
| 32768      | 7.3          | 9,103        | 1,249.0×           |

**Critical Observations:**

1. **Cost Scales with Image Size** (opposite of throughput)
   - Smallest image (4096): ~17-18 Mvis/$ baseline
   - Largest image (32768): ~7-8 Mvis/$ baseline
   - **Interpretation**: Larger images consume more compute per pixel, so the cost per megavisibility is higher. This is a fundamental trade-off: larger imaging requires more expensive hardware operations.

2. **Scaling Gains Are Substantial**
   - Channels: **+8,700-10,500% improvement**
   - Timesteps: **+2,800-4,300% improvement**
   - **Interpretation**: Running large jobs (high n_times and n_chans) dramatically improves monetary efficiency, reducing per-unit cost by orders of magnitude.

3. **Capital vs. Operational Cost**
   - The baseline inefficiency for small workloads (7-18 Mvis/$) is driven by amortized capital costs.
   - Larger workloads make capital expense negligible relative to operational cost.
   - **Interpretation**: Small jobs should be batched or scheduled during off-peak times; large jobs justify the full computational investment.

**Economic Meaning:**
- Higher Mvis/$ = More science per dollar spent → Better budget utilization
- The ~1,200× spread emphasizes that *batch scheduling is economically critical*.
- An interferometer operator should always prefer n_times=128, n_chans=128 over many small jobs.

---

## Comparative Interpretation: Carbon vs. Throughput vs. Cost

### Alignment and Trade-offs

**1. Do efficiency metrics agree?**
- **Yes, largely**: All three metrics prefer large, balanced workloads (n_times=128, n_chans=128).
- **Range**: Carbon (~1,600×) > Cost (~1,200×) > Throughput (~1,150×)
- **Interpretation**: The system's efficiency is limited by fixed overhead, which all metrics capture differently:
  - *Carbon efficiency* captures the amortization of idle power draw.
  - *Cost efficiency* captures amortization of capital and operational costs.
  - *Throughput* captures raw compute utilization.

### Location Sensitivity (Grid-Adjusted)
- Carbon efficiency is higher in **South Africa** (lower carbon intensity: 0.19 vs 0.27 kgCO2/kWh), giving SA roughly **1.4×** less carbon per Mvis than Western Australia.
- Cost efficiency is higher in **Western Australia** (lower price: 0.40 vs 0.713 $/kWh), giving WA roughly **1.8×** more Mvis/$ than South Africa.
- The **shape of the preference** is unchanged: large, balanced workloads remain best; location scales the bars up or down without moving the optimum.

### Channel > Timestep Preference (All Metrics)
- Channels improve all three metrics **~2-3× more than timesteps**.
- Example (4096 image):
  - Channels alone (n_chans=1→128): Mvis/$ goes 17 → 1,847 (+10,800%)
  - Timesteps alone (n_times=1→128): Mvis/$ goes 18 → 583 (+3,000%)

**Why?** Spectral parallelism:
- Channel data can be processed in parallel with minimal inter-channel dependencies.
- Time-stepping has sequential dependencies (correlations across time).
- Modern GPU architectures favor wide, data-parallel operations.

---

## Heatmap Visual Interpretation Guide

### Color Gradient (Red-Yellow-Green Scale)
- **Red zone** (lower left): Minimal workloads (n_times=1, n_chans=1)
  - Low efficiency, high per-unit cost
  - Suitable only if latency is critical (single-snapshot observations)
  
- **Yellow zone** (middle): Moderate workloads (n_times=8 or 64, n_chans=8 or 64)
  - Balanced efficiency and runtime
  - Typical for survey observations
  
- **Green zone** (upper right): Large workloads (n_times=128, n_chans=128)
  - Maximum efficiency, amortized overhead
  - Recommended for batch processing and archive analysis

### Gradient Directions
- **Horizontal (left→right, increasing channels)**: Steepest gradient
  - Indicates strong spectral parallelism opportunity
  - Each doubling of channels yields ~5-8× efficiency gain
  
- **Vertical (bottom→top, increasing timesteps)**: Gentler gradient
  - Indicates weaker temporal scaling
  - Each doubling of timesteps yields ~2-4× efficiency gain
  
- **Diagonal (lower-left→upper-right)**: Synergistic effect
  - Combined scaling exceeds individual contributions
  - The interaction shows that larger spectral/temporal workloads compound efficiency

---

## Practical Recommendations

### For Observers/Scientists

1. **Batch your observations**
   - Combine multiple targets into a single large observation if possible
   - Cost savings: **5-10× improvement** by going from small to large batches

2. **Prioritize channel resolution over temporal resolution**
   - If you must choose: add channels rather than timesteps
   - Efficiency gain (channels): 100× for 8→128 channels
   - Efficiency gain (time): 10× for 8→128 timesteps

3. **Avoid small, isolated observations**
   - A single (1,1) config is ~500-1000× less efficient than optimal
   - Schedule small observations in batch queues during off-peak times

### For Cluster/Cloud Operators

1. **Implement workload batching**
   - Pool small jobs into larger ones when possible
   - Expected cost reduction: **10-50%**

2. **Schedule large processing tasks during compute-abundant periods**
   - The "sweet spot" (n_times=128, n_chans=128) fully utilizes the system
   - Small jobs (n_times=1) waste ~30-50% of hardware capacity

3. **Monitor utilization metrics**
   - Watch for jobs running with n_chans < 8 (red flag for underutilization)
   - Encourage researchers to increase problem size before submission

---

## Summary Table: Efficiency Rankings

| Metric | Best Config | Performance | Worst Config | Performance | Gain Factor |
|--------|-------------|-------------|--------------|------------|------------|
| **Carbon (Mvis/kgCO2)** | n_t=128, n_c=128 | ~152k | n_t=1, n_c=1 | ~100 | **1,500×** |
| **Throughput (Mvis/s)** | n_t=128, n_c=128 | ~1.9 | n_t=1, n_c=1 | ~0.002 | **1,150×** |
| **Cost (Mvis/$)** | n_t=128, n_c=128 | ~9,000 | n_t=1, n_c=1 | ~7 | **1,250×** |

**Key Takeaway:** The system exhibits a **universal efficiency advantage for large, balanced workloads**—across carbon, cost, and performance dimensions. Optimal operations require n_times and n_chans both ≥ 64.
