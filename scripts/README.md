# scripts/

This directory contains all analysis and plotting scripts used to produce the figures in the PASC 2026 paper.

> **Raw data:** Download the benchmark datasets and measurement traces from Zenodo ([DOI: 10.5281/zenodo.20093790](https://doi.org/10.5281/zenodo.20093790)) and place the primary CSV files in `data/` (repo root) before running any scripts.

---

## Running all figures at once

```bash
cd <repo-root>
source astrocamp-env/bin/activate   # or: conda activate astrocamp
python scripts/regenerate_all_plots.py
```

All outputs are written to `results/` (repo root).

---

## Primary input data files (from Zenodo — place in `data/`)

| File | Description |
|---|---|
| `benchmarks_comprehensive.csv` | Full benchmark results: all image sizes, timesteps, channels, locations |
| `benchmarks.csv` | Core benchmark results subset |
| `cpu_scaling.csv` | CPU thread-scaling measurements |
| `kernel_breakdown_pasc25_16c_summary.csv` | Per-kernel time breakdown (16 channels, GPU) |
| `idg_gpu_parsed_log_summary.csv` | IDG GPU raw log summary |
| `idg_cpu_gpu_walltime_from_bench.csv` | CPU vs GPU wall-time comparison |
| `darshan_io_summary.csv` | Darshan I/O profiling summary |
| `uprof_hotspots_summary.csv` | AMD uProf hotspot profiling summary |
| `gpu_bytes_per_joule_summary.csv` | GPU data movement efficiency |
| `locations.csv` | Carbon intensity and electricity cost by location |
| `machines.csv` | Hardware specifications |

Generated summary CSVs (produced by scripts, tracked in `data/derived/`) are committed and do not need to be re-generated manually.

---

## Plotting scripts

### Paper figures

| Script | Output (in `results/`) | Description |
|---|---|---|
| `pasc2025_paper_analysis_plots.py` | `plot1_*` … `plot10_*` | Main analysis: energy stacks, throughput, carbon, cost |
| `plot9b_flagship_energy_stack_throughput_facets_linear.py` | `plot9b_*` | Flagship energy + throughput, linear right axis |
| `plot10b_carbon_cost_efficiency_facets_conference.py` | `plot10b_*` | Carbon & cost efficiency — conference layout |
| `plot17b_kernel_time_breakdown_with_disk.py` | `plot17b_*` | Kernel time breakdown with disk I/O |
| `plot19e_largest_image_kernel_wall_saturation_signals_active_by_c.py` | `plot19e_*` | GPU saturation signals, largest image |
| `plot25_cpu_roofline_stacking.py` | `plot25_*` | CPU roofline with stacking |
| `plot25b_cpu_roofline_scaling_gap.py` | `plot25b_*` | CPU roofline scaling gap |
| `plot25c_cpu_roofline_scalability_comparison.py` | `plot25c_*` | CPU roofline scalability comparison |
| `plot25d_cpu_scalability_only.py` | `plot25d_*` | CPU scalability (speedup) |
| `plot25e_cpu_scalability_with_gpu_speedup.py` | `plot25e_*` | CPU scalability + GPU speedup overlay |
| `plot25f_cpu_roofline_with_scaling_gap.py` | `plot25f_*` | CPU roofline + scaling gap combined |
| `plot26_gpu_cpu_roofline_comparison_conference.py` | `plot26_*` | GPU vs CPU roofline — conference layout |
| `plot26b_cpu_execution_comparison_conference.py` | `plot26b_conf_*` | CPU execution time comparison |
| `plot26b_cpu_gpu_execution_comparison.py` | `plot26b_*` | CPU vs GPU execution comparison |
| `plot26c_cpu_gpu_speedup_utilization.py` | `plot26c_*` | CPU/GPU speedup and utilization |
| `plot26d_cpu_gpu_execution_comparison_gibj.py` | `plot26d_*` | CPU/GPU execution comparison (GiB/J) |
| `plot27_hotspots_summary.py` | `plot27_*` | AMD uProf hotspot summary |
| `plot28_darshan_io_summary.py` | `plot28_*` | Darshan I/O summary |
| `plot29_gpu_bytes_per_joule.py` | `plot29_*` | GPU bytes-per-joule efficiency |
| `plot30_rebuttal_gpu_roofline_hotspots.py` | `plot30_*` | GPU roofline with hotspot overlay |
| `plot31_cpu_scaling_phase_limits.py` | `plot31_*` | CPU scaling with phase limits |
| `plot31b_cpu_scaling_gap_only.py` | `plot31b_*` | CPU scaling gap |
| `plot32_problem_footprint_gib_per_joule.py` | `plot32_*` | Problem footprint (GiB/J) |
| `plot32b_idg_data_movement_gb_per_joule.py` | `plot32b_*` | IDG data movement (GB/J) |
| `plot33_problem_sizes_memory_io_overview.py` | `plot33_*` | Problem size × memory × I/O overview |
| `plot33b_problem_sizes_memory_io_heatmap.py` | `plot33b_*` | Problem size × memory × I/O heatmap |
| `plot34_problem_size_stacked_linear.py` | `plot34_*` | Problem sizes stacked (linear) |
| `plot34b_large_problem_size_stacked_linear.py` | `plot34b_*` | Large problem sizes stacked |
| `plot34c_large_problem_size_with_gbj.py` | `plot34c_*` | Large problem sizes + GB/J |
| `plot34d_large_problem_size_with_gbj_by_c.py` | `plot34d_*` | Large problem sizes + GB/J by channels |
| `plot34e_large_problem_size_with_gbj_by_c_single_column.py` | `plot34e_*` | Large problem sizes + GB/J single column |
| `plot34f_largest_problem_size_with_gbj_by_c.py` | `plot34f_*` | Largest (32k²) problem size + GB/J |
| `plot_cpu_scaling.py` | `cpu_scalability*` | CPU thread scalability: time (bars) + speedup (lines) |
| `plot_key_takeaway.py` | `key_takeaway_*` | Key takeaway composite efficiency figure |
| `plot_largest_bars_and_lines.py` | `largest_32768_*` | Component energy breakdown for largest image |
| `plot_largest_image_metrics.py` | `largest_image_metrics*` | Efficiency metrics for the largest image |
| `plot_largest_image_metrics_comparison.py` | `largest_image_metrics_comparison*` | Efficiency metrics: WA vs SA location |
| `plot_lifetime_breakdown.py` | `lifetime_breakdown*` | Lifetime carbon & cost breakdown |
| `plot_lifetime_breakdown_comparison.py` | `lifetime_breakdown_comparison*` | Lifetime breakdown by location |
| `plot_paper_views.py` | `paper_views*` | Paper-view summary plots |
| `plot_performance_vs_workload.py` | `performance_vs_workload*` | Throughput & energy efficiency vs workload |
| `plot_regime_heatmaps.py` | `regime_heatmaps*` | Regime heatmaps (throughput, energy, carbon, cost) |
| `plot_scalability_analysis.py` | `scalability_*` | Scalability analysis plots |
| `location_comparison_summary.py` | `location_comparison_*` | WA vs SA location comparison summary |

### Utility scripts

| Script | Description |
|---|---|
| `regenerate_all_plots.py` | Runs all plotting scripts in sequence |
| `results_dataframe.py` | Shared data-loading utilities used by plot scripts |
| `cea.py` | Carbon emissions accounting helpers |
| `table_problem_sizes_memory.py` | Generates problem size table (LaTeX/Markdown) |
| `table_problem_sizes_memory_darshan.py` | Generates problem size + I/O table |
| `extract_darshan_io_summary.py` | Parses Darshan logs → `darshan_io_summary.csv` |
| `extract_uprof_hotspots_summary.py` | Parses uProf logs → `uprof_hotspots_summary.csv` |
| `figure_caption_article_notebook.ipynb` | Notebook with figure captions for the article |
