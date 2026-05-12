# astroCAMP — PASC 2026 Companion Repository

[![License](https://img.shields.io/badge/license-CC%20BY%204.0-lightgrey.svg)](LICENSE)

<img width="682" height="425" alt="astroCAMP overview" src="https://github.com/user-attachments/assets/20cc4fcc-3351-45a1-a5c9-6c1f0e505557" />

This repository is the companion code release for the paper published at **PASC 2026**. It contains all analysis and plotting scripts needed to reproduce the figures in the paper.


---

## What is astroCAMP?

**astroCAMP (Astronomical Co-design Analysis and Metrics Platform)** is a framework for evaluating radio-interferometric imaging pipelines under **performance**, **energy**, **scientific quality**, and **sustainability** constraints.

It supports **hardware–software co-design** for SKA-scale workloads (SKA1-Low and SKA1-Mid) by enabling reproducible, quantitative exploration of:

- system-level behaviour (runtime, energy, throughput),
- platform-level device utilisation (CPU/GPU),
- algorithmic scientific fidelity (RMS, PSNR, astrometry, photometry),
- carbon and cost efficiency.

---

## Repository Structure

| Directory               | Contents                                                              |
| ----------------------- | --------------------------------------------------------------------- |
| `scripts/`            | All analysis and plotting scripts (Python only)                       |
| `data/`               | Primary input CSVs from Zenodo (tracked in git)                       |
| `data/derived/`       | Generated summary CSVs produced by scripts (tracked in git)           |
| `results/`            | Generated figure outputs (PNG/PDF, gitignored)                        |
| `docs/`               | Metric specification, benchmarking protocol, and documentation index  |
| `baselines/`          | Reference baseline pipeline descriptions (WSClean, IDG, BIPP)         |
| `configs/`            | Configuration file templates for controlled experiments               |
| `datasets/`           | Dataset generation recipe and descriptions (data itself is on Zenodo) |
| `figures/paper_figs/` | Final publication-ready figures used in the paper                     |
| `metrics/`            | Metric definitions and evaluation utilities                           |

---

# 📐 **Core astroCAMP Co-Design Metrics**

astroCAMP defines **four co-design layers**, each quantifying a different aspect of imaging performance and scientific validity.
All symbols are defined **inline** so the table is fully self-contained.

## **1. System-Level Metrics (End-to-End Execution on Heterogeneous Nodes)**

| ID           | Metric             | Formula              | Unit           | Meaning & Notation                                           |
| ------------ | ------------------ | -------------------- | -------------- | ------------------------------------------------------------ |
| **A1** | Time-to-solution   | `T_c`              | s              | Total job runtime.`T_c` = wall-clock time.                 |
| **A2** | Energy-to-solution | `E_c = ∫ P(t) dt` | J              | Total energy.`P(t)` = instantaneous power.                 |
| **A3** | Throughput         | `Θ = N / T_c`     | vis/s or img/s | Science processed per second.`N` = visibilities or images. |
| **A4** | Energy efficiency  | `η_E = N / E_c`   | vis/J          | Visibilities per joule.                                      |

## **2. Platform-Level Metrics (CPU / GPU / FPGA / ASIC Devices)**

| ID           | Metric            | Formula                    | Unit | Meaning & Notation                                 |
| ------------ | ----------------- | -------------------------- | ---- | -------------------------------------------------- |
| **A5** | Utilisation       | `U = t_active / t_total` | –   | Device activity.`t_active` = active kernel time. |
| **A6** | Memory bandwidth  | `B_mem = Bytes / T_c`    | GB/s | Sustained device memory throughput.                |
| **A7** | Peak memory usage | `M_peak`                 | GB   | Maximum resident memory footprint.                 |

## **3. Algorithmic Quality Metrics (Scientific Validity)**

| ID | Metric            | Formula                                           | Unit         | Meaning & Notation (Self-contained)                                                                                                                                                                                  |
| -- | ----------------- | ------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B1 | Dirty-image RMS   | `σ_dirty = sqrt( (1/N) Σ (I_i – Ī)² )`     | Jy/beam      | Noise + artefacts in the dirty image.`I_i` = pixel values; `Ī` = mean pixel intensity; `N` = number of pixels; `σ_dirty` = root-mean-square deviation from the mean.                                       |
| B2 | PSNR / SSIM       | `PSNR = 10 log10( I_max² / MSE )`              | dB / –      | Fidelity vs reference image `I_ref`. `I_max` = maximum pixel value; `MSE` = mean squared error between reconstruction `Ĩ` and `I_ref`; SSIM = structural similarity between them.                         |
| B3 | Dynamic range     | `DR = I_max / σ_res`                           | –           | Ratio of peak brightness to residual noise.`I_max` = brightest pixel in the image; `σ_res` = RMS of the residual image; higher `DR` = better faint-source detectability.                                      |
| B4 | Astrometric error | `ε_astro = (1/N) Σ L2(x_i – x_i_ref)`        | arcsec or px | Position error of detected sources.`x_i` = measured source positions; `x_i_ref` = reference (catalogue) positions; `ε_astro` = mean positional offset over `N` sources.                                     |
| B5 | Photometric error | `ε_photo = (1/N) Σ L1(S_i – S_i_ref)`        | Jy           | Flux-density error.`S_i` = measured flux densities; `S_i_ref` = reference fluxes; `ε_photo` = mean absolute flux difference over `N` matched sources.                                                       |
| B6 | Spectral fidelity | `ε_spec = (1/N_ν) Σ L1 (I(ν) – I_ref(ν))` | Jy           | Per-channel spectral error.`I(ν)` = reconstructed intensity at frequency `ν`; `I_ref(ν)` = reference intensity; `N_ν` = number of frequency channels; `ε_spec` = mean absolute per-channel deviation. |

## **4. Sustainability Metrics (Energy → Carbon)**

| ID           | Metric             | Formula                 | Unit       | Meaning & Notation                                    |
| ------------ | ------------------ | ----------------------- | ---------- | ----------------------------------------------------- |
| **C1** | Carbon-to-solution | `C_c = E_c * κ(t,r)` | gCO₂e     | Carbon footprint.`κ(t,r)` = grid carbon intensity. |
| **C2** | Carbon efficiency  | `η_C = N / C_c`      | vis/gCO₂e | Science per gram CO₂ emitted.                        |

## **5. Economic Metrics (Cost-Aware Co-Design)**

| ID           | Metric                  | Formula                      | Unit   | Meaning & Notation                                   |
| ------------ | ----------------------- | ---------------------------- | ------ | ---------------------------------------------------- |
| **E1** | Total cost of ownership | `C_TTO = C_capex + C_opex` | €     | Hardware lifetime cost.                              |
| **E2** | Cost per job            | `C_E = E_c * p_E`          | €     | Monetary execution cost.`p_E` = electricity price. |
| **E3** | Cost efficiency         | `Θ / C_TTO`               | ops/€ | Science per euro invested.                           |

# Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/SEAMS-Project/astroCAMP.git
cd astroCAMP

# 2. Create and activate a virtual environment
python3 -m venv astrocamp-env
source astrocamp-env/bin/activate        # macOS / Linux
# astrocamp-env\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data from Zenodo (DOI: 10.5281/zenodo.20093790)
#    Place the CSV files in data/

# 5. Regenerate all figures
python scripts/regenerate_all_plots.py
#    Output images are written to results/
```

Alternatively, using conda:

```bash
conda env create -f environment.yml
conda activate astrocamp
```

---

## Reproducing astroCAMP Paper Figures and Analysis

All figures are regenerated by `python scripts/regenerate_all_plots.py`.
Individual scripts can also be run directly from the repo root:

| Script                                                                  | Description                                                 |
| ----------------------------------------------------------------------- | ----------------------------------------------------------- |
| `pasc2025_paper_analysis_plots.py`                                    | Main analysis: energy, throughput, carbon, cost facet plots |
| `plot9b_flagship_energy_stack_throughput_facets_linear.py`            | Flagship energy + throughput (linear scale)                 |
| `plot10b_carbon_cost_efficiency_facets_conference.py`                 | Carbon & cost efficiency — conference layout               |
| `plot17b_kernel_time_breakdown_with_disk.py`                          | Kernel time breakdown including disk I/O                    |
| `plot19e_largest_image_kernel_wall_saturation_signals_active_by_c.py` | GPU saturation signals for largest image                    |
| `plot25_cpu_roofline_stacking.py`                                     | CPU roofline model with stacking                            |
| `plot25c_cpu_roofline_scalability_comparison.py`                      | CPU roofline scalability comparison                         |
| `plot26_gpu_cpu_roofline_comparison_conference.py`                    | GPU vs CPU roofline — conference layout                    |
| `plot26b_cpu_execution_comparison_conference.py`                      | CPU execution time comparison                               |
| `plot27_hotspots_summary.py`                                          | AMD uProf hotspot summary                                   |
| `plot28_darshan_io_summary.py`                                        | Darshan I/O summary                                         |
| `plot29_gpu_bytes_per_joule.py`                                       | GPU data movement efficiency (GB/J)                         |
| `plot32b_idg_data_movement_gb_per_joule.py`                           | IDG data movement efficiency (GB/J)                         |
| `plot33_problem_sizes_memory_io_overview.py`                          | Problem size × memory × I/O overview                      |
| `plot34e_large_problem_size_with_gbj_by_c_single_column.py`           | Large problem sizes with GB/J                               |
| `plot34f_largest_problem_size_with_gbj_by_c.py`                       | Largest (32k²) problem size with GB/J                      |
| `plot_cpu_scaling.py`                                                 | CPU thread scalability: time and speedup                    |
| `plot_key_takeaway.py`                                                | Key takeaway composite efficiency figure                    |
| `plot_largest_bars_and_lines.py`                                      | Component energy breakdown for largest image                |
| `plot_largest_image_metrics_comparison.py`                            | Efficiency metrics: WA vs SA location                       |
| `plot_lifetime_breakdown_comparison.py`                               | Lifetime carbon & cost breakdown by location                |
| `plot_performance_vs_workload.py`                                     | Throughput and energy efficiency vs workload                |
| `plot_regime_heatmaps.py`                                             | Regime heatmaps (throughput, energy, carbon, cost)          |

---

## Documentation

- [docs/metrics_spec.md](docs/metrics_spec.md) — full metric definitions
- [docs/protocol_v0.1.md](docs/protocol_v0.1.md) — benchmarking protocol
- [docs/index.md](docs/index.md) — documentation index

## PREESM

The simulator used to generate heterogeneous configurations is available under [simulateur](simulateur) and mainatined on gitlab : https://gitlab.insa-rennes.fr/jamorin/simulateur.

---

## Code of Conduct

We follow the [NumFOCUS Code of Conduct](https://numfocus.org/code-of-conduct).

---

## Datasets & Traces (Zenodo)

> The benchmark datasets and raw measurement traces used to produce all figures in this paper are published separately on Zenodo.
>

>
> Download the archive and place the CSV files under `data/` before running any plot scripts (see Quick Start below).


## astroCAMP Dataset Contributors


| Name                           | Affiliation                                               | Role           |
| ------------------------------ | --------------------------------------------------------- | -------------- |
| Orliac, Etienne Jacques        | École Polytechnique Fédérale de Lausanne               | Data collector |
| Constantinescu, Denisa-Andreea | EPFL-EcoCloud, Laboratoire des systèmes embarqués, EPFL | Project leader |
| Rodriguez Alvarez, Ruben       | École Polytechnique Fédérale de Lausanne               | Researcher     |
| Morin, Jacques                 | Université de Rennes                                     | Researcher     |
| Dardaillon, Mickael            | Institut National des Sciences Appliquées de Rennes      | Project member |
| Wang, Sunrise                  | Observatoire de la Côte d'Azur                           | Researcher     |
| Miomandre, Hugo                | Université de Rennes                                     | Project member |
| Mbuyi, Junior                  | Swiss Federal Institute of Technology in Lausanne         | Other          |
| Lopes, Yves                    | Swiss Federal Institute of Technology in Lausanne         | Other          |
| Ouvrard, Xavier                | Swiss Federal Institute of Technology in Lausanne         | Data collector |
| Peón-Quirós, Miguel            | École Polytechnique Fédérale de Lausanne               | Project leader |
| Nezan, Jean-François           | Université de Rennes                                  | Project member |
| Atienza Alonso, David          | École Polytechnique Fédérale de Lausanne (EPFL)        | Project member |

---

## Citation

If you use astroCAMP or the companion dataset in your research, please cite:

```bibtex

@inproceedings{astroCAMPpasc26,
  authors   = {Denisa-Andreea Constantinescu, Rubén Rodríguez Álvarez, Jacques Morin, Etienne Orliac, Mickaël Dardaillon, Sunrise Wang, Hugo Miomandre, Miguel Peón-Quirós, Jean-François Nezan, David Atienza},
  title     = {astroCAMP: A Co-design Analysis and Metrics Platform for SKA-scale Radio Interferometric Imaging},
  booktitle = {Platform for Advanced Scientific Computing Conference (PASC '26), June 29-July 01, 2026, Bern, Switzerland},
  year      = {2026},
  doi       = {https://doi.org/10.48550/arXiv.2512.13591},
}

@dataset{astroCAMPdata2026,
  author       = { Etienne Orliac,  Denisa-Andreea Constantinescu,  Rubén Rodríguez Álvarez, Jacques Morin,  Mickaël Dardaillon, Sunrise Wang, Hugo Miomandre, Junior Mbuyi, Yves Lopes, Xavier Ouvrard,  Miguel Peón-Quirós, Jean-François Nezan, David Atienza},
  title        = {{astroCAMP-data-v1.0}: {SKA}-{Low} visibility datasets and reference dirty images for cross-layer co-design benchmarking},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.20093790},
  url          = {https://doi.org/10.5281/zenodo.20093790}
}

```

---

## License

This work is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
