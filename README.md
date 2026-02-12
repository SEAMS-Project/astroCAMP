
# 🌌 **astroCAMP Framework**

<img width="682" height="425" alt="image" src="https://github.com/user-attachments/assets/20cc4fcc-3351-45a1-a5c9-6c1f0e505557" />

## 🔍 **What is astroCAMP?**

**astroCAMP (Astronomical Co-design Analysis and Metrics Platform)** is a community-driven framework for evaluating radio-interferometric imaging pipelines under **performance**, **energy**, **scientific quality**, and **sustainability** constraints.

Its goal is to support **hardware–software co-design** for SKA-scale workloads (SKA1-Low and SKA1-Mid) by enabling reproducible, quantitative exploration of:

* system-level behaviour (runtime, energy, throughput),
* platform-level device utilisation (CPU/GPU/FPGA/ASIC),
* algorithmic scientific fidelity (RMS, PSNR, astrometry, photometry, spectra),
* carbon and cost efficiency.

astroCAMP provides **datasets, baseline implementations, standard metrics, and evaluation tools**, allowing fair comparison across heterogeneous architectures and imaging approaches.



# 🎯 **Why astroCAMP?**

The Square Kilometre Array (SKA) will operate under **strict power (2–5 MW)** and **cost** envelopes while processing **petascale imaging workloads**.
Most existing imaging benchmarks:

* measure performance **only**,
* neglect scientific fidelity,
* ignore carbon and economic constraints,
* use inconsistent metrics across tools,
* lack reproducibility across HPC systems.

**astroCAMP fills this gap** by introducing a **unified, multi-layer metric suite** and a reproducible **benchmarking protocol** for co-design.



# 🧩 **What’s in this Repository?**

* **Standardised benchmark datasets** for SKA-like workloads
* **Reference output dirty images** for quality comparison
* **A unified suite of performance–quality–sustainability–economic metrics**
* **Baseline imaging pipelines** (e.g., WSClean, IDG)
* **Scripts for power, memory, throughput, and fidelity evaluation**
* **Configuration files** for running controlled experiments
* **Documentation and reproducibility protocol**



# 🚀 **Quick Start**

```bash
# Clone the repository
git clone git@github.com:SEAMS-Project/astroCAMP.git
cd astrocamp

# Run a benchmark configuration
./scripts/run_benchmark.sh configs/wsclean_ska_low.yaml

# Evaluate quality and system metrics
./scripts/evaluate_metrics.py results/wsclean_ska_low/
```

Outputs include:

* system-level logs
* energy traces
* quality metrics 
* sustainability and cost metrics
* comparison plots vs. reference images

---

# 📐 **Core astroCAMP Co-Design Metrics**

astroCAMP defines **four co-design layers**, each quantifying a different aspect of imaging performance and scientific validity.
All symbols are defined **inline** so the table is fully self-contained.



## **1. System-Level Metrics (End-to-End Execution on Heterogeneous Nodes)**

| ID     | Metric             | Formula           | Unit           | Meaning & Notation                                          |
| ------ | ------------------ | ----------------- | -------------- | ----------------------------------------------------------- |
| **A1** | Time-to-solution   | `T_c`             | s              | Total job runtime. `T_c` = wall-clock time.                 |
| **A2** | Energy-to-solution | `E_c = ∫ P(t) dt` | J              | Total energy. `P(t)` = instantaneous power.                 |
| **A3** | Throughput         | `Θ = N / T_c`     | vis/s or img/s | Science processed per second. `N` = visibilities or images. |
| **A4** | Energy efficiency  | `η_E = N / E_c`   | vis/J          | Visibilities per joule.                                     |


## **2. Platform-Level Metrics (CPU / GPU / FPGA / ASIC Devices)**

| ID     | Metric            | Formula                  | Unit | Meaning & Notation                                |
| ------ | ----------------- | ------------------------ | ---- | ------------------------------------------------- |
| **A5** | Utilisation       | `U = t_active / t_total` | –    | Device activity. `t_active` = active kernel time. |
| **A6** | Memory bandwidth  | `B_mem = Bytes / T_c`    | GB/s | Sustained device memory throughput.               |
| **A7** | Peak memory usage | `M_peak`                 | GB   | Maximum resident memory footprint.                |



## **3. Algorithmic Quality Metrics (Scientific Validity)**

| ID   | Metric                 | Formula                                           | Unit    | Meaning & Notation (Self-contained)                                                                                                                                               |
|------|------------------------|---------------------------------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| B1   | Dirty-image RMS        | `σ_dirty = sqrt( (1/N) Σ (I_i – Ī)² )`            | Jy/beam | Noise + artefacts in the dirty image. `I_i` = pixel values; `Ī` = mean pixel intensity; `N` = number of pixels; `σ_dirty` = root-mean-square deviation from the mean.             |
| B2   | PSNR / SSIM            | `PSNR = 10 log10( I_max² / MSE )`                 | dB / –  | Fidelity vs reference image `I_ref`. `I_max` = maximum pixel value; `MSE` = mean squared error between reconstruction `Ĩ` and `I_ref`; SSIM = structural similarity between them. |
| B3   | Dynamic range          | `DR = I_max / σ_res`                              | –       | Ratio of peak brightness to residual noise. `I_max` = brightest pixel in the image; `σ_res` = RMS of the residual image; higher `DR` = better faint-source detectability.         |
| B4   | Astrometric error      | `ε_astro = (1/N) Σ L2(x_i – x_i_ref)`           | arcsec or px | Position error of detected sources. `x_i` = measured source positions; `x_i_ref` = reference (catalogue) positions; `ε_astro` = mean positional offset over `N` sources.     |
| B5   | Photometric error      | `ε_photo = (1/N) Σ L1(S_i – S_i_ref)`             | Jy      | Flux-density error. `S_i` = measured flux densities; `S_i_ref` = reference fluxes; `ε_photo` = mean absolute flux difference over `N` matched sources.                            |
| B6   | Spectral fidelity      | `ε_spec = (1/N_ν) Σ L1 (I(ν) – I_ref(ν))`          | Jy      | Per-channel spectral error. `I(ν)` = reconstructed intensity at frequency `ν`; `I_ref(ν)` = reference intensity; `N_ν` = number of frequency channels; `ε_spec` = mean absolute per-channel deviation. |



## **4. Sustainability Metrics (Energy → Carbon)**

| ID     | Metric             | Formula              | Unit      | Meaning & Notation                                  |
| ------ | ------------------ | -------------------- | --------- | --------------------------------------------------- |
| **C1** | Carbon-to-solution | `C_c = E_c * κ(t,r)` | gCO₂e     | Carbon footprint. `κ(t,r)` = grid carbon intensity. |
| **C2** | Carbon efficiency  | `η_C = N / C_c`      | vis/gCO₂e | Science per gram CO₂ emitted.                       |



## **5. Economic Metrics (Cost-Aware Co-Design)**

| ID     | Metric                  | Formula                    | Unit  | Meaning & Notation                                  |
| ------ | ----------------------- | -------------------------- | ----- | --------------------------------------------------- |
| **E1** | Total cost of ownership | `C_TTO = C_capex + C_opex` | €     | Hardware lifetime cost.                             |
| **E2** | Cost per job            | `C_E = E_c * p_E`          | €     | Monetary execution cost. `p_E` = electricity price. |
| **E3** | Cost efficiency         | `Θ / C_TTO`                | ops/€ | Science per euro invested.                          |


# 🧪 **Benchmark Datasets**

AstroCAMP includes curated datasets representing:

* **SKA1-Low** visibility volumes
* [TODO] **SKA1-Mid** continuum datasets
* **Dirty-image references** for quality verification

All datasets include metadata describing:
* reference outputs,
* [TODO] numerical precision requirements,
* [TODO] acceptable tolerances


---

# 🏗️ **Repository Structure**

```
astrocamp/
│
├── datasets/        # Standard benchmark datasets + references
├── metrics/         # Metric definitions, measurement tools, analysis scripts
├── baselines/       # Baseline pipelines (WSClean, IDG, etc.)
├── tools/           # Power measurement, GPU/CPU monitoring, image stats
├── configs/         # YAML/JSON benchmark configs
├── scripts/         # Benchmark runners + evaluation utilities
├── results/         # Local results directory (auto-generated)
└── docs/            # Protocol, methodology, design notes
```

---

# 📄 **Benchmarking Protocol (Short Summary)**

astroCAMP’s protocol ensures:

* **Reproducibility:** fixed configs, standard datasets, controlled measurement tools
* **Comparability:** consistent metrics across algorithms/tools/architectures
* **Scientific validity:** quality metrics tied to astronomical requirements
* **Co-design relevance:** integrates performance, energy, carbon, and cost

A full specification is provided in `/docs/protocol.md`.

---

# 🤝 **Contribute**

astroCAMP is a **community benchmark**.
Contributions welcome for:

* new datasets
* new imaging pipelines
* new target architectures (FPGA, ASIC, RISC-V)
* improved metrics
* documentation, tutorials, results

# Code of Conduct

We follow the [NumFOCUS Code of Conduct](https://numfocus.org/code-of-conduct).


# 📬 **Contact**

For questions, collaborations, or adding your pipeline to the benchmark suite, please open an issue or contact the maintainers.



