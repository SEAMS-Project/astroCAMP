# astroCAMP Community Benchmark Protocol v0.1

This document defines the initial astroCAMP Community Benchmark Protocol
(version 0.1). The protocol specifies benchmark tasks, tracks, metrics,
quality constraints, required artefacts, and submission rules.

The goal is to provide a transparent, reproducible, and science-aligned
framework for evaluating radio interferometric imaging pipelines under the
performance, energy, carbon, and scientific-quality constraints relevant
to SKA-like workloads.

---

## 1. Objectives and Scope

astroCAMP aims to:

1. Quantify trade-offs between scientific fidelity, performance, energy use,
   carbon cost, and resource utilisation.
2. Enable fair comparison of imaging pipelines across algorithms and hardware.
3. Support hardware–software co-design under realistic SKA constraints.
4. Provide a stable metric and dataset backbone for community-defined
   imaging-quality thresholds.

Scope of protocol v0.1:

- SKA-like continuum and spectral-line imaging tasks (C1, C2, L1).
- Single-node and small multi-node CPU/GPU systems.
- Calibrated visibilities only (no calibration pipelines yet).
- Offline imaging (no real-time streaming constraints).

Later versions (v0.2+) may include calibration, transient imaging, and larger
multi-node experiments.

---

## 2. Benchmark Tasks and Tracks

Each benchmark task is defined by:

- a dataset (calibrated visibilities),
- a reference image or data product, and
- a set of scientific and technical metrics.

### 2.1 Tasks in v0.1

- **C1 – Continuum Widefield**  
  SKA1-Mid-like deep field, moderate dynamic range. Used to assess general
  continuum imaging performance.

- **C2 – High Dynamic Range (HDR)**  
  Bright-source field (e.g. calibrator or bright cluster) stressing PSF control,
  precision, and deconvolution depth.

- **L1 – Spectral Line / HI Cube**  
  SKA1-Low or SKA1-Mid-like spectral-line data (e.g. HI cube), used to assess
  spectral fidelity and line recovery.

The exact datasets are documented in `docs/datasets.md` and in the
`datasets/<task>/README.md` files.

### 2.2 Tracks

Each task supports two benchmark tracks: **closed** and **open**.

#### Closed Track

The closed track fixes the imaging configuration. This includes:

- image size and pixel scale,
- weighting scheme,
- gridder and major/minor cycle structure,
- deconvolution depth and stopping criteria.

Participants may optimise the implementation (parallelisation, memory layout,
libraries, etc.) but may **not** change these high-level imaging parameters.

The closed track primarily measures system and implementation efficiency.

#### Open Track

The open track allows algorithmic innovation and approximations, as long as:

- the same input data are used,
- the required metrics are produced, and
- task-specific quality constraints are satisfied.

Participants may change:

- gridder or imaging kernel (e.g. NUFFT, IDG, alternative w-handling),
- precision (e.g. mixed-precision, reduced-precision),
- deconvolution strategies,
- other algorithmic degrees of freedom.

The open track is designed to explore fidelity–efficiency trade-offs and
co-design opportunities.

---

## 3. Core Metrics and Quality Constraints

astroCAMP defines a multi-layer metric taxonomy (performance, energy/carbon,
system, algorithm/quality, economics, FAIRness). This protocol specifies a
subset of **core metrics** that are mandatory for valid submissions.

### 3.1 Mandatory Core Metrics

For each run and task, report:

#### Performance

- **Runtime (Tc)**  
  Total wall-clock time from start of imaging to final product.
  - Unit: seconds
  - Measurement: scheduler logs, workflow timestamps, or `/usr/bin/time`.

- **Throughput** (optional but recommended)  
  Domain-specific throughput (e.g. visibilities per second, or pixels per second).
  - Computed as number_of_visibilities / Tc.

- **Peak Memory Usage**  
  Maximum resident set size during the run.
  - Unit: GB
  - Measurement: `/proc/meminfo`, job scheduler telemetry, or `time -v`.

#### Energy and Carbon

- **Energy to Solution (Ec)**  
  Integral of power over the duration of the run.
  - Unit: Joules
  - Measurement: from power logs using RAPL (CPU), NVML (GPU), IPMI/PDU (node).

- **Average Power**  
  Ec / Tc.
  - Unit: Watts

- **Carbon to Solution (Cc)**  
  Ec multiplied by a location/time-dependent carbon intensity factor.
  - Unit: gCO2e
  - Calculation:  
    `Cc = Ec * kappa(t, region)`  
    where `kappa` is obtained from a grid carbon-intensity dataset or API.

#### Scientific Quality

All quality metrics are computed using reference scripts in `metrics/`.

- **Dirty Image RMS (sigma_dirty)**  
  RMS of image values in a source-masked background region of the dirty
  or residual image.
  - Unit: Jy/beam
  - Interpretation: achieved noise floor; captures thermal noise plus artefacts.

- **Peak Signal-to-Noise Ratio (PSNR)**  
  PSNR between the submission image and the reference image.
  - Unit: dB
  - Computed using MSE against the reference image.

- **Structural Similarity Index (SSIM)**  
  SSIM between submission and reference image.
  - Unit: dimensionless (0 to 1)

- **Flux Error**  
  Median relative flux error for matched sources:
  - Definition: median(|S - S_ref| / S_ref)

- **Astrometric Error**  
  Positional error for matched sources (e.g. in pixels or fraction of beam).
  - Typical unit: fraction of synthesized beam FWHM.

- **Dynamic Range (DR)**  
  Peak brightness divided by residual RMS:
  - DR = I_max / sigma_residual

- **Spectral Fidelity** (for L1)  
  Channel-wise RMS or other spectral metrics between submission and reference
  cubes (e.g. line profiles, integrated flux per source).

#### System and Workflow

- **Utilisation**  
  Fraction of walltime during which the main compute devices (CPUs/GPUs) are
  actively executing.
  - Derived from monitoring tools or scheduler logs.

- **Parallel Efficiency** (for parallel runs)  
  Defined as T_seq / (n * T_n), where T_seq is the best known sequential time
  and T_n is the run time with n resources.

### 3.2 Task-Specific Quality Constraints

Each task defines minimum acceptable quality thresholds. Example constraints
for v0.1 (to be refined with community input):

- **C1 (continuum)**  
  - PSNR within 0.5 dB of the reference solution.  
  - Dirty RMS difference within 5 percent of reference.  
  - Median flux error within 5 percent.  
  - Astrometric error within 0.1 beam FWHM.

- **C2 (high dynamic range)**  
  - Similar constraints to C1, with additional DR requirement (e.g. DR above
    a specified threshold relative to reference).

- **L1 (spectral line)**  
  - Spectral RMS within 5 percent of reference in line-free channels.  
  - Integrated line flux within a few percent of the reference for key sources.

Submissions that fail quality constraints are labelled as **quality-violating**.
They may still be listed but are not ranked on efficiency-focused leaderboards.

---

## 4. Benchmark Artefacts

Each astroCAMP-bench release must include the following artefacts:

1. **Datasets**  
   - Calibrated visibilities for each task in standard formats (e.g. MeasurementSet, HDF5).
   - Documentation of telescope setup and simulation parameters.

2. **Reference Images and Catalogs**  
   - High-fidelity reference images (e.g. from conservative high-precision runs).
   - Reference source catalogs (positions, fluxes, spectra) for flux/astrometry metrics.

3. **Metric Evaluation Scripts**  
   - PSNR/SSIM computation.  
   - Dirty RMS, dynamic range.  
   - Catalog comparison (flux and astrometry).  
   - Spectral fidelity for cubes.  
   - Energy and carbon calculation from power logs.

4. **Power Measurement Utilities**  
   - Example scripts for RAPL, NVML, IPMI, or PDU logging.

5. **Baseline Submissions**  
   - Reference runs for representative algorithms (e.g. WSClean, IDG/WS-Snapshot, BIPP) on common CPU/GPU systems.

All artefacts must be versioned (e.g. `astroCAMP-bench v0.1`) and stored with
persistent identifiers (e.g. DOIs).

---

## 5. Submission Rules

### 5.1 Environment Disclosure

Each submission must include a machine-readable `system.json` (or similar) with:

- CPU and GPU models, counts, and clock speeds.
- Memory, storage, and node interconnect details.
- Operating system and version.
- Compiler, MPI, and library versions.
- Imaging code version (commit hash, release tag).
- Power measurement method and sampling frequency.
- Carbon-intensity model or dataset used.

### 5.2 Run Protocol

- Each reported result must be the **median of at least three runs** with
  identical configuration.
- For each run, log:
  - start and end timestamps,
  - runtime (Tc),
  - integrated energy (Ec),
  - peak memory usage,
  - CPU/GPU utilisation (if available).

- Quality metrics must be computed using the official scripts in `metrics/`.

### 5.3 Allowed Modifications by Track

- **Closed Track**
  - Must use the provided baseline imaging configuration.
  - May optimise implementation (e.g. threading, kernel tuning).
  - May adjust precision only if quality constraints remain satisfied.

- **Open Track**
  - May change algorithmic strategies (e.g. gridder, deconvolution, precision).
  - Must use the same input data and metric definitions.
  - Must satisfy the same quality constraints for “valid” status.

---

## 6. Reporting and Leaderboards

Submissions should be summarised in a `metrics.json` file containing at least:

- runtime (Tc),
- energy to solution (Ec),
- carbon to solution (Cc),
- dirty RMS, PSNR, SSIM,
- flux and astrometric error,
- dynamic range,
- utilisation and parallel efficiency (if applicable).

Public leaderboards may:

- Rank submissions by runtime, energy, carbon, or composite metrics (e.g. work per gCO2e).
- Distinguish **quality-valid** from **quality-violating** submissions.
- Highlight Pareto frontiers (e.g. Ec vs. PSNR, Cc vs. RMS).

---

## 7. FAIRness, Reproducibility, and Governance

To support reproducibility and long-term value:

- Submissions should include runnable scripts or containers when possible.
- Datasets must be versioned, and metric scripts kept backward compatible or
  versioned alongside protocol updates.
- Changes to tasks, metrics, or quality thresholds are made in versioned
  protocol documents (e.g. `protocol_v0.2.md`) and discussed openly.

A lightweight steering group is encouraged to:

- review new tasks and metrics,
- define and revise quality thresholds,
- validate baseline submissions,
- coordinate major version updates.

---

## 8. Roadmap

Planned future extensions:

- v0.2: add RM-grid and transient imaging tasks; multi-node scaling benchmarks; more detailed economic metrics.
- v1.0: establish community-agreed quality thresholds per science case; annual benchmark cycles; more formal submission and review flow.
