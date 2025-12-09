
# astroCAMP Metrics Specification (v0.1)

This document defines the astroCAMP metrics used in the benchmark protocol.
For each metric we provide:

- a formal **definition / formula**  
- **how to measure** it in practice  
- **how to interpret** it for SKA-like imaging  
- selected **references** for further detail  

The structure follows the metric layers used in the astroCAMP paper:

1. Performance and scalability  
2. Energy and carbon  
3. Algorithm / scientific quality  
4. System and workflow  
5. Economic and lifecycle  
6. FAIRness and reproducibility  

---

## 1. Performance and Scalability Metrics

### 1.1 Runtime (Time to Completion, Tc)

**Definition**

Total wall-clock time from the start of the imaging job to the availability of
the final science-ready image or cube.

- Unit: seconds (s)  
- Symbol: Tc  

**How to measure**

- Use `/usr/bin/time -v <command>` for simple runs; record the "Elapsed (wall clock) time".
- For scheduled jobs (SLURM, PBS, etc.), use job start/end timestamps.
- For workflow systems (e.g. CWL, Nextflow), log timestamps at workflow start and completion.

**Interpretation**

- Primary indicator of **time to solution**.
- Used directly as a core metric and in derived metrics (e.g. average power).
- Lower Tc is better, subject to **scientific quality constraints**.

**References**

- Hennessy and Patterson, *Computer Architecture: A Quantitative Approach* (6th ed).  
- SKA Science Data Challenge documents (for typical runtime scales).

---

### 1.2 Throughput (Theta)

**Definition**

Domain-specific throughput, e.g.:

- visibilities processed per second, or  
- image pixels updated per second.

A typical choice for radio imaging is:

$$
\Theta_V = \frac{N_{\text{vis}}}{T_c}
$$

- Unit: visibilities / second, or pixels / second  

**How to measure**

- Count total number of visibilities ingested or processed (N_vis) from the MeasurementSet or visibility file.
- Divide by measured Tc.

**Interpretation**

- Higher throughput indicates better utilisation of hardware for a given workload.
- Allows comparison of different systems or algorithms independent of absolute problem size.
- Useful for normalising energy and carbon metrics (e.g. visibilities per joule).

**References**

- Frigo and Johnson, *The Design and Implementation of FFTW3*, Proceedings of the IEEE 93(2), 2005.  
- Roofline model literature for performance per byte or per operation.

---

### 1.3 Peak Memory Usage (M_peak)

**Definition**

Maximum resident memory (RAM) used by the imaging process during the run.

- Unit: gigabytes (GB)  

**How to measure**

- Use `/usr/bin/time -v` (“Maximum resident set size”).
- Or use job scheduler telemetry (SLURM `MaxRSS`, etc.).
- For multi-process runs, report either:
  - per-process peak, or  
  - node-level peak (document which in the submission).

**Interpretation**

- Constrains problem sizes and influences scaling to larger surveys.
- High M_peak may limit deployment on memory-constrained nodes.
- Combined with performance and energy metrics, helps identify memory-bound workloads.

**References**

- Hennessy and Patterson, *Computer Architecture* (sections on memory hierarchy and capacity).  

---

### 1.4 Parallel Efficiency (E_p)

**Definition**

Parallel efficiency compares ideal linear speedup to observed speedup:

$$
E_p = \frac{T_{\text{seq}}}{n \, T_n}
$$

- T_seq: best-known sequential runtime  
- T_n: runtime with n parallel resources (cores, GPUs, or nodes)  
- Unit: dimensionless (0 to 1)  

**How to measure**

- Run the same configuration with a single resource (or minimal parallelism) to obtain T_seq.
- Run with n resources to obtain T_n.
- Compute E_p using the formula above.

**Interpretation**

- E_p close to 1 indicates good scalability.
- Lower E_p indicates overheads (communication, synchronisation, load imbalance).
- Helps distinguish algorithmic limits from system or implementation issues.

**References**

- Amdahl, “Validity of the Single Processor Approach to Achieving Large-Scale Computing Capabilities,” AFIPS 1967.  
- Gustafson, “Reevaluating Amdahl’s law,” Communications of the ACM, 1988.  

---

## 2. Energy and Carbon Metrics

### 2.1 Energy to Solution (E_c)

**Definition**

Total energy consumed during the job:

$$
E_c = \int_{t_0}^{t_1} P(t) \, dt
$$

- P(t): instantaneous power (Watts)  
- t0, t1: start and end times of the job  
- Unit: Joules (J)  

**How to measure**

1. Choose a power measurement method:
   - CPU: RAPL (Intel)  
   - GPU: NVIDIA NVML, ROCm SMI  
   - Node: IPMI, PDU, or external power meter  
2. Sample power at regular intervals (e.g. every 100 ms to 1 s).  
3. Integrate using discrete samples:

$$
E_c \approx \sum_i P_i \, \Delta t_i
$$

4. Ensure you document:
   - sampling frequency,  
   - what is measured (CPU package, GPU only, full node),  
   - any calibration or offset corrections.

**Interpretation**

- Core metric for **energy to solution**.
- Used to derive:
  - average power (P_avg = E_c / T_c),  
  - energy efficiency (e.g. visibilities per joule).  
- Lower E_c is better at fixed quality and problem size.

**References**

- Intel RAPL documentation and white papers.  
- NVIDIA NVML documentation.  
- Top500/Green500 methodology descriptions.  

---

### 2.2 Average Power (P_avg)

**Definition**

Average power consumption during the job:

$$
\bar{P} = \frac{E_c}{T_c}
$$

- Unit: Watts (W)  

**How to measure**

- Compute from measured E_c and T_c.

**Interpretation**

- Indicates how “hard” the system is driven during the workload.
- Useful for cross-checking against system power limits and facility constraints.
- Distinguish between:
  - power caps (hardware-enforced),  
  - average vs. peak power during bursts.

**References**

- SKA power and energy design documents (for 2–5 MW site constraints).  

---

### 2.3 Carbon to Solution (C_c)

**Definition**

Carbon footprint associated with E_c, using a carbon intensity factor κ(t, r):

$$
C_c = E_c \, \kappa(t,r)
$$

- κ(t, r): grid carbon intensity [gCO2e per joule] as a function of time and region  
- Unit: grams of CO2 equivalent (gCO2e)  

**How to measure**

1. Measure E_c as above.
2. Obtain κ(t, r) from:
   - national grid statistics,  
   - electricityMap or similar APIs,  
   - facility-provided carbon-intensity data.  
3. If κ is time-dependent, integrate:

$$
C_c \approx \sum_i P_i \, \Delta t_i \, \kappa(t_i, r)
$$

If only a time-averaged κ is available, use the simpler E_c * κ.

**Interpretation**

- Translates energy into environmental impact.
- Enables comparison across sites and energy mixes.
- Should be interpreted alongside **renewable fraction** and **PUE** where available.

**References**

- SKA Observatory sustainability reports.  
- IPCC guidelines for greenhouse gas inventories.  

---

### 2.4 Energy Efficiency (eta_E)

**Definition**

Energy efficiency in domain-specific units, e.g.:

$$
\eta_E = \frac{N_{\text{vis}}}{E_c}
$$

- Unit: visibilities per joule (vis/J)  

**How to measure**

- Count total visibilities processed (N_vis).  
- Measure E_c.  
- Compute N_vis / E_c.

**Interpretation**

- Higher η_E means more work per unit energy.
- Good for comparing systems where runtime may differ but energy budgets matter.
- Relates to Green500-style “flops per watt”, but in domain-specific terms.

**References**

- Green500 list methodology.  
- Energy-efficient HPC surveys.  

---

## 3. Algorithm / Scientific Quality Metrics

### 3.1 Dirty Image RMS (sigma_dirty)

**Definition**

The root mean square (RMS) of pixel intensities in a source-masked background
region of the dirty or residual image:

$$
\sigma_{\text{dirty}} =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N} (I_i - \bar{I})^2
}
$$

- I_i: pixel values in the background region  
- \bar{I}: mean of those pixel values (often ~0)  
- N: number of pixels in mask  
- Unit: Jy/beam  

**How to measure**

1. Load the dirty or residual image (FITS) as a 2D array.
2. Generate a background mask:
   - either from a provided mask file,  
   - or via a source finder (e.g. PyBDSF) masking sources and bright residuals.  
3. Compute σ_dirty on masked pixels with the formula above.
4. Use provided `metrics/dirty_rms.py` script for standardisation.

**Interpretation**

- Proxy for achieved sensitivity and noise floor.
- Deviations from theoretical thermal noise indicate:
  - calibration errors,  
  - residual sidelobes,  
  - numerical approximations,  
  - suboptimal deconvolution.  
- Used as a core quality metric in astroCAMP, often with allowed relative deviation (e.g. within 5% of reference).

**References**

- Thompson, Moran, and Swenson, *Interferometry and Synthesis in Radio Astronomy*, 3rd ed.  
- CASA documentation for image statistics and noise estimation.  

---

### 3.2 Peak Signal-to-Noise Ratio (PSNR)

**Definition**

For a reconstructed image $\hat{I}$ and reference image $I_{\text{ref}}$,

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \big( \hat{I}_i - I_{\text{ref},i} \big)^2
$$

$$
\text{PSNR} = 10 \log_{10} \left( \frac{I_{\text{max}}^2}{\text{MSE}} \right)
$$

- I_max is typically the maximum absolute value in I_ref (or a known dynamic range reference).  
- Unit: decibels (dB)  

**How to measure**

- Align and regrid images if necessary so they are pixel-registered.
- Use `metrics/psnr_ssim.py` with:
  - submission image,  
  - reference image.  

**Interpretation**

- Higher PSNR indicates closer agreement to reference in an L2 sense.
- Differences of a few tenths of a dB may already be significant for high-dynamic-range data.
- PSNR alone is not sufficient (it is global and can miss structural differences), but combined with SSIM and dirty RMS it forms a robust fidelity set.

**References**

- Huynh-Thu and Ghanbari, “Scope of validity of PSNR in image/video quality assessment,” Electronics Letters, 2008.  

---

### 3.3 Structural Similarity Index (SSIM)

**Definition**

SSIM measures structural similarity between images and is defined (for a window)
using local means, variances, and covariances. Global SSIM is typically the mean
over windows.

Range: [0, 1], with 1 being identical.

**How to measure**

- Use `skimage.metrics.structural_similarity` or equivalent in `metrics/psnr_ssim.py`.
- Use consistent parameters (window size, Gaussian weighting) across submissions.

**Interpretation**

- More sensitive than PSNR to structural distortions (e.g. shape changes, artefacts).
- High SSIM (close to 1) with good PSNR and acceptable σ_dirty is a strong indicator of scientific fidelity.

**References**

- Wang et al., “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Processing, 2004.  

---

### 3.4 Flux Error

**Definition**

For a set of matched sources, with measured flux S and reference flux S_ref:

$$
\epsilon_{\text{flux}} = \text{median}\left( \frac{|S - S_{\text{ref}}|}{S_{\text{ref}}} \right)
$$

- Unit: dimensionless (often reported as a percentage).  

**How to measure**

1. Use a source finder (e.g. PyBDSF) to extract source catalogs from:
   - submission image,  
   - reference image.  
2. Cross-match sources by position.
3. Compute absolute relative flux errors and take the median.

**Interpretation**

- Direct probe of photometric accuracy.
- Important for continuum surveys and H I science that depend on integrated flux.
- Tolerances (e.g. ≤ 5%) should be aligned with science requirements.

**References**

- Offringa et al., WSClean documentation and validation papers.  
- PyBDSF documentation.  

---

### 3.5 Astrometric Error

**Definition**

Positional error for matched sources, typically:

$$
\epsilon_{\text{astro}} = \text{median}\left( \frac{\sqrt{(\Delta \alpha)^2 + (\Delta \delta)^2}}{\theta_{\text{beam}}} \right)
$$

- Δα, Δδ: RA and Dec differences (in radians or arcseconds)  
- θ_beam: synthesized beam FWHM (converted to the same units)  
- Unit: fraction of beam FWHM  

**How to measure**

- Use cross-matched source catalogs as for flux error.
- Compute position offsets for each matched source and normalise by beam FWHM.
- Report the median (and optionally distribution).

**Interpretation**

- Measures astrometric fidelity and direction-dependent calibration quality.
- Critical for cross-matching with other surveys and for stacking analyses.

**References**

- Smirnov, “Revisiting the radio interferometer measurement equation,” A&A series.  

---

### 3.6 Dynamic Range (DR)

**Definition**

Dynamic range of an image:

$$
\text{DR} = \frac{I_{\text{max}}}{\sigma_{\text{residual}}}
$$

- I_max: peak brightness in the image  
- σ_residual: RMS of the residual image (or dirty background)  

**How to measure**

- Find the maximum pixel intensity in the cleaned image.
- Compute residual RMS using a source-masked region.
- Compute the ratio.

**Interpretation**

- High DR is necessary for accurate imaging in fields with bright and faint structure.
- Sensitive to calibration errors, deconvolution performance, and numerical precision.

**References**

- Perley, “Imaging,” in Synthesis Imaging in Radio Astronomy, ASP Conf. Ser. 6.  

---

### 3.7 Spectral Fidelity (epsilon_spec) – L1

**Definition**

Spectral fidelity measures how well the spectrum or cube matches a reference.
Implementation may vary; a simple option is:

$$
\epsilon_{\text{spec}} =
\sqrt{
\frac{1}{N_{\text{vox}}}
\sum_{k} \big( I_k - I_{\text{ref},k} \big)^2
}
$$

where the sum can be over:

- spatially integrated spectra,  
- selected voxels, or  
- per-source line profiles.

**How to measure**

- Use `metrics/spectral_fidelity.py`.
- Define a region or set of sources.
- Compute per-channel or aggregate RMS difference between submission and reference cube.

**Interpretation**

- Crucial for H I, EoR, and other line-science cases.
- Sensitive to:
  - spectral leakage,  
  - channel misregistration,  
  - non-linear effects of approximations.

**References**

- Braun et al., “Survey Speed Requirements for SKA HI Surveys,” SKA technical reports.  
- EoR simulation and analysis literature (for spectral smoothness metrics).  

---

## 4. System and Workflow Metrics

### 4.1 Utilisation (U)

**Definition**

Fraction of walltime during which major compute devices are actively executing
the workload:

$$
U = \frac{t_{\text{active}}}{t_{\text{total}}}
$$

- Unit: dimensionless (0 to 1)  

**How to measure**

- Use monitoring tools (e.g. `nvidia-smi`, `top`, Prometheus/Grafana) to obtain utilisation time series.
- Define t_active as the total time where utilisation exceeds a threshold (e.g. 10%).
- t_total ≈ T_c.

**Interpretation**

- Low U suggests I/O-bound or synchronisation-heavy workloads.
- Higher U indicates better resource usage but might increase P_avg.

**References**

- HPC system monitoring and performance analysis best practices (e.g. PAPI, LIKWID, VTune docs).  

---

### 4.2 I/O Throughput and Imbalance

**Definition (Throughput)**

Average sustained I/O rate:

$$
B_{\text{I/O}} = \frac{V_{\text{I/O}}}{T_c}
$$

- V_IO: total volume of data read/written  
- Unit: MB/s or GB/s  

**Definition (Imbalance)**

Imbalance across nodes or ranks:

$$
\Delta_{\text{I/O}} =
\frac{\max_i b_i - \min_i b_i}{\bar{b}}
$$

- b_i: per-node bandwidth  
- \bar{b}: mean bandwidth  

**How to measure**

- Use `iostat`, Lustre/GPFS stats, or application-level logs.
- For imbalance, collect per-node metrics and compute Δ_IO.

**Interpretation**

- Low throughput with high Tc may indicate I/O bottlenecks.
- High imbalance indicates stragglers and poor load distribution.

**References**

- Carns et al., Darshan I/O characterization tool.  

---

## 5. Economic and Lifecycle Metrics

### 5.1 Total Cost of Ownership (C_TCO)

**Definition**

Sum of capital and operational costs over the system lifetime:

$$
C_{\text{TCO}} = C_{\text{capex}} + C_{\text{opex}}
$$

- Unit: currency (e.g. EUR)  

**How to measure**

- Use facility or project accounting data.
- Allocate costs per node or per job using a reasonable costing model.

**Interpretation**

- Enables comparison of algorithm/hardware choices on financial grounds.
- Used with performance and carbon metrics to identify economically viable solutions.

**References**

- TCO models from major HPC centres and cloud providers.  

---

### 5.2 Energy Cost per Job (C_E)

**Definition**

Energy cost of a job:

$$
C_E = E_c \, p_E
$$

- p_E: electricity price (e.g. EUR per kWh)  
- Unit: currency (e.g. EUR)  

**How to measure**

- Choose p_E from facility or national tariffs.
- Convert E_c from Joules to kWh if needed (1 kWh = 3.6e6 J).

**Interpretation**

- Direct link between energy usage and operational budget.
- Useful for long-term planning of SKA computing operations.

**References**

- Facility-specific cost models; national energy price statistics.  

---

## 6. FAIRness and Reproducibility Metrics

### 6.1 Determinism Ratio (D)

**Definition**

Fraction of runs that produce bit-identical outputs:

$$
D = \frac{N_{\text{ident}}}{N_{\text{runs}}}
$$

- Unit: dimensionless (0 to 1)  

**How to measure**

- Run the same configuration multiple times (e.g. N_runs = 5).
- Compute checksums (e.g. SHA-256) of output images/cubes.
- Count N_ident, the number of runs that match a reference checksum.

**Interpretation**

- D = 1 indicates fully deterministic execution.
- D < 1 indicates non-determinism (e.g. from race conditions, non-deterministic reductions, random seeds).
- Non-determinism is not always problematic, but should be understood and documented.

**References**

- Reproducible research guidelines in computational science (e.g. ACM, Nature).  

---

### 6.2 Metric Variance (V_m)

**Definition**

Relative variance of a metric M across repeated runs:

$$
V_m = \frac{\sigma_M}{\mu_M}
$$

- σ_M: standard deviation over runs  
- μ_M: mean over runs  
- Unit: dimensionless  

**How to measure**

- Repeat the same benchmark configuration multiple times.
- Compute mean and standard deviation for runtime, energy, or quality metrics.

**Interpretation**

- Low V_m indicates stable measurements (good for benchmark comparisons).
- High V_m suggests variability due to shared infrastructure, thermal throttling, or non-deterministic behaviour.

**References**

- Statistical best practices in benchmarking and performance evaluation.  

---

This specification is expected to evolve as new metrics are proposed and validated
by the astroCAMP community. All changes should be versioned and documented
in future `metrics_spec` revisions (v0.2, v1.0, etc.).
