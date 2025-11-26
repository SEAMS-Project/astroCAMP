# metrics/

This directory contains definitions and reference implementations of the
astroCAMP metrics used in the benchmark protocol.

Core metrics include:

- performance: time to completion (Tc), throughput
- energy and carbon: energy to solution (Ec), carbon to solution (Cc)
- scientific quality: dirty image RMS, PSNR, SSIM, flux error, astrometric error,
  dynamic range, spectral fidelity
- system and workflow: utilisation, parallel efficiency

Typical contents:

- `psnr_ssim.py` – computes PSNR and SSIM from two images
- `dirty_rms.py` – computes dirty/residual RMS in a source-masked region
- `flux_astrometry.py` – compares source catalogs to compute flux and position errors
- `dynamic_range.py` – measures peak brightness and residual RMS
- `spectral_fidelity.py` – computes metrics across a spectral cube
- `energy_carbon.py` – integrates power logs and applies a carbon intensity model

All scripts should accept command-line arguments and write results in both
human-readable and machine-readable formats (JSON/CSV).
