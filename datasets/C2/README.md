# TODO future release: datasets/C2 – High Dynamic Range

C2 is a high dynamic range (HDR) benchmark task with bright compact sources
that stress PSF control, deconvolution, and dynamic range.

Typical choices include:

- VLA calibrator fields (e.g. 3C286, 3C147), or
- bright MeerKAT cluster fields (MGCLS) with strong compact and diffuse emission

This folder should contain:

- `data/` – symlink or path to the local HDR dataset
- `reference/` – reference HDR image(s) and catalog(s)
- `C2_manifest.json` – description of files
- `C2_config_example.yaml` – baseline imaging configuration

Document the exact dataset used, including:

- archive or DOI
- observing setup (band, frequency, time, configuration)
- any pre-processing required before imaging
