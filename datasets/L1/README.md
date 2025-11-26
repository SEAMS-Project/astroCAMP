# datasets/L1 – Spectral Line / HI Cube

L1 is a spectral line benchmark task for HI and other line emission, suitable
for testing spectral fidelity, cube imaging, and line recovery.

Recommended primary dataset:

- SKA Science Data Challenge 2 (SDC2) HI galaxy challenge

This folder should contain:

- `data/` – symlink or path to the local spectral dataset
- `reference/` – reference cube(s) and line catalogs
- `L1_manifest.json` – description of files
- `L1_config_example.yaml` – baseline imaging configuration

Document:

- where to obtain the data (URL, DOI)
- the spectral setup (channels, bandwidth, resolution)
- any regridding or format conversion needed before running astroCAMP metrics
