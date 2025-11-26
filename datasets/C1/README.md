# datasets/C1 – Continuum Widefield

C1 is a continuum widefield benchmark task designed to emulate SKA1-Mid deep
extragalactic imaging.

Recommended primary dataset:

- SKA Science Data Challenge 1 (SDC1) continuum field
- Format: MeasurementSet or equivalent
- Source: SKA SDC1 website or archive (see paper/docs for link)

This folder should contain:

- `data/` – symlink or path to the local dataset (not tracked by git)
- `reference/` – reference image(s) and catalog(s) for C1
- `C1_manifest.json` – machine-readable description of files
- `C1_config_example.yaml` – example imaging configuration

Do not commit full data products. Instead, document:

1. How to download the dataset
2. Any required pre-processing
3. How to verify integrity (checksum, file size, etc.)
