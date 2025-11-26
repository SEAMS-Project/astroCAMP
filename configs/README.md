# configs/

This directory contains benchmark configuration files (YAML or JSON) that
describe how to run a given task and track.

Typical examples:

- `C1_closed_wsclean.yaml` – C1, closed track, WSClean baseline
- `C1_open_bipp.yaml` – C1, open track, BIPP variant
- `L1_closed_idg.yaml` – L1 spectral line benchmark

Each config should include:

- dataset and task (C1, C2, L1)
- imaging parameters (image size, cell size, weighting, etc.)
- algorithm-specific settings
- resource configuration (threads, GPUs, MPI ranks)
