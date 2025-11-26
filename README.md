# astroCAMP-bench

astroCAMP-bench is a community benchmark suite for radio interferometric imaging,
inspired by MLPerf. It provides standardised datasets, reference images, metric
definitions, and a submission protocol for evaluating imaging pipelines across:

- scientific fidelity
- performance and scalability
- energy use and carbon footprint
- system utilisation and cost
- FAIRness and reproducibility

The goal is to enable scientifically valid, feasible, and sustainable imaging
pipelines for SKA-like workloads (SKA1-Low and SKA1-Mid).

## Repository layout

- `datasets/` – benchmark datasets and references (C1, C2, L1)
- `metrics/` – metric definitions and evaluation scripts
- `tools/` – helper tools (power measurement, image statistics, etc.)
- `baselines/` – reference implementations (WSClean, IDG, BIPP)
- `configs/` – benchmark configuration files (YAML/JSON)
- `scripts/` – helper scripts to run the benchmarks
- `results/` – local output directory for benchmark runs
- `submissions/` – example submission packages
- `docs/` – protocol, design notes, and extended documentation

For the full protocol, see `docs/README.md`.
