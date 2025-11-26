# datasets/

This directory contains all benchmark datasets used by astroCAMP-bench.

Each task has its own subfolder:

- `C1/` – continuum widefield benchmark (SKA1-Mid–like deep field)
- `C2/` – high dynamic range benchmark (bright-source, PSF stress test)
- `L1/` – spectral line / HI cube benchmark (SKA1-Low / SKA1-Mid line science)

Datasets are not committed directly to the repository if they are large.
Instead, each subfolder contains:

- download instructions
- checksums or DOIs
- a description of the science case and usage in the benchmark

See the `README.md` in each subfolder for details.
