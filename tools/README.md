# tools/

This directory contains helper tools that support the benchmark, but are not
metrics themselves.

Subfolders:

- `power_measurements/` – scripts for reading hardware power sensors
- `image_stats/` – generic image inspection and plotting tools

These tools are optional but recommended for reproducing the reference
measurement workflow described in the astroCAMP paper and docs.

## Additional tools
Additional tools were used to produce our results : 

### PREESM
[PREESM](https://preesm.github.io/) is a rapid prototyping tool we used to generate design configuration. We used its "heterogeneous" branch, under active development.

## Vitis
We used the [Vitis](https://www.amd.com/fr/products/software/adaptive-socs-and-fpgas/vitis.html) 2024.1 high-level synthesis and hardware implementation tool to synthesize our designs and implement them on FPGA.
