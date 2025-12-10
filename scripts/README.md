# scripts/

Helper scripts to orchestrate benchmark runs end-to-end.

Examples:

- `run_benchmark.sh` – generic wrapper to run a config and collect metrics
- `collect_metrics.sh` – post-processing of logs into JSON/CSV
- `make_submission_package.sh` – bundle results into a submission archive

These scripts should be kept simple and portable, ideally POSIX shell or
Python with minimal dependencies.

## Carbon and economy analisys scripts
- `machines.csv` – List of machines with their cost ($) and embodied carbon (kgCO2eq).
- `benchmarks.csv` – Profiling results for carbon and economy analysis. Time (s) and energy consumption (J) for each benchmark.
- `locations.csv` – List of locations with their carbon intensity (kgCO2eq/kWh) and electricity price ($/kWh).
- `cea.csv` – Carbon and Economy Analysis results combining data from machines, benchmarks, and locations. generates a LATEX table with the results.
    - `-l` or `--lifetime`: Lifetime of the machines in years (default: 5).

### Usage
To run the carbon and economy analysis, execute the following command:
```sh
python cea.py [-l <years>]
```

### Dependencies
```sh
pip install pandas numpy
```
