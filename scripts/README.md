# scripts/

Helper scripts to orchestrate benchmark runs end-to-end.

Examples:

- `run_benchmark.sh` – generic wrapper to run a config and collect metrics
- `collect_metrics.sh` – post-processing of logs into JSON/CSV
- `make_submission_package.sh` – bundle results into a submission archive

These scripts should be kept simple and portable, ideally POSIX shell or
Python with minimal dependencies.
