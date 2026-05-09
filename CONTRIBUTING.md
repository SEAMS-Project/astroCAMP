# Contributing to astroCAMP

Thank you for your interest in astroCAMP. This project is a research companion repository for a conference publication. We welcome bug reports, corrections, and discussion via GitHub Issues.

## Reporting Issues

- Use [GitHub Issues](https://github.com/SEAMS-Project/astroCAMP/issues) to report bugs or ask questions.
- Please include your Python version, OS, and the exact error message.

## Reproducing Figures

1. Follow the [Quick Start](README.md#quick-start) in the main README.
2. Download the data from Zenodo: [DOI 10.5281/zenodo.20093790](https://doi.org/10.5281/zenodo.20093790).
3. Place the CSV files in `scripts/` and run `python scripts/regenerate_all_plots.py`.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Keep plot scripts self-contained: each script should read its own input CSV(s) and write outputs to `scripts/results/`.
- Do not commit generated images or PDFs — they are gitignored and regenerated from scripts.

## Submitting Changes

1. Fork the repository and create a feature branch.
2. Make your changes and verify all scripts still run (`python scripts/regenerate_all_plots.py`).
3. Open a pull request with a clear description of the change.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
