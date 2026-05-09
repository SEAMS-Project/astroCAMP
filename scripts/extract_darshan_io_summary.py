#!/usr/bin/env python3
"""
Extract Darshan I/O summary metrics from summary.dxt.csv files when present.
Bundles without summary exports are still recorded in the output.
"""

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = REPO_ROOT.parent / "astroCAMP-bench" / "profiling_gpu2"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "darshan_io_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Darshan I/O summaries when present.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def iter_darshan_bundles(root: Path):
    for bundle_dir in sorted(root.glob("*_darshan")):
        yield bundle_dir


def parse_summary_file(path: Path) -> tuple[float, float] | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return None
    if not rows:
        return None
    row = rows[0]
    try:
        total_logs = float(row.get("total_logs", "nan"))
        runtime = float(row.get("runtime", "nan"))
    except ValueError:
        return None
    return total_logs, runtime


def build_rows(input_root: Path) -> list[dict]:
    rows = []
    for bundle_dir in iter_darshan_bundles(input_root):
        summary_files = sorted(bundle_dir.rglob("*.summary.dxt.csv"))
        parsed = [parsed for path in summary_files if (parsed := parse_summary_file(path)) is not None]
        row = {
            "run_bundle": bundle_dir.name,
            "bundle_dir": str(bundle_dir),
            "summary_file_count": len(summary_files),
            "parsed_summary_count": len(parsed),
            "mean_total_logs": "",
            "mean_runtime_s": "",
            "max_runtime_s": "",
            "status": "empty_bundle" if not summary_files else "ready_for_parser",
            "note": "No Darshan summary.dxt.csv files found in bundle" if not summary_files else "Found summary.dxt.csv files",
        }
        if parsed:
            total_logs = [item[0] for item in parsed]
            runtimes = [item[1] for item in parsed]
            row.update(
                {
                    "mean_total_logs": sum(total_logs) / len(total_logs),
                    "mean_runtime_s": sum(runtimes) / len(runtimes),
                    "max_runtime_s": max(runtimes),
                    "status": "parsed_summary_dxt_csv",
                    "note": "Parsed Darshan summary.dxt.csv files",
                }
            )
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_bundle",
        "bundle_dir",
        "summary_file_count",
        "parsed_summary_count",
        "mean_total_logs",
        "mean_runtime_s",
        "max_runtime_s",
        "status",
        "note",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = build_rows(args.input_root)
    write_csv(rows, args.output_csv)
    print(f"Scanned {len(rows)} Darshan bundle(s)")
    print(f"Wrote summary to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
