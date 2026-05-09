#!/usr/bin/env python3
"""
Extract AMD uProf hotspot summaries from session bundles when report.csv exports
are present. Bundles without exports are still recorded in the output.
"""

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = REPO_ROOT.parent / "astroCAMP-bench" / "profiling_gpu2"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "uprof_hotspots_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AMD uProf hotspot summaries when present.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def iter_hotspot_bundles(root: Path):
    for bundle_root in sorted(root.glob("*_uprof_profile_hotspots")):
        for session_dir in sorted(bundle_root.glob("AMDuProf-*Hotspots*")):
            yield bundle_root, session_dir


def extract_table(lines: list[str], header: str) -> list[list[str]]:
    try:
        start = lines.index(header) + 1
    except ValueError:
        return []
    rows = []
    for line in lines[start:]:
        if not line.strip():
            break
        rows.append(next(csv.reader([line])))
    return rows


def load_report_summary(report_path: Path) -> dict:
    lines = report_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    duration_s = None
    for line in lines:
        if line.startswith("Profile Duration:"):
            try:
                duration_s = float(next(csv.reader([line]))[1].split()[0].strip('"'))
            except Exception:
                duration_s = None
            break

    hottest_functions = extract_table(lines, '"10 HOTTEST FUNCTIONS (Sort Event - CPU_TIME)"')
    hottest_modules = extract_table(lines, '"10 HOTTEST MODULES (Sort Event - CPU_TIME)"')

    top_function = hottest_functions[1] if len(hottest_functions) > 1 else None
    top_module = hottest_modules[1] if len(hottest_modules) > 1 else None
    return {
        "profile_duration_s": duration_s,
        "top_function": top_function[0] if top_function else "",
        "top_function_cpu_time_s": top_function[1] if top_function else "",
        "top_function_total_cpu_time_s": top_function[2] if top_function and len(top_function) > 2 else "",
        "top_module": top_module[0] if top_module else "",
        "top_module_cpu_time_s": top_module[1] if top_module and len(top_module) > 1 else "",
    }


def build_rows(input_root: Path) -> list[dict]:
    rows = []
    for bundle_root, session_dir in iter_hotspot_bundles(input_root):
        report_path = session_dir / "report.csv"
        row = {
            "run_bundle": bundle_root.name,
            "session_dir": str(session_dir),
            "status": "empty_bundle",
            "profile_duration_s": "",
            "top_function": "",
            "top_function_cpu_time_s": "",
            "top_function_total_cpu_time_s": "",
            "top_module": "",
            "top_module_cpu_time_s": "",
            "note": "No report.csv export found inside hotspot bundle",
        }
        if report_path.exists():
            row.update(load_report_summary(report_path))
            row["status"] = "parsed_report_csv"
            row["note"] = "Parsed hotspot report.csv export"
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_bundle",
        "session_dir",
        "status",
        "profile_duration_s",
        "top_function",
        "top_function_cpu_time_s",
        "top_function_total_cpu_time_s",
        "top_module",
        "top_module_cpu_time_s",
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
    print(f"Scanned {len(rows)} hotspot session bundle(s)")
    print(f"Wrote summary to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
