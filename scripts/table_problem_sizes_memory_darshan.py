#!/usr/bin/env python3
"""
Build a side-by-side problem-size table with:
- payload estimates from benchmarks.csv
- measured Darshan I/O bytes where available

The measured Darshan values are transfer volumes, not resident dataset sizes.
They are useful as a verification/measurement companion to the payload estimates,
but they should not be interpreted as exact Measurement Set directory sizes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = BASE_DIR.parent / "results"
REPO_ROOT = BASE_DIR.parent
BENCH_ROOT = REPO_ROOT.parent / "astroCAMP-bench"

BENCHMARKS_CSV = DATA_DIR / "benchmarks.csv"
OUT_CSV = DERIVED_DIR / "problem_size_memory_darshan_table.csv"
OUT_MD = RESULTS_DIR / "problem_size_memory_darshan_table.md"
OUT_TEX = RESULTS_DIR / "problem_size_memory_darshan_table.tex"

BENCHMARK_COLUMNS = [
    "im_size",
    "n_times",
    "n_chans",
    "wall_time",
    "wall_time_sec",
    "n_rows",
    "n_vis",
    "n_idg",
    "idg_h_sec",
    "idg_h_watt",
    "idg_h_jou",
    "idg_d_sec",
    "idg_d_watt",
    "idg_d_jou",
    "idg_grid_mvs",
    "cpu_j",
    "cpu_bsl_j",
    "cpu_bsl_std_j",
    "gpu0_j",
    "gpu1_j",
    "gpu2_j",
    "gpu3_j",
    "gpu_j",
    "gpu_bsl_j",
    "gpu_bsl_std_j",
    "tot_sys_j",
    "tot_pdu_j",
    "pdu_bsl_j",
    "pdu_bsl_std_j",
    "abs_cpu_j",
    "abs_gpu_j",
    "abs_pdu_j",
]

INPUT_BYTES_PER_VIS = 8
OUTPUT_BYTES_PER_PIXEL = 4

EXE_RE = re.compile(r"^# exe:\s+(?P<exe>.*)$")
RUNTIME_RE = re.compile(r"^# run time:\s+(?P<runtime>[0-9.]+)$")
BUNDLE_RE = re.compile(
    r"slurm-\d+_wsc_dirty_t0-(?P<t>\d+)_c0-(?P<c>\d+)_(?P<img>\d+)(?:pix|p)_.*_darshan$"
)


def to_gib(num_bytes: pd.Series) -> pd.Series:
    return num_bytes / (1024 ** 3)


def load_payload_table() -> pd.DataFrame:
    df = pd.read_csv(BENCHMARKS_CSV, header=None, names=BENCHMARK_COLUMNS)
    table = (
        df[["im_size", "n_times", "n_chans", "n_rows", "n_vis"]]
        .drop_duplicates()
        .sort_values(["im_size", "n_times", "n_chans"])
        .reset_index(drop=True)
    )
    table["pixels"] = table["im_size"] ** 2
    table["input_payload_bytes"] = table["n_vis"] * INPUT_BYTES_PER_VIS
    table["output_payload_bytes"] = table["pixels"] * OUTPUT_BYTES_PER_PIXEL
    table["total_payload_bytes"] = table["input_payload_bytes"] + table["output_payload_bytes"]
    return pd.DataFrame(
        {
            "Image size": table["im_size"].astype(int),
            "Timesteps": table["n_times"].astype(int),
            "Channels": table["n_chans"].astype(int),
            "Rows": table["n_rows"].astype(int),
            "Visibilities": table["n_vis"].astype(int),
            "Input payload est. (GiB)": to_gib(table["input_payload_bytes"]),
            "Output payload est. (GiB)": to_gib(table["output_payload_bytes"]),
            "Total payload est. (GiB)": to_gib(table["total_payload_bytes"]),
        }
    )


def classify_mode_from_exe(exe_line: str | None, bundle_name: str) -> str:
    if exe_line:
        if "-idg-mode cpu" in exe_line:
            return "cpu"
        if "-idg-mode gpu" in exe_line:
            return "gpu"
    if "profiling_cpu" in bundle_name:
        return "cpu"
    return "gpu"


def parse_job_characterization(path: Path) -> dict | None:
    bundle_dir = path.parent
    m = BUNDLE_RE.match(bundle_dir.name)
    if not m:
        return None

    exe_line = None
    mode = None
    runtime_s = None
    ms_read_bytes = 0
    fits_write_bytes = 0
    total_io_bytes = 0
    output_file_write_bytes = 0
    posix_transfer_bytes = 0
    stdio_transfer_bytes = 0

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        em = EXE_RE.match(raw)
        if em:
            exe_line = em.group("exe")
            mode = classify_mode_from_exe(exe_line, str(bundle_dir))
            continue

        rm = RUNTIME_RE.match(raw)
        if rm:
            runtime_s = float(rm.group("runtime"))
            continue

        if not raw.startswith(("POSIX\t", "STDIO\t")):
            continue
        parts = raw.split("\t")
        if len(parts) < 6:
            continue

        module = parts[0]
        counter = parts[3]
        value_str = parts[4]
        file_path = parts[5]

        if file_path.startswith("<") and file_path.endswith(">"):
            continue

        try:
            value = int(float(value_str))
        except ValueError:
            continue

        is_read = counter in {"POSIX_BYTES_READ", "STDIO_BYTES_READ"}
        is_write = counter in {"POSIX_BYTES_WRITTEN", "STDIO_BYTES_WRITTEN"}
        if not (is_read or is_write):
            continue

        total_io_bytes += value
        if module == "POSIX":
            posix_transfer_bytes += value
        elif module == "STDIO":
            stdio_transfer_bytes += value

        is_ms = (".ms/" in file_path) or file_path.endswith(".ms")
        is_fits = file_path.endswith(".fits") or file_path.endswith(".fits.tmp")
        is_output_file = (
            "/gleam_benchmarks/" in file_path
            and file_path.endswith(".fits")
        )

        if is_ms and is_read:
            ms_read_bytes += value
        if is_fits and is_write:
            fits_write_bytes += value
        if is_output_file and is_write:
            output_file_write_bytes += value

    if mode is None:
        mode = classify_mode_from_exe(exe_line, str(bundle_dir))

    posix_transfer_mib = posix_transfer_bytes / (1024 ** 2)
    stdio_transfer_mib = stdio_transfer_bytes / (1024 ** 2)
    posix_throughput_mib_s = (
        posix_transfer_mib / runtime_s if runtime_s and runtime_s > 0 else float("nan")
    )
    stdio_throughput_mib_s = (
        stdio_transfer_mib / runtime_s if runtime_s and runtime_s > 0 else float("nan")
    )

    return {
        "Image size": int(m.group("img")),
        "Timesteps": int(m.group("t")) + 1,
        "Channels": int(m.group("c")) + 1,
        "mode": mode,
        "Darshan run time (s)": runtime_s,
        "Darshan MS read (GiB)": ms_read_bytes / (1024 ** 3),
        "Darshan FITS writes (GiB)": fits_write_bytes / (1024 ** 3),
        "Darshan named output writes (GiB)": output_file_write_bytes / (1024 ** 3),
        "Darshan total I/O (GiB)": total_io_bytes / (1024 ** 3),
        "Darshan POSIX transferred (MiB)": posix_transfer_mib,
        "Darshan POSIX throughput (MiB/s)": posix_throughput_mib_s,
        "Darshan STDIO transferred (MiB)": stdio_transfer_mib,
        "Darshan STDIO throughput (MiB/s)": stdio_throughput_mib_s,
        "job_characterization": str(path),
    }


def collect_darshan_measurements() -> pd.DataFrame:
    rows = []
    for job_file in BENCH_ROOT.rglob("job-characterization.txt"):
        parsed = parse_job_characterization(job_file)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby(["Image size", "Timesteps", "Channels", "mode"], as_index=False)
        .agg(
            **{
                "Darshan run time (s)": ("Darshan run time (s)", "median"),
                "Darshan MS read (GiB)": ("Darshan MS read (GiB)", "median"),
                "Darshan FITS writes (GiB)": ("Darshan FITS writes (GiB)", "median"),
                "Darshan named output writes (GiB)": ("Darshan named output writes (GiB)", "median"),
                "Darshan total I/O (GiB)": ("Darshan total I/O (GiB)", "median"),
                "Darshan POSIX transferred (MiB)": ("Darshan POSIX transferred (MiB)", "median"),
                "Darshan POSIX throughput (MiB/s)": ("Darshan POSIX throughput (MiB/s)", "median"),
                "Darshan STDIO transferred (MiB)": ("Darshan STDIO transferred (MiB)", "median"),
                "Darshan STDIO throughput (MiB/s)": ("Darshan STDIO throughput (MiB/s)", "median"),
                "Darshan runs": ("job_characterization", "count"),
            }
        )
        .sort_values(["Image size", "Timesteps", "Channels", "mode"])
        .reset_index(drop=True)
    )

    wide = grouped.pivot(
        index=["Image size", "Timesteps", "Channels"],
        columns="mode",
        values=[
            "Darshan run time (s)",
            "Darshan MS read (GiB)",
            "Darshan FITS writes (GiB)",
            "Darshan named output writes (GiB)",
            "Darshan total I/O (GiB)",
            "Darshan POSIX transferred (MiB)",
            "Darshan POSIX throughput (MiB/s)",
            "Darshan STDIO transferred (MiB)",
            "Darshan STDIO throughput (MiB/s)",
            "Darshan runs",
        ],
    )
    wide.columns = [
        f"{'CPU' if mode == 'cpu' else 'GPU'} {metric}"
        for metric, mode in wide.columns.to_flat_index()
    ]
    return wide.reset_index()


def build_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "Image size",
        "Timesteps",
        "Channels",
        "Visibilities",
        "Rows",
        "Input payload est. (GiB)",
        "Output payload est. (GiB)",
        "CPU Darshan POSIX transferred (MiB)",
        "CPU Darshan POSIX throughput (MiB/s)",
        "CPU Darshan STDIO transferred (MiB)",
        "CPU Darshan STDIO throughput (MiB/s)",
        "GPU Darshan POSIX transferred (MiB)",
        "GPU Darshan POSIX throughput (MiB/s)",
        "GPU Darshan STDIO transferred (MiB)",
        "GPU Darshan STDIO throughput (MiB/s)",
        "CPU Darshan MS read (GiB)",
        "CPU Darshan named output writes (GiB)",
        "GPU Darshan MS read (GiB)",
        "GPU Darshan named output writes (GiB)",
    ]
    display = df[cols].copy()
    for col in ["Rows", "Visibilities"]:
        display[col] = display[col].map(lambda x: f"{int(x):,}")
    for col in display.columns:
        if "(GiB)" in col:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        if "(MiB)" in col:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        if "(MiB/s)" in col:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    note = "\n".join(
        [
            "# Problem Size Table with Darshan Measurements",
            "",
            "Interpretation notes:",
            "- `Input/Output payload est.` columns are analytical payload estimates from `benchmarks.csv`.",
            "- `Darshan MS read` is measured bytes read from files under the input `.ms` directory.",
            "- `Darshan named output writes` is measured bytes written to named `.fits` output files under the benchmark output path.",
            "- `Darshan POSIX/STDIO transferred` and `throughput` are derived from Darshan byte counters divided by Darshan job runtime.",
            "- Darshan values are transfer volumes, not exact resident dataset sizes.",
            "- Empty Darshan cells mean no matching measured bundle was available for that workload/mode.",
            "",
        ]
    )
    return note + "\n".join([header, divider] + rows) + "\n"


def build_latex_table(df: pd.DataFrame) -> str:
    cols = [
        "Image size",
        "Timesteps",
        "Channels",
        "Visibilities",
        "Rows",
        "Input payload est. (GiB)",
        "Output payload est. (GiB)",
        "CPU Darshan POSIX transferred (MiB)",
        "CPU Darshan POSIX throughput (MiB/s)",
        "GPU Darshan POSIX transferred (MiB)",
        "GPU Darshan POSIX throughput (MiB/s)",
        "CPU Darshan MS read (GiB)",
        "CPU Darshan named output writes (GiB)",
        "GPU Darshan MS read (GiB)",
        "GPU Darshan named output writes (GiB)",
    ]
    lines = [
        "% Requires \\usepackage{booktabs,longtable}",
        "\\begin{longtable}{rrrrrrrrrrrrrrr}",
        "\\caption{Problem-size payload estimates together with measured Darshan transfer volumes and derived POSIX throughput estimates where available. Payload estimates are analytical; Darshan columns are measured transfer bytes and therefore represent I/O volume, not exact resident dataset size.}\\\\",
        "\\toprule",
        "Image size & Timesteps & Channels & Visibilities & Rows & Input est. (GiB) & Output est. (GiB) & CPU POSIX (MiB) & CPU POSIX (MiB/s) & GPU POSIX (MiB) & GPU POSIX (MiB/s) & CPU MS read (GiB) & CPU output writes (GiB) & GPU MS read (GiB) & GPU output writes (GiB)\\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Image size & Timesteps & Channels & Visibilities & Rows & Input est. (GiB) & Output est. (GiB) & CPU POSIX (MiB) & CPU POSIX (MiB/s) & GPU POSIX (MiB) & GPU POSIX (MiB/s) & CPU MS read (GiB) & CPU output writes (GiB) & GPU MS read (GiB) & GPU output writes (GiB)\\\\",
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]
    for _, row in df[cols].iterrows():
        def fmt_num(x):
            return "" if pd.isna(x) else f"{x:.3f}"

        lines.append(
            f"{int(row['Image size'])} & {int(row['Timesteps'])} & {int(row['Channels'])} & "
            f"{int(row['Visibilities'])} & {int(row['Rows'])} & "
            f"{row['Input payload est. (GiB)']:.3f} & {row['Output payload est. (GiB)']:.3f} & "
            f"{fmt_num(row['CPU Darshan POSIX transferred (MiB)'])} & {fmt_num(row['CPU Darshan POSIX throughput (MiB/s)'])} & "
            f"{fmt_num(row['GPU Darshan POSIX transferred (MiB)'])} & {fmt_num(row['GPU Darshan POSIX throughput (MiB/s)'])} & "
            f"{fmt_num(row['CPU Darshan MS read (GiB)'])} & {fmt_num(row['CPU Darshan named output writes (GiB)'])} & "
            f"{fmt_num(row['GPU Darshan MS read (GiB)'])} & {fmt_num(row['GPU Darshan named output writes (GiB)'])}\\\\"
        )
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = load_payload_table()
    darshan = collect_darshan_measurements()
    if not darshan.empty:
        merged = payload.merge(darshan, on=["Image size", "Timesteps", "Channels"], how="left")
    else:
        merged = payload.copy()

    # Ensure stable column order.
    ordered_cols = [
        "Image size",
        "Timesteps",
        "Channels",
        "Rows",
        "Visibilities",
        "Input payload est. (GiB)",
        "Output payload est. (GiB)",
        "Total payload est. (GiB)",
        "CPU Darshan run time (s)",
        "CPU Darshan POSIX transferred (MiB)",
        "CPU Darshan POSIX throughput (MiB/s)",
        "CPU Darshan STDIO transferred (MiB)",
        "CPU Darshan STDIO throughput (MiB/s)",
        "CPU Darshan MS read (GiB)",
        "CPU Darshan FITS writes (GiB)",
        "CPU Darshan named output writes (GiB)",
        "CPU Darshan total I/O (GiB)",
        "CPU Darshan runs",
        "GPU Darshan run time (s)",
        "GPU Darshan POSIX transferred (MiB)",
        "GPU Darshan POSIX throughput (MiB/s)",
        "GPU Darshan STDIO transferred (MiB)",
        "GPU Darshan STDIO throughput (MiB/s)",
        "GPU Darshan MS read (GiB)",
        "GPU Darshan FITS writes (GiB)",
        "GPU Darshan named output writes (GiB)",
        "GPU Darshan total I/O (GiB)",
        "GPU Darshan runs",
    ]
    for col in ordered_cols:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[ordered_cols].sort_values(["Image size", "Timesteps", "Channels"]).reset_index(drop=True)

    merged.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_markdown_table(merged))
    OUT_TEX.write_text(build_latex_table(merged))

    print(f"Wrote {len(merged)} rows to {OUT_CSV}")
    print(f"Wrote markdown table to {OUT_MD}")
    print(f"Wrote LaTeX table to {OUT_TEX}")
    cpu_rows = int(merged["CPU Darshan runs"].fillna(0).astype(float).gt(0).sum())
    gpu_rows = int(merged["GPU Darshan runs"].fillna(0).astype(float).gt(0).sum())
    print(f"Workloads with CPU Darshan measurements: {cpu_rows}")
    print(f"Workloads with GPU Darshan measurements: {gpu_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
