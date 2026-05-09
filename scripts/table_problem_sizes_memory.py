#!/usr/bin/env python3
"""
Export a table of unique problem sizes with approximate input/output sizes.

Assumptions:
- Input on-disk visibility payload is counted from n_vis as one complex FP32
  visibility value (8 bytes).
- Output payload is one FP32 Stokes-I image plane (4 bytes/pixel).

These are payload estimates, not full Measurement Set directory sizes or peak RAM
usage.
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
BENCHMARKS_CSV = BASE_DIR / "benchmarks.csv"
OUT_CSV = BASE_DIR / "problem_size_memory_table.csv"
OUT_MD = BASE_DIR / "problem_size_memory_table.md"
OUT_TEX = BASE_DIR / "problem_size_memory_table.tex"

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


def to_gib(num_bytes: pd.Series) -> pd.Series:
    return num_bytes / (1024**3)


def build_markdown_table(df: pd.DataFrame) -> str:
    display_df = df[
        [
            "Image size",
            "Timesteps",
            "Channels",
            "Visibilities",
            "Rows",
            "Mvis",
            "Input on-disk est. (GiB)",
            "Output (GiB)",
            "Input + Output (GiB)",
        ]
    ].copy()

    display_df["Rows"] = display_df["Rows"].map(lambda x: f"{int(x):,}")
    display_df["Visibilities"] = display_df["Visibilities"].map(lambda x: f"{int(x):,}")
    for col in ["Mvis", "Input on-disk est. (GiB)", "Output (GiB)", "Input + Output (GiB)"]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.3f}")

    header = "| " + " | ".join(display_df.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display_df.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in display_df.columns) + " |"
        for _, row in display_df.iterrows()
    ]

    note = "\n".join(
        [
            "# Problem Size Memory Table",
            "",
            "Assumptions:",
            "- `Visibilities` is the total `n_vis` count from `benchmarks.csv`.",
            "- Input on-disk estimate = `n_vis x 8 B` for one complex FP32 visibility value in the `_single.ms` inputs.",
            "- Output payload = `im_size^2 x 4 B` for one FP32 Stokes-I image plane.",
            "- Full Measurement Set directories would be larger because of metadata, subtables, flags, and other columns.",
            "",
        ]
    )
    return note + "\n".join([header, divider] + rows) + "\n"


def build_latex_table(df: pd.DataFrame) -> str:
    lines = [
        "% Requires \\usepackage{booktabs,longtable}",
        "\\begin{longtable}{rrrrrrrr}",
        "\\caption{Problem sizes and approximate input/output sizes. The input on-disk estimate assumes one complex FP32 visibility value per $n_{vis}$ entry in the `_single.ms` inputs; the output assumes one FP32 Stokes-I image plane. Full Measurement Set directories would be larger because of metadata and auxiliary columns.}\\\\",
        "\\toprule",
        "Image size & Timesteps & Channels & Visibilities & Rows & Input on-disk est. (GiB) & Output (GiB) & Input+Output (GiB)\\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Image size & Timesteps & Channels & Visibilities & Rows & Input on-disk est. (GiB) & Output (GiB) & Input+Output (GiB)\\\\",
        "\\midrule",
        "\\endhead",
        "\\bottomrule",
        "\\endfoot",
    ]

    for _, row in df.iterrows():
        lines.append(
            (
                f"{int(row['Image size'])} & {int(row['Timesteps'])} & {int(row['Channels'])} & "
                f"{int(row['Visibilities'])} & {int(row['Rows'])} & "
                f"{row['Input on-disk est. (GiB)']:.3f} & {row['Output (GiB)']:.3f} & "
                f"{row['Input + Output (GiB)']:.3f}\\\\"
            )
        )

    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def main() -> int:
    df = pd.read_csv(BENCHMARKS_CSV, header=None, names=BENCHMARK_COLUMNS)
    table = (
        df[["im_size", "n_times", "n_chans", "n_rows", "n_vis"]]
        .drop_duplicates()
        .sort_values(["im_size", "n_times", "n_chans"])
        .reset_index(drop=True)
    )

    table["pixels"] = table["im_size"] ** 2
    table["mvis"] = table["n_vis"] / 1e6
    table["input_on_disk_est_bytes"] = table["n_vis"] * INPUT_BYTES_PER_VIS
    table["output_bytes"] = table["pixels"] * OUTPUT_BYTES_PER_PIXEL
    table["total_bytes"] = table["input_on_disk_est_bytes"] + table["output_bytes"]

    export_df = pd.DataFrame(
        {
            "Image size": table["im_size"].astype(int),
            "Timesteps": table["n_times"].astype(int),
            "Channels": table["n_chans"].astype(int),
            "Visibilities": table["n_vis"].astype(int),
            "Rows": table["n_rows"].astype(int),
            "Pixels": table["pixels"].astype(int),
            "Mvis": table["mvis"],
            "Input on-disk est. bytes": table["input_on_disk_est_bytes"].astype(int),
            "Output bytes": table["output_bytes"].astype(int),
            "Input + Output bytes": table["total_bytes"].astype(int),
            "Input on-disk est. (GiB)": to_gib(table["input_on_disk_est_bytes"]),
            "Output (GiB)": to_gib(table["output_bytes"]),
            "Input + Output (GiB)": to_gib(table["total_bytes"]),
        }
    )

    export_df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(build_markdown_table(export_df))
    OUT_TEX.write_text(build_latex_table(export_df))

    print(f"Wrote {len(export_df)} rows to {OUT_CSV}")
    print(f"Wrote markdown table to {OUT_MD}")
    print(f"Wrote LaTeX table to {OUT_TEX}")
    print(
        "Assumptions: input on-disk estimate = n_vis x 8 B (complex FP32), "
        "output = im_size^2 x 4 B (FP32 Stokes-I image)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
