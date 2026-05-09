#!/usr/bin/env python3
"""
Plot the extracted Darshan I/O summary.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
INPUT_CSV = DATA_DIR / "darshan_io_summary.csv"
OUTPUT_STEM = RESULTS_DIR / "plot28_darshan_io_summary"


def main() -> int:
    df = pd.read_csv(INPUT_CSV)
    parsed = df[df["status"] == "parsed_summary_dxt_csv"].copy()

    if parsed.empty:
        raise SystemExit(f"No parsed Darshan rows found in {INPUT_CSV}")

    parsed["summary_file_count"] = pd.to_numeric(parsed["summary_file_count"], errors="coerce")
    parsed["parsed_summary_count"] = pd.to_numeric(parsed["parsed_summary_count"], errors="coerce")
    parsed["mean_total_logs"] = pd.to_numeric(parsed["mean_total_logs"], errors="coerce")
    parsed["mean_runtime_s"] = pd.to_numeric(parsed["mean_runtime_s"], errors="coerce")
    parsed["max_runtime_s"] = pd.to_numeric(parsed["max_runtime_s"], errors="coerce")
    parsed["run_label"] = parsed["run_bundle"].str.extract(r"_(\d+)cores_")[0].fillna("?").map(lambda v: f"p={v}")

    fig, (ax_runtime, ax_logs) = plt.subplots(1, 2, figsize=(14.5, 6.0))
    x = np.arange(len(parsed), dtype=float)

    ax_runtime.bar(x - 0.16, parsed["mean_runtime_s"], width=0.32, color="#4c72b0", label="Mean runtime")
    ax_runtime.bar(x + 0.16, parsed["max_runtime_s"], width=0.32, color="#9ecae9", label="Max runtime")
    ax_runtime.set_xticks(x)
    ax_runtime.set_xticklabels(parsed["run_label"], fontsize=11)
    ax_runtime.set_ylabel("Runtime (s)", fontsize=12, fontweight="bold")
    ax_runtime.set_title("Darshan Runtime Summary", fontsize=13, fontweight="bold")
    ax_runtime.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)
    for xi, mean_v, max_v in zip(x, parsed["mean_runtime_s"], parsed["max_runtime_s"]):
        ax_runtime.text(xi - 0.16, mean_v * 1.01, f"{mean_v:.0f}", ha="center", va="bottom", fontsize=9.5)
        ax_runtime.text(xi + 0.16, max_v * 1.01, f"{max_v:.0f}", ha="center", va="bottom", fontsize=9.5)

    ax_logs.bar(x, parsed["mean_total_logs"], color="#dd8452", edgecolor="black", linewidth=0.7)
    ax_logs.set_xticks(x)
    ax_logs.set_xticklabels(parsed["run_label"], fontsize=11)
    ax_logs.set_ylabel("Mean total logs per summary file", fontsize=12, fontweight="bold")
    ax_logs.set_title("Darshan Summary File Activity", fontsize=13, fontweight="bold")
    for xi, logs_v, count_v in zip(x, parsed["mean_total_logs"], parsed["parsed_summary_count"]):
        ax_logs.text(xi, logs_v * 1.02, f"{logs_v:.1f}\n(n={int(count_v)})", ha="center", va="bottom", fontsize=9.5)

    fig.suptitle(
        "Darshan I/O Summary for Available WSClean GPU-Mode Runs",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Only runs with parsed Darshan summary.dxt.csv files are shown. Current data contains p=1 and p=16 only.",
        ha="center",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.18, wspace=0.22)

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
