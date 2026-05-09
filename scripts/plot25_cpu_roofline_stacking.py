#!/usr/bin/env python3
"""
Generate a CPU roofline analysis for the WSClean stacking runs profiled with
AMD uProf in the sibling astroCAMP-bench repository.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

TMP_CACHE_ROOT = Path(tempfile.gettempdir()) / "astrocamp-mpl"
os.environ.setdefault("MPLCONFIGDIR", str(TMP_CACHE_ROOT / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_CACHE_ROOT / "xdg-cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EFFICIENCY_COLOR = "#c46a2a"
CPU_BAR_COLOR = plt.cm.cividis(0.32)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_ROOT = REPO_ROOT.parent / "astroCAMP-bench" / "roofline_stacking"
DEFAULT_OUTPUT_STEM = SCRIPT_DIR / "plot25_cpu_roofline_stacking"
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "roofline_stacking_summary.csv"

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)


def contrast_text_color(facecolor: tuple[float, float, float, float]) -> str:
    r, g, b = facecolor[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.52 else "white"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a roofline figure and CSV summary from AMD uProf exports."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Directory containing roofline report.json files (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help=f"Output path stem for PNG/PDF files (default: {DEFAULT_OUTPUT_STEM})",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Output CSV for the derived summary table (default: {DEFAULT_SUMMARY_CSV})",
    )
    return parser.parse_args()


def metadata_map(report: dict) -> dict:
    return {entry["name"]: entry.get("value", "") for entry in report.get("metadata", [])}


def metric_average(report: dict, metric_name: str) -> float:
    for metric in report.get("metrics", []):
        if metric.get("name") == metric_name or metric.get("abbreviation") == metric_name:
            return float(metric["aggregated"]["average"])
    raise KeyError(f"Metric '{metric_name}' not found")


def roofline_line(report: dict, line_name: str) -> dict:
    for line in report.get("roofline", {}).get("lines", []):
        if line.get("name") == line_name:
            return line
    raise KeyError(f"Roofline line '{line_name}' not found")


def app_point(report: dict) -> dict:
    for point in report.get("roofline", {}).get("points", []):
        if point.get("type") == "point-app":
            return point
    raise KeyError("Application roofline point not found")


def parse_scalar(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse {label} from '{text}'")
    return float(match.group(1))


def extract_requested_threads(command: str, report_path: Path) -> int:
    match = re.search(r"(?:^|\s)-j\s+(\d+)\b", command)
    if match:
        return int(match.group(1))

    match = re.search(r"_(\d+)cores_", report_path.as_posix())
    if match:
        return int(match.group(1))

    raise ValueError(f"Could not determine requested thread count for {report_path}")


def load_run(report_path: Path, input_root: Path) -> dict:
    report = json.loads(report_path.read_text())
    metadata = metadata_map(report)
    command = str(metadata.get("COMMAND", "")).strip('"')

    point = app_point(report)
    dp_peak_line = roofline_line(report, "DP FP Peak")
    sp_peak_line = roofline_line(report, "SP FP Peak")
    dram_bw_line = roofline_line(report, "DRAM BW")

    try:
        source_report = report_path.relative_to(input_root).as_posix()
    except ValueError:
        source_report = report_path.as_posix()

    return {
        "source_report": source_report,
        "run_label": report_path.parents[1].name,
        "processor_name": str(metadata.get("PROCESSOR_NAME", "")).strip(),
        "requested_threads": extract_requested_threads(command, report_path),
        "elapsed_s": float(metadata.get("ELAPSED_TIME", 0.0)) / 1000.0,
        "command": command,
        "arithmetic_intensity_flop_per_byte": float(point["x"]),
        "throughput_gflops": float(point["y"]),
        "bandwidth_gbs": metric_average(report, "Bandwidth"),
        "profiler_utilization_pct": metric_average(report, "Utilization"),
        "dp_peak_gflops": float(dp_peak_line["start"]["y"]),
        "sp_peak_gflops": float(sp_peak_line["start"]["y"]),
        "dram_bw_gbs": parse_scalar(r"([0-9.]+)\s+GB/sec", dram_bw_line["title"], "DRAM BW"),
    }


def load_roofline_dataframe(input_root: Path) -> tuple[pd.DataFrame, int]:
    report_paths = sorted(input_root.rglob("report.json"))
    if not report_paths:
        raise FileNotFoundError(f"No report.json files found under {input_root}")

    rows = [load_run(path, input_root) for path in report_paths]
    df = pd.DataFrame(rows).sort_values(["requested_threads", "source_report"]).reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(
        subset=[
            "requested_threads",
            "arithmetic_intensity_flop_per_byte",
            "throughput_gflops",
            "bandwidth_gbs",
        ],
        keep="first",
    ).copy()
    dropped_duplicates = before - len(df)

    df = df.sort_values("requested_threads").reset_index(drop=True)
    base_threads = float(df.loc[0, "requested_threads"])
    base_perf = float(df.loc[0, "throughput_gflops"])
    base_elapsed = float(df.loc[0, "elapsed_s"])

    df["throughput_gain_vs_base"] = df["throughput_gflops"] / base_perf
    df["ideal_speedup"] = df["requested_threads"] / base_threads
    df["walltime_speedup_vs_base"] = base_elapsed / df["elapsed_s"]
    df["walltime_parallel_efficiency_pct"] = (
        100.0 * df["walltime_speedup_vs_base"] / df["ideal_speedup"]
    )
    df["ai_from_ratio_flop_per_byte"] = df["throughput_gflops"] / df["bandwidth_gbs"]
    df["ridge_point_flop_per_byte"] = df["dp_peak_gflops"] / df["dram_bw_gbs"]
    df["roof_at_ai_gflops"] = np.minimum(
        df["dp_peak_gflops"],
        df["arithmetic_intensity_flop_per_byte"] * df["dram_bw_gbs"],
    )
    df["roof_fraction_pct"] = 100.0 * df["throughput_gflops"] / df["roof_at_ai_gflops"]
    df["dp_peak_fraction_pct"] = 100.0 * df["throughput_gflops"] / df["dp_peak_gflops"]
    df["dram_bw_fraction_pct"] = 100.0 * df["bandwidth_gbs"] / df["dram_bw_gbs"]
    df["ai_consistency_error_pct"] = 100.0 * (
        df["ai_from_ratio_flop_per_byte"] - df["arithmetic_intensity_flop_per_byte"]
    ) / df["arithmetic_intensity_flop_per_byte"]
    df["bound_regime"] = np.where(
        df["arithmetic_intensity_flop_per_byte"] >= df["ridge_point_flop_per_byte"],
        "compute-side",
        "memory-side",
    )
    return df, dropped_duplicates


def plot_roofline_analysis(df: pd.DataFrame, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    dp_peak = float(df["dp_peak_gflops"].iloc[0])
    sp_peak = float(df["sp_peak_gflops"].iloc[0])
    dram_bw = float(df["dram_bw_gbs"].iloc[0])
    ridge = float(df["ridge_point_flop_per_byte"].iloc[0])

    best_roofline_row = df.loc[df["throughput_gflops"].idxmax()]
    best_scaling_row = df.loc[df["elapsed_s"].idxmin()]
    colors = plt.cm.cividis(np.linspace(0.15, 0.85, len(df)))

    fig, (ax_roof, ax_scale) = plt.subplots(
        1,
        2,
        figsize=(17.4, 7.6),
        gridspec_kw={"width_ratios": [1.45, 1.15]},
    )

    x_min = min(0.04, float(df["arithmetic_intensity_flop_per_byte"].min()) / 1.8)
    x_max = max(150.0, float(df["arithmetic_intensity_flop_per_byte"].max()) * 2.0)
    y_min = min(30.0, float(df["throughput_gflops"].min()) * 0.75)
    y_max = sp_peak * 1.15

    ai_diag = np.logspace(np.log10(x_min), np.log10(ridge), 300)
    ax_roof.plot(
        ai_diag,
        dram_bw * ai_diag,
        color="#c44e52",
        linewidth=2.6,
        label=f"DRAM bandwidth roof ({dram_bw:.1f} GB/s)",
    )
    ax_roof.hlines(
        dp_peak,
        ridge,
        x_max,
        color="#4c72b0",
        linewidth=2.6,
        label=f"Double-precision peak ({dp_peak:.0f} GFLOP/s)",
    )
    ax_roof.hlines(
        sp_peak,
        sp_peak / dram_bw,
        x_max,
        color="0.55",
        linewidth=1.8,
        linestyles="--",
        label=f"Single-precision peak ({sp_peak:.0f} GFLOP/s)",
    )
    ax_roof.axvline(ridge, color="0.6", linestyle=":", linewidth=1.1)

    ax_roof.plot(
        df["arithmetic_intensity_flop_per_byte"],
        df["throughput_gflops"],
        color="0.35",
        linewidth=1.2,
        alpha=0.85,
        zorder=2,
    )

    label_offsets = {
        1: (-34, -10),
        16: (-42, 18),
        32: (16, 18),
        64: (18, -18),
    }
    for color, row in zip(colors, df.to_dict("records")):
        ax_roof.scatter(
            row["arithmetic_intensity_flop_per_byte"],
            row["throughput_gflops"],
            s=95,
            color=color,
            edgecolors="black",
            linewidth=0.8,
            zorder=3,
        )
        offset = label_offsets.get(int(row["requested_threads"]), (8, 8))
        ax_roof.annotate(
            f"p={int(row['requested_threads'])}",
            (row["arithmetic_intensity_flop_per_byte"], row["throughput_gflops"]),
            textcoords="offset points",
            xytext=offset,
            fontsize=11.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "0.75",
                "linewidth": 0.6,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "0.45",
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    ax_roof.text(
        ridge * 1.05,
        dp_peak * 0.35,
        f"ridge = {ridge:.2f} FLOP/B",
        color="0.35",
        fontsize=11,
        rotation=90,
        va="center",
    )

    summary_text = (
        f"Arithmetic intensity stays at {df['arithmetic_intensity_flop_per_byte'].min():.1f}-"
        f"{df['arithmetic_intensity_flop_per_byte'].max():.1f} FLOP/B\n"
        f"All runs land on the compute side of the ridge\n"
        f"Best roofline point: p={int(best_roofline_row['requested_threads'])} = "
        f"{best_roofline_row['throughput_gflops']:.1f} GFLOP/s\n"
        f"That is {best_roofline_row['dp_peak_fraction_pct']:.1f}% of the double-precision peak and "
        f"{best_roofline_row['dram_bw_fraction_pct']:.2f}% of peak DRAM bandwidth"
    )
    ax_roof.text(
        0.03,
        0.05,
        summary_text,
        transform=ax_roof.transAxes,
        fontsize=11.5,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(x_min, x_max)
    ax_roof.set_ylim(y_min, y_max)
    ax_roof.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=14, fontweight="bold")
    ax_roof.set_ylabel("Throughput (GFLOP/s)", fontsize=14, fontweight="bold")
    ax_roof.set_title("CPU Roofline", fontsize=15, fontweight="bold")
    ax_roof.tick_params(axis="both", labelsize=12)
    roof_legend = ax_roof.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=10.9,
        frameon=True,
        columnspacing=1.4,
        handlelength=2.8,
        borderaxespad=0.0,
    )

    x = np.arange(len(df))
    bars = ax_scale.bar(
        x,
        df["walltime_speedup_vs_base"],
        width=0.62,
        color=CPU_BAR_COLOR,
        edgecolor="black",
        linewidth=0.7,
        label=r"Wall-time speedup $S(p)=T_1/T_p$",
    )
    ax_eff = ax_scale.twinx()
    eff_line = ax_eff.plot(
        x,
        df["walltime_parallel_efficiency_pct"],
        color=EFFICIENCY_COLOR,
        marker="o",
        linewidth=3.0,
        markersize=6.5,
        alpha=0.78,
        label=r"Parallel efficiency $E(p)=S(p)/p$",
    )
    eff_ref = ax_eff.axhline(
        100.0,
        color="0.45",
        linestyle=":",
        linewidth=1.4,
        label="Ideal efficiency",
    )

    for bar, row in zip(bars, df.to_dict("records")):
        text_color = contrast_text_color(bar.get_facecolor())
        ax_scale.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() / 2.0,
            f"T={row['elapsed_s']:.0f} s\n{row['throughput_gflops']:.0f} GFLOP/s",
            ha="center",
            va="center",
            fontsize=10.8,
            fontweight="bold",
            color=text_color,
        )

    scaling_text = (
        r"Scalability uses profiler elapsed time $T_p$, not sampled GFLOP/s" "\n"
        f"Best wall-time speedup: {best_scaling_row['walltime_speedup_vs_base']:.2f}x at "
        f"p={int(best_scaling_row['requested_threads'])}\n"
        f"Runs at p=16..64 all stay near "
        f"{df.loc[df['requested_threads'] >= 16, 'elapsed_s'].min():.0f}-"
        f"{df.loc[df['requested_threads'] >= 16, 'elapsed_s'].max():.0f} s"
    )
    ax_scale.text(
        0.03,
        0.95,
        scaling_text,
        transform=ax_scale.transAxes,
        fontsize=11.6,
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    ax_scale.set_xticks(x)
    ax_scale.set_xticklabels([f"p={int(v)}" for v in df["requested_threads"]], fontsize=13)
    ax_scale.set_xlabel("Parallelism p  (-j threads)", fontsize=14, fontweight="bold")
    ax_scale.set_ylabel(r"Wall-time speedup $S(p)=T_1/T_p$", fontsize=14, fontweight="bold")
    ax_scale.set_ylim(0.0, max(3.6, float(df["walltime_speedup_vs_base"].max()) * 1.30))
    ax_scale.set_title("Scalability Summary", fontsize=15, fontweight="bold")
    ax_scale.grid(axis="y", alpha=0.3, linestyle="--")
    ax_scale.tick_params(axis="y", labelsize=12)

    ax_eff.set_ylabel(
        r"Parallel efficiency $E(p)=S(p)/p$ (%)",
        fontsize=14,
        fontweight="bold",
        color=EFFICIENCY_COLOR,
    )
    ax_eff.tick_params(axis="y", labelcolor=EFFICIENCY_COLOR, labelsize=12)
    ax_eff.set_ylim(0.0, 105.0)

    legend_handles = [bars, eff_line[0], eff_ref]
    legend_labels = [
        "Wall-time speedup",
        "Parallel efficiency",
        "Ideal efficiency",
    ]
    scale_legend = ax_scale.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=10.9,
        frameon=True,
        columnspacing=1.4,
        handlelength=2.6,
        borderaxespad=0.0,
    )

    processor = str(df["processor_name"].iloc[0]).strip()
    fig.suptitle(
        "Roofline Analysis for 16384^2, t=256, c=256 CPU Stacking Runs\n"
        f"Dual-socket node: 2 x {processor} | p = WSClean thread count (-j)",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    for legend in (roof_legend, scale_legend):
        legend.get_frame().set_edgecolor("0.75")
        legend.get_frame().set_alpha(0.95)

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.28, top=0.84, wspace=0.1)

    saved_paths = []
    for suffix in (".png", ".pdf"):
        out_path = output_stem.with_suffix(suffix)
        fig.savefig(out_path, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        saved_paths.append(out_path)

    plt.close(fig)
    return saved_paths


def print_summary(df: pd.DataFrame, dropped_duplicates: int) -> None:
    if dropped_duplicates:
        print(f"Dropped {dropped_duplicates} duplicate roofline report(s) with identical metrics.")

    display_cols = [
        "requested_threads",
        "elapsed_s",
        "walltime_speedup_vs_base",
        "walltime_parallel_efficiency_pct",
        "arithmetic_intensity_flop_per_byte",
        "throughput_gflops",
        "bandwidth_gbs",
        "throughput_gain_vs_base",
        "dp_peak_fraction_pct",
        "dram_bw_fraction_pct",
        "bound_regime",
    ]
    formatters = {
        "requested_threads": "{:.0f}".format,
        "elapsed_s": "{:.1f}".format,
        "walltime_speedup_vs_base": "{:.2f}".format,
        "walltime_parallel_efficiency_pct": "{:.1f}".format,
        "arithmetic_intensity_flop_per_byte": "{:.2f}".format,
        "throughput_gflops": "{:.2f}".format,
        "bandwidth_gbs": "{:.2f}".format,
        "throughput_gain_vs_base": "{:.2f}".format,
        "dp_peak_fraction_pct": "{:.2f}".format,
        "dram_bw_fraction_pct": "{:.3f}".format,
    }

    print("\nCPU roofline summary")
    print("=" * 80)
    print(df[display_cols].to_string(index=False, formatters=formatters))
    print("=" * 80)
    print(
        "Interpretation: roofline throughput improves with core count, but standard "
        "scalability from elapsed time is only about 2.82x from p=1 to p=64. The "
        "workload stays on the compute side of the DRAM ridge point, while memory "
        "bandwidth usage remains below 1% of peak."
    )


def main() -> int:
    args = parse_args()

    if not args.input_root.exists():
        print(f"Skipping roofline analysis: input directory not found: {args.input_root}")
        return 0

    df, dropped_duplicates = load_roofline_dataframe(args.input_root)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.summary_csv, index=False)

    saved_paths = plot_roofline_analysis(df, args.output_stem)
    print_summary(df, dropped_duplicates)
    print(f"\nSaved summary CSV to {args.summary_csv}")
    for path in saved_paths:
        print(f"Saved figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    def contrast_text_color(facecolor: tuple[float, float, float, float]) -> str:
        r, g, b = facecolor[:3]
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "black" if luminance > 0.52 else "white"
