#!/usr/bin/env python3
"""
Compare the CPU roofline from AMD uProf with a GPU empirical roofline-style
view derived from kernel-level instruction collection logs.
"""

import argparse
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
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = SCRIPT_DIR.parent / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
DEFAULT_CPU_SUMMARY = DERIVED_DIR / "roofline_stacking_summary.csv"
DEFAULT_GPU_ROOT = REPO_ROOT.parent / "astroCAMP-bench" / "profiling_gpu2"
DEFAULT_OUTPUT_STEM = RESULTS_DIR / "plot26_gpu_cpu_roofline_comparison"
DEFAULT_SUMMARY_CSV = DERIVED_DIR / "gpu_cpu_roofline_comparison_summary.csv"

ROOF_DRAM_COLOR = "#c44e52"
ROOF_COMPUTE_COLOR = "#4c72b0"
ROOF_SP_COLOR = "0.55"
GPU_GRIIDER_COLOR = plt.cm.cividis(0.78)
GPU_SUBFFT_COLOR = plt.cm.cividis(0.58)
GPU_HARDWARE_COLOR = "0.35"

plt.rcParams.update(
    {
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GPU-vs-CPU roofline comparison using local profiling logs."
    )
    parser.add_argument("--cpu-summary", type=Path, default=DEFAULT_CPU_SUMMARY)
    parser.add_argument("--gpu-root", type=Path, default=DEFAULT_GPU_ROOT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    return parser.parse_args()


def load_cpu_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["source"] = "CPU application"
    df["kind"] = "cpu-application"
    df["threads"] = df["requested_threads"]
    df["ai"] = df["arithmetic_intensity_flop_per_byte"]
    df["gflops"] = df["throughput_gflops"]
    df["gbs"] = df["bandwidth_gbs"]
    df["sp_peak_fraction_pct"] = 100.0 * df["gflops"] / df["sp_peak_gflops"]
    return df


def load_gpu_kernel_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pattern = re.compile(
        r"(?P<kind>gridder|sub-fft):\s+"
        r"(?P<time>[0-9.e+-]+)\s+s,\s+"
        r"(?P<gflops>[0-9.]+)\s+GFLOPS,\s+"
        r"(?P<gbs>[0-9.]+)\s+GB/s"
        r"(?:,\s+(?P<watt>[0-9.]+)\s+Watt)?"
        r"(?:,\s+(?P<joules>[0-9.]+)\s+Joules)?"
    )

    rows = []
    for path in sorted(root.glob("*_uprof_collect_inst.txt")):
        match = re.search(r"_(\d+)cores_", path.name)
        if not match:
            continue
        threads = int(match.group(1))
        for line in path.read_text(errors="ignore").splitlines():
            m = pattern.match(line.strip())
            if not m:
                continue
            row = m.groupdict()
            row["threads"] = threads
            row["source_file"] = path.name
            rows.append(row)

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise FileNotFoundError(f"No GPU kernel traces found under {root}")

    for col in ["time", "gflops", "gbs", "watt", "joules"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    grouped = (
        raw.groupby(["threads", "kind"], as_index=False)
        .agg(
            sample_count=("kind", "size"),
            time_s_median=("time", "median"),
            time_s_min=("time", "min"),
            time_s_max=("time", "max"),
            gflops_median=("gflops", "median"),
            gflops_max=("gflops", "max"),
            gbs_median=("gbs", "median"),
            gbs_max=("gbs", "max"),
            watt_median=("watt", "median"),
            joules_median=("joules", "median"),
        )
        .sort_values(["kind", "threads"])
        .reset_index(drop=True)
    )
    grouped["ai_median"] = grouped["gflops_median"] / grouped["gbs_median"]
    grouped["gb_per_joule_median"] = grouped["gbs_median"] / grouped["watt_median"]
    grouped["mib_per_joule_median"] = grouped["gb_per_joule_median"] * 1024.0
    grouped["source"] = "GPU kernel"
    return raw, grouped


def plot_comparison(cpu_df: pd.DataFrame, gpu_raw: pd.DataFrame, gpu_df: pd.DataFrame, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    cpu_dp_peak = float(cpu_df["dp_peak_gflops"].iloc[0])
    cpu_sp_peak = float(cpu_df["sp_peak_gflops"].iloc[0])
    cpu_bw = float(cpu_df["dram_bw_gbs"].iloc[0])
    cpu_dp_ridge = cpu_dp_peak / cpu_bw
    cpu_sp_ridge = cpu_sp_peak / cpu_bw

    # EPFL Kuma H100 partition: NVIDIA H100 SXM5 94 GB, 2.4 TB/s memory
    # bandwidth according to the Kuma hardware page. FP32 peak is taken from
    # NVIDIA's H100 product specification.
    kuma_h100 = {
        "peak_gflops": 67000.0,
        "bw_gbs": 2400.0,
        "ridge": 67000.0 / 2400.0,
        "color": GPU_HARDWARE_COLOR,
        "linestyle": ":",
    }

    fig, (ax_roof, ax_cmp) = plt.subplots(
        1,
        2,
        figsize=(18.2, 8.2),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    x_min = 0.02
    x_max = max(420.0, float(max(cpu_df["ai"].max(), gpu_df["ai_median"].max()) * 2.1))
    y_min = 12.0
    y_max = max(cpu_sp_peak, kuma_h100["peak_gflops"], float(gpu_df["gflops_median"].max())) * 1.8

    # CPU hardware roofline
    ai_cpu_dp_diag = np.logspace(np.log10(x_min), np.log10(cpu_dp_ridge), 300)
    ax_roof.plot(
        ai_cpu_dp_diag,
        cpu_bw * ai_cpu_dp_diag,
        color=ROOF_DRAM_COLOR,
        linewidth=2.4,
        label=f"CPU DRAM roof ({cpu_bw:.0f} GB/s)",
    )
    ax_roof.hlines(
        cpu_dp_peak,
        cpu_dp_ridge,
        x_max,
        color=ROOF_COMPUTE_COLOR,
        linewidth=2.0,
        linestyles="-.",
        label=f"CPU FP64 peak ({cpu_dp_peak:.0f} GFLOP/s)",
    )
    ax_roof.hlines(
        cpu_sp_peak,
        cpu_sp_ridge,
        x_max,
        color=ROOF_SP_COLOR,
        linewidth=2.4,
        linestyles="--",
        label=f"CPU FP32 peak ({cpu_sp_peak:.0f} GFLOP/s)",
    )
    ax_roof.axvline(cpu_dp_ridge, color="0.6", linestyle=":", linewidth=1.1, alpha=0.85)
    ax_roof.axvline(cpu_sp_ridge, color="0.6", linestyle=":", linewidth=1.1, alpha=0.85)
    ax_roof.scatter(
        [cpu_dp_ridge, cpu_sp_ridge],
        [cpu_dp_peak, cpu_sp_peak],
        s=[58, 64],
        color=ROOF_COMPUTE_COLOR,
        edgecolors="black",
        linewidth=0.7,
        zorder=4,
    )

    ai_h100_diag = np.logspace(np.log10(x_min), np.log10(kuma_h100["ridge"]), 300)
    ax_roof.plot(
        ai_h100_diag,
        kuma_h100["bw_gbs"] * ai_h100_diag,
        color=kuma_h100["color"],
        linewidth=1.9,
        linestyle=kuma_h100["linestyle"],
        alpha=0.95,
        label=f"Kuma H100 roof ({kuma_h100['bw_gbs']:.0f} GB/s, 67 TFLOP/s FP32)",
    )
    ax_roof.hlines(
        kuma_h100["peak_gflops"],
        kuma_h100["ridge"],
        x_max,
        color=kuma_h100["color"],
        linewidth=1.9,
        linestyles=kuma_h100["linestyle"],
        alpha=0.95,
    )

    cpu_colors = plt.cm.cividis(np.linspace(0.15, 0.85, len(cpu_df)))
    gpu_kind_styles = {
        "gridder": {"color": GPU_GRIIDER_COLOR, "marker": "o", "label": "GPU gridder kernel"},
        "sub-fft": {"color": GPU_SUBFFT_COLOR, "marker": "s", "label": "GPU sub-FFT kernel"},
    }
    cpu_label_offsets = {1: (22, 16), 16: (-58, 8), 32: (18, 20), 64: (18, -24)}
    gpu_label_offsets = {
        "gridder": {1: (-62, -22), 16: (-62, 8), 32: (20, -22), 64: (20, 10)},
        "sub-fft": {1: (-58, 18), 16: (-58, -26), 32: (18, 18), 64: (18, -26)},
    }

    ax_roof.plot(cpu_df["ai"], cpu_df["gflops"], color="0.35", alpha=0.75, linewidth=1.0)
    for color, row in zip(cpu_colors, cpu_df.to_dict("records")):
        ax_roof.scatter(row["ai"], row["gflops"], s=75, color=color, edgecolors="black", linewidth=0.7, zorder=3)
        offset = cpu_label_offsets.get(int(row["threads"]), (6, 6))
        ax_roof.annotate(
            f"CPU p={int(row['threads'])}",
            (row["ai"], row["gflops"]),
            textcoords="offset points",
            xytext=offset,
            ha="right" if offset[0] < 0 else "left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.75", "linewidth": 0.6},
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.8, "shrinkA": 2, "shrinkB": 4},
        )

    for kind, sub in gpu_df.groupby("kind"):
        style = gpu_kind_styles[kind]
        ax_roof.plot(sub["ai_median"], sub["gflops_median"], color=style["color"], linewidth=1.3, alpha=0.9)
        ax_roof.scatter(
            sub["ai_median"],
            sub["gflops_median"],
            s=82,
            color=style["color"],
            marker=style["marker"],
            edgecolors="black",
            linewidth=0.7,
            label=style["label"],
            zorder=3,
        )
        for row in sub.to_dict("records"):
            offset = gpu_label_offsets.get(kind, {}).get(int(row["threads"]), (6, 6))
            ax_roof.annotate(
                f"p={int(row['threads'])}",
                (row["ai_median"], row["gflops_median"]),
                textcoords="offset points",
                xytext=offset,
                ha="right" if offset[0] < 0 else "left",
                fontsize=8.5,
                color=style["color"],
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.75", "linewidth": 0.6},
                arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.8, "shrinkA": 2, "shrinkB": 4},
            )

    # roof_text = (
    #     "CPU points are full-application AMD uProf roofline points.\n"
    #     "GPU points are kernel medians from `*_uprof_collect_inst.txt`.\n"
    #     "CPU hardware roofline shows shared DRAM slope with both FP64 and FP32 ceilings.\n"
    #     "GPU hardware roof uses FP32 because the application is single precision.\n"
    #     "The GPU points are shown against theoretical hardware roofs only.\n"
    #     "No empirical GPU bandwidth roof is drawn because the profiler GB/s values\n"
    #     "are not directly comparable to H100 HBM bandwidth."
    # )
    # ax_roof.text(
    #     0.02,
    #     0.03,
    #     roof_text,
    #     transform=ax_roof.transAxes,
    #     fontsize=9.3,
    #     bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
    # )

    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(x_min, x_max)
    ax_roof.set_ylim(y_min, y_max)
    ax_roof.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=13, fontweight="bold")
    ax_roof.set_ylabel("Throughput (GFLOP/s)", fontsize=13, fontweight="bold")
    ax_roof.set_title("CPU Hardware Roofline vs GPU Kernel Roofline-Style View", fontsize=14, fontweight="bold")
    ax_roof.tick_params(axis="both", labelsize=11)
    ax_roof.text(
        0.985,
        0.965,
        "Kuma H100 theory: 67 TFLOP/s FP32, 2.4 TB/s",
        transform=ax_roof.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        color=kuma_h100["color"],
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    ax_roof.text(
        cpu_dp_ridge * 1.06,
        cpu_dp_peak * 0.55,
        f"CPU FP64 ridge = {cpu_dp_ridge:.2f} FLOP/B",
        color="0.35",
        fontsize=9.1,
        rotation=90,
        va="center",
        ha="left",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    ax_roof.text(
        cpu_sp_ridge * 1.04,
        cpu_sp_peak * 0.54,
        f"CPU FP32 ridge = {cpu_sp_ridge:.2f} FLOP/B",
        color="0.35",
        fontsize=9.1,
        rotation=90,
        va="center",
        ha="left",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    ax_roof.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=5,
        fontsize=8.0,
        frameon=True,
        columnspacing=1.1,
        handlelength=2.4,
    )

    cmp_threads = sorted(set(cpu_df["threads"]).intersection(set(gpu_df["threads"])))
    cpu_cmp = cpu_df.set_index("threads").reindex(cmp_threads)
    x = np.arange(len(cmp_threads), dtype=float)
    ax_cmp_eff = ax_cmp.twinx()

    ax_cmp.plot(
        x,
        cpu_cmp["gflops"],
        marker="o",
        linewidth=2.0,
        color=ROOF_COMPUTE_COLOR,
        label="CPU application throughput",
    )
    for kind, style in gpu_kind_styles.items():
        sub = gpu_df[gpu_df["kind"] == kind].set_index("threads").reindex(cmp_threads)
        ax_cmp.plot(
            x,
            sub["gflops_median"],
            marker=style["marker"],
            linewidth=2.0,
            color=style["color"],
            label=f"{style['label']} throughput",
        )
        ax_cmp_eff.plot(
            x,
            sub["mib_per_joule_median"],
            marker=style["marker"],
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=1.8,
            linestyle="--",
            alpha=0.75,
            color=style["color"],
            label=f"{style['label']} MiB/J",
        )

    cpu_vals = cpu_cmp["gflops"].tolist()
    gridder_vals = gpu_df[gpu_df["kind"] == "gridder"].set_index("threads").reindex(cmp_threads)["gflops_median"].tolist()
    subfft_vals = gpu_df[gpu_df["kind"] == "sub-fft"].set_index("threads").reindex(cmp_threads)["gflops_median"].tolist()
    for i, (cv, gv, sv) in enumerate(zip(cpu_vals, gridder_vals, subfft_vals)):
        ax_cmp.annotate(f"{cv:.0f}", (x[i], cv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=ROOF_COMPUTE_COLOR)
        ax_cmp.annotate(f"{gv:.0f}", (x[i], gv), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5, color=GPU_GRIIDER_COLOR)
        ax_cmp.annotate(f"{sv:.0f}", (x[i], sv), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8.5, color=GPU_SUBFFT_COLOR)

    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels([f"p={int(v)}" for v in cmp_threads], fontsize=12)
    ax_cmp.set_xlim(-0.35, len(cmp_threads) - 0.65)
    ax_cmp.set_ylim(2.5e1, max(max(cpu_vals), max(gridder_vals), max(subfft_vals)) * 1.45)
    ax_cmp.set_ylabel("Achieved throughput (GFLOP/s)", fontsize=13, fontweight="bold")
    ax_cmp.set_xlabel("Parallelism p", fontsize=13, fontweight="bold")
    ax_cmp.set_yscale("log")
    ax_cmp_eff.set_ylabel("GPU kernel data movement efficiency (MiB/J)", fontsize=13, fontweight="bold")
    ax_cmp_eff.set_yscale("log")
    gpu_mib_min = float(gpu_df["mib_per_joule_median"].min())
    gpu_mib_max = float(gpu_df["mib_per_joule_median"].max())
    ax_cmp_eff.set_ylim(gpu_mib_min * 0.8, gpu_mib_max * 1.35)
    ax_cmp_eff.tick_params(axis="y", labelsize=11)
    ax_cmp_eff.grid(False)
    ax_cmp.set_title("Matched-p Throughput and GPU MiB/J", fontsize=14, fontweight="bold")
    ax_cmp.tick_params(axis="y", labelsize=11)
    ax_cmp.grid(True, axis="y", alpha=0.25)
    throughput_handles = [
        Line2D([0], [0], color=ROOF_COMPUTE_COLOR, marker="o", linewidth=2.0, label="CPU application throughput"),
        Line2D([0], [0], color=GPU_GRIIDER_COLOR, marker="o", linewidth=2.0, label="GPU gridder throughput"),
        Line2D([0], [0], color=GPU_SUBFFT_COLOR, marker="s", linewidth=2.0, label="GPU sub-FFT throughput"),
    ]
    efficiency_handles = [
        Line2D(
            [0],
            [0],
            color=GPU_GRIIDER_COLOR,
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=1.8,
            linestyle="--",
            alpha=0.75,
            label="GPU gridder MiB/J",
        ),
        Line2D(
            [0],
            [0],
            color=GPU_SUBFFT_COLOR,
            marker="s",
            markerfacecolor="white",
            markeredgewidth=1.3,
            linewidth=1.8,
            linestyle="--",
            alpha=0.75,
            label="GPU sub-FFT MiB/J",
        ),
    ]
    cmp_handles = throughput_handles + efficiency_handles
    ax_cmp.legend(
        cmp_handles,
        [handle.get_label() for handle in cmp_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=8.2,
        frameon=True,
        columnspacing=1.15,
        handlelength=2.5,
    )

    # speed_text = (
    #     f"At p=64, CPU app = {float(cpu_cmp.loc[64, 'gflops']):.0f} GFLOP/s\n"
    #     f"At p=64, GPU gridder median = {float(gpu_df[(gpu_df['threads']==64)&(gpu_df['kind']=='gridder')]['gflops_median'].iloc[0]):.0f} GFLOP/s\n"
    #     f"At p=64, GPU sub-FFT median = {float(gpu_df[(gpu_df['threads']==64)&(gpu_df['kind']=='sub-fft')]['gflops_median'].iloc[0]):.0f} GFLOP/s"
    # )
    # ax_cmp.text(
    #     0.03,
    #     0.08,
    #     speed_text,
    #     transform=ax_cmp.transAxes,
    #     va="bottom",
    #     fontsize=9.4,
    #     bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
    # )

    # fig.suptitle(
    #     # "GPU-vs-CPU Roofline Comparison for WSClean Stacking (16384^2, t=256, c=256)\n"
    #     "CPU from AMD uProf application roofline | GPU from kernel instruction-collection logs",
    #     fontsize=15,
    #     fontweight="bold",
    #     y=0.98,
    # )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.28, top=0.86, wspace=0.30)

    saved_paths = []
    for suffix in (".png", ".pdf"):
        out = output_stem.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        saved_paths.append(out)
    plt.close(fig)
    return saved_paths


def build_summary(cpu_df: pd.DataFrame, gpu_df: pd.DataFrame) -> pd.DataFrame:
    cpu_summary = cpu_df[
        ["threads", "source", "kind", "ai", "gflops", "gbs", "elapsed_s", "sp_peak_fraction_pct", "dram_bw_fraction_pct"]
    ].copy()
    cpu_summary = cpu_summary.rename(
        columns={
            "ai": "arithmetic_intensity_flop_per_byte",
            "gflops": "throughput_gflops",
            "gbs": "bandwidth_gbs",
            "sp_peak_fraction_pct": "compute_peak_fraction_pct",
        }
    )
    cpu_summary["gb_per_joule_median"] = np.nan
    cpu_summary["mib_per_joule_median"] = np.nan

    gpu_summary = gpu_df.copy()
    gpu_summary["arithmetic_intensity_flop_per_byte"] = gpu_summary["ai_median"]
    gpu_summary["throughput_gflops"] = gpu_summary["gflops_median"]
    gpu_summary["bandwidth_gbs"] = gpu_summary["gbs_median"]
    gpu_summary["elapsed_s"] = gpu_summary["time_s_median"]
    gpu_summary["compute_peak_fraction_pct"] = 100.0 * gpu_summary["throughput_gflops"] / gpu_df["gflops_max"].max()
    gpu_summary["dram_bw_fraction_pct"] = 100.0 * gpu_summary["bandwidth_gbs"] / gpu_df["gbs_max"].max()
    gpu_summary = gpu_summary[
        [
            "threads",
            "source",
            "kind",
            "arithmetic_intensity_flop_per_byte",
            "throughput_gflops",
            "bandwidth_gbs",
            "elapsed_s",
            "compute_peak_fraction_pct",
            "dram_bw_fraction_pct",
            "gb_per_joule_median",
            "mib_per_joule_median",
            "sample_count",
        ]
    ]

    return pd.concat([cpu_summary, gpu_summary], ignore_index=True)


def main() -> int:
    args = parse_args()
    cpu_df = load_cpu_summary(args.cpu_summary)
    gpu_raw, gpu_df = load_gpu_kernel_summary(args.gpu_root)
    summary = build_summary(cpu_df, gpu_df)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    saved = plot_comparison(cpu_df, gpu_raw, gpu_df, args.output_stem)

    print("GPU-vs-CPU roofline comparison summary")
    print("=" * 80)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=" * 80)
    print(f"Saved summary CSV to {args.summary_csv}")
    for path in saved:
        print(f"Saved figure to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
