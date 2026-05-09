#!/usr/bin/env python3
"""
Plot a representative GPU kernel timing breakdown from the instruction-collection
logs. This replaces the CPU hotspot view, which is not representative of the GPU
execution path.
"""

import re
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
REPO_ROOT = SCRIPT_DIR.parent
INPUT_ROOT = REPO_ROOT.parent / "astroCAMP-bench" / "profiling_gpu2"
OUTPUT_STEM = RESULTS_DIR / "plot27_hotspots_summary"

KERNELS = ["average-beam", "gridder", "sub-fft", "wtiling"]
COLORS = {
    "average-beam": plt.cm.cividis(0.15),
    "gridder": plt.cm.cividis(0.30),
    "sub-fft": plt.cm.cividis(0.55),
    "wtiling": plt.cm.cividis(0.80),
}


def load_kernel_stage_summary(root: Path) -> pd.DataFrame:
    pattern = re.compile(r"^\|(?P<kind>[a-zA-Z0-9_-]+):\s+(?P<time>[0-9.e+-]+)\s+s")
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
            kind = m.group("kind")
            if kind not in KERNELS + ["host", "device"]:
                continue
            rows.append({"threads": threads, "kind": kind, "time_s": float(m.group("time"))})

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No representative kernel-stage lines found under {root}")

    summary = (
        df.groupby(["threads", "kind"], as_index=False)
        .agg(
            sample_count=("time_s", "size"),
            median_time_s=("time_s", "median"),
            mean_time_s=("time_s", "mean"),
        )
        .sort_values(["threads", "kind"])
        .reset_index(drop=True)
    )
    return summary


def main() -> int:
    summary = load_kernel_stage_summary(INPUT_ROOT)
    kernel_df = summary[summary["kind"].isin(KERNELS)].copy()
    ref_df = summary[summary["kind"].isin(["host", "device"])].copy()

    threads = sorted(kernel_df["threads"].unique())
    x = np.arange(len(threads), dtype=float)
    width = 0.18

    fig, (ax_time, ax_rank) = plt.subplots(1, 2, figsize=(15.0, 7.2), gridspec_kw={"width_ratios": [1.25, 0.95]})

    for i, kind in enumerate(KERNELS):
        sub = kernel_df[kernel_df["kind"] == kind].set_index("threads").reindex(threads)
        xpos = x + (i - 1.5) * width
        ax_time.bar(
            xpos,
            sub["median_time_s"],
            width=width,
            color=COLORS[kind],
            edgecolor="black",
            linewidth=0.6,
            label=kind,
        )
        for xi, val in zip(xpos, sub["median_time_s"]):
            ax_time.text(xi, val * 1.04, f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, rotation=90)

    host_ref = ref_df[ref_df["kind"] == "host"].set_index("threads").reindex(threads)
    device_ref = ref_df[ref_df["kind"] == "device"].set_index("threads").reindex(threads)
    ax_time.plot(x, host_ref["median_time_s"], color="0.35", linestyle="--", marker="o", linewidth=1.6, label="host median")
    ax_time.plot(x, device_ref["median_time_s"], color="0.15", linestyle="-.", marker="s", linewidth=1.6, label="device median")

    ax_time.set_xticks(x)
    ax_time.set_xticklabels([f"p={p}" for p in threads], fontsize=11)
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Median stage time per invocation (s)", fontsize=12.5, fontweight="bold")
    ax_time.set_xlabel("Parallelism p", fontsize=12.5, fontweight="bold")
    ax_time.set_title("Representative GPU Kernel Stages Across Runs", fontsize=14, fontweight="bold")
    ax_time.grid(True, axis="y", alpha=0.25)
    ax_time.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, fontsize=9.7)

    rank_df = (
        kernel_df.groupby("kind", as_index=False)
        .agg(
            overall_median_time_s=("median_time_s", "median"),
            overall_mean_time_s=("mean_time_s", "mean"),
        )
        .sort_values("overall_median_time_s", ascending=True)
        .reset_index(drop=True)
    )
    y = np.arange(len(rank_df), dtype=float)
    ax_rank.barh(
        y,
        rank_df["overall_median_time_s"],
        color=[COLORS[k] for k in rank_df["kind"]],
        edgecolor="black",
        linewidth=0.6,
    )
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels(rank_df["kind"], fontsize=11)
    ax_rank.set_xscale("log")
    ax_rank.set_xlabel("Across-run median stage time (s)", fontsize=12.5, fontweight="bold")
    ax_rank.set_title("Kernel Ranking by Median Time", fontsize=14, fontweight="bold")
    ax_rank.grid(True, axis="x", alpha=0.25)
    for yi, med, mean in zip(y, rank_df["overall_median_time_s"], rank_df["overall_mean_time_s"]):
        ax_rank.text(med * 1.05, yi, f"median {med:.3f}s | mean {mean:.3f}s", va="center", fontsize=9.1)

    note_text = (
        "This plot uses the `|kernel:` stage timings from `*_uprof_collect_inst.txt`\n"
        "for p={1,16,32,64}. It replaces the CPU hotspot-function view because\n"
        "the AMD hotspot export mainly profiles CPU-side casacore/WSClean work,\n"
        "which is not representative of the GPU execution path."
    )
    fig.text(
        0.5,
        0.03,
        note_text,
        ha="center",
        fontsize=10.1,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
    )

    fig.suptitle(
        "Representative GPU Kernel Timing Breakdown for WSClean Stacking",
        fontsize=15.5,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.22, wspace=0.28)

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
