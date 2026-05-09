#!/usr/bin/env python3
"""
Enhanced CPU roofline figure that makes the gap between achieved throughput gain
and real wall-time scaling explicit.
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
INPUT_CSV = DERIVED_DIR / "roofline_stacking_summary.csv"
OUTPUT_STEM = RESULTS_DIR / "plot25b_cpu_roofline_scaling_gap"


def main() -> int:
    df = pd.read_csv(INPUT_CSV).sort_values("requested_threads").reset_index(drop=True)

    dp_peak = float(df["dp_peak_gflops"].iloc[0])
    sp_peak = float(df["sp_peak_gflops"].iloc[0])
    dram_bw = float(df["dram_bw_gbs"].iloc[0])
    dp_ridge = dp_peak / dram_bw
    sp_ridge = sp_peak / dram_bw

    fig, (ax_roof, ax_scale) = plt.subplots(
        1,
        2,
        figsize=(15.0, 7.2),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    x_min = 0.03
    x_max = 140.0
    y_min = 20.0
    y_max = sp_peak * 1.35

    ai_diag = np.logspace(np.log10(x_min), np.log10(sp_ridge), 400)
    ax_roof.plot(ai_diag, dram_bw * ai_diag, color="#4c72b0", linewidth=2.3, label=f"DRAM roof ({dram_bw:.0f} GB/s)")
    ax_roof.hlines(dp_peak, dp_ridge, x_max, color="#4c72b0", linestyle="-.", linewidth=2.0, label=f"CPU FP64 peak ({dp_peak:.0f} GFLOP/s)")
    ax_roof.hlines(sp_peak, sp_ridge, x_max, color="#4c72b0", linestyle="--", linewidth=2.3, label=f"CPU FP32 peak ({sp_peak:.0f} GFLOP/s)")
    ax_roof.axvline(dp_ridge, color="#4c72b0", linestyle=":", linewidth=1.1, alpha=0.85)
    ax_roof.axvline(sp_ridge, color="#4c72b0", linestyle=":", linewidth=1.1, alpha=0.85)

    eff = df["walltime_parallel_efficiency_pct"].to_numpy()
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=eff.min(), vmax=eff.max())
    colors = cmap(norm(eff))

    ax_roof.plot(df["arithmetic_intensity_flop_per_byte"], df["throughput_gflops"], color="0.45", linewidth=1.1, alpha=0.7)
    label_offsets = {1: (-38, -10), 16: (-42, 12), 32: (10, 16), 64: (10, -18)}
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
            ha="right" if offset[0] < 0 else "left",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8", "linewidth": 0.5},
            arrowprops={"arrowstyle": "-", "color": "0.45", "linewidth": 0.7, "shrinkA": 2, "shrinkB": 4},
        )

    ax_roof.text(dp_ridge * 1.06, dp_peak * 0.55, f"FP64 ridge = {dp_ridge:.2f}", rotation=90, color="#4c72b0", fontsize=9.6, va="center")
    ax_roof.text(sp_ridge * 1.04, sp_peak * 0.52, f"FP32 ridge = {sp_ridge:.2f}", rotation=90, color="#4c72b0", fontsize=9.6, va="center")

    roof_text = (
        f"Arithmetic intensity stays at {df['arithmetic_intensity_flop_per_byte'].min():.1f}-{df['arithmetic_intensity_flop_per_byte'].max():.1f} FLOP/B\n"
        "All runs stay to the right of the DRAM ridge\n"
        f"Best CPU throughput is {df['throughput_gflops'].max():.1f} GFLOP/s at p={int(df.loc[df['throughput_gflops'].idxmax(), 'requested_threads'])}\n"
        "Point color shows wall-time parallel efficiency"
    )
    ax_roof.text(
        0.03,
        0.05,
        roof_text,
        transform=ax_roof.transAxes,
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
    )

    ax_roof.set_xscale("log")
    ax_roof.set_yscale("log")
    ax_roof.set_xlim(x_min, x_max)
    ax_roof.set_ylim(y_min, y_max)
    ax_roof.set_xlabel("Arithmetic intensity (FLOP/byte)", fontsize=13, fontweight="bold")
    ax_roof.set_ylabel("Throughput (GFLOP/s)", fontsize=13, fontweight="bold")
    ax_roof.set_title("CPU Roofline with Efficiency-Coded Points", fontsize=14, fontweight="bold")
    ax_roof.tick_params(axis="both", labelsize=11)
    ax_roof.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True, fontsize=9.8)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax_roof, fraction=0.046, pad=0.02)
    cbar.set_label("Wall-time parallel efficiency (%)", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9.5)

    threads = df["requested_threads"].to_numpy()
    x = np.arange(len(threads), dtype=float)
    ax_scale.plot(x, df["throughput_gain_vs_base"], color="#55a868", marker="o", linewidth=2.2, label="Throughput gain")
    ax_scale.plot(x, df["walltime_speedup_vs_base"], color="#c44e52", marker="s", linewidth=2.2, label="Wall-time speedup")
    ax_scale.plot(x, df["ideal_speedup"], color="0.4", linestyle="--", linewidth=1.6, label="Ideal speedup")

    ax_eff = ax_scale.twinx()
    ax_eff.plot(x, df["walltime_parallel_efficiency_pct"], color="#dd8452", marker="D", linewidth=2.0, label="Parallel efficiency")

    for xi, tg, ws in zip(x, df["throughput_gain_vs_base"], df["walltime_speedup_vs_base"]):
        ax_scale.text(xi, tg * 1.03, f"{tg:.1f}x", ha="center", va="bottom", fontsize=9.4, color="#55a868")
        ax_scale.text(xi, ws * 0.95, f"{ws:.2f}x", ha="center", va="top", fontsize=9.4, color="#c44e52")
    for xi, eff_v in zip(x, df["walltime_parallel_efficiency_pct"]):
        ax_eff.text(xi, eff_v + 2.2, f"{eff_v:.1f}%", ha="center", va="bottom", fontsize=9.0, color="#dd8452")

    ax_scale.set_xticks(x)
    ax_scale.set_xticklabels([f"p={int(v)}" for v in threads], fontsize=11.5)
    ax_scale.set_xlabel("Parallelism p", fontsize=13, fontweight="bold")
    ax_scale.set_ylabel("Relative gain over p=1", fontsize=13, fontweight="bold")
    ax_eff.set_ylabel("Parallel efficiency (%)", fontsize=13, fontweight="bold")
    ax_scale.set_title("Throughput Gain vs Real Wall-Time Scaling", fontsize=14, fontweight="bold")
    ax_scale.grid(True, axis="y", alpha=0.25)
    ax_scale.set_ylim(0, max(df["ideal_speedup"]) * 1.1)
    ax_eff.set_ylim(0, 110)

    scale_text = (
        f"At p=64, throughput gain is {df.loc[df['requested_threads']==64, 'throughput_gain_vs_base'].iloc[0]:.2f}x\n"
        f"but wall-time speedup is only {df.loc[df['requested_threads']==64, 'walltime_speedup_vs_base'].iloc[0]:.2f}x\n"
        f"and efficiency falls to {df.loc[df['requested_threads']==64, 'walltime_parallel_efficiency_pct'].iloc[0]:.1f}%"
    )
    ax_scale.text(
        0.03,
        0.05,
        scale_text,
        transform=ax_scale.transAxes,
        fontsize=10.2,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.8"},
    )

    lines1, labels1 = ax_scale.get_legend_handles_labels()
    lines2, labels2 = ax_eff.get_legend_handles_labels()
    ax_scale.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, fontsize=9.8)

    fig.suptitle("CPU Roofline v2: Compute-Side Position vs Weak Real Scaling", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.22, wspace=0.30)

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
