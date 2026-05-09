#!/usr/bin/env python3
"""
Plot 9b — same as plot9 but with linear (not log) right y-axis for Mvis/h.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
DERIVED_DIR = DATA_DIR / "derived"
RESULTS_DIR = BASE_DIR / "results"
CSV_PATH = BASE_DIR / "data" / "benchmarks_comprehensive.csv"
OUTPUT_STEM = RESULTS_DIR / "plot9b_flagship_energy_stack_throughput_facets_linear"

MVIS_COLOR = "#d95f5f"


def main() -> int:
    df = pd.read_csv(CSV_PATH)
    df["WorkProxy"] = (df["Image Size"] ** 2) * df["Timesteps"] * df["Channels"]

    img_sizes = sorted(df["Image Size"].unique())
    locs = ["WA"]
    nrows = len(img_sizes)
    ncols = len(locs)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 3.2 * nrows), squeeze=False)

    for i, img in enumerate(img_sizes):
        for j, loc in enumerate(locs):
            ax = axes[i][j]
            sub = df[(df["Image Size"] == img) & (df["Location"] == loc)].copy()
            if sub.empty:
                ax.axis("off")
                continue

            sub["combo"] = sub.apply(lambda r: f"t{int(r['Timesteps'])}-c{int(r['Channels'])}", axis=1)
            sub = sub.sort_values(["Timesteps", "Channels"])

            t_series = sub["Timesteps"].values
            t_breaks = [k - 0.5 for k in range(1, len(t_series)) if t_series[k] != t_series[k - 1]]
            t_groups: list[tuple[int, int, int]] = []
            start = 0
            for k in range(1, len(t_series) + 1):
                if k == len(t_series) or t_series[k] != t_series[k - 1]:
                    t_groups.append((int(t_series[start]), start, k - 1))
                    start = k

            x = np.arange(len(sub))
            dyn = sub["Dynamic Energy (Wh)"].values
            sta = sub["Static Energy (Wh)"].values

            ax.bar(x, dyn, label="Dynamic Energy (Wh)", hatch="///", edgecolor="black", linewidth=0.4)
            ax.bar(x, sta, bottom=dyn, label="Static Energy (Wh)", hatch="...", edgecolor="black", linewidth=0.4)
            ax.set_ylabel("Energy (Wh)")
            ax.set_xticks(x)
            if i == nrows - 1:
                ax.set_xticklabels(sub["combo"], rotation=90, ha="center")
            else:
                ax.set_xticklabels([])
            ax.set_title(f"Image {img}² — {loc}")
            for xb in t_breaks:
                ax.axvline(xb, linestyle=":", linewidth=1.0, color="0.45", alpha=0.85)

            secax = ax.secondary_xaxis("top")
            secax.set_xticks([0.5 * (i0 + i1) for _, i0, i1 in t_groups])
            secax.set_xticklabels([f"t={tval}" for tval, _, _ in t_groups])
            secax.tick_params(axis="x", labelsize=8, length=0, pad=2)

            ax2 = ax.twinx()
            ax2.plot(x, sub["Mvis/h"].values, marker="o",
                     color=MVIS_COLOR, markeredgecolor="black", markeredgewidth=0.5)
            ax2.set_ylabel("Mvis/h", color=MVIS_COLOR)
            ax2.tick_params(axis="y", colors=MVIS_COLOR)
            ax2.spines["right"].set_edgecolor(MVIS_COLOR)
            ax2.yaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=False))
            # linear scale (no set_yscale call)

            # "insight" marker: best energy-efficiency point
            best_idx = int(sub["Mvis/kWh"].values.argmax())
            y_best = sub["Mvis/h"].values[best_idx]
            x_frac = best_idx / max(1, len(sub) - 1)
            x_frac = min(max(x_frac, 0.08), 0.92)
            ax2.annotate(
                "best Mvis/kWh",
                xy=(best_idx, y_best),
                xycoords="data",
                xytext=(x_frac, 1.06),
                textcoords="axes fraction",
                ha="center",
                va="bottom",
                fontsize=8,
                clip_on=False,
                annotation_clip=False,
                arrowprops=dict(arrowstyle="->", lw=0.8),
            )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(
        "Energy breakdown (stacked) + throughput (right axis, linear) across (Timesteps, Channels)\n"
        "Faceted by Image Size — WA Location",
        y=0.99,
    )
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(str(out), dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
