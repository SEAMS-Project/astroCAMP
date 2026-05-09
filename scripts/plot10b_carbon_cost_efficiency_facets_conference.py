#!/usr/bin/env python3
"""
Conference-friendly single-column variant of Plot 10.

Largest image only, grouped by channels for clearer trend visibility.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "benchmarks_comprehensive.csv"
OUTPUT_STEM = SCRIPT_DIR / "plot10b_carbon_cost_efficiency_facets_conference"
SUMMARY_CSV = SCRIPT_DIR / "plot10b_carbon_cost_efficiency_facets_conference_summary.csv"

CARBON_OPERATIONAL = "#d95f5f"
CARBON_EMBODIED = "#5b8ecb"
COST_OPERATIONAL = "#d95f5f"
COST_CAPITAL = "#5b8ecb"
WA_LINE = "#1a7d3a"
SA_LINE = "#74c476"
EFF_AXIS_COLOR = "#1a7d3a"

TITLE_FS = 11.8
LABEL_FS = 10.6
TICK_FS = 8.4
LEGEND_FS = 7.4


def add_group_guides(ax: plt.Axes, xvals: np.ndarray, sub: pd.DataFrame) -> None:
    c_groups = list(sub.groupby("Channels", sort=True))
    c_positions = []
    c_labels = []
    for idx, (channels, group) in enumerate(c_groups):
        start = group.index.min()
        end = group.index.max()
        c_positions.append(float(xvals[start:end + 1].mean()))
        c_labels.append(f"c={int(channels)}")
        ax.axvspan(
            xvals[start] - 0.55,
            xvals[end] + 0.55,
            color="#f7f7f7" if idx % 2 == 0 else "#ffffff",
            alpha=0.55,
            zorder=0,
        )
        if idx < len(c_groups) - 1:
            ax.axvline(xvals[end] + 0.55, color="#7a7a7a", linestyle=":", linewidth=1.0, alpha=0.85, zorder=1)

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(c_positions)
    secax.set_xticklabels(c_labels, fontsize=TICK_FS)
    secax.tick_params(axis="x", length=0, pad=2)


def main() -> int:
    df = pd.read_csv(CSV_PATH)
    largest_img = int(df["Image Size"].max())
    sub = (
        df[df["Image Size"] == largest_img]
        .copy()
        .sort_values(["Channels", "Timesteps", "Location"])
        .reset_index(drop=True)
    )
    sub["operational_carbon"] = sub["Total Carbon (g CO2)"] * sub["Operational Carbon (%)"] / 100.0
    sub["embodied_carbon"] = sub["Total Carbon (g CO2)"] * sub["Embodied Carbon (%)"] / 100.0
    sub["operational_cost"] = sub["Total Cost ($)"] * sub["Operational Cost (%)"] / 100.0
    sub["capital_cost"] = sub["Total Cost ($)"] * sub["Capital Cost (%)"] / 100.0
    sub.to_csv(SUMMARY_CSV, index=False)

    locs = sorted(sub["Location"].unique())
    ref = sub[sub["Location"] == locs[0]].copy().sort_values(["Channels", "Timesteps"]).reset_index(drop=True)
    x_base = np.arange(len(ref), dtype=float) * 1.10
    bar_width = 0.32

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.0), sharex=True)
    ax1, ax2 = axes

    loc_offsets = {loc: (-0.17 if idx == 0 else 0.17) for idx, loc in enumerate(locs)}
    loc_hatches = {locs[0]: "///", locs[1]: "\\\\\\\\"} if len(locs) >= 2 else {locs[0]: "///"}
    line_styles = {locs[0]: ("o", "-"), locs[1]: ("s", "--")} if len(locs) >= 2 else {locs[0]: ("o", "-")}
    line_colors = {locs[0]: WA_LINE, locs[1]: SA_LINE} if len(locs) >= 2 else {locs[0]: WA_LINE}

    for loc in locs:
        loc_sub = sub[sub["Location"] == loc].copy().sort_values(["Channels", "Timesteps"]).reset_index(drop=True)
        x = x_base + loc_offsets[loc]

        ax1.bar(
            x,
            loc_sub["embodied_carbon"],
            width=bar_width,
            color=CARBON_EMBODIED,
            edgecolor="black",
            linewidth=0.45,
            hatch=loc_hatches[loc],
            alpha=0.82,
            zorder=2,
        )
        ax1.bar(
            x,
            loc_sub["operational_carbon"],
            width=bar_width,
            bottom=loc_sub["embodied_carbon"],
            color=CARBON_OPERATIONAL,
            edgecolor="black",
            linewidth=0.45,
            hatch=loc_hatches[loc],
            alpha=0.82,
            zorder=2,
        )

        marker, linestyle = line_styles[loc]
        ax1r = ax1.twinx() if loc == locs[0] else ax1._right_ax  # type: ignore[attr-defined]
        if loc == locs[0]:
            ax1._right_ax = ax1r  # type: ignore[attr-defined]
        ax1r.plot(
            x_base,
            loc_sub["Mvis/kgCO2"],
            color=line_colors[loc],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=4.0,
            markeredgecolor="black",
            markeredgewidth=0.45,
            zorder=4,
        )

        ax2.bar(
            x,
            loc_sub["capital_cost"],
            width=bar_width,
            color=COST_CAPITAL,
            edgecolor="black",
            linewidth=0.45,
            hatch=loc_hatches[loc],
            alpha=0.82,
            zorder=2,
        )
        ax2.bar(
            x,
            loc_sub["operational_cost"],
            width=bar_width,
            bottom=loc_sub["capital_cost"],
            color=COST_OPERATIONAL,
            edgecolor="black",
            linewidth=0.45,
            hatch=loc_hatches[loc],
            alpha=0.82,
            zorder=2,
        )

        ax2r = ax2.twinx() if loc == locs[0] else ax2._right_ax  # type: ignore[attr-defined]
        if loc == locs[0]:
            ax2._right_ax = ax2r  # type: ignore[attr-defined]
        ax2r.plot(
            x_base,
            loc_sub["Mvis/$"],
            color=line_colors[loc],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.7,
            markersize=4.0,
            markeredgecolor="black",
            markeredgewidth=0.45,
            zorder=4,
        )

    ax1r = ax1._right_ax  # type: ignore[attr-defined]
    ax2r = ax2._right_ax  # type: ignore[attr-defined]

    add_group_guides(ax1, x_base, ref)
    add_group_guides(ax2, x_base, ref)

    for ax in (ax1, ax2):
        ax.set_xlim(x_base.min() - 0.65, x_base.max() + 0.65)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", labelsize=TICK_FS)

    ax1r.tick_params(axis="y", labelsize=TICK_FS, colors=EFF_AXIS_COLOR)
    ax2r.tick_params(axis="y", labelsize=TICK_FS, colors=EFF_AXIS_COLOR)
    ax1r.spines["top"].set_visible(False)
    ax2r.spines["top"].set_visible(False)
    ax1r.spines["right"].set_edgecolor(EFF_AXIS_COLOR)
    ax2r.spines["right"].set_edgecolor(EFF_AXIS_COLOR)
    ax1r.grid(False)
    ax2r.grid(False)

    ax1.set_ylabel("Carbon (g CO$_2$)", fontsize=LABEL_FS, fontweight="bold")
    ax1r.set_ylabel("Mvis/kgCO$_2$", fontsize=LABEL_FS, fontweight="bold", color=EFF_AXIS_COLOR)
    ax1.set_title(f"Carbon Breakdown and Carbon Efficiency ({largest_img}$^2$)", fontsize=TITLE_FS, fontweight="bold")

    ax2.set_ylabel("Cost ($)", fontsize=LABEL_FS, fontweight="bold")
    ax2r.set_ylabel("Mvis/$", fontsize=LABEL_FS, fontweight="bold", color=EFF_AXIS_COLOR)
    ax2.set_title("Cost Breakdown and Cost Efficiency", fontsize=TITLE_FS, fontweight="bold")

    ax2.set_xticks(x_base)
    ax2.set_xticklabels([f"t={int(v)}" for v in ref["Timesteps"]], rotation=90, ha="center", fontsize=TICK_FS)
    ax2.set_xlabel("Timesteps within channel groups", fontsize=LABEL_FS, fontweight="bold")

    carbon_handles = [
        Patch(facecolor=CARBON_EMBODIED, edgecolor="black", linewidth=0.45, label="Embodied"),
        Patch(facecolor=CARBON_OPERATIONAL, edgecolor="black", linewidth=0.45, label="Operational"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.45, hatch=loc_hatches[locs[0]], label=f"{locs[0]} bars"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.45, hatch=loc_hatches[locs[1]], label=f"{locs[1]} bars"),
        Line2D([0], [0], color=line_colors[locs[0]], marker=line_styles[locs[0]][0], linestyle=line_styles[locs[0]][1], linewidth=1.7, markersize=4.0, label=f"{locs[0]} efficiency"),
        Line2D([0], [0], color=line_colors[locs[1]], marker=line_styles[locs[1]][0], linestyle=line_styles[locs[1]][1], linewidth=1.7, markersize=4.0, label=f"{locs[1]} efficiency"),
    ]
    ax1.legend(
        carbon_handles,
        [h.get_label() for h in carbon_handles],
        loc="upper left",
        fontsize=LEGEND_FS,
        frameon=True,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    cost_handles = [
        Patch(facecolor=COST_CAPITAL, edgecolor="black", linewidth=0.45, label="Capital"),
        Patch(facecolor=COST_OPERATIONAL, edgecolor="black", linewidth=0.45, label="Operational"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.45, hatch=loc_hatches[locs[0]], label=f"{locs[0]} bars"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.45, hatch=loc_hatches[locs[1]], label=f"{locs[1]} bars"),
        Line2D([0], [0], color=line_colors[locs[0]], marker=line_styles[locs[0]][0], linestyle=line_styles[locs[0]][1], linewidth=1.7, markersize=4.0, label=f"{locs[0]} efficiency"),
        Line2D([0], [0], color=line_colors[locs[1]], marker=line_styles[locs[1]][0], linestyle=line_styles[locs[1]][1], linewidth=1.7, markersize=4.0, label=f"{locs[1]} efficiency"),
    ]
    ax2.legend(
        cost_handles,
        [h.get_label() for h in cost_handles],
        loc="upper left",
        fontsize=LEGEND_FS,
        frameon=True,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.12, right=0.89, top=0.94, bottom=0.14, hspace=0.32)

    for suffix in (".png", ".pdf"):
        out = OUTPUT_STEM.with_suffix(suffix)
        fig.savefig(out, dpi=300 if suffix == ".png" else None, bbox_inches="tight")
        print(f"Saved figure to {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
