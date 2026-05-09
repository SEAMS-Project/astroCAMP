import os
import re
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Resolve CSV path relative to repo structure
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CSV_PATH = DATA_DIR / "benchmarks_comprehensive.csv"

# Load data
df = pd.read_csv(CSV_PATH)

# Derived fields
df["WorkProxy"] = (df["Image Size"]**2) * df["Timesteps"] * df["Channels"]
df["J_per_Mvis"] = (df["Energy (Wh)"] * 3600.0) / df["Mvis"]
df["Wh_per_Mvis"] = df["Energy (Wh)"] / df["Mvis"]
df["StaticFrac"] = df["Static Energy (Wh)"] / df["Energy (Wh)"]
df["DynamicFrac"] = df["Dynamic Energy (Wh)"] / df["Energy (Wh)"]
df["AvgPower_W"] = df["Power (W)"]  # already average power per run

# Save outputs next to this script (results folder)
out_dir = str(RESULTS_DIR)
os.makedirs(out_dir, exist_ok=True)

def savefig(name):
    path=os.path.join(out_dir, name)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path

# 1) Wall-time vs work proxy (log-log), per location
plt.figure()
for loc, g in df.groupby("Location"):
    g2 = g.sort_values("WorkProxy")
    plt.loglog(g2["WorkProxy"], g2["Time (s)"], marker="o", linestyle="-", label=str(loc))
plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
plt.ylabel("Wall time (s)")
plt.title("Wall-time scaling vs work proxy (log–log)")
plt.legend()
savefig("plot1_walltime_vs_workproxy.png")

# 2) Throughput vs work proxy (log-x), per location
plt.figure()
for loc, g in df.groupby("Location"):
    g2=g.sort_values("WorkProxy")
    plt.semilogx(g2["WorkProxy"], g2["Mvis/h"], marker="o", linestyle="-", label=str(loc))
plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
plt.ylabel("Throughput (Mvis/h)")
plt.title("Throughput vs work proxy")
plt.legend()
savefig("plot2_throughput_vs_workproxy.png")

# 3) Energy vs work proxy (log-log), per location
plt.figure()
for loc, g in df.groupby("Location"):
    g2=g.sort_values("WorkProxy")
    plt.loglog(g2["WorkProxy"], g2["Energy (Wh)"], marker="o", linestyle="-", label=str(loc))
plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
plt.ylabel("Total energy (Wh)")
plt.title("Energy scaling vs work proxy (log–log)")
plt.legend()
savefig("plot3_energy_vs_workproxy.png")

# 4) Static energy fraction vs scale (log-x), per location
plt.figure()
for loc, g in df.groupby("Location"):
    g2=g.sort_values("WorkProxy")
    plt.semilogx(g2["WorkProxy"], g2["StaticFrac"], marker="o", linestyle="-", label=str(loc))
plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
plt.ylabel("Static energy fraction (Static / Total)")
plt.ylim(0, 1)
plt.title("Energy proportionality: static fraction vs scale")
plt.legend()
savefig("plot4_static_fraction_vs_workproxy.png")

# 5) Energy efficiency vs scale (log-x), per location
plt.figure()
for loc, g in df.groupby("Location"):
    g2=g.sort_values("WorkProxy")
    plt.semilogx(g2["WorkProxy"], g2["Mvis/kWh"], marker="o", linestyle="-", label=str(loc))
plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
plt.ylabel("Energy efficiency (Mvis/kWh)")
plt.title("Energy efficiency vs scale")
plt.legend()
savefig("plot5_mvis_per_kwh_vs_workproxy.png")

# 6) Energy–Time scatter with marker size proportional to throughput
plt.figure()
sizes = 30 + 70*(df["Mvis/h"] - df["Mvis/h"].min())/(df["Mvis/h"].max() - df["Mvis/h"].min() + 1e-12)
plt.scatter(df["Time (s)"], df["Energy (Wh)"], s=sizes)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Wall time (s) [log]")
plt.ylabel("Total energy (Wh) [log]")
plt.title("Energy–time tradeoff (marker size ∝ throughput)")
savefig("plot6_energy_time_scatter.png")

# 7) Average power vs throughput
plt.figure()
plt.scatter(df["Mvis/h"], df["AvgPower_W"])
plt.xscale("log")
plt.xlabel("Throughput (Mvis/h) [log]")
plt.ylabel("Average power (W)")
plt.title("Power vs throughput")
savefig("plot7_power_vs_throughput.png")

# 8) Carbon efficiency vs scale (if present)
if "Mvis/kgCO2" in df.columns:
    plt.figure()
    for loc, g in df.groupby("Location"):
        g2=g.sort_values("WorkProxy")
        plt.semilogx(g2["WorkProxy"], g2["Mvis/kgCO2"], marker="o", linestyle="-", label=str(loc))
    plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
    plt.ylabel("Carbon efficiency (Mvis/kgCO2)")
    plt.title("Carbon efficiency vs scale")
    plt.legend()
    savefig("plot8_mvis_per_kgco2_vs_workproxy.png")

# 9) Flagship: facets by Image Size × Location with stacked (Dynamic+Static Wh) and right-axis Mvis/h
img_sizes = sorted(df["Image Size"].unique())
locs = ["WA"]  # Only WA location
nrows = len(img_sizes)
ncols = len(locs)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*ncols, 3.2*nrows), squeeze=False)

for i, img in enumerate(img_sizes):
    for j, loc in enumerate(locs):
        ax = axes[i][j]
        sub = df[(df["Image Size"]==img) & (df["Location"]==loc)].copy()
        if sub.empty:
            ax.axis("off")
            continue

        sub["combo"] = sub.apply(lambda r: f"t{int(r['Timesteps'])}-c{int(r['Channels'])}", axis=1)
        sub = sub.sort_values(["Timesteps","Channels"])

        t_series = sub["Timesteps"].values
        t_breaks = [k - 0.5 for k in range(1, len(t_series)) if t_series[k] != t_series[k - 1]]
        t_groups = []
        start = 0
        for k in range(1, len(t_series) + 1):
            if k == len(t_series) or t_series[k] != t_series[k - 1]:
                t_groups.append((int(t_series[start]), start, k - 1))
                start = k

        x = np.arange(len(sub))
        dyn = sub["Dynamic Energy (Wh)"].values
        sta = sub["Static Energy (Wh)"].values

        ax.bar(x, dyn, label="Dynamic Energy (Wh)")
        ax.bar(x, sta, bottom=dyn, label="Static Energy (Wh)")
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
        ax2.plot(x, sub["Mvis/h"].values, marker="o", color="#d95f5f", markeredgecolor="black", markeredgewidth=0.5)
        ax2.set_ylabel("Mvis/h", color="#d95f5f")
        ax2.tick_params(axis="y", colors="#d95f5f")
        ax2.spines["right"].set_edgecolor("#d95f5f")
        ax2.set_yscale("log")

        # "insight" marker: best energy-efficiency point
        best_idx = sub["Mvis/kWh"].values.argmax()
        y_best = sub["Mvis/h"].values[best_idx]
        x_frac = best_idx / max(1, (len(sub) - 1))
        x_frac = min(max(x_frac, 0.08), 0.92)
        ax2.annotate("best Mvis/kWh",
                 xy=(best_idx, y_best),
                 xycoords="data",
                 xytext=(x_frac, 1.06),
                 textcoords="axes fraction",
                 ha="center",
                 va="bottom",
                 fontsize=8,
                 clip_on=False,
                 annotation_clip=False,
                 arrowprops=dict(arrowstyle="->", lw=0.8))

handles, labels = axes[0][0].get_legend_handles_labels()
fig.suptitle("Energy breakdown (stacked) + throughput (right axis) across (Timesteps, Channels)\nFaceted by Image Size — WA Location", y=0.99)
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))
fig.tight_layout(rect=[0,0,1,0.95])

fig.savefig(str(Path(out_dir) / "plot9_flagship_energy_stack_throughput_facets.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# 15) Explain Plot9: why throughput saturates while cost grows.
wa = df[df["Location"] == "WA"].copy()
if not wa.empty:
    wa = wa.sort_values(["Image Size", "Timesteps", "Channels"]).reset_index(drop=True)
    wa["StaticFrac"] = wa["Static Energy (Wh)"] / wa["Energy (Wh)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left: throughput flattens as work proxy grows.
    for img, sub in wa.groupby("Image Size"):
        sub = sub.sort_values("WorkProxy")
        ax1.semilogx(
            sub["WorkProxy"],
            sub["Mvis/h"],
            marker="o",
            linewidth=1.4,
            label=f"image {int(img)}",
        )
    ax1.set_xlabel("Work proxy = (ImageSize^2) x Timesteps x Channels")
    ax1.set_ylabel("Throughput (Mvis/h)")
    ax1.set_title("Throughput plateaus as workload scale increases")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)

    # Right: saturation aligns with higher static fraction and longer wall time.
    pts = ax2.scatter(
        wa["StaticFrac"],
        wa["Mvis/h"],
        c=wa["Time (s)"],
        s=34,
        cmap="viridis",
        alpha=0.9,
        edgecolors="none",
    )
    ax2.set_xlabel("Static energy fraction (Static/Total)")
    ax2.set_ylabel("Throughput (Mvis/h)")
    ax2.set_title("Higher static share and longer runtime limit throughput gains")
    ax2.grid(True, alpha=0.25)

    cbar = fig.colorbar(pts, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Wall time (s)")

    fig.suptitle("Why Plot9 Saturates: Throughput Plateau Under Rising Static Burden (WA)", y=1.01)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "plot15_explain_plot9_saturation.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "plot15_explain_plot9_saturation.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 24) Article panel: simplified regime-level gains to identify saturation knee.
    step = wa.sort_values(["WorkProxy", "Image Size", "Timesteps", "Channels"]).reset_index(drop=True).copy()
    step["thr_gain_pct"] = 100.0 * step["Mvis/h"].pct_change()
    step["energy_gain_pct"] = 100.0 * step["Energy (Wh)"].pct_change()
    step = step.iloc[1:].copy()  # first row has no previous step

    # Group ordered steps into a few workload regimes to avoid overcrowded bars.
    step["regime"] = pd.qcut(step["WorkProxy"], q=6, labels=False, duplicates="drop")
    regime = (
        step.groupby("regime", as_index=False)
        .agg(
            thr_gain_pct=("thr_gain_pct", "median"),
            energy_gain_pct=("energy_gain_pct", "median"),
            n_steps=("thr_gain_pct", "count"),
            wp_min=("WorkProxy", "min"),
            wp_max=("WorkProxy", "max"),
        )
        .sort_values("regime")
        .reset_index(drop=True)
    )
    regime["label"] = [f"R{i+1}" for i in range(len(regime))]

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    x = np.arange(len(regime), dtype=float)
    bw = 0.42

    thr_gain = regime["thr_gain_pct"].values
    en_gain = regime["energy_gain_pct"].values
    ax.bar(x - bw / 2, thr_gain, width=bw, color="#1f77b4", alpha=0.85, label="throughput gain per step (%)")
    ax.bar(x + bw / 2, en_gain, width=bw, color="#d62728", alpha=0.85, label="energy gain per step (%)")
    ax.axhline(0.0, color="0.55", linewidth=1.0, linestyle="--")

    # Knee: first regime where throughput gain is small and lower than energy gain.
    knee_mask = (regime["thr_gain_pct"] <= 5.0) & (regime["thr_gain_pct"] < regime["energy_gain_pct"])
    if knee_mask.any():
        knee_idx = int(np.where(knee_mask.values)[0][0])
        ax.axvline(knee_idx, color="0.35", linestyle=":", linewidth=1.3, label="saturation knee")
        ax.annotate(
            "saturation knee",
            xy=(knee_idx - bw / 2, float(regime["thr_gain_pct"].iloc[knee_idx])),
            xytext=(knee_idx + 0.6, float(regime["thr_gain_pct"].iloc[knee_idx]) + 6.0),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="0.3"),
            fontsize=8,
            color="0.25",
        )

    # Show compact regime labels and counts.
    ax.set_xticks(x)
    ax.set_xticklabels(regime["label"].tolist(), fontsize=9)
    for xi, n in zip(x, regime["n_steps"].values):
        ax.text(xi, 0.02, f"n={int(n)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color="0.35")

    ax.set_ylabel("Median step-to-step relative gain (%)")
    ax.set_xlabel("Workload regime (low to high WorkProxy, WA)")
    ax.set_title("Saturation Knee Quantification with Regime-Level Gains")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "plot24_plot9_step_gain_knee_overlay.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "plot24_plot9_step_gain_knee_overlay.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

# 10) Carbon and Cost breakdown: largest image size, both locations overlaid
img_sizes = sorted(df["Image Size"].unique())
largest_img = img_sizes[-1]  # Get largest image size
locs = sorted(df["Location"].unique())

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), squeeze=False)

# Prepare data for both locations
data_by_loc = {}
for loc in locs:
    sub = df[(df["Image Size"]==largest_img) & (df["Location"]==loc)].copy()
    if not sub.empty:
        sub["combo"] = sub.apply(lambda r: f"t{int(r['Timesteps'])}-c{int(r['Channels'])}", axis=1)
        sub = sub.sort_values(["Timesteps","Channels"])
        data_by_loc[loc] = sub

if not data_by_loc:
    fig.text(0.5, 0.5, "No data available", ha="center", va="center")
else:
    # Use first location's x-axis as reference
    ref_loc = locs[0]
    ref_sub = data_by_loc[ref_loc]
    x_base = np.arange(len(ref_sub))
    bar_width = 0.35
    
    # --- Subplot 1: Carbon breakdown (bars) + Mvis/kgCO2 efficiency (line) ---
    ax1 = axes[0][0]
    
    for idx, loc in enumerate(locs):
        if loc not in data_by_loc:
            continue
        sub = data_by_loc[loc]
        x = x_base + idx * bar_width
        
        # Compute carbon components from percentages
        total_carbon = sub["Total Carbon (g CO2)"].values
        operational_pct = sub["Operational Carbon (%)"].values / 100.0
        embodied_pct = sub["Embodied Carbon (%)"].values / 100.0
        
        operational_carbon = total_carbon * operational_pct
        embodied_carbon = total_carbon * embodied_pct
        
        # Stacked bars: operational + embodied with hatching for differentiation
        color_operational = 'lightcoral' if loc == 'WA' else 'coral'
        color_embodied = 'lightblue' if loc == 'WA' else 'steelblue'
        hatch_operational = '///' if loc == 'WA' else '\\\\\\'
        hatch_embodied = '...' if loc == 'WA' else 'xxx'
        
        ax1.bar(x, operational_carbon, bar_width, label=f"{loc} Operational Carbon", color=color_operational, hatch=hatch_operational, alpha=0.8)
        ax1.bar(x, embodied_carbon, bar_width, bottom=operational_carbon, label=f"{loc} Embodied Carbon", color=color_embodied, hatch=hatch_embodied, alpha=0.8)
    
    ax1.set_ylabel("Carbon (g CO2)")
    ax1.set_xlabel("Timesteps × Channels")
    ax1.set_xticks(x_base + bar_width / 2)
    ax1.set_xticklabels(ref_sub["combo"], rotation=90, ha="center")
    ax1.set_title("Carbon Breakdown")
    ax1.tick_params(axis='y')
    
    # Right axis: Mvis/kgCO2 efficiency (line)
    ax1_right = ax1.twinx()
    for loc in locs:
        if loc not in data_by_loc:
            continue
        sub = data_by_loc[loc]
        carbon_eff = sub["Mvis/kgCO2"].values
        linestyle = '-' if loc == 'WA' else '--'
        marker_style = 'o' if loc == 'WA' else 's'
        ax1_right.plot(x_base, carbon_eff, marker=marker_style, linestyle=linestyle, linewidth=2, label=f"{loc} Mvis/kgCO2")
    
    ax1_right.set_ylabel("Mvis/kgCO2")
    ax1_right.tick_params(axis='y')
    
    # --- Subplot 2: Cost breakdown (bars) + Mvis/$ efficiency (line) ---
    ax2 = axes[0][1]
    
    for idx, loc in enumerate(locs):
        if loc not in data_by_loc:
            continue
        sub = data_by_loc[loc]
        x = x_base + idx * bar_width
        
        # Compute cost components from percentages
        total_cost = sub["Total Cost ($)"].values
        operational_cost_pct = sub["Operational Cost (%)"].values / 100.0
        capital_cost_pct = sub["Capital Cost (%)"].values / 100.0
        
        operational_cost = total_cost * operational_cost_pct
        capital_cost = total_cost * capital_cost_pct
        
        # Stacked bars: operational + capital with hatching for differentiation
        color_operational_cost = 'lightgreen' if loc == 'WA' else 'mediumseagreen'
        color_capital_cost = 'plum' if loc == 'WA' else 'mediumpurple'
        hatch_operational_cost = '///' if loc == 'WA' else '\\\\\\'
        hatch_capital_cost = '...' if loc == 'WA' else 'xxx'
        
        ax2.bar(x, operational_cost, bar_width, label=f"{loc} Operational Cost", color=color_operational_cost, hatch=hatch_operational_cost, alpha=0.8)
        ax2.bar(x, capital_cost, bar_width, bottom=operational_cost, label=f"{loc} Capital Cost", color=color_capital_cost, hatch=hatch_capital_cost, alpha=0.8)
    
    ax2.set_ylabel("Cost (%)")
    ax2.set_xlabel("Timesteps × Channels")
    ax2.set_xticks(x_base + bar_width / 2)
    ax2.set_xticklabels(ref_sub["combo"], rotation=90, ha="center")
    ax2.set_title("Cost Breakdown")
    ax2.tick_params(axis='y')
    
    # Right axis: Mvis/$ efficiency (line)
    ax2_right = ax2.twinx()
    for loc in locs:
        if loc not in data_by_loc:
            continue
        sub = data_by_loc[loc]
        cost_eff = sub["Mvis/$"].values
        linestyle = '-' if loc == 'WA' else '--'
        marker_style = 'o' if loc == 'WA' else 's'
        ax2_right.plot(x_base, cost_eff, marker=marker_style, linestyle=linestyle, linewidth=2, label=f"{loc} Mvis/$", color='blue')
    
    ax2_right.set_ylabel("Mvis/$", color='blue')
    ax2_right.tick_params(axis='y', labelcolor='blue')
    
    # Collect all handles and labels for unified legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1r, labels1r = ax1_right.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2r, labels2r = ax2_right.get_legend_handles_labels()
    
    all_handles = handles1 + handles1r + handles2 + handles2r
    all_labels = labels1 + labels1r + labels2 + labels2r
    
    fig.legend(all_handles, all_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.97), fontsize=8)

fig.suptitle(f"Carbon and Cost Breakdown with Efficiency Metrics — Image {largest_img}²", y=1.00, fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])

fig.savefig(str(Path(out_dir) / "plot10_carbon_cost_efficiency_facets.png"), dpi=300, bbox_inches="tight")
fig.savefig(str(Path(out_dir) / "plot10_carbon_cost_efficiency_facets.pdf"), format="pdf", bbox_inches="tight")
plt.close(fig)

from matplotlib.backends.backend_pdf import PdfPages

# Reuse df and out_dir computed above

# Precompute work proxy
df["WorkProxy"] = (df["Image Size"]**2) * df["Timesteps"] * df["Channels"]

# ---------- Flagship (vector PDF) ----------
img_sizes = sorted(df["Image Size"].unique())
locs = sorted(df["Location"].unique())
nrows, ncols = len(img_sizes), len(locs)

fig_w = 7.2 * max(1, ncols)
fig_h = 2.2 * nrows + 1.0

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)

for i, img in enumerate(img_sizes):
    for j, loc in enumerate(locs):
        ax = axes[i][j]
        sub = df[(df["Image Size"]==img) & (df["Location"]==loc)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        
        sub["combo"] = sub.apply(lambda r: f"t{int(r['Timesteps'])}-c{int(r['Channels'])}", axis=1)
        sub = sub.sort_values(["Timesteps","Channels"])
        
        x = np.arange(len(sub))
        dyn = sub["Dynamic Energy (Wh)"].values
        sta = sub["Static Energy (Wh)"].values
        
        ax.bar(x, dyn, label="Dynamic")
        ax.bar(x, sta, bottom=dyn, label="Static")
        ax.set_ylabel("Energy (Wh)")
        ax.set_xticks(x)
        if i == nrows - 1:
            ax.set_xticklabels(sub["combo"], rotation=90, ha="center", fontsize=8)
        else:
            ax.set_xticklabels([])
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(f"{loc} — {img}²", fontsize=10)

        # Carbon annotations atop total stacked bars (g CO2)
        total_wh = dyn + sta
        carbon_g = None
        try:
            if "S" in sub.columns:
                carbon_g = pd.to_numeric(sub["S"], errors="coerce").values
            elif "Total Carbon (g CO2)" in sub.columns:
                carbon_g = pd.to_numeric(sub["Total Carbon (g CO2)"], errors="coerce").values
            elif "Total Carbon (gCO2)" in sub.columns:
                carbon_g = pd.to_numeric(sub["Total Carbon (gCO2)"], errors="coerce").values
            else:
                ci_candidates = [
                    "CI (kgCO2/kWh)",
                    "CI kgCO2/kWh",
                    "CI_kgCO2_per_kWh",
                    "CI (gCO2/kWh)",
                    "CI_gCO2_per_kWh",
                ]
                ci_col = next((c for c in ci_candidates if c in sub.columns), None)
                if ci_col is not None:
                    ci_vals = pd.to_numeric(sub[ci_col], errors="coerce").values
                    if "gCO2" in ci_col:
                        ci_vals = ci_vals / 1000.0
                    carbon_g = (total_wh/1000.0) * ci_vals * 1000.0
        except Exception:
            carbon_g = None
        if carbon_g is not None:
            y_offset = max(total_wh) * 0.015 if len(total_wh) else 0.0
            for xi, yh, cg in zip(x, total_wh, carbon_g):
                if np.isfinite(cg):
                    ax.text(xi, yh + y_offset, f"{cg:.0f}", ha="center", va="bottom", fontsize=7, clip_on=False)
        
        ax2 = ax.twinx()
        ax2.plot(x, sub["Mvis/h"].values, marker="o", linewidth=1.2)
        ax2.set_ylabel("Mvis/h")
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelsize=8)
        
        # Insight marker: best energy efficiency
        if "Mvis/kWh" in sub.columns and len(sub) > 0:
            best_idx = int(np.argmax(sub["Mvis/kWh"].values))
            y_best = sub["Mvis/h"].values[best_idx]
            x_frac = best_idx / max(1, (len(sub) - 1))
            x_frac = min(max(x_frac, 0.08), 0.92)
            ax2.annotate("best",
                         xy=(best_idx, y_best),
                         xycoords="data",
                         xytext=(x_frac, 1.06),
                         textcoords="axes fraction",
                         ha="center",
                         va="bottom",
                         fontsize=8,
                         clip_on=False,
                         annotation_clip=False,
                         arrowprops=dict(arrowstyle="->", lw=0.8))

handles, labels = axes[0][0].get_legend_handles_labels()
fig.suptitle(
    "Energy breakdown (stacked) and throughput (right axis) across (Timesteps, Channels)\n"
    "Faceted by Image Size and Location",
    y=0.99, fontsize=11
)
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.935))
fig.tight_layout(rect=[0,0,1,0.955])

flagship_pdf = os.path.join(out_dir, "flagship_energy_throughput_facets_paper.pdf")
fig.savefig(flagship_pdf, format="pdf", bbox_inches="tight")  # vector PDF
plt.close(fig)

# ---------- Multipage PDF with all plots (vector) ----------
multipage_pdf = os.path.join(out_dir, "all_scalability_utilisation_plots.pdf")
with PdfPages(multipage_pdf) as pdf:
    # 1) Wall-time vs work proxy
    plt.figure()
    for loc, g in df.groupby("Location"):
        g2 = g.sort_values("WorkProxy")
        plt.loglog(g2["WorkProxy"], g2["Time (s)"], marker="o", linestyle="-", label=str(loc))
    plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
    plt.ylabel("Wall time (s)")
    plt.title("Wall-time scaling vs work proxy (log–log)")
    plt.legend()
    pdf.savefig(bbox_inches="tight"); plt.close()

    # 2) Throughput vs work proxy
    plt.figure()
    for loc, g in df.groupby("Location"):
        g2 = g.sort_values("WorkProxy")
        plt.semilogx(g2["WorkProxy"], g2["Mvis/h"], marker="o", linestyle="-", label=str(loc))
    plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
    plt.ylabel("Throughput (Mvis/h)")
    plt.title("Throughput vs work proxy")
    plt.legend()
    pdf.savefig(bbox_inches="tight"); plt.close()

    # 3) Energy vs work proxy
    plt.figure()
    for loc, g in df.groupby("Location"):
        g2 = g.sort_values("WorkProxy")
        plt.loglog(g2["WorkProxy"], g2["Energy (Wh)"], marker="o", linestyle="-", label=str(loc))
    plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
    plt.ylabel("Total energy (Wh)")
    plt.title("Energy scaling vs work proxy (log–log)")
    plt.legend()
    pdf.savefig(bbox_inches="tight"); plt.close()

    # 4) Static energy fraction vs scale
    plt.figure()
    for loc, g in df.groupby("Location"):
        g2 = g.sort_values("WorkProxy")
        static_frac = g2["Static Energy (Wh)"] / g2["Energy (Wh)"]
        plt.semilogx(g2["WorkProxy"], static_frac, marker="o", linestyle="-", label=str(loc))
    plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
    plt.ylabel("Static energy fraction")
    plt.ylim(0, 1)
    plt.title("Energy proportionality: static fraction vs scale")
    plt.legend()
    pdf.savefig(bbox_inches="tight"); plt.close()

    # 5) Energy efficiency
    if "Mvis/kWh" in df.columns:
        plt.figure()
        for loc, g in df.groupby("Location"):
            g2 = g.sort_values("WorkProxy")
            plt.semilogx(g2["WorkProxy"], g2["Mvis/kWh"], marker="o", linestyle="-", label=str(loc))
        plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
        plt.ylabel("Energy efficiency (Mvis/kWh)")
        plt.title("Energy efficiency vs scale")
        plt.legend()
        pdf.savefig(bbox_inches="tight"); plt.close()

    # 6) Energy–time scatter (marker size ∝ throughput)
    plt.figure()
    sizes = 30 + 70*(df["Mvis/h"] - df["Mvis/h"].min())/(df["Mvis/h"].max() - df["Mvis/h"].min() + 1e-12)
    plt.scatter(df["Time (s)"], df["Energy (Wh)"], s=sizes)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Wall time (s) [log]")
    plt.ylabel("Total energy (Wh) [log]")
    plt.title("Energy–time tradeoff (marker size ∝ throughput)")
    pdf.savefig(bbox_inches="tight"); plt.close()

    # 7) Power vs throughput
    plt.figure()
    plt.scatter(df["Mvis/h"], df["Power (W)"])
    plt.xscale("log")
    plt.xlabel("Throughput (Mvis/h) [log]")
    plt.ylabel("Average power (W)")
    plt.title("Power vs throughput")
    pdf.savefig(bbox_inches="tight"); plt.close()

    # 8) Carbon efficiency vs scale
    if "Mvis/kgCO2" in df.columns:
        plt.figure()
        for loc, g in df.groupby("Location"):
            g2 = g.sort_values("WorkProxy")
            plt.semilogx(g2["WorkProxy"], g2["Mvis/kgCO2"], marker="o", linestyle="-", label=str(loc))
        plt.xlabel("Work proxy = (ImageSize^2) × Timesteps × Channels")
        plt.ylabel("Carbon efficiency (Mvis/kgCO2)")
        plt.title("Carbon efficiency vs scale")
        plt.legend()
        pdf.savefig(bbox_inches="tight"); plt.close()

    # 9) Flagship again inside the multipage PDF (rasterized within PDF but still vector elements)
    img_sizes = sorted(df["Image Size"].unique())
    locs = sorted(df["Location"].unique())
    nrows, ncols = len(img_sizes), len(locs)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), squeeze=False)
    for i, img in enumerate(img_sizes):
        for j, loc in enumerate(locs):
            ax = axes[i][j]
            sub = df[(df["Image Size"]==img) & (df["Location"]==loc)].copy()
            if sub.empty:
                ax.axis("off")
                continue
            sub["combo"] = sub.apply(lambda r: f"t{int(r['Timesteps'])}-c{int(r['Channels'])}", axis=1)
            sub = sub.sort_values(["Timesteps","Channels"])
            t_series = sub["Timesteps"].values
            t_breaks = [k - 0.5 for k in range(1, len(t_series)) if t_series[k] != t_series[k - 1]]
            t_groups = []
            start = 0
            for k in range(1, len(t_series) + 1):
                if k == len(t_series) or t_series[k] != t_series[k - 1]:
                    t_groups.append((int(t_series[start]), start, k - 1))
                    start = k
            x = np.arange(len(sub))
            dyn = sub["Dynamic Energy (Wh)"].values
            sta = sub["Static Energy (Wh)"].values
            ax.bar(x, dyn, label="Dynamic")
            ax.bar(x, sta, bottom=dyn, label="Static")
            ax.set_ylabel("Energy (Wh)")
            ax.set_xticks(x)
            if i == nrows - 1:
                ax.set_xticklabels(sub["combo"], rotation=90, ha="center", fontsize=8)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=8)
            ax.set_title(f"{loc} — {img}²", fontsize=10)
            for xb in t_breaks:
                ax.axvline(xb, linestyle=":", linewidth=1.0, color="0.45", alpha=0.85)

            secax = ax.secondary_xaxis("top")
            secax.set_xticks([0.5 * (i0 + i1) for _, i0, i1 in t_groups])
            secax.set_xticklabels([f"t={tval}" for tval, _, _ in t_groups])
            secax.tick_params(axis="x", labelsize=8, length=0, pad=2)

            # Carbon annotations atop total stacked bars (g CO2)
            total_wh = dyn + sta
            carbon_g = None
            try:
                if "S" in sub.columns:
                    carbon_g = pd.to_numeric(sub["S"], errors="coerce").values
                elif "Total Carbon (g CO2)" in sub.columns:
                    carbon_g = pd.to_numeric(sub["Total Carbon (g CO2)"], errors="coerce").values
                elif "Total Carbon (gCO2)" in sub.columns:
                    carbon_g = pd.to_numeric(sub["Total Carbon (gCO2)"], errors="coerce").values
                else:
                    ci_candidates = [
                        "CI (kgCO2/kWh)",
                        "CI kgCO2/kWh",
                        "CI_kgCO2_per_kWh",
                        "CI (gCO2/kWh)",
                        "CI_gCO2_per_kWh",
                    ]
                    ci_col = next((c for c in ci_candidates if c in sub.columns), None)
                    if ci_col is not None:
                        ci_vals = pd.to_numeric(sub[ci_col], errors="coerce").values
                        if "gCO2" in ci_col:
                            ci_vals = ci_vals / 1000.0
                        carbon_g = (total_wh/1000.0) * ci_vals * 1000.0
            except Exception:
                carbon_g = None
            if carbon_g is not None:
                y_offset = max(total_wh) * 0.015 if len(total_wh) else 0.0
                for xi, yh, cg in zip(x, total_wh, carbon_g):
                    if np.isfinite(cg):
                        ax.text(xi, yh + y_offset, f"{cg:.0f}", ha="center", va="bottom", fontsize=7, clip_on=False)
            ax2 = ax.twinx()
            ax2.plot(x, sub["Mvis/h"].values, marker="o", linewidth=1.2)
            ax2.set_ylabel("Mvis/h")
            ax2.set_yscale("log")
            ax2.tick_params(axis="y", labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Flagship: Energy (stacked) + Throughput", y=0.99, fontsize=11)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.935))
    fig.tight_layout(rect=[0,0,1,0.945])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# 11) New data parser: IDG GPU logs (throughput + gridder timings + wall clock)
RAW_IDG_GPU_LOGS = """
64 timesteps:
- 4794: <path-redacted>
|gridding:  40.80 Mvisibilities/s
gridder:    1.9049e-02 s,  4023.81 GFLOPS,  1062.14 GB/s,   159.49 Watt,    25.23 GFLOPS/W,     3.04 Joules
gridder:    2.1158e-02 s,  3753.06 GFLOPS,  1082.75 GB/s,   159.47 Watt,    23.54 GFLOPS/W,     3.37 Joules
gridder:    2.5347e-02 s,  3308.75 GFLOPS,  1081.27 GB/s,   161.46 Watt,    20.49 GFLOPS/W,     4.09 Joules
gridder:    3.4822e-02 s,  2700.42 GFLOPS,  1091.91 GB/s,   161.66 Watt,    16.70 GFLOPS/W,     5.63 Joules
gridder:    1.2702e-02 s,  2327.62 GFLOPS,  1107.67 GB/s,   161.68 Watt,    14.40 GFLOPS/W,     2.05 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:27.58
- 4795: <path-redacted>
|gridding:  301.05 Mvisibilities/s
gridder:    2.4272e-02 s, 12580.49 GFLOPS,   842.92 GB/s,   159.07 Watt,    79.09 GFLOPS/W,     3.86 Joules
gridder:    2.6411e-02 s, 11552.84 GFLOPS,   876.38 GB/s,   159.46 Watt,    72.45 GFLOPS/W,     4.21 Joules
gridder:    3.0616e-02 s,  9921.17 GFLOPS,   903.21 GB/s,   159.47 Watt,    62.21 GFLOPS/W,     4.88 Joules
gridder:    3.7996e-02 s,  7858.94 GFLOPS,  1009.02 GB/s,   308.52 Watt,    25.47 GFLOPS/W,    11.72 Joules
gridder:    1.4788e-02 s,  5524.62 GFLOPS,   957.98 GB/s,   398.04 Watt,    13.88 GFLOPS/W,     5.89 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:59.96
- 4796: <path-redacted>
|gridding:  959.96 Mvisibilities/s
|gridding:  889.91 Mvisibilities/s
|gridding:  351.00 Mvisibilities/s
gridder:    7.8270e-02 s, 24776.05 GFLOPS,   302.03 GB/s,   159.19 Watt,   155.64 GFLOPS/W,    12.46 Joules
gridder:    7.8316e-02 s, 23600.25 GFLOPS,   402.35 GB/s,   465.35 Watt,    50.72 GFLOPS/W,    36.44 Joules
gridder:    2.2215e-02 s, 20293.03 GFLOPS,   577.25 GB/s,   465.40 Watt,    43.60 GFLOPS/W,    10.34 Joules
gridder:    9.3100e-02 s, 24737.64 GFLOPS,   309.78 GB/s,   159.02 Watt,   155.56 GFLOPS/W,    14.80 Joules
gridder:    6.2405e-02 s, 22254.74 GFLOPS,   480.91 GB/s,   348.34 Watt,    63.89 GFLOPS/W,    21.74 Joules
gridder:    3.8015e-02 s, 21805.08 GFLOPS,   501.90 GB/s,   168.62 Watt,   129.31 GFLOPS/W,     6.41 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 5:35.11
- 4797: <path-redacted>
|gridding:  958.61 Mvisibilities/s
|gridding:  917.29 Mvisibilities/s
|gridding:  920.33 Mvisibilities/s
|gridding:  918.17 Mvisibilities/s
|gridding:  642.22 Mvisibilities/s
gridder:    1.1845e-01 s, 26182.02 GFLOPS,   221.18 GB/s,   166.47 Watt,   157.27 GFLOPS/W,    19.72 Joules
gridder:    3.9169e-02 s, 24234.09 GFLOPS,   330.34 GB/s,   345.16 Watt,    70.21 GFLOPS/W,    13.52 Joules
gridder:    1.2273e-01 s, 26040.48 GFLOPS,   226.20 GB/s,   166.31 Watt,   156.58 GFLOPS/W,    20.41 Joules
gridder:    2.4343e-02 s, 23975.73 GFLOPS,   364.30 GB/s,   479.64 Watt,    49.99 GFLOPS/W,    11.68 Joules
gridder:    1.2249e-01 s, 26093.62 GFLOPS,   221.47 GB/s,   192.72 Watt,   135.40 GFLOPS/W,    23.61 Joules
gridder:    2.4262e-02 s, 24046.19 GFLOPS,   364.01 GB/s,   306.47 Watt,    78.46 GFLOPS/W,     7.44 Joules
gridder:    1.2274e-01 s, 26059.43 GFLOPS,   222.28 GB/s,   239.21 Watt,   108.94 GFLOPS/W,    29.36 Joules
gridder:    2.4256e-02 s, 24016.44 GFLOPS,   363.76 GB/s,   472.18 Watt,    50.86 GFLOPS/W,    11.45 Joules
gridder:    7.6016e-02 s, 24956.74 GFLOPS,   289.92 GB/s,   163.26 Watt,   152.87 GFLOPS/W,    12.41 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 9:48.26
- 4857: <path-redacted>
|gridding:  948.85 Mvisibilities/s
|gridding:  950.30 Mvisibilities/s
|gridding:  950.93 Mvisibilities/s
|gridding:  950.25 Mvisibilities/s
|gridding:  951.69 Mvisibilities/s
|gridding:  949.14 Mvisibilities/s
|gridding:  950.59 Mvisibilities/s
|gridding:  947.80 Mvisibilities/s
gridder:    1.4953e-01 s, 26723.92 GFLOPS,   158.50 GB/s,   164.17 Watt,   162.78 GFLOPS/W,    24.55 Joules
gridder:    1.1677e-02 s, 25371.21 GFLOPS,   247.62 GB/s,   346.44 Watt,    73.23 GFLOPS/W,     4.05 Joules
gridder:    1.5025e-01 s, 26579.32 GFLOPS,   154.09 GB/s,   227.27 Watt,   116.95 GFLOPS/W,    34.15 Joules
gridder:    1.2718e-02 s, 23316.69 GFLOPS,   230.07 GB/s,   305.15 Watt,    76.41 GFLOPS/W,     3.88 Joules
gridder:    1.4925e-01 s, 26754.28 GFLOPS,   157.22 GB/s,   333.25 Watt,    80.28 GFLOPS/W,    49.74 Joules
gridder:    1.1645e-02 s, 25460.89 GFLOPS,   251.62 GB/s,   470.12 Watt,    54.16 GFLOPS/W,     5.47 Joules
gridder:    1.4919e-01 s, 26768.75 GFLOPS,   156.96 GB/s,   272.06 Watt,    98.39 GFLOPS/W,    40.59 Joules
gridder:    1.1621e-02 s, 25576.98 GFLOPS,   252.40 GB/s,   476.64 Watt,    53.66 GFLOPS/W,     5.54 Joules
gridder:    1.5031e-01 s, 26576.39 GFLOPS,   156.51 GB/s,   163.27 Watt,   162.78 GFLOPS/W,    24.54 Joules
gridder:    1.2652e-02 s, 23422.99 GFLOPS,   230.90 GB/s,   166.26 Watt,   140.88 GFLOPS/W,     2.10 Joules
gridder:    1.4911e-01 s, 26797.35 GFLOPS,   156.10 GB/s,   245.28 Watt,   109.25 GFLOPS/W,    36.57 Joules
gridder:    1.1654e-02 s, 25412.57 GFLOPS,   249.95 GB/s,   473.96 Watt,    53.62 GFLOPS/W,     5.52 Joules
gridder:    1.5023e-01 s, 26609.30 GFLOPS,   155.74 GB/s,   176.19 Watt,   151.02 GFLOPS/W,    26.47 Joules
gridder:    1.2712e-02 s, 23212.90 GFLOPS,   228.80 GB/s,   468.61 Watt,    49.54 GFLOPS/W,     5.96 Joules
gridder:    1.5023e-01 s, 26613.23 GFLOPS,   155.89 GB/s,   340.56 Watt,    78.15 GFLOPS/W,    51.16 Joules
gridder:    1.1654e-02 s, 25522.45 GFLOPS,   254.74 GB/s,   467.59 Watt,    54.58 GFLOPS/W,     5.45 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 17:58.63

256 timesteps:
- 4838: <path-redacted>
|gridding:  128.34 Mvisibilities/s
gridder:    2.2143e-02 s,  8575.93 GFLOPS,   889.40 GB/s,   159.56 Watt,    53.75 GFLOPS/W,     3.53 Joules
gridder:    2.3289e-02 s,  8190.08 GFLOPS,   931.93 GB/s,   163.49 Watt,    50.09 GFLOPS/W,     3.81 Joules
gridder:    2.6381e-02 s,  7281.36 GFLOPS,   936.23 GB/s,   195.08 Watt,    37.32 GFLOPS/W,     5.15 Joules
gridder:    2.9562e-02 s,  6512.84 GFLOPS,   985.41 GB/s,   195.07 Watt,    33.39 GFLOPS/W,     5.77 Joules
gridder:    3.9055e-02 s,  5031.92 GFLOPS,  1039.82 GB/s,   280.63 Watt,    17.93 GFLOPS/W,    10.96 Joules
gridder:    1.4755e-02 s,  3969.76 GFLOPS,  1075.21 GB/s,   342.64 Watt,    11.59 GFLOPS/W,     5.06 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 4:02.97
- 4839: <path-redacted>
|gridding:  721.89 Mvisibilities/s
|gridding:  93.83 Mvisibilities/s
gridder:    4.6442e-02 s, 21333.12 GFLOPS,   468.48 GB/s,   159.35 Watt,   133.88 GFLOPS/W,     7.40 Joules
gridder:    4.8535e-02 s, 20225.82 GFLOPS,   503.21 GB/s,   159.36 Watt,   126.92 GFLOPS/W,     7.73 Joules
gridder:    4.9597e-02 s, 19546.24 GFLOPS,   568.05 GB/s,   388.05 Watt,    50.37 GFLOPS/W,    19.25 Joules
gridder:    5.2849e-02 s, 17734.43 GFLOPS,   662.46 GB/s,   433.71 Watt,    40.89 GFLOPS/W,    22.92 Joules
gridder:    4.6567e-02 s, 14173.23 GFLOPS,   841.78 GB/s,   440.25 Watt,    32.19 GFLOPS/W,    20.50 Joules
gridder:    1.6887e-02 s, 12701.80 GFLOPS,   877.37 GB/s,   159.20 Watt,    79.78 GFLOPS/W,     2.69 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 6:06.90
- 4840: <path-redacted>
|gridding:  936.21 Mvisibilities/s
|gridding:  899.13 Mvisibilities/s
|gridding:  894.97 Mvisibilities/s
|gridding:  890.96 Mvisibilities/s
|gridding:  895.16 Mvisibilities/s
|gridding:  894.48 Mvisibilities/s
|gridding:  895.27 Mvisibilities/s
|gridding:  891.41 Mvisibilities/s
|gridding:  896.99 Mvisibilities/s
|gridding:  487.47 Mvisibilities/s
gridder:    1.4583e-01 s, 26741.97 GFLOPS,   175.20 GB/s,   163.02 Watt,   164.04 GFLOPS/W,    23.77 Joules
gridder:    1.1643e-02 s, 24857.49 GFLOPS,   278.93 GB/s,   166.07 Watt,   149.68 GFLOPS/W,     1.93 Joules
gridder:    1.2772e-01 s, 26591.97 GFLOPS,   179.52 GB/s,   203.00 Watt,   131.00 GFLOPS/W,    25.93 Joules
gridder:    1.0546e-02 s, 23923.31 GFLOPS,   268.79 GB/s,   338.92 Watt,    70.59 GFLOPS/W,     3.57 Joules
gridder:    1.2793e-01 s, 26538.35 GFLOPS,   177.39 GB/s,   275.45 Watt,    96.35 GFLOPS/W,    35.24 Joules
gridder:    9.5329e-03 s, 26495.86 GFLOPS,   306.50 GB/s,   463.44 Watt,    57.17 GFLOPS/W,     4.42 Joules
gridder:    1.2806e-01 s, 26516.57 GFLOPS,   179.99 GB/s,   187.74 Watt,   141.24 GFLOPS/W,    24.04 Joules
gridder:    1.0580e-02 s, 23913.57 GFLOPS,   276.66 GB/s,   472.22 Watt,    50.64 GFLOPS/W,     5.00 Joules
gridder:    1.2839e-01 s, 26454.25 GFLOPS,   180.50 GB/s,   195.82 Watt,   135.10 GFLOPS/W,    25.14 Joules
gridder:    1.0581e-02 s, 23879.82 GFLOPS,   272.84 GB/s,   479.93 Watt,    49.76 GFLOPS/W,     5.08 Joules
gridder:    1.2805e-01 s, 26528.69 GFLOPS,   180.08 GB/s,   177.17 Watt,   149.73 GFLOPS/W,    22.69 Joules
gridder:    9.5251e-03 s, 26472.97 GFLOPS,   304.04 GB/s,   471.62 Watt,    56.13 GFLOPS/W,     4.49 Joules
gridder:    1.2811e-01 s, 26523.68 GFLOPS,   180.12 GB/s,   183.43 Watt,   144.60 GFLOPS/W,    23.50 Joules
gridder:    9.4678e-03 s, 26624.80 GFLOPS,   307.04 GB/s,   481.07 Watt,    55.35 GFLOPS/W,     4.55 Joules
gridder:    1.2779e-01 s, 26599.71 GFLOPS,   181.70 GB/s,   168.43 Watt,   157.93 GFLOPS/W,    21.52 Joules
gridder:    1.0527e-02 s, 23856.29 GFLOPS,   271.08 GB/s,   479.43 Watt,    49.76 GFLOPS/W,     5.05 Joules
gridder:    1.2823e-01 s, 26511.41 GFLOPS,   180.96 GB/s,   290.06 Watt,    91.40 GFLOPS/W,    37.20 Joules
gridder:    1.0593e-02 s, 23834.23 GFLOPS,   272.99 GB/s,   398.26 Watt,    59.85 GFLOPS/W,     4.22 Joules
gridder:    4.9774e-02 s, 24634.70 GFLOPS,   298.45 GB/s,   169.99 Watt,   144.92 GFLOPS/W,     8.46 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 20:30.00
- 4841: <path-redacted>
|gridding:  939.03 Mvisibilities/s
|gridding:  912.93 Mvisibilities/s
|gridding:  920.05 Mvisibilities/s
|gridding:  915.01 Mvisibilities/s
|gridding:  915.76 Mvisibilities/s
|gridding:  919.27 Mvisibilities/s
|gridding:  916.02 Mvisibilities/s
|gridding:  916.33 Mvisibilities/s
|gridding:  916.19 Mvisibilities/s
|gridding:  914.13 Mvisibilities/s
|gridding:  915.00 Mvisibilities/s
|gridding:  916.96 Mvisibilities/s
|gridding:  919.01 Mvisibilities/s
|gridding:  918.47 Mvisibilities/s
|gridding:  920.07 Mvisibilities/s
|gridding:  917.79 Mvisibilities/s
|gridding:  915.14 Mvisibilities/s
|gridding:  916.76 Mvisibilities/s
|gridding:  380.82 Mvisibilities/s
gridder:    1.4788e-01 s, 27233.83 GFLOPS,   136.38 GB/s,   163.07 Watt,   167.01 GFLOPS/W,    24.11 Joules
gridder:    1.3846e-01 s, 27149.87 GFLOPS,   142.92 GB/s,   333.04 Watt,    81.52 GFLOPS/W,    46.11 Joules
gridder:    1.3844e-01 s, 27143.21 GFLOPS,   136.81 GB/s,   281.30 Watt,    96.49 GFLOPS/W,    38.94 Joules
gridder:    1.3843e-01 s, 27149.34 GFLOPS,   142.50 GB/s,   163.98 Watt,   165.57 GFLOPS/W,    22.70 Joules
gridder:    1.3828e-01 s, 27174.55 GFLOPS,   141.28 GB/s,   352.77 Watt,    77.03 GFLOPS/W,    48.78 Joules
gridder:    1.3847e-01 s, 27131.50 GFLOPS,   139.66 GB/s,   188.05 Watt,   144.28 GFLOPS/W,    26.04 Joules
gridder:    1.3843e-01 s, 27143.11 GFLOPS,   140.01 GB/s,   229.70 Watt,   118.17 GFLOPS/W,    31.80 Joules
gridder:    1.3934e-01 s, 26974.04 GFLOPS,   140.40 GB/s,   162.26 Watt,   166.24 GFLOPS/W,    22.61 Joules
gridder:    1.3831e-01 s, 27174.66 GFLOPS,   141.40 GB/s,   218.04 Watt,   124.63 GFLOPS/W,    30.16 Joules
gridder:    1.3854e-01 s, 27125.69 GFLOPS,   140.88 GB/s,   168.15 Watt,   161.31 GFLOPS/W,    23.30 Joules
gridder:    1.3840e-01 s, 27155.85 GFLOPS,   140.47 GB/s,   159.92 Watt,   169.81 GFLOPS/W,    22.13 Joules
gridder:    1.3843e-01 s, 27154.69 GFLOPS,   141.95 GB/s,   262.08 Watt,   103.61 GFLOPS/W,    36.28 Joules
gridder:    1.3835e-01 s, 27173.43 GFLOPS,   140.32 GB/s,   188.50 Watt,   144.16 GFLOPS/W,    26.08 Joules
gridder:    1.3948e-01 s, 26956.13 GFLOPS,   140.13 GB/s,   199.23 Watt,   135.30 GFLOPS/W,    27.79 Joules
gridder:    1.3940e-01 s, 26969.71 GFLOPS,   140.12 GB/s,   363.66 Watt,    74.16 GFLOPS/W,    50.70 Joules
gridder:    1.3835e-01 s, 27176.72 GFLOPS,   141.23 GB/s,   265.75 Watt,   102.26 GFLOPS/W,    36.77 Joules
gridder:    1.3857e-01 s, 27142.79 GFLOPS,   140.92 GB/s,   297.62 Watt,    91.20 GFLOPS/W,    41.24 Joules
gridder:    1.3855e-01 s, 27162.95 GFLOPS,   141.24 GB/s,   233.62 Watt,   116.27 GFLOPS/W,    32.37 Joules
gridder:    3.2795e-02 s, 24934.04 GFLOPS,   388.94 GB/s,   168.90 Watt,   147.63 GFLOPS/W,     5.54 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 36:36.39
- 4859: <path-redacted>
|gridding:  953.06 Mvisibilities/s
|gridding:  953.00 Mvisibilities/s
|gridding:  952.89 Mvisibilities/s
|gridding:  952.64 Mvisibilities/s
|gridding:  952.88 Mvisibilities/s
|gridding:  952.67 Mvisibilities/s
|gridding:  950.22 Mvisibilities/s
|gridding:  952.69 Mvisibilities/s
|gridding:  948.58 Mvisibilities/s
|gridding:  954.15 Mvisibilities/s
|gridding:  951.99 Mvisibilities/s
|gridding:  950.21 Mvisibilities/s
|gridding:  955.87 Mvisibilities/s
|gridding:  948.38 Mvisibilities/s
|gridding:  953.47 Mvisibilities/s
|gridding:  948.46 Mvisibilities/s
|gridding:  952.59 Mvisibilities/s
|gridding:  952.23 Mvisibilities/s
|gridding:  952.20 Mvisibilities/s
|gridding:  954.56 Mvisibilities/s
|gridding:  951.87 Mvisibilities/s
|gridding:  950.33 Mvisibilities/s
|gridding:  953.82 Mvisibilities/s
|gridding:  952.59 Mvisibilities/s
|gridding:  952.73 Mvisibilities/s
|gridding:  954.41 Mvisibilities/s
|gridding:  953.06 Mvisibilities/s
|gridding:  954.27 Mvisibilities/s
|gridding:  951.18 Mvisibilities/s
|gridding:  953.71 Mvisibilities/s
|gridding:  956.02 Mvisibilities/s
|gridding:  951.76 Mvisibilities/s
gridder:    1.5725e-01 s, 27224.84 GFLOPS,   110.46 GB/s,   217.14 Watt,   125.38 GFLOPS/W,    34.14 Joules
gridder:    1.5622e-01 s, 27399.20 GFLOPS,   109.78 GB/s,   273.83 Watt,   100.06 GFLOPS/W,    42.78 Joules
gridder:    1.5620e-01 s, 27402.00 GFLOPS,   111.73 GB/s,   164.08 Watt,   167.01 GFLOPS/W,    25.63 Joules
gridder:    1.5623e-01 s, 27394.19 GFLOPS,   111.36 GB/s,   304.05 Watt,    90.10 GFLOPS/W,    47.50 Joules
gridder:    1.5572e-01 s, 27478.91 GFLOPS,   109.58 GB/s,   208.58 Watt,   131.75 GFLOPS/W,    32.48 Joules
gridder:    1.5629e-01 s, 27375.59 GFLOPS,   109.95 GB/s,   296.96 Watt,    92.19 GFLOPS/W,    46.41 Joules
gridder:    1.5728e-01 s, 27204.51 GFLOPS,   109.84 GB/s,   228.74 Watt,   118.93 GFLOPS/W,    35.98 Joules
gridder:    1.5629e-01 s, 27373.81 GFLOPS,   109.70 GB/s,   288.16 Watt,    94.99 GFLOPS/W,    45.04 Joules
gridder:    1.5739e-01 s, 27181.34 GFLOPS,   111.01 GB/s,   163.07 Watt,   166.69 GFLOPS/W,    25.67 Joules
gridder:    1.5671e-01 s, 27297.57 GFLOPS,   109.61 GB/s,   260.39 Watt,   104.83 GFLOPS/W,    40.80 Joules
gridder:    1.5635e-01 s, 27359.02 GFLOPS,   109.43 GB/s,   228.00 Watt,   119.99 GFLOPS/W,    35.65 Joules
gridder:    1.5623e-01 s, 27384.89 GFLOPS,   110.62 GB/s,   220.03 Watt,   124.46 GFLOPS/W,    34.38 Joules
gridder:    1.5728e-01 s, 27205.03 GFLOPS,   108.64 GB/s,   180.75 Watt,   150.51 GFLOPS/W,    28.43 Joules
gridder:    1.5623e-01 s, 27393.65 GFLOPS,   111.06 GB/s,   163.25 Watt,   167.80 GFLOPS/W,    25.50 Joules
gridder:    1.5622e-01 s, 27393.23 GFLOPS,   111.12 GB/s,   283.23 Watt,    96.72 GFLOPS/W,    44.25 Joules
gridder:    1.5628e-01 s, 27381.15 GFLOPS,   109.86 GB/s,   205.09 Watt,   133.51 GFLOPS/W,    32.05 Joules
gridder:    1.5623e-01 s, 27388.97 GFLOPS,   109.96 GB/s,   317.07 Watt,    86.38 GFLOPS/W,    49.54 Joules
gridder:    1.5736e-01 s, 27192.17 GFLOPS,   109.52 GB/s,   210.33 Watt,   129.28 GFLOPS/W,    33.10 Joules
gridder:    1.5659e-01 s, 27326.84 GFLOPS,   109.33 GB/s,   218.33 Watt,   125.16 GFLOPS/W,    34.19 Joules
gridder:    1.5646e-01 s, 27356.15 GFLOPS,   111.35 GB/s,   243.12 Watt,   112.52 GFLOPS/W,    38.04 Joules
gridder:    1.5620e-01 s, 27401.70 GFLOPS,   110.08 GB/s,   356.68 Watt,    76.82 GFLOPS/W,    55.71 Joules
gridder:    1.5728e-01 s, 27214.93 GFLOPS,   109.16 GB/s,   237.05 Watt,   114.81 GFLOPS/W,    37.28 Joules
gridder:    1.5646e-01 s, 27360.30 GFLOPS,   110.01 GB/s,   362.63 Watt,    75.45 GFLOPS/W,    56.74 Joules
gridder:    1.5664e-01 s, 27329.29 GFLOPS,   109.97 GB/s,   163.90 Watt,   166.74 GFLOPS/W,    25.67 Joules
gridder:    1.5669e-01 s, 27320.92 GFLOPS,   109.85 GB/s,   290.69 Watt,    93.99 GFLOPS/W,    45.55 Joules
gridder:    1.5630e-01 s, 27387.40 GFLOPS,   110.34 GB/s,   200.24 Watt,   136.77 GFLOPS/W,    31.30 Joules
gridder:    1.5670e-01 s, 27318.96 GFLOPS,   109.97 GB/s,   180.60 Watt,   151.26 GFLOPS/W,    28.30 Joules
gridder:    1.5652e-01 s, 27351.89 GFLOPS,   109.92 GB/s,   172.40 Watt,   158.65 GFLOPS/W,    26.98 Joules
gridder:    1.5759e-01 s, 27170.79 GFLOPS,   109.50 GB/s,   320.27 Watt,    84.84 GFLOPS/W,    50.47 Joules
gridder:    1.5728e-01 s, 27233.51 GFLOPS,   109.69 GB/s,   161.96 Watt,   168.15 GFLOPS/W,    25.47 Joules
gridder:    1.5670e-01 s, 27341.68 GFLOPS,   110.01 GB/s,   282.71 Watt,    96.71 GFLOPS/W,    44.30 Joules
gridder:    1.5660e-01 s, 27371.20 GFLOPS,   110.43 GB/s,   186.96 Watt,   146.40 GFLOPS/W,    29.28 Joules
        Elapsed (wall clock) time (h:mm:ss or m:ss): 1:09:34
"""


def _parse_hms_to_seconds(time_str):
    parts = time_str.strip().split(":")
    vals = [float(p) for p in parts]
    if len(vals) == 3:
        h, m, s = vals
        return h * 3600.0 + m * 60.0 + s
    if len(vals) == 2:
        m, s = vals
        return m * 60.0 + s
    return vals[0]


def _parse_idg_gpu_blob(raw_text):
    run_re = re.compile(r"^-\s*(\d+):\s+.*_t0-(\d+)_c0-(\d+)_.*\.log")
    gridding_re = re.compile(r"\|gridding:\s*([0-9.]+)\s*Mvisibilities/s")
    gridder_re = re.compile(
        r"gridder:\s*([0-9.eE+-]+)\s*s,\s*"
        r"([0-9.eE+-]+)\s*GFLOPS,\s*"
        r"([0-9.eE+-]+)\s*GB/s,\s*"
        r"([0-9.eE+-]+)\s*Watt,\s*"
        r"([0-9.eE+-]+)\s*GFLOPS/W,\s*"
        r"([0-9.eE+-]+)\s*Joules"
    )
    elapsed_re = re.compile(r"Elapsed .*?:\s*([0-9:.]+)\s*$")

    runs = {}
    current = None
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        m_run = run_re.match(line)
        if m_run:
            run_id = int(m_run.group(1))
            runs[run_id] = {
                "run_id": run_id,
                "timesteps": int(m_run.group(2)),
                "channels": int(m_run.group(3)),
                "throughputs": [],
                "gridder_times_s": [],
                "gridder_gflops": [],
                "gridder_gbs": [],
                "gridder_watt": [],
                "gridder_gflops_w": [],
                "gridder_joules": [],
                "wall_s": np.nan,
            }
            current = run_id
            continue

        if current is None:
            continue

        m_thr = gridding_re.search(line)
        if m_thr:
            runs[current]["throughputs"].append(float(m_thr.group(1)))
            continue

        m_grid = gridder_re.search(line)
        if m_grid:
            runs[current]["gridder_times_s"].append(float(m_grid.group(1)))
            runs[current]["gridder_gflops"].append(float(m_grid.group(2)))
            runs[current]["gridder_gbs"].append(float(m_grid.group(3)))
            runs[current]["gridder_watt"].append(float(m_grid.group(4)))
            runs[current]["gridder_gflops_w"].append(float(m_grid.group(5)))
            runs[current]["gridder_joules"].append(float(m_grid.group(6)))
            continue

        m_elapsed = elapsed_re.search(line)
        if m_elapsed:
            runs[current]["wall_s"] = _parse_hms_to_seconds(m_elapsed.group(1))

    rows = []
    for rec in runs.values():
        thr = np.array(rec["throughputs"], dtype=float)
        grid = np.array(rec["gridder_times_s"], dtype=float)
        gflops = np.array(rec["gridder_gflops"], dtype=float)
        gbs = np.array(rec["gridder_gbs"], dtype=float)
        watt = np.array(rec["gridder_watt"], dtype=float)
        gflops_w = np.array(rec["gridder_gflops_w"], dtype=float)
        joules = np.array(rec["gridder_joules"], dtype=float)
        wall_s = float(rec["wall_s"])
        grid_sum_s = float(grid.sum()) if grid.size else np.nan
        rows.append({
            "run_id": rec["run_id"],
            "timesteps": rec["timesteps"],
            "channels": rec["channels"],
            "n_gridding_samples": int(thr.size),
            "n_gridder_calls": int(grid.size),
            "throughput_median": float(np.median(thr)) if thr.size else np.nan,
            "throughput_min": float(np.min(thr)) if thr.size else np.nan,
            "throughput_max": float(np.max(thr)) if thr.size else np.nan,
            "gridder_total_s": grid_sum_s,
            "gridder_mean_s": float(np.mean(grid)) if grid.size else np.nan,
            "gridder_std_s": float(np.std(grid, ddof=0)) if grid.size else np.nan,
            "gridder_var_s2": float(np.var(grid, ddof=0)) if grid.size else np.nan,
            "gridder_gflops_mean": float(np.mean(gflops)) if gflops.size else np.nan,
            "gridder_gflops_std": float(np.std(gflops, ddof=0)) if gflops.size else np.nan,
            "gridder_gbs_mean": float(np.mean(gbs)) if gbs.size else np.nan,
            "gridder_gbs_std": float(np.std(gbs, ddof=0)) if gbs.size else np.nan,
            "gridder_watt_mean": float(np.mean(watt)) if watt.size else np.nan,
            "gridder_watt_std": float(np.std(watt, ddof=0)) if watt.size else np.nan,
            "gridder_gflops_w_mean": float(np.mean(gflops_w)) if gflops_w.size else np.nan,
            "gridder_gflops_w_std": float(np.std(gflops_w, ddof=0)) if gflops_w.size else np.nan,
            "gridder_joules_total": float(np.sum(joules)) if joules.size else np.nan,
            "gridder_joules_mean": float(np.mean(joules)) if joules.size else np.nan,
            "gridder_joules_std": float(np.std(joules, ddof=0)) if joules.size else np.nan,
            "wall_s": wall_s,
            "nongridder_s": wall_s - grid_sum_s if np.isfinite(wall_s) and np.isfinite(grid_sum_s) else np.nan,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["timesteps", "channels"]).reset_index(drop=True)
    return out


idg_gpu_df = _parse_idg_gpu_blob(RAW_IDG_GPU_LOGS)

if not idg_gpu_df.empty:
    # Attach workload visibility counts so we can compute end-to-end throughput
    # from total work / wall time for each parsed run.
    idg_gpu_df["image_size"] = 8192
    nvis_lookup_path = DATA_DIR / "benchmarks.csv"
    if nvis_lookup_path.exists():
        nvis_raw = pd.read_csv(
            nvis_lookup_path,
            header=None,
            usecols=[0, 1, 2, 6],
            names=["image_size", "timesteps", "channels", "n_vis"],
        )
        nvis_lookup = (
            nvis_raw.groupby(["image_size", "timesteps", "channels"], as_index=False)["n_vis"]
            .median()
            .reset_index(drop=True)
        )
        idg_gpu_df = idg_gpu_df.merge(
            nvis_lookup,
            on=["image_size", "timesteps", "channels"],
            how="left",
        )
    else:
        idg_gpu_df["n_vis"] = np.nan

    idg_gpu_df["e2e_mvis_s"] = (idg_gpu_df["n_vis"] / 1e6) / idg_gpu_df["wall_s"]
    idg_gpu_df["gridder_fraction_pct"] = 100.0 * idg_gpu_df["gridder_total_s"] / idg_gpu_df["wall_s"]

    channels = sorted(idg_gpu_df["channels"].unique())
    timesteps = sorted(idg_gpu_df["timesteps"].unique())
    colors = {t: f"C{idx}" for idx, t in enumerate(timesteps)}

    x_cat = np.arange(len(channels), dtype=float)
    bw = 0.38
    for t in timesteps:
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13.5, 4.6))

        sub = idg_gpu_df[idg_gpu_df["timesteps"] == t].sort_values("channels")

        # Left: one-timestep throughput view.
        ax_l2 = ax_l.twinx()
        xvals = sub["channels"].values
        y = sub["throughput_median"].values
        y_min = sub["throughput_min"].values
        y_max = sub["throughput_max"].values

        ax_l.plot(xvals, y, marker="o", linewidth=1.7, color=colors[t], label="gridding median")
        ax_l.fill_between(xvals, y_min, y_max, color=colors[t], alpha=0.14, label="gridding min-max")
        ax_l2.plot(
            xvals,
            sub["e2e_mvis_s"].values,
            marker="s",
            linestyle="--",
            linewidth=1.2,
            color="black",
            label="end-to-end",
        )

        ax_l.axhspan(900.0, 960.0, color="gray", alpha=0.15, label="plateau band")
        ax_l.set_xscale("log", base=2)
        ax_l.set_xticks(channels)
        ax_l.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_l.set_xlabel("Channels per call")
        ax_l.set_ylabel("Gridding throughput (Mvis/s)")
        ax_l2.set_ylabel("End-to-end throughput (Mvis/s)")
        ax_l.set_title(f"t={t}: Per-call throughput saturates before end-to-end throughput")
        ax_l.grid(True, alpha=0.25)

        hl1, ll1 = ax_l.get_legend_handles_labels()
        hl2, ll2 = ax_l2.get_legend_handles_labels()
        uniq_l = {}
        for h, l in zip(hl1 + hl2, ll1 + ll2):
            if l not in uniq_l:
                uniq_l[l] = h
        ax_l.legend(list(uniq_l.values()), list(uniq_l.keys()), frameon=False, fontsize=8, loc="upper left")

        # Right: bar relation between gridder and wall time (no non-gridder bar, no wall line).
        ax_r2 = ax_r.twinx()
        sub_idx = sub.set_index("channels").reindex(channels)
        grid_s = sub_idx["gridder_total_s"].values
        wall_s = sub_idx["wall_s"].values

        ax_r.bar(x_cat - bw / 2, grid_s, width=bw, color=colors[t], alpha=0.80, label="gridder s")
        ax_r.bar(x_cat + bw / 2, wall_s, width=bw, color="gray", alpha=0.45, label="wall s")
        ax_r2.plot(x_cat, sub_idx["gridder_fraction_pct"].values, marker="o", linewidth=1.3, color="black", label="gridder share %")

        ax_r.set_xticks(x_cat)
        ax_r.set_xticklabels([str(c) for c in channels])
        ax_r.set_xlabel("Channels per call")
        ax_r.set_ylabel("Time (s)")
        ax_r.set_yscale("log")
        ax_r2.set_ylabel("Gridder fraction (%)")
        ax_r.set_title(f"t={t}: Wall time outgrows gridder time (log scale)")
        ax_r.grid(True, axis="y", alpha=0.25)

        hr1, lr1 = ax_r.get_legend_handles_labels()
        hr2, lr2 = ax_r2.get_legend_handles_labels()
        uniq_r = {}
        for h, l in zip(hr1 + hr2, lr1 + lr2):
            if l not in uniq_r:
                uniq_r[l] = h
        ax_r.legend(list(uniq_r.values()), list(uniq_r.keys()), frameon=False, fontsize=8, loc="upper left")

        # Force plain formatting on linear y-axes.
        for a in (ax_l, ax_l2, ax_r, ax_r2):
            if a.get_yscale() == "linear":
                a.ticklabel_format(style="plain", axis="y", useOffset=False)

        fig.suptitle(f"Kernel Throughput Saturation and End-to-End Time Growth (IDG GPU, t={t})", y=1.02)
        fig.tight_layout()
        fig.savefig(str(Path(out_dir) / f"plot11_idg_gpu_saturation_overhead_from_logs_t{int(t)}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(out_dir) / f"plot11_idg_gpu_saturation_overhead_from_logs_t{int(t)}.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

    # 12) Sanity-check diagnostics for parsing and timing interpretation.

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # (a) Wall vs summed gridder time by run, with identity line.
    finite = idg_gpu_df[np.isfinite(idg_gpu_df["wall_s"]) & np.isfinite(idg_gpu_df["gridder_total_s"])].copy()
    if not finite.empty:
        for t in timesteps:
            sub = finite[finite["timesteps"] == t]
            ax1.scatter(sub["wall_s"], sub["gridder_total_s"], s=48, label=f"t={t}")
            for _, r in sub.iterrows():
                ax1.annotate(f"c{int(r['channels'])}", (r["wall_s"], r["gridder_total_s"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
        lim = max(float(finite["wall_s"].max()), float(finite["gridder_total_s"].max()))
        ax1.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0, color="gray", label="y=x")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Total wall time (s)")
    ax1.set_ylabel("Summed gridder time (s)")
    ax1.set_title("Summed gridder time tracks total wall time")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)

    # (b) Fraction of runtime spent in gridder.
    for t in timesteps:
        sub = idg_gpu_df[idg_gpu_df["timesteps"] == t].sort_values("channels")
        ax2.plot(sub["channels"], sub["gridder_fraction_pct"], marker="o", linewidth=1.4, label=f"t={t}")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(channels)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_xlabel("Channels")
    ax2.set_ylabel("100 × sum(gridder)/wall (%)")
    ax2.set_title("Gridder runtime share across channel scaling")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False, fontsize=8)

    # (c) Gridder call counts from both line types.
    for t in timesteps:
        sub = idg_gpu_df[idg_gpu_df["timesteps"] == t].sort_values("channels")
        ax3.plot(sub["channels"], sub["n_gridder_calls"], marker="o", linewidth=1.4, label=f"gridder lines t={t}")
        ax3.plot(sub["channels"], sub["n_gridding_samples"], marker="s", linewidth=1.1, linestyle="--", label=f"gridding lines t={t}")
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(channels)
    ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax3.set_xlabel("Channels")
    ax3.set_ylabel("Count")
    ax3.set_title("Parsed call counts from gridding and gridder lines")
    ax3.grid(True, alpha=0.25)
    ax3.legend(frameon=False, fontsize=7, ncol=2)

    # (d) Mean gridder time per call with +-1 sigma spread.
    for t in timesteps:
        sub = idg_gpu_df[idg_gpu_df["timesteps"] == t].sort_values("channels")
        y = sub["gridder_mean_s"].values
        yerr = sub["gridder_std_s"].values
        ax4.errorbar(
            sub["channels"],
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.4,
            capsize=3,
            label=f"t={t}",
        )
    ax4.set_xscale("log", base=2)
    ax4.set_xticks(channels)
    ax4.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax4.set_xlabel("Channels")
    ax4.set_ylabel("Mean gridder time per call (s)")
    ax4.set_title("Per-call gridder duration and variability")
    ax4.grid(True, alpha=0.25)
    ax4.legend(frameon=False, fontsize=8)

    fig.suptitle("Consistency Checks for Parsed IDG GPU Runtime Metrics", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(Path(out_dir) / "plot12_idg_gpu_sanity_checks_from_logs.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "plot12_idg_gpu_sanity_checks_from_logs.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 13) Focus plot: ratio of summed gridder time to wall time across channels and timesteps.
    ratio_pivot = idg_gpu_df.pivot(index="timesteps", columns="channels", values="gridder_fraction_pct")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # (a) Ratio vs channels for each timestep.
    for t in timesteps:
        sub = idg_gpu_df[idg_gpu_df["timesteps"] == t].sort_values("channels")
        ax_a.plot(sub["channels"], sub["gridder_fraction_pct"], marker="o", linewidth=1.6, label=f"t={t}")
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(channels)
    ax_a.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_a.set_xlabel("Channels")
    ax_a.set_ylabel("Gridder time / Wall time (%)")
    ax_a.set_title("Gridder fraction versus channels")
    ax_a.grid(True, alpha=0.25)
    ax_a.legend(frameon=False)

    # (b) Heatmap to emphasize joint dependency on timesteps and channels.
    if not ratio_pivot.empty:
        hm = ax_b.imshow(ratio_pivot.values, cmap="viridis", aspect="auto")
        ax_b.set_xticks(np.arange(len(ratio_pivot.columns)))
        ax_b.set_xticklabels([str(int(c)) for c in ratio_pivot.columns])
        ax_b.set_yticks(np.arange(len(ratio_pivot.index)))
        ax_b.set_yticklabels([str(int(t)) for t in ratio_pivot.index])
        ax_b.set_xlabel("Channels")
        ax_b.set_ylabel("Timesteps")
        ax_b.set_title("Gridder fraction heatmap by timesteps and channels")

        for i in range(ratio_pivot.shape[0]):
            for j in range(ratio_pivot.shape[1]):
                v = ratio_pivot.values[i, j]
                if np.isfinite(v):
                    ax_b.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=8)

        cbar = fig.colorbar(hm, ax=ax_b, fraction=0.046, pad=0.04)
        cbar.set_label("Gridder/Wall (%)")

    fig.suptitle("Gridder Contribution to End-to-End Runtime Across Workload Scale", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(Path(out_dir) / "plot13_idg_gpu_gridder_wall_ratio_focus.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "plot13_idg_gpu_gridder_wall_ratio_focus.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 16) Rich RAW-IDG explainer: tie plateau to kernel efficiency and non-gridder overhead.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.0, 4.9))

    # (a) Throughput plateau with min/max band.
    for t in timesteps:
        sub = idg_gpu_df[idg_gpu_df["timesteps"] == t].sort_values("channels")
        xvals = sub["channels"].values
        med = sub["throughput_median"].values
        tmin = sub["throughput_min"].values
        tmax = sub["throughput_max"].values
        ax1.plot(xvals, med, marker="o", linewidth=1.6, label=f"t={t} median")
        ax1.fill_between(xvals, tmin, tmax, alpha=0.15)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(channels)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("Channels per call")
    ax1.set_ylabel("Gridding throughput (Mvis/s)")
    ax1.set_title("Kernel throughput plateau with run-to-run spread")
    ax1.grid(True, alpha=0.25)
    ax1.legend(frameon=False, fontsize=8)

    # (b) End-to-end decomposition by channels.
    width = 0.36 if len(timesteps) == 2 else min(0.78 / max(1, len(timesteps)), 0.36)
    x = np.arange(len(channels), dtype=float)
    for idx, t in enumerate(timesteps):
        sub = idg_gpu_df[idg_gpu_df["timesteps"] == t].set_index("channels").reindex(channels)
        offs = (idx - (len(timesteps) - 1) / 2.0) * width
        xpos = x + offs
        ax2.bar(xpos, sub["gridder_total_s"].values, width=width, alpha=0.8, label=f"gridder s (t={t})")
        ax2.bar(
            xpos,
            sub["nongridder_s"].values,
            width=width,
            bottom=sub["gridder_total_s"].values,
            alpha=0.4,
            label=f"non-gridder s (t={t})",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in channels])
    ax2.set_xlabel("Channels per call")
    ax2.set_ylabel("Wall time decomposition (s)")
    ax2.set_title("Runtime decomposition across channel scaling")
    ax2.grid(True, axis="y", alpha=0.25)
    h2, l2 = ax2.get_legend_handles_labels()
    # Deduplicate legend labels.
    uniq = {}
    for h, l in zip(h2, l2):
        if l not in uniq:
            uniq[l] = h
    ax2.legend(list(uniq.values()), list(uniq.keys()), frameon=False, fontsize=7)

    # (c) Kernel efficiency and power trend from gridder lines.
    ax3r = ax3.twinx()
    for t in timesteps:
        sub = idg_gpu_df[idg_gpu_df["timesteps"] == t].sort_values("channels")
        xvals = sub["channels"].values
        eff = sub["gridder_gflops_w_mean"].values
        eff_std = sub["gridder_gflops_w_std"].values
        pwr = sub["gridder_watt_mean"].values
        ax3.errorbar(xvals, eff, yerr=eff_std, marker="o", linewidth=1.4, capsize=3, label=f"GFLOPS/W t={t}")
        ax3r.plot(xvals, pwr, marker="s", linewidth=1.1, linestyle="--", label=f"Watt t={t}")

    ax3.set_xscale("log", base=2)
    ax3.set_xticks(channels)
    ax3.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax3.set_xlabel("Channels per call")
    ax3.set_ylabel("Kernel efficiency (GFLOPS/W)")
    ax3r.set_ylabel("Kernel power (W)")
    ax3.set_title("Efficiency-power trade-off reveals diminishing returns")
    ax3.grid(True, alpha=0.25)
    h3a, l3a = ax3.get_legend_handles_labels()
    h3b, l3b = ax3r.get_legend_handles_labels()
    ax3.legend(h3a + h3b, l3a + l3b, frameon=False, fontsize=7, loc="upper right")

    fig.suptitle("IDG GPU Mechanistic View: Kernel Plateau, Runtime Decomposition, and Efficiency", y=1.02)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / "plot16_idg_gpu_raw_log_explainer.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "plot16_idg_gpu_raw_log_explainer.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    idg_gpu_df.to_csv(str(DATA_DIR / "idg_gpu_parsed_log_summary.csv"), index=False)


# 14) CPU-only vs GPU wall-time comparison from external benchmark folder.
# def _parse_monit_wall_seconds(monit_path):
#     ts_re = re.compile(r"^=====\s*([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?)\s*=====")
#     first_ts = None
#     last_ts = None

#     try:
#         with open(monit_path, "r", encoding="utf-8", errors="ignore") as f:
#             for line in f:
#                 m = ts_re.match(line.strip())
#                 if not m:
#                     continue
#                 dt = datetime.fromisoformat(m.group(1))
#                 if first_ts is None:
#                     first_ts = dt
#                 last_ts = dt
#     except OSError:
#         return np.nan

#     if first_ts is None or last_ts is None:
#         return np.nan
#     return (last_ts - first_ts).total_seconds()


# def _parse_external_cpu_gpu_walltimes(bench_root):
#     bench_path = Path(bench_root)
#     if not bench_path.exists():
#         return pd.DataFrame()

#     filename_re = re.compile(
#         r"slurm-(\d+)_wsc_dirty_t0-(\d+)_c0-(\d+)_([0-9]+)pix_.*_([0-9]+)cores\.monit$"
#     )

#     rows = []
#     for mode_dir, mode_name in [("profiling_cpu", "CPU"), ("profiling_gpu2", "GPU")]:
#         mode_path = bench_path / mode_dir
#         if not mode_path.exists():
#             continue

#         for monit_path in sorted(mode_path.glob("*.monit")):
#             m = filename_re.match(monit_path.name)
#             if not m:
#                 continue

#             rows.append(
#                 {
#                     "run_id": int(m.group(1)),
#                     "timesteps": int(m.group(2)) + 1,
#                     "channels": int(m.group(3)) + 1,
#                     "image_size": int(m.group(4)),
#                     "threads": int(m.group(5)),
#                     "mode": mode_name,
#                     "wall_s": _parse_monit_wall_seconds(monit_path),
#                     "monit_file": str(monit_path),
#                 }
#             )

#     out = pd.DataFrame(rows)
#     if out.empty:
#         return out

#     return out.sort_values(["image_size", "timesteps", "channels", "threads", "mode"]).reset_index(drop=True)


def _load_nvis_lookup():
    bench_csv = DATA_DIR / "benchmarks.csv"
    if not bench_csv.exists():
        return pd.DataFrame(columns=["image_size", "timesteps", "channels", "n_vis"])

    # benchmarks.csv has no header; column 6 stores n_vis.
    raw = pd.read_csv(
        bench_csv,
        header=None,
        usecols=[0, 1, 2, 6],
        names=["image_size", "timesteps", "channels", "n_vis"],
    )
    return (
        raw.groupby(["image_size", "timesteps", "channels"], as_index=False)["n_vis"]
        .median()
        .reset_index(drop=True)
    )


idg_cpu_gpu_wall_df = pd.DataFrame()

if not idg_cpu_gpu_wall_df.empty:
    valid = idg_cpu_gpu_wall_df[np.isfinite(idg_cpu_gpu_wall_df["wall_s"])].copy()
    if not valid.empty:
        nvis_lookup = _load_nvis_lookup()
        valid = valid.merge(nvis_lookup, on=["image_size", "timesteps", "channels"], how="left")
        valid["mvis_s_eff"] = (valid["n_vis"] / 1e6) / valid["wall_s"]

        # Focus on workload(s) common to CPU and GPU modes.
        workload_cols = ["image_size", "timesteps", "channels"]
        mode_counts = (
            valid.groupby(workload_cols)["mode"]
            .nunique()
            .reset_index(name="n_modes")
        )
        common_workloads = mode_counts[mode_counts["n_modes"] >= 2][workload_cols]

        if not common_workloads.empty:
            merged = valid.merge(common_workloads, on=workload_cols, how="inner")

            # Use the largest workload by default when multiple exist.
            merged = merged.sort_values(workload_cols)
            workload_key = tuple(merged[workload_cols].drop_duplicates().iloc[-1].tolist())
            sub = merged[
                (merged["image_size"] == workload_key[0])
                & (merged["timesteps"] == workload_key[1])
                & (merged["channels"] == workload_key[2])
            ].copy()

            pivot = sub.pivot_table(index="threads", columns="mode", values="wall_s", aggfunc="mean")
            pivot = pivot.sort_index()

            if {"CPU", "GPU"}.issubset(set(pivot.columns)):
                threads = pivot.index.values
                cpu_wall = pivot["CPU"].values
                gpu_wall = pivot["GPU"].values
                speedup = cpu_wall / gpu_wall

                thr_pivot = sub.pivot_table(index="threads", columns="mode", values="mvis_s_eff", aggfunc="mean")
                thr_pivot = thr_pivot.sort_index()
                cpu_thr = thr_pivot["CPU"].values if "CPU" in thr_pivot.columns else np.full_like(cpu_wall, np.nan)
                gpu_thr = thr_pivot["GPU"].values if "GPU" in thr_pivot.columns else np.full_like(gpu_wall, np.nan)
                thr_speedup = gpu_thr / cpu_thr

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

                x = np.arange(len(threads), dtype=float)
                bw = 0.38
                ax1.bar(x - bw / 2, cpu_wall, width=bw, label="CPU-only", alpha=0.8)
                ax1.bar(x + bw / 2, gpu_wall, width=bw, label="GPU", alpha=0.8)
                ax1.set_xticks(x)
                ax1.set_xticklabels([str(int(t)) for t in threads])
                ax1.set_xlabel("Threads")
                ax1.set_ylabel("Wall time (s)")
                ax1.set_title("Wall-time comparison by thread count")
                ax1.grid(True, axis="y", alpha=0.25)
                ax1.legend(frameon=False)

                ax2.bar(x - bw / 2, cpu_thr, width=bw, label="CPU-only", alpha=0.8)
                ax2.bar(x + bw / 2, gpu_thr, width=bw, label="GPU", alpha=0.8)
                ax2.set_xticks(x)
                ax2.set_xticklabels([str(int(t)) for t in threads])
                ax2.set_xlabel("Threads")
                ax2.set_ylabel("Effective throughput (Mvis/s)")
                ax2.set_title("Effective throughput comparison by thread count")
                ax2.grid(True, axis="y", alpha=0.25)

                ax2r = ax2.twinx()
                ax2r.plot(x, thr_speedup, color="black", marker="o", linewidth=1.4, label="GPU/CPU throughput")
                for xi, s in zip(x, thr_speedup):
                    if np.isfinite(s):
                        ax2r.text(xi, s * 1.01, f"{s:.2f}x", ha="center", va="bottom", fontsize=8)
                ax2r.set_ylabel("GPU/CPU throughput ratio")

                h1, l1 = ax2.get_legend_handles_labels()
                h2, l2 = ax2r.get_legend_handles_labels()
                ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

                fig.suptitle(
                    f"CPU vs GPU End-to-End Performance for Matched Workload\n"
                    f"image={int(workload_key[0])}, timesteps={int(workload_key[1])}, channels={int(workload_key[2])}",
                    y=1.02,
                )
                fig.tight_layout()
                fig.savefig(str(Path(out_dir) / "plot14_cpu_vs_gpu_walltime_from_bench.png"), dpi=300, bbox_inches="tight")
                fig.savefig(str(Path(out_dir) / "plot14_cpu_vs_gpu_walltime_from_bench.pdf"), format="pdf", bbox_inches="tight")
                plt.close(fig)

        valid.to_csv(str(DATA_DIR / "idg_cpu_gpu_walltime_from_bench.csv"), index=False)


# 17) Kernel time breakdown from pasc25_16c raw logs across image size, timesteps, and channels.
def _parse_kernel_breakdown_from_logs(log_root):
    root = Path(log_root)
    if not root.exists():
        return pd.DataFrame()

    run_re = re.compile(r"slurm-(\d+)_wsc_dirty_t0-(\d+)_c0-(\d+)_([0-9]+)p_.*\.log$")
    timed_re = re.compile(r"^\s*(gridder|sub-fft|wtiling):\s*([0-9.eE+-]+)\s*s,")
    gridding_re = re.compile(r"\|gridding:\s*([0-9.]+)\s*Mvisibilities/s")
    elapsed_re = re.compile(r"Elapsed .*?:\s*([0-9:.]+)\s*$")

    rows = []
    for p in sorted(root.glob("slurm-*_wsc_dirty_t0-*_c0-*_*p_*.log")):
        m = run_re.match(p.name)
        if not m:
            continue

        rec = {
            "run_id": int(m.group(1)),
            "timesteps": int(m.group(2)),
            "channels": int(m.group(3)),
            "image_size": int(m.group(4)),
            "gridder_s": 0.0,
            "sub_fft_s": 0.0,
            "wtiling_s": 0.0,
            "wall_s": np.nan,
            "n_gridder": 0,
            "n_sub_fft": 0,
            "n_wtiling": 0,
            "gridding_samples": [],
            "log_file": str(p),
        }

        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m_time = timed_re.match(line)
                    if m_time:
                        label = m_time.group(1)
                        secs = float(m_time.group(2))
                        if label == "gridder":
                            rec["gridder_s"] += secs
                            rec["n_gridder"] += 1
                        elif label == "sub-fft":
                            rec["sub_fft_s"] += secs
                            rec["n_sub_fft"] += 1
                        elif label == "wtiling":
                            rec["wtiling_s"] += secs
                            rec["n_wtiling"] += 1
                        continue

                    m_thr = gridding_re.search(line)
                    if m_thr:
                        rec["gridding_samples"].append(float(m_thr.group(1)))
                        continue

                    m_elapsed = elapsed_re.search(line)
                    if m_elapsed:
                        rec["wall_s"] = _parse_hms_to_seconds(m_elapsed.group(1))
        except OSError:
            continue

        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["throughput_mean_mvis_s"] = out["gridding_samples"].apply(lambda v: float(np.mean(v)) if len(v) else np.nan)
    out["throughput_std_mvis_s"] = out["gridding_samples"].apply(lambda v: float(np.std(v, ddof=0)) if len(v) else np.nan)
    out["n_gridding_samples"] = out["gridding_samples"].apply(len)
    out = out.drop(columns=["gridding_samples"])

    out["known_kernel_s"] = out["gridder_s"] + out["sub_fft_s"] + out["wtiling_s"]
    out["other_s"] = out["wall_s"] - out["known_kernel_s"]
    return out.sort_values(["image_size", "timesteps", "channels", "run_id"]).reset_index(drop=True)


def _parse_monit_gpu_resource_metrics(monit_root):
    root = Path(monit_root)
    if not root.exists():
        return pd.DataFrame()

    fname_re = re.compile(r"slurm-(\d+)\.monit$")
    rows = []
    for p in sorted(root.glob("slurm-*.monit")):
        m = fname_re.match(p.name)
        if not m:
            continue

        run_id = int(m.group(1))
        sm_means = []
        mem_util_means = []
        mem_used_sum_mb = []
        power_sum_w = []
        cpu_power_sum_w = []
        active_sm_means = []
        active_mem_util_means = []
        active_mem_used_sum_mb = []
        active_power_sum_w = []

        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("CPU_POWER:"):
                        payload = line.split("CPU_POWER:", 1)[1].strip()
                        try:
                            obj = json.loads(payload)
                        except json.JSONDecodeError:
                            obj = None
                        if isinstance(obj, dict):
                            p0 = float(obj.get("package_0_W", np.nan))
                            p1 = float(obj.get("package_1_W", np.nan))
                            cpu_power_sum_w.append(float(np.nansum([p0, p1])))
                        continue

                    if not line.startswith("GPU_STATS:"):
                        continue
                    payload = line.split("GPU_STATS:", 1)[1].strip()
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    gpus = obj.get("data", [])
                    if not gpus:
                        continue

                    sm_vals = [float(g.get("sm_util", np.nan)) for g in gpus]
                    mem_util_vals = [float(g.get("mem_util", np.nan)) for g in gpus]
                    mem_used_vals = [float(g.get("mem_used_MB", np.nan)) for g in gpus]
                    power_vals = [float(g.get("power_W", np.nan)) for g in gpus]

                    sm_means.append(float(np.nanmean(sm_vals)))
                    mem_util_means.append(float(np.nanmean(mem_util_vals)))
                    mem_used_sum_mb.append(float(np.nansum(mem_used_vals)))
                    power_sum_w.append(float(np.nansum(power_vals)))

                    # Active-window samples: any visible GPU work in SM or memory util.
                    if (np.nanmax(sm_vals) > 0.0) or (np.nanmax(mem_util_vals) > 0.0):
                        active_sm_means.append(float(np.nanmean(sm_vals)))
                        active_mem_util_means.append(float(np.nanmean(mem_util_vals)))
                        active_mem_used_sum_mb.append(float(np.nansum(mem_used_vals)))
                        active_power_sum_w.append(float(np.nansum(power_vals)))
        except OSError:
            continue

        rows.append(
            {
                "run_id": run_id,
                "gpu_sm_util_mean_pct": float(np.nanmean(sm_means)) if sm_means else np.nan,
                "gpu_sm_util_peak_pct": float(np.nanmax(sm_means)) if sm_means else np.nan,
                "gpu_mem_util_mean_pct": float(np.nanmean(mem_util_means)) if mem_util_means else np.nan,
                "gpu_mem_util_peak_pct": float(np.nanmax(mem_util_means)) if mem_util_means else np.nan,
                "gpu_mem_used_sum_mb_mean": float(np.nanmean(mem_used_sum_mb)) if mem_used_sum_mb else np.nan,
                "gpu_mem_used_sum_mb_peak": float(np.nanmax(mem_used_sum_mb)) if mem_used_sum_mb else np.nan,
                "gpu_power_sum_w_mean": float(np.nanmean(power_sum_w)) if power_sum_w else np.nan,
                "gpu_power_sum_w_peak": float(np.nanmax(power_sum_w)) if power_sum_w else np.nan,
                "cpu_power_sum_w_mean": float(np.nanmean(cpu_power_sum_w)) if cpu_power_sum_w else np.nan,
                "cpu_power_sum_w_peak": float(np.nanmax(cpu_power_sum_w)) if cpu_power_sum_w else np.nan,
                "gpu_sm_util_active_mean_pct": float(np.nanmean(active_sm_means)) if active_sm_means else np.nan,
                "gpu_mem_util_active_mean_pct": float(np.nanmean(active_mem_util_means)) if active_mem_util_means else np.nan,
                "gpu_mem_used_sum_mb_active_mean": float(np.nanmean(active_mem_used_sum_mb)) if active_mem_used_sum_mb else np.nan,
                "gpu_power_sum_w_active_mean": float(np.nanmean(active_power_sum_w)) if active_power_sum_w else np.nan,
                "n_gpu_samples": int(len(sm_means)),
                "n_gpu_active_samples": int(len(active_sm_means)),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("run_id").reset_index(drop=True)


kernel_logs_root = os.environ.get("ASTROCAMP_BENCH_PASC_DIR", "")
kernel_df = _parse_kernel_breakdown_from_logs(kernel_logs_root)

if not kernel_df.empty:
    monit_df = _parse_monit_gpu_resource_metrics(kernel_logs_root)
    if not monit_df.empty:
        kernel_df = kernel_df.merge(monit_df, on="run_id", how="left")

    kernel_summary = (
        kernel_df.groupby(["image_size", "timesteps", "channels"], as_index=False)
        .agg(
            gridder_s=("gridder_s", "median"),
            sub_fft_s=("sub_fft_s", "median"),
            wtiling_s=("wtiling_s", "median"),
            known_kernel_s=("known_kernel_s", "median"),
            wall_s=("wall_s", "median"),
            wall_s_mean=("wall_s", "mean"),
            wall_s_std=("wall_s", "std"),
            throughput_mvis_s_mean=("throughput_mean_mvis_s", "mean"),
            throughput_mvis_s_std=("throughput_mean_mvis_s", "std"),
            gpu_sm_util_mean_pct=("gpu_sm_util_mean_pct", "mean"),
            gpu_sm_util_peak_pct=("gpu_sm_util_peak_pct", "max"),
            gpu_mem_util_mean_pct=("gpu_mem_util_mean_pct", "mean"),
            gpu_mem_util_peak_pct=("gpu_mem_util_peak_pct", "max"),
            gpu_mem_used_sum_mb_peak=("gpu_mem_used_sum_mb_peak", "mean"),
            gpu_sm_util_active_mean_pct=("gpu_sm_util_active_mean_pct", "mean"),
            gpu_mem_util_active_mean_pct=("gpu_mem_util_active_mean_pct", "mean"),
            gpu_mem_used_sum_mb_active_mean=("gpu_mem_used_sum_mb_active_mean", "mean"),
            cpu_power_sum_w_mean=("cpu_power_sum_w_mean", "mean"),
            other_s=("other_s", "median"),
            n_runs=("run_id", "count"),
        )
        .sort_values(["image_size", "timesteps", "channels"])
        .reset_index(drop=True)
    )

    def _build_timestep_group_layout(sub, gap=0.55):
        x_positions = []
        channel_labels = []
        groups = []
        cursor = 0.0
        for tval, tsub in sub.groupby("timesteps", sort=True):
            start = cursor
            for _, row in tsub.iterrows():
                x_positions.append(cursor)
                channel_labels.append(f"c={int(row['channels'])}")
                cursor += 1.0
            end = cursor - 1.0
            groups.append((int(tval), start, end))
            cursor += gap
        return np.array(x_positions, dtype=float), channel_labels, groups

    def _add_timestep_subgroups(ax, groups, labelsize=16):
        for idx in range(1, len(groups)):
            left = groups[idx - 1][2]
            right = groups[idx][1]
            ax.axvline(0.5 * (left + right), linestyle=":", linewidth=1.2, color="0.45", alpha=0.95)

        secax = ax.secondary_xaxis("top")
        secax.set_xticks([0.5 * (start + end) for _, start, end in groups])
        secax.set_xticklabels([f"t={tval}" for tval, _, _ in groups])
        secax.tick_params(axis="x", labelsize=labelsize, length=0, pad=2)
        return secax

    image_sizes = sorted(kernel_summary["image_size"].unique())
    fig, axes = plt.subplots(len(image_sizes), 1, figsize=(7.2, 3.5 * len(image_sizes)), squeeze=False)

    for row, img in enumerate(image_sizes):
        ax = axes[row][0]
        sub = kernel_summary[kernel_summary["image_size"] == img].copy()
        sub = sub.sort_values(["timesteps", "channels"]).reset_index(drop=True)
        x, channel_labels, t_groups = _build_timestep_group_layout(sub)
        bw = 0.38
        stack_offset = bw * 0.06
        wall_offset = bw * 0.12
        stack_width = bw * 0.62
        wall_width = bw * 0.68

        k0 = sub["gridder_s"].values
        k1 = sub["sub_fft_s"].values
        k2 = sub["wtiling_s"].values
        wall = sub["wall_s"].values

        ax.bar(
            x + wall_offset,
            wall,
            width=wall_width,
            color="0.6",
            alpha=0.28,
            label="WSClean total",
            zorder=1,
        )
        ax.bar(x - stack_offset, k0, width=stack_width, color="#1f77b4", label="gridder", zorder=2)
        ax.bar(x - stack_offset, k1, width=stack_width, bottom=k0, color="#ff7f0e", label="sub-FFT", zorder=2)
        ax.bar(x - stack_offset, k2, width=stack_width, bottom=k0 + k1, color="#2ca02c", label="w-tiling", zorder=2)

        ax.set_yscale("log")
        ax.set_ylabel("Time (s)", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(channel_labels, fontsize=16, rotation=90, ha="center", va="top")
        ax.set_xlabel("Channels", fontsize=16)
        ax.set_title(f"Image {int(img)}", fontsize=16, pad=8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xlim(x[0] - 0.75, x[-1] + 0.75)
        _add_timestep_subgroups(ax, t_groups, labelsize=16)

        if row != len(image_sizes) - 1:
            ax.tick_params(axis="x", labelbottom=True)

    handles, labels = axes[0][0].get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    fig.legend(
        list(uniq.values()),
        list(uniq.keys()),
        ncol=2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        fontsize=16,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.suptitle(
        "IDG Kernel Breakdown and WSClean Total Time",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(Path(out_dir) / "plot17_kernel_time_breakdown_pasc25_16c.png"), dpi=300, bbox_inches="tight")
    fig.savefig(str(Path(out_dir) / "plot17_kernel_time_breakdown_pasc25_16c.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Largest-image detailed view: kernel breakdown + wall and throughput with std deviation.
    largest_img = int(max(image_sizes))
    largest_sub = kernel_summary[kernel_summary["image_size"] == largest_img].copy()
    if not largest_sub.empty:
        largest_sub = largest_sub.sort_values(["timesteps", "channels"]).reset_index(drop=True)
        largest_sub["combo"] = largest_sub.apply(lambda r: f"t{int(r['timesteps'])}-c{int(r['channels'])}", axis=1)

        def _add_timestep_group_guides(ax, t_series):
            t_series = np.asarray(t_series)
            breaks = [i - 0.5 for i in range(1, len(t_series)) if t_series[i] != t_series[i - 1]]
            groups = []
            start = 0
            for i in range(1, len(t_series) + 1):
                if i == len(t_series) or t_series[i] != t_series[i - 1]:
                    groups.append((int(t_series[start]), start, i - 1))
                    start = i

            for xb in breaks:
                ax.axvline(xb, linestyle=":", linewidth=1.1, color="0.45", alpha=0.9)

            # Top axis labels keep t-group text away from data and legends.
            secax = ax.secondary_xaxis("top")
            secax.set_xticks([0.5 * (i0 + i1) for _, i0, i1 in groups])
            secax.set_xticklabels([f"t={tval}" for tval, _, _ in groups])
            secax.tick_params(axis="x", labelsize=8, length=0, pad=2)

        def _add_channel_group_guides(ax, c_series):
            c_series = np.asarray(c_series)
            breaks = [i - 0.5 for i in range(1, len(c_series)) if c_series[i] != c_series[i - 1]]
            groups = []
            start = 0
            for i in range(1, len(c_series) + 1):
                if i == len(c_series) or c_series[i] != c_series[i - 1]:
                    groups.append((int(c_series[start]), start, i - 1))
                    start = i

            for xb in breaks:
                ax.axvline(xb, linestyle=":", linewidth=1.1, color="0.45", alpha=0.9)

            secax = ax.secondary_xaxis("top")
            secax.set_xticks([0.5 * (i0 + i1) for _, i0, i1 in groups])
            secax.set_xticklabels([f"c={cval}" for cval, _, _ in groups])
            secax.tick_params(axis="x", labelsize=8, length=0, pad=2)

        t_series_all = largest_sub["timesteps"].values

        x = np.arange(len(largest_sub), dtype=float)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.2))

        # Left: stacked kernel-time breakdown (median per workload).
        g = largest_sub["gridder_s"].values
        f = largest_sub["sub_fft_s"].values
        w = largest_sub["wtiling_s"].values
        ax1.bar(x, g, label="gridder", color="#1f77b4")
        ax1.bar(x, f, bottom=g, label="sub-fft", color="#ff7f0e")
        ax1.bar(x, w, bottom=g + f, label="wtiling", color="#2ca02c")
        ax1.set_yscale("log")
        ax1.set_ylabel("Kernel time (s)")
        ax1.set_xlabel("Workload (timesteps, channels)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax1.set_title("Kernel-time composition of IDG")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend(frameon=False, fontsize=8)
        _add_timestep_group_guides(ax1, t_series_all)

        # Right: total wall time and average throughput with std deviation.
        ax2r = ax2.twinx()
        wall_mean = largest_sub["wall_s_mean"].values
        wall_std = largest_sub["wall_s_std"].fillna(0.0).values
        thr_mean = largest_sub["throughput_mvis_s_mean"].values
        thr_std = largest_sub["throughput_mvis_s_std"].fillna(0.0).values

        ax2.bar(x, wall_mean, alpha=0.45, color="gray", label="wall time mean")
        ax2.set_yscale("log")
        ax2.set_ylabel("Total wall time (s)")

        ax2r.plot(
            x,
            thr_mean,
            marker="o",
            linewidth=1.5,
            color="black",
            label="avg throughput mean",
        )
        ax2r.set_ylabel("Average throughput (Mvis/s)")

        ax2.set_xlabel("Workload (timesteps, channels)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax2.set_title("Total wall time and average throughput")
        ax2.grid(True, axis="y", alpha=0.25)
        _add_timestep_group_guides(ax2, t_series_all)

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2r.get_legend_handles_labels()
        legend_lines20 = ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
        ax2.add_artist(legend_lines20)

        fig.suptitle(
            f"Largest Image ({largest_img}): Kernel Breakdown, Wall Time, and Throughput Variability",
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(str(Path(out_dir) / "plot18_largest_image_kernel_wall_throughput.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(out_dir) / "plot18_largest_image_kernel_wall_throughput.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Plot19: merged kernel+wall in left panel, throughput + saturation indicators in right panel.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4))

        # Left panel: merged view (stacked kernel bars + wall bars).
        g = largest_sub["gridder_s"].values
        f = largest_sub["sub_fft_s"].values
        w = largest_sub["wtiling_s"].values
        wall = largest_sub["wall_s_mean"].values
        wall_std = largest_sub["wall_s_std"].fillna(0.0).values

        bw = 0.38
        ax1.bar(x - bw / 2, g, width=bw, label="gridder", color="#1f77b4")
        ax1.bar(x - bw / 2, f, width=bw, bottom=g, label="sub-fft", color="#ff7f0e")
        ax1.bar(x - bw / 2, w, width=bw, bottom=g + f, label="wtiling", color="#2ca02c")
        ax1.bar(x + bw / 2, wall, width=bw, label="wall mean", color="0.6", alpha=0.45)
        ax1.set_yscale("log")
        ax1.set_ylabel("Time (s)")
        ax1.set_xlabel("Workload (timesteps, channels)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax1.set_title("Kernel breakdown merged with total wall time")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend(frameon=False, fontsize=8)
        _add_timestep_group_guides(ax1, t_series_all)

        # Right panel: throughput + resource indicators (memory/SM utilization, memory footprint).
        ax2r = ax2.twinx()
        thr_mean = largest_sub["throughput_mvis_s_mean"].values
        thr_std = largest_sub["throughput_mvis_s_std"].fillna(0.0).values
        sm_util = largest_sub["gpu_sm_util_mean_pct"].values
        mem_util = largest_sub["gpu_mem_util_mean_pct"].values
        mem_peak_gb = largest_sub["gpu_mem_used_sum_mb_peak"].values / 1024.0

        ax2.errorbar(
            x,
            thr_mean,
            yerr=thr_std,
            marker="o",
            linewidth=1.5,
            capsize=3,
            color="black",
            label="throughput mean ± std",
        )
        # Memory footprint as marker size on throughput points.
        finite_mem = np.where(np.isfinite(mem_peak_gb), mem_peak_gb, np.nan)
        ms19 = None
        if np.any(np.isfinite(finite_mem)):
            mmin = np.nanmin(finite_mem)
            mmax = np.nanmax(finite_mem)
            ms19 = 40.0 + 120.0 * (finite_mem - mmin) / (mmax - mmin + 1e-12)
            ax2.scatter(x, thr_mean, s=ms19, color="black", alpha=0.28, label="memory footprint (marker size)")

        ax2.set_ylabel("Average throughput (Mvis/s)")
        ax2.set_xlabel("Workload (timesteps, channels)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax2.set_title("Throughput with GPU saturation indicators")
        ax2.grid(True, axis="y", alpha=0.25)
        _add_timestep_group_guides(ax2, t_series_all)

        ax2r.plot(x, sm_util, marker="s", linewidth=1.2, color="#d62728", label="GPU SM util mean (%)")
        ax2r.plot(x, mem_util, marker="^", linewidth=1.2, color="#9467bd", label="GPU mem util mean (%)")
        ax2r.set_ylabel("GPU utilization (%)")

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2r.get_legend_handles_labels()
        legend_lines20 = ax2.legend(
            h1 + h2,
            l1 + l2,
            frameon=False,
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
        )
        ax2.add_artist(legend_lines20)

        if ms19 is not None and np.any(np.isfinite(mem_peak_gb)):
            mvals19 = np.array([np.nanmin(mem_peak_gb), np.nanmedian(mem_peak_gb), np.nanmax(mem_peak_gb)], dtype=float)
            mlabels19 = [f"{v:.1f} GB" for v in mvals19]
            msizes19 = 40.0 + 120.0 * (mvals19 - np.nanmin(mem_peak_gb)) / (np.nanmax(mem_peak_gb) - np.nanmin(mem_peak_gb) + 1e-12)
            bubble_handles19 = [
                ax2.scatter([], [], s=float(s), color="black", alpha=0.28)
                for s in msizes19
            ]
            legend_bubbles19 = ax2.legend(
                bubble_handles19,
                mlabels19,
                title="Memory footprint",
                frameon=False,
                fontsize=8,
                title_fontsize=8,
                loc="upper left",
                bbox_to_anchor=(0.0, 0.66),
            )
            ax2.add_artist(legend_bubbles19)

        ax2.text(
            0.99,
            0.02,
            "I/O read/write counters not present in .monit logs",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

        fig.suptitle(
            f"Largest Image ({largest_img}): Kernel-Wall Coupling and Throughput Saturation Signals",
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(str(Path(out_dir) / "plot19_largest_image_kernel_wall_saturation_signals.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(out_dir) / "plot19_largest_image_kernel_wall_saturation_signals.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Plot19b: same structure as plot19 but with active-window GPU metrics.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4))

        # Left panel matches plot19 for direct comparability.
        ax1.bar(x - bw / 2, g, width=bw, label="gridder", color="#1f77b4")
        ax1.bar(x - bw / 2, f, width=bw, bottom=g, label="sub-fft", color="#ff7f0e")
        ax1.bar(x - bw / 2, w, width=bw, bottom=g + f, label="wtiling", color="#2ca02c")
        ax1.bar(x + bw / 2, wall, width=bw, label="wall mean", color="0.6", alpha=0.45)
        ax1.set_yscale("log")
        ax1.set_ylabel("Time (s)")
        ax1.set_xlabel("Workload (timesteps, channels)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax1.set_title("Kernel breakdown merged with total wall time")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend(frameon=False, fontsize=8)
        _add_timestep_group_guides(ax1, t_series_all)

        # Right panel: active-window utilization and active memory footprint.
        ax2r = ax2.twinx()
        sm_util_active_19b = largest_sub["gpu_sm_util_active_mean_pct"].values
        mem_util_active_19b = largest_sub["gpu_mem_util_active_mean_pct"].values
        mem_active_gb_19b = largest_sub["gpu_mem_used_sum_mb_active_mean"].values / 1024.0

        ax2.plot(
            x,
            thr_mean,
            marker="o",
            linewidth=1.5,
            color="black",
            label="throughput mean",
        )

        finite_mem_19b = np.where(np.isfinite(mem_active_gb_19b), mem_active_gb_19b, np.nan)
        ms19b = None
        if np.any(np.isfinite(finite_mem_19b)):
            mmin = np.nanmin(finite_mem_19b)
            mmax = np.nanmax(finite_mem_19b)
            ms19b = 40.0 + 120.0 * (finite_mem_19b - mmin) / (mmax - mmin + 1e-12)
            ax2.scatter(x, thr_mean, s=ms19b, color="black", alpha=0.28, label="active memory footprint (marker size)")

        ax2.set_ylabel("Average throughput (Mvis/s)")
        ax2.set_xlabel("Workload (timesteps, channels)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax2.set_title("Throughput with active-window GPU saturation indicators")
        ax2.grid(True, axis="y", alpha=0.25)
        _add_timestep_group_guides(ax2, t_series_all)

        ax2r.plot(x, sm_util_active_19b, marker="s", linewidth=1.2, color="#d62728", label="GPU SM util active mean (%)")
        ax2r.plot(x, mem_util_active_19b, marker="^", linewidth=1.2, color="#9467bd", label="GPU mem util active mean (%)")
        ax2r.set_ylabel("GPU utilization during active windows (%)")

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2r.get_legend_handles_labels()
        legend_lines19b = ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
        ax2.add_artist(legend_lines19b)

        if ms19b is not None and np.any(np.isfinite(mem_active_gb_19b)):
            mvals19b = np.array([np.nanmin(mem_active_gb_19b), np.nanmedian(mem_active_gb_19b), np.nanmax(mem_active_gb_19b)], dtype=float)
            mlabels19b = [f"{v:.1f} GB" for v in mvals19b]
            msizes19b = 40.0 + 120.0 * (mvals19b - np.nanmin(mem_active_gb_19b)) / (np.nanmax(mem_active_gb_19b) - np.nanmin(mem_active_gb_19b) + 1e-12)
            bubble_handles19b = [
                ax2.scatter([], [], s=float(s), color="black", alpha=0.28)
                for s in msizes19b
            ]
            legend_bubbles19b = ax2.legend(
                bubble_handles19b,
                mlabels19b,
                title="Active memory footprint",
                frameon=False,
                fontsize=8,
                title_fontsize=8,
                loc="upper left",
                bbox_to_anchor=(0.0, 0.62),
            )
            ax2.add_artist(legend_bubbles19b)

        ax2.text(
            0.99,
            0.02,
            "Active window = any GPU has SM util > 0 or mem util > 0",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

        fig.suptitle(
            f"Largest Image ({largest_img}): Kernel-Wall Coupling with Active-Window Saturation Signals",
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(str(Path(out_dir) / "plot19b_largest_image_kernel_wall_saturation_signals_active.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(out_dir) / "plot19b_largest_image_kernel_wall_saturation_signals_active.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Plot19d: same structure as plot19b but grouped by channel count instead of timestep count.
        largest_sub_c = largest_sub.sort_values(["channels", "timesteps"]).reset_index(drop=True)
        largest_sub_c["combo"] = largest_sub_c.apply(lambda r: f"t{int(r['timesteps'])}-c{int(r['channels'])}", axis=1)
        x_c = np.arange(len(largest_sub_c), dtype=float)
        c_series_all = largest_sub_c["channels"].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4))

        g_c = largest_sub_c["gridder_s"].values
        f_c = largest_sub_c["sub_fft_s"].values
        w_c = largest_sub_c["wtiling_s"].values
        wall_c = largest_sub_c["wall_s_mean"].values
        thr_c = largest_sub_c["throughput_mvis_s_mean"].values
        sm_util_active_c = largest_sub_c["gpu_sm_util_active_mean_pct"].values
        mem_util_active_c = largest_sub_c["gpu_mem_util_active_mean_pct"].values
        mem_active_gb_c = largest_sub_c["gpu_mem_used_sum_mb_active_mean"].values / 1024.0

        ax1.bar(x_c - bw / 2, g_c, width=bw, label="gridder", color="#1f77b4")
        ax1.bar(x_c - bw / 2, f_c, width=bw, bottom=g_c, label="sub-fft", color="#ff7f0e")
        ax1.bar(x_c - bw / 2, w_c, width=bw, bottom=g_c + f_c, label="wtiling", color="#2ca02c")
        ax1.bar(x_c + bw / 2, wall_c, width=bw, label="wall mean", color="0.6", alpha=0.45)
        ax1.set_yscale("log")
        ax1.set_ylabel("Time (s)")
        ax1.set_xlabel("Workload (timesteps, channels)")
        ax1.set_xticks(x_c)
        ax1.set_xticklabels(largest_sub_c["combo"].tolist(), rotation=90, ha="center")
        ax1.set_title("Kernel breakdown merged with total wall time")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend(frameon=False, fontsize=8)
        _add_channel_group_guides(ax1, c_series_all)

        ax2r = ax2.twinx()
        ax2.plot(
            x_c,
            thr_c,
            marker="o",
            linewidth=1.5,
            color="black",
            label="throughput mean",
        )

        finite_mem_19d = np.where(np.isfinite(mem_active_gb_c), mem_active_gb_c, np.nan)
        ms19d = None
        if np.any(np.isfinite(finite_mem_19d)):
            mmin = np.nanmin(finite_mem_19d)
            mmax = np.nanmax(finite_mem_19d)
            ms19d = 40.0 + 120.0 * (finite_mem_19d - mmin) / (mmax - mmin + 1e-12)
            ax2.scatter(x_c, thr_c, s=ms19d, color="black", alpha=0.28, label="active memory footprint (marker size)")

        ax2.set_ylabel("Average throughput (Mvis/s)")
        ax2.set_xlabel("Workload (timesteps, channels)")
        ax2.set_xticks(x_c)
        ax2.set_xticklabels(largest_sub_c["combo"].tolist(), rotation=90, ha="center")
        ax2.set_title("Throughput with active-window GPU saturation indicators")
        ax2.grid(True, axis="y", alpha=0.25)
        _add_channel_group_guides(ax2, c_series_all)

        ax2r.plot(x_c, sm_util_active_c, marker="s", linewidth=1.2, color="#d62728", label="GPU SM util active mean (%)")
        ax2r.plot(x_c, mem_util_active_c, marker="^", linewidth=1.2, color="#9467bd", label="GPU mem util active mean (%)")
        ax2r.set_ylabel("GPU utilization during active windows (%)")

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2r.get_legend_handles_labels()
        legend_lines19d = ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
        ax2.add_artist(legend_lines19d)

        if ms19d is not None and np.any(np.isfinite(mem_active_gb_c)):
            mvals19d = np.array([np.nanmin(mem_active_gb_c), np.nanmedian(mem_active_gb_c), np.nanmax(mem_active_gb_c)], dtype=float)
            mlabels19d = [f"{v:.1f} GB" for v in mvals19d]
            msizes19d = 40.0 + 120.0 * (mvals19d - np.nanmin(mem_active_gb_c)) / (np.nanmax(mem_active_gb_c) - np.nanmin(mem_active_gb_c) + 1e-12)
            bubble_handles19d = [
                ax2.scatter([], [], s=float(s), color="black", alpha=0.28)
                for s in msizes19d
            ]
            legend_bubbles19d = ax2.legend(
                bubble_handles19d,
                mlabels19d,
                title="Active memory footprint",
                frameon=False,
                fontsize=8,
                title_fontsize=8,
                loc="upper left",
                bbox_to_anchor=(0.0, 0.62),
            )
            ax2.add_artist(legend_bubbles19d)

        ax2.text(
            0.99,
            0.02,
            "Active window = any GPU has SM util > 0 or mem util > 0",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

        fig.suptitle(
            f"Largest Image ({largest_img}): Kernel-Wall Coupling with Active-Window Saturation Signals Grouped by Channels",
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(str(Path(out_dir) / "plot19d_largest_image_kernel_wall_saturation_signals_active_by_c.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(out_dir) / "plot19d_largest_image_kernel_wall_saturation_signals_active_by_c.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Plot19c: faceted active-metric version of plot19b, separated by timestep size.
        tvals = sorted(largest_sub["timesteps"].unique())
        fig, axes = plt.subplots(len(tvals), 2, figsize=(15.0, 4.2 * len(tvals)), squeeze=False)

        for i, t in enumerate(tvals):
            row = largest_sub[largest_sub["timesteps"] == t].sort_values("channels").reset_index(drop=True)
            x_t = np.arange(len(row), dtype=float)
            ch_lbl = [str(int(c)) for c in row["channels"].values]

            # Left facet: kernel + wall time for fixed timestep.
            axl = axes[i][0]
            g_t = row["gridder_s"].values
            f_t = row["sub_fft_s"].values
            w_t = row["wtiling_s"].values
            wall_t = row["wall_s_mean"].values
            wall_t_std = row["wall_s_std"].fillna(0.0).values

            axl.bar(x_t - bw / 2, g_t, width=bw, color="#1f77b4", label="gridder")
            axl.bar(x_t - bw / 2, f_t, width=bw, bottom=g_t, color="#ff7f0e", label="sub-fft")
            axl.bar(x_t - bw / 2, w_t, width=bw, bottom=g_t + f_t, color="#2ca02c", label="wtiling")
            axl.bar(x_t + bw / 2, wall_t, width=bw, yerr=wall_t_std, capsize=3, color="0.6", alpha=0.45, label="wall mean ± std")
            axl.set_yscale("log")
            axl.set_ylabel("Time (s)")
            axl.set_xticks(x_t)
            axl.set_xticklabels(ch_lbl)
            axl.set_title(f"t={int(t)}: Kernel-wall decomposition")
            axl.grid(True, axis="y", alpha=0.25)

            # Right facet: throughput + active-window utilization for fixed timestep.
            axr = axes[i][1]
            axr2 = axr.twinx()
            thr_t = row["throughput_mvis_s_mean"].values
            thr_t_std = row["throughput_mvis_s_std"].fillna(0.0).values
            sm_t = row["gpu_sm_util_active_mean_pct"].values
            mem_t = row["gpu_mem_util_active_mean_pct"].values
            mem_gb_t = row["gpu_mem_used_sum_mb_active_mean"].values / 1024.0

            axr.errorbar(x_t, thr_t, yerr=thr_t_std, marker="o", linewidth=1.5, capsize=3, color="black", label="throughput mean ± std")

            finite_mem_t = np.where(np.isfinite(mem_gb_t), mem_gb_t, np.nan)
            if np.any(np.isfinite(finite_mem_t)):
                mmin_t = np.nanmin(finite_mem_t)
                mmax_t = np.nanmax(finite_mem_t)
                ms_t = 40.0 + 120.0 * (finite_mem_t - mmin_t) / (mmax_t - mmin_t + 1e-12)
                axr.scatter(x_t, thr_t, s=ms_t, color="black", alpha=0.28, label="active memory footprint (marker size)")

            axr2.plot(x_t, sm_t, marker="s", linewidth=1.2, color="#d62728", label="GPU SM util active mean (%)")
            axr2.plot(x_t, mem_t, marker="^", linewidth=1.2, color="#9467bd", label="GPU mem util active mean (%)")

            axr.set_ylabel("Throughput (Mvis/s)")
            axr2.set_ylabel("GPU utilization active-window (%)")
            axr.set_xticks(x_t)
            axr.set_xticklabels(ch_lbl)
            axr.set_title(f"t={int(t)}: Throughput and active-window saturation")
            axr.grid(True, axis="y", alpha=0.25)

            if i == len(tvals) - 1:
                axl.set_xlabel("Channels")
                axr.set_xlabel("Channels")

            if i == 0:
                h1, l1 = axl.get_legend_handles_labels()
                axl.legend(h1, l1, frameon=False, fontsize=8, loc="upper left")

                h2, l2 = axr.get_legend_handles_labels()
                h3, l3 = axr2.get_legend_handles_labels()
                axr.legend(h2 + h3, l2 + l3, frameon=False, fontsize=8, loc="upper left")

        fig.suptitle(
            f"Largest Image ({largest_img}): Active-Window Saturation Signals Separated by Timesteps",
            y=1.0,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(str(Path(out_dir) / "plot19c_largest_image_kernel_wall_saturation_signals_active_by_t.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(out_dir) / "plot19c_largest_image_kernel_wall_saturation_signals_active_by_t.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Plot20: active-window saturation signals to reduce idle-time dilution.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.4))

        # Left: merged kernel + wall view (same baseline as plot19).
        ax1.bar(x - bw / 2, g, width=bw, label="gridder", color="#1f77b4")
        ax1.bar(x - bw / 2, f, width=bw, bottom=g, label="sub-fft", color="#ff7f0e")
        ax1.bar(x - bw / 2, w, width=bw, bottom=g + f, label="wtiling", color="#2ca02c")
        ax1.bar(x + bw / 2, wall, width=bw, yerr=wall_std, capsize=3, label="wall mean ± std", color="0.6", alpha=0.45)
        ax1.set_yscale("log")
        ax1.set_ylabel("Time (s)")
        ax1.set_xlabel("Workload (timesteps, channels)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax1.set_title("Kernel breakdown merged with total wall time")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend(frameon=False, fontsize=8)
        _add_timestep_group_guides(ax1, t_series_all)

        # Right: throughput + active-window resource signals.
        ax2r = ax2.twinx()
        sm_util_active = largest_sub["gpu_sm_util_active_mean_pct"].values
        mem_util_active = largest_sub["gpu_mem_util_active_mean_pct"].values
        mem_active_gb = largest_sub["gpu_mem_used_sum_mb_active_mean"].values / 1024.0

        ax2.errorbar(
            x,
            thr_mean,
            yerr=thr_std,
            marker="o",
            linewidth=1.5,
            capsize=3,
            color="black",
            label="throughput mean ± std",
        )

        finite_mem = np.where(np.isfinite(mem_active_gb), mem_active_gb, np.nan)
        ms = None
        if np.any(np.isfinite(finite_mem)):
            mmin = np.nanmin(finite_mem)
            mmax = np.nanmax(finite_mem)
            ms = 40.0 + 120.0 * (finite_mem - mmin) / (mmax - mmin + 1e-12)
            ax2.scatter(x, thr_mean, s=ms, color="black", alpha=0.28, label="active memory footprint (marker size)")

        ax2.set_ylabel("Average throughput (Mvis/s)")
        ax2.set_xlabel("Workload (timesteps, channels)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax2.set_title("Throughput with active-window saturation signals")
        ax2.grid(True, axis="y", alpha=0.25)
        _add_timestep_group_guides(ax2, t_series_all)

        ax2r.plot(x, sm_util_active, marker="s", linewidth=1.2, color="#d62728", label="GPU SM util active mean (%)")
        ax2r.plot(x, mem_util_active, marker="^", linewidth=1.2, color="#9467bd", label="GPU mem util active mean (%)")
        ax2r.set_ylabel("GPU utilization during active windows (%)")

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2r.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")

        if ms is not None and np.any(np.isfinite(mem_active_gb)):
            mvals = np.array([np.nanmin(mem_active_gb), np.nanmedian(mem_active_gb), np.nanmax(mem_active_gb)], dtype=float)
            mlabels = [f"{v:.1f} GB" for v in mvals]
            msizes = 40.0 + 120.0 * (mvals - np.nanmin(mem_active_gb)) / (np.nanmax(mem_active_gb) - np.nanmin(mem_active_gb) + 1e-12)
            bubble_handles = [
                ax2.scatter([], [], s=float(s), color="black", alpha=0.28)
                for s in msizes
            ]
            legend_bubbles = ax2.legend(
                bubble_handles,
                mlabels,
                title="Active memory",
                frameon=False,
                fontsize=8,
                title_fontsize=8,
                loc="upper left",
                bbox_to_anchor=(0.0, 0.56),
            )
            ax2.add_artist(legend_bubbles)

        ax2.text(
            0.99,
            0.02,
            "Active window = any GPU has SM util > 0 or mem util > 0",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

        fig.suptitle(
            f"Largest Image ({largest_img}): Active-Window Saturation Signals",
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(str(Path(out_dir) / "plot20_largest_image_active_window_saturation.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(out_dir) / "plot20_largest_image_active_window_saturation.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Plot21: publication-oriented version of plot19 with clearer visual hierarchy.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.2))

        # Left panel: normalized time composition as percent of wall time.
        wall_safe = np.where(np.isfinite(wall) & (wall > 0.0), wall, np.nan)
        g_pct = 100.0 * g / wall_safe
        f_pct = 100.0 * f / wall_safe
        w_pct = 100.0 * w / wall_safe
        other_pct = 100.0 * largest_sub["other_s"].values / wall_safe
        total_kernel_pct = g_pct + f_pct + w_pct

        ax1.bar(x, g_pct, label="gridder", color="#1f77b4")
        ax1.bar(x, f_pct, bottom=g_pct, label="sub-fft", color="#ff7f0e")
        ax1.bar(x, w_pct, bottom=g_pct + f_pct, label="wtiling", color="#2ca02c")
        ax1.bar(x, other_pct, bottom=total_kernel_pct, label="other", color="#9e9e9e", alpha=0.8)
        ax1.set_ylabel("Share of total wall time (%)")
        ax1.set_xlabel("Workload (timesteps, channels)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax1.set_ylim(0.0, 100.0)
        ax1.set_title("Wall-time decomposition (normalized)")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
        _add_timestep_group_guides(ax1, t_series_all)

        # Right panel: throughput and GPU utilization only (no marker-size encoding).
        ax2r = ax2.twinx()
        ax2.plot(
            x,
            thr_mean,
            marker="o",
            linewidth=1.8,
            color="black",
            label="throughput mean",
        )
        ax2.fill_between(
            x,
            thr_mean - thr_std,
            thr_mean + thr_std,
            color="black",
            alpha=0.12,
            linewidth=0.0,
            label="throughput ± std",
        )
        ax2.set_ylabel("Average throughput (Mvis/s)")
        ax2.set_xlabel("Workload (timesteps, channels)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax2.set_title("Throughput and GPU utilization")
        ax2.grid(True, axis="y", alpha=0.25)
        _add_timestep_group_guides(ax2, t_series_all)

        ax2r.plot(x, sm_util, marker="s", linewidth=1.4, color="#d62728", label="GPU SM util mean (%)")
        ax2r.plot(x, mem_util, marker="^", linewidth=1.4, color="#9467bd", label="GPU mem util mean (%)")
        ax2r.set_ylabel("GPU utilization (%)")

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2r.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")

        fig.suptitle(
            f"Largest Image ({largest_img}): Readable Saturation View for Publication",
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(str(Path(out_dir) / "plot21_largest_image_readable_article_view.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(out_dir) / "plot21_largest_image_readable_article_view.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Plot22: dedicated view of peak GPU utilization signals.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.2))

        sm_mean = largest_sub["gpu_sm_util_mean_pct"].values
        sm_peak = largest_sub["gpu_sm_util_peak_pct"].values
        mem_mean = largest_sub["gpu_mem_util_mean_pct"].values
        mem_peak = largest_sub["gpu_mem_util_peak_pct"].values

        # Left: SM utilization mean vs peak.
        ax1.plot(x, sm_mean, marker="o", linewidth=1.5, color="#d62728", label="SM util mean (%)")
        ax1.plot(x, sm_peak, marker="s", linewidth=1.5, linestyle="--", color="#8c1d1d", label="SM util peak (%)")
        ax1.set_ylim(0.0, 100.0)
        ax1.set_ylabel("SM utilization (%)")
        ax1.set_xlabel("Workload (timesteps, channels)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax1.set_title("GPU SM utilization: mean vs peak")
        ax1.grid(True, axis="y", alpha=0.25)
        ax1.legend(frameon=False, fontsize=8, loc="upper left")
        _add_timestep_group_guides(ax1, t_series_all)

        # Right: memory utilization mean vs peak with throughput context.
        ax2r = ax2.twinx()
        ax2.plot(x, mem_mean, marker="o", linewidth=1.5, color="#9467bd", label="Mem util mean (%)")
        ax2.plot(x, mem_peak, marker="^", linewidth=1.5, linestyle="--", color="#5e3c99", label="Mem util peak (%)")
        ax2.set_ylim(0.0, 100.0)
        ax2.set_ylabel("Memory utilization (%)")
        ax2.set_xlabel("Workload (timesteps, channels)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(largest_sub["combo"].tolist(), rotation=90, ha="center")
        ax2.set_title("GPU memory utilization: mean vs peak")
        ax2.grid(True, axis="y", alpha=0.25)
        _add_timestep_group_guides(ax2, t_series_all)

        ax2r.plot(x, thr_mean, marker="D", linewidth=1.2, color="black", alpha=0.8, label="throughput mean (Mvis/s)")
        ax2r.set_ylabel("Throughput (Mvis/s)")

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = ax2r.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")

        fig.suptitle(
            f"Largest Image ({largest_img}): Peak GPU Utilization Diagnostics",
            y=1.02,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(str(Path(out_dir) / "plot22_largest_image_peak_gpu_utilization.png"), dpi=300, bbox_inches="tight")
        fig.savefig(str(Path(out_dir) / "plot22_largest_image_peak_gpu_utilization.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

        # Plot23: article-focused relationship used to explain plot11 saturation behavior.
        plot11_like = kernel_summary[
            (kernel_summary["image_size"] == 8192)
            & (kernel_summary["timesteps"].isin([64, 256]))
        ].copy()
        if not plot11_like.empty:
            plot11_like = plot11_like.sort_values(["timesteps", "channels"]).reset_index(drop=True)
            plot11_like["active_mem_gb"] = plot11_like["gpu_mem_used_sum_mb_active_mean"] / 1024.0
            am = plot11_like["active_mem_gb"].values
            amin = np.nanmin(am)
            amax = np.nanmax(am)
            plot11_like["active_mem_marker_size"] = 36.0 + 140.0 * (am - amin) / (amax - amin + 1e-12)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.0))

            # Left: direct relationship (throughput vs active memory utilization).
            for t, sub in plot11_like.groupby("timesteps"):
                (ln,) = ax1.plot(
                    sub["gpu_mem_util_active_mean_pct"].values,
                    sub["throughput_mvis_s_mean"].values,
                    marker="o",
                    linewidth=1.5,
                    label=f"t={int(t)}",
                )
                ax1.scatter(
                    sub["gpu_mem_util_active_mean_pct"].values,
                    sub["throughput_mvis_s_mean"].values,
                    s=sub["active_mem_marker_size"].values,
                    color=ln.get_color(),
                    alpha=0.22,
                    edgecolors="none",
                    zorder=2,
                )

            ax1.set_xlabel("Active memory utilization (%)")
            ax1.set_ylabel("Throughput (Mvis/s)")
            ax1.set_title("Throughput vs active memory utilization")
            ax1.grid(True, alpha=0.25)
            ax1.legend(frameon=False, fontsize=8, loc="upper left")

            # Bubble legend: total active memory indicator (GB).
            mvals = np.array([np.nanmin(am), np.nanmedian(am), np.nanmax(am)], dtype=float)
            mlabs = [f"{v:.1f} GB" for v in mvals]
            msizes = 36.0 + 140.0 * (mvals - np.nanmin(am)) / (np.nanmax(am) - np.nanmin(am) + 1e-12)
            mhandles = [
                ax1.scatter([], [], s=float(s), color="black", alpha=0.22)
                for s in msizes
            ]
            mem_legend = ax1.legend(
                mhandles,
                mlabs,
                title="Total active memory",
                frameon=False,
                fontsize=8,
                title_fontsize=8,
                loc="lower right",
            )
            ax1.add_artist(mem_legend)

            # Right: channel scaling view for the same workloads as plot11.
            ax2r = ax2.twinx()
            chs = sorted(plot11_like["channels"].unique())
            for t, sub in plot11_like.groupby("timesteps"):
                sub = sub.set_index("channels").reindex(chs)
                (ln2,) = ax2.plot(
                    chs,
                    sub["throughput_mvis_s_mean"].values,
                    marker="o",
                    linewidth=1.5,
                    label=f"throughput t={int(t)}",
                )
                ax2.scatter(
                    chs,
                    sub["throughput_mvis_s_mean"].values,
                    s=sub["active_mem_marker_size"].values,
                    color=ln2.get_color(),
                    alpha=0.20,
                    edgecolors="none",
                    zorder=2,
                )
                ax2r.plot(
                    chs,
                    sub["gpu_mem_util_active_mean_pct"].values,
                    marker="s",
                    linewidth=1.3,
                    linestyle="--",
                    label=f"active mem util t={int(t)}",
                )

            ax2.set_xscale("log", base=2)
            ax2.set_xticks(chs)
            ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax2.set_xlabel("Channels")
            ax2.set_ylabel("Throughput (Mvis/s)")
            ax2r.set_ylabel("Active memory utilization (%)")
            ax2.set_title("Same channel scaling as plot11")
            ax2.grid(True, alpha=0.25)

            h1, l1 = ax2.get_legend_handles_labels()
            h2, l2 = ax2r.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")
            fig.suptitle(
                "Why Plot11 Saturates: Throughput, Active Memory Utilization, and Active Memory Footprint (Image 8192)",
                y=1.02,
            )
            fig.tight_layout()
            fig.savefig(str(Path(out_dir) / "plot23_plot11_explanation_throughput_active_memory_util.png"), dpi=300, bbox_inches="tight")
            fig.savefig(str(Path(out_dir) / "plot23_plot11_explanation_throughput_active_memory_util.pdf"), format="pdf", bbox_inches="tight")
            plt.close(fig)

    kernel_summary.to_csv(str(DATA_DIR / "kernel_breakdown_pasc25_16c_summary.csv"), index=False)


(flagship_pdf, multipage_pdf)
