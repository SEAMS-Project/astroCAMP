import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Resolve CSV path relative to repo structure
BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "scripts" / "benchmarks_comprehensive.csv"

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
out_dir = str(Path(__file__).resolve().parent)
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
locs = sorted(df["Location"].unique())
nrows = len(img_sizes)
ncols = len(locs)

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.5*ncols, 3.2*nrows), squeeze=False)

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

        ax.bar(x, dyn, label="Dynamic Energy (Wh)")
        ax.bar(x, sta, bottom=dyn, label="Static Energy (Wh)")
        ax.set_ylabel("Energy (Wh)")
        ax.set_xticks(x)
        if i == nrows - 1:
            ax.set_xticklabels(sub["combo"], rotation=90, ha="center")
        else:
            ax.set_xticklabels([])
        ax.set_title(f"Image {img}² — {loc}")

        # Carbon annotations atop total stacked bars (g CO2)
        total_wh = dyn + sta
        carbon_g = None
        try:
            if "S" in sub.columns:
                # Column S provided by user; treat as grams CO2 total per run
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
        ax2.plot(x, sub["Mvis/h"].values, marker="o")
        ax2.set_ylabel("Mvis/h")
        ax2.set_yscale("log")

        # "insight" marker: best energy-efficiency point
        best_idx = sub["Mvis/kWh"].values.argmax()
        ax2.annotate("best Mvis/kWh",
                 xy=(best_idx, sub["Mvis/h"].values[best_idx]),
                 xytext=(best_idx, sub["Mvis/h"].values[best_idx]*2.0),
                 arrowprops=dict(arrowstyle="->"))

handles, labels = axes[0][0].get_legend_handles_labels()
fig.suptitle("Energy breakdown (stacked) + throughput (right axis) across (Timesteps, Channels)\nFaceted by Image Size and Location", y=0.99)
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))
fig.tight_layout(rect=[0,0,1,0.96])

fig.savefig(str(Path(out_dir) / "plot9_flagship_energy_stack_throughput_facets.png"), dpi=300, bbox_inches="tight")
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
            ax2.annotate("best",
                         xy=(best_idx, sub["Mvis/h"].values[best_idx]),
                         xytext=(best_idx, sub["Mvis/h"].values[best_idx]*2.0),
                         fontsize=8,
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
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Flagship: Energy (stacked) + Throughput", y=0.99, fontsize=11)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.935))
    fig.tight_layout(rect=[0,0,1,0.955])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

(flagship_pdf, multipage_pdf)
