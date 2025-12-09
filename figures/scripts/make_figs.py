import matplotlib
# Comment the next line if you want the plot window to pop up interactively:
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Data
# ----------------------------
# Power caps (MW) - 2 to 5 MW
power_caps = np.array([1, 2, 3, 4, 5])

# Compute requirements in PFLOPS for SKA1-Low and SKA1-Mid
ska_low_pflops = [16, 41.5]
ska_mid_pflops = [20, 72]

# Convert PFLOPS to GFLOPS (1 PFLOP = 1e6 GFLOPS)
ska_low_gflops = np.array(ska_low_pflops) * 1e6
ska_mid_gflops = np.array(ska_mid_pflops) * 1e6

# Efficiency (GFLOPS/W) = performance (GFLOPS) / power (W)
eff_low_min = ska_low_gflops[0] / (power_caps * 1e6)
eff_low_max = ska_low_gflops[1] / (power_caps * 1e6)
eff_mid_min = ska_mid_gflops[0] / (power_caps * 1e6)
eff_mid_max = ska_mid_gflops[1] / (power_caps * 1e6)

# Top500 / Green500 reference values (June 2025)
top500_avg = 35
top500_best = 73

# Astronomy pipeline utilization range (5% - 15% of HW)
astro_low_best = 0.05 * top500_best
astro_high_best = 0.15 * top500_best
astro_low_avg = 0.05 * top500_avg
astro_high_avg = 0.15 * top500_avg

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8, 4.8))

# SKA1-Low range - light blue band with diagonal hatch
plt.fill_between(
    power_caps, eff_low_min, eff_low_max,
    color="#8cc3e1", alpha=0.5, label="SKA1-Low (required)",
    hatch='//', edgecolor="#084594", linewidth=0.0
)

# SKA1-Mid range - orange band with dotted hatch
plt.fill_between(
    power_caps, eff_mid_min, eff_mid_max,
    color="#dfb536", alpha=0.25, label="SKA1-Mid (required)",
    hatch='..', edgecolor="#7f2704", linewidth=0.0
)

# Astronomy utilization bands (hatching for texture)
plt.axhspan(
    astro_low_avg, astro_high_avg,
    color="#fbb4b9", alpha=0.5, label="Astronomy avg deployment (5–15%)",
    hatch='++', edgecolor="#7a0177", linewidth=0.0
)

# plt.axhspan(
#     astro_low_best, astro_high_best,
#     color="#c7e9c0", alpha=0.5, label="Astronomy best deployment (5–15%)",
#     hatch='xx', edgecolor="#00441b", linewidth=0.0
# )

# Reference lines (made semi-transparent)
plt.axhline(top500_avg, linestyle="--", linewidth=2, color="#030303",
            alpha=0.4, label="Top500 avg (LINPACK) ~35 GFLOPS/W")
plt.axhline(top500_best, linestyle="-", linewidth=2, color="#006837",
            alpha=0.4, label="Top500 best (Green500) ~73 GFLOPS/W")

# Annotate efficiency values for 2–5 MW points
for i, p in enumerate(power_caps):
    plt.text(p, eff_low_max[i] + 1, f"{eff_low_max[i]:.0f}",
             color="#084594", fontsize=12, ha="center")
    plt.text(p, eff_mid_max[i] + 1, f"{eff_mid_max[i]:.0f}",
             color="#7f2704", fontsize=12, ha="center")

# Labels and formatting
plt.xlabel("Power Cap per Site (MW)")
plt.ylabel("Required Efficiency (GFLOPS/W)")
plt.grid(alpha=0.3, zorder=0)
plt.ylim(0, 80)
plt.xticks(power_caps)

# Legend: smaller, single column, positioned in the white space
# Using framealpha and edgecolor so hatched patches show nicely in legend too
leg = plt.legend(
    loc="lower right",
    fontsize=12,
    framealpha=0.9,
    bbox_to_anchor=(1.0, 0.48),
    ncol=1
)
# Improve legend patch edges for hatched items
for patch in leg.legend_handles:
    try:
        patch.set_linewidth(0.8)
        patch.set_edgecolor("black")
    except Exception:
        pass

plt.tight_layout()

# Save both PNG and PDF to current working directory
png_path = "/Users/nisa/code/astroCAMP/figures/ska_efficiency_final_singlecol_legend_textured.png"
pdf_path = "/Users/nisa/code/astroCAMP/figures/ska_efficiency_final_singlecol_legend_textured.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight", format="pdf")

# Show interactively (comment out if running headless)
plt.show()

print(f"Saved:\n  {png_path}\n  {pdf_path}")
