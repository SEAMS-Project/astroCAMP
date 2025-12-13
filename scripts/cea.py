import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Carbon and Economic analysis.')
parser.add_argument('-l', '--lifetime', type=int, default=6, help='Lifetime in years (default: 5)')
args = parser.parse_args()

print(f"Using lifetime of {args.lifetime} years for all machines.")

# Parameters
Lifetime = args.lifetime * 365 * 24  # Lifetime in hours
location_ids = ['SA', 'WA']

# Plot benchmarks
benchmarks_df = pd.read_csv("benchmarks.csv",
                 header=None,
                 names=["im_size", "n_times", "n_chans", "wall_time", "wall_time_sec", "n_rows", "n_vis",
                        "n_idg",
                        "idg_h_sec", "idg_h_watt", "idg_h_jou",
                        "idg_d_sec", "idg_d_watt", "idg_d_jou",
                        "idg_grid_mvs",
                        "cpu_j", "cpu_bsl_j", "cpu_bsl_std_j",
                        "gpu0_j", "gpu1_j", "gpu2_j", "gpu3_j",
                        "gpu_j", "gpu_bsl_j", "gpu_bsl_std_j",
                        "tot_sys_j",
                        "tot_pdu_j", "pdu_bsl_j", "pdu_bsl_std_j",
                        "abs_cpu_j", "abs_gpu_j", "abs_pdu_j"])

print(benchmarks_df.to_string())

idle_pdu_watt = benchmarks_df['pdu_bsl_j'].mean() # W

benchmarks_df['machine'] = 'R675 V3 + 4xH100 96GB'  # Assign machine name to all benchmarks
benchmarks_df['benchmark'] = benchmarks_df['im_size'].astype(str) + '_' + benchmarks_df['n_times'].astype(str) + '_' + benchmarks_df['n_chans'].astype(str)
benchmarks_df['time'] = benchmarks_df['wall_time_sec']
# Temporal
benchmarks_df['mvis'] = benchmarks_df['n_vis'] / 1e6

# Read the three CSV files
machines_df = pd.read_csv('machines.csv').set_index('machine')
locations_df = pd.read_csv('locations.csv').set_index('id').reset_index()
locations_df = locations_df[locations_df['id'].isin(location_ids)]

# Create a comprehensive table with benchmarks in rows
# and operational and capital expenditures (carbon and costs) in columns
results = []

for _, benchmark in benchmarks_df.iterrows():
    benchmark_name = benchmark['benchmark']
    machine_name = benchmark['machine']  # Machine is now specified in the benchmark
    time = benchmark['time'] / 3600 # from seconds to hours
    energy_dynamic = benchmark['tot_sys_j'] / 3.6e6 # Static energy in kWh
    energy_static = idle_pdu_watt / 4 * time / 1000  # Dynamic energy in kWh
    energy = energy_dynamic + energy_static # Total energy in kWh
    
    # Get the specific machine's cost and embodied carbon
    if machine_name not in machines_df.index:
        raise ValueError(f"Machine '{machine_name}' not found in machines.csv")
    
    machine_cost = machines_df.loc[machine_name, 'cost'] # in $
    machine_embodied = machines_df.loc[machine_name, 'embodied'] # in kg CO2
    
    for _, location in locations_df.iterrows():
        location_id = location['id']
        location_name = location['location']
        ci = location['ci']  # Carbon intensity in kg CO2/kWh
        ep = location['ep']  # Electricity price in $/kWh
        
        # Calculate operational and capital expenditures
        operational_energy_cost = energy * ep # in $
        operational_carbon = energy * ci # in kg CO2
        capital_cost = machine_cost * (time / Lifetime)
        capital_carbon = machine_embodied * (time / Lifetime)
        mvis = benchmark['mvis']
        
        results.append({
            'Image Size': benchmark['im_size'],
            'Timesteps': benchmark['n_times'],
            'Channels': benchmark['n_chans'],
            'Machine': machine_name,
            'Location': location_id,
            'Mvis': mvis,
            'Time (s)': time * 3600,
            'Dynamic Energy (Wh)': energy_dynamic * 1e3,
            'Static Energy (Wh)': energy_static * 1e3,
            'Energy (Wh)': energy * 1e3,
            'Power (W)': energy * 1e3 / time,
            'Operational Carbon (g CO2)': operational_carbon * 1e3,
            'Embodied Carbon (g CO2)': capital_carbon * 1e3,
            'Total Carbon (g CO2)': (operational_carbon + capital_carbon) * 1e3,
            'Operational Cost ($)': operational_energy_cost,
            'Capital Cost ($)': capital_cost,
            'Total Cost ($)': operational_energy_cost + capital_cost,
            'Mvis/h': mvis / time,
            'Mvis/kWh': mvis / energy,
            'Mvis/kgCO2': mvis / (operational_carbon + capital_carbon),
            'Mvis/$': mvis / (operational_energy_cost + capital_cost),
        })

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Sort by Location, then Benchmark, then Machine
results_df = results_df.sort_values(by=[
    'Location',
    'Image Size',
    'Mvis',
    'Channels',
    'Timesteps',
    'Machine',
    ]).reset_index(drop=True)

# Create a display DataFrame with percentages for operational and capital, absolute for totals
display_df = results_df.copy()
display_df['Operational Carbon (%)'] = (display_df['Operational Carbon (g CO2)'] / display_df['Total Carbon (g CO2)'] * 100).round(1)
display_df['Embodied Carbon (%)'] = (display_df['Embodied Carbon (g CO2)'] / display_df['Total Carbon (g CO2)'] * 100).round(1)
display_df['Operational Cost (%)'] = (display_df['Operational Cost ($)'] / display_df['Total Cost ($)'] * 100).round(1)
display_df['Capital Cost (%)'] = (display_df['Capital Cost ($)'] / display_df['Total Cost ($)'] * 100).round(1)
display_df['Dynamic Power (%)'] = (display_df['Dynamic Energy (Wh)'] / display_df['Energy (Wh)'] * 100).round(1)
display_df['Static Power (%)'] = (display_df['Static Energy (Wh)'] / display_df['Energy (Wh)'] * 100).round(1)

# Print ranges for each numerical field
print("=" * 150)
print("FIELD RANGES")
print("=" * 150)
numerical_cols = [
    'Operational Carbon (%)',
    'Embodied Carbon (%)',
    'Operational Cost (%)',
    'Capital Cost (%)',
    'Dynamic Power (%)',
    'Static Power (%)',
]
for col in numerical_cols:
    min_val = display_df[col].min()
    max_val = display_df[col].max()
    print(f"{col:40s}: {min_val:12.2f} to {max_val:12.2f}")
print()

display_df['Mvis/h'] = display_df['Mvis/h'].round(2)
display_df['Mvis/kWh'] = display_df['Mvis/kWh'].round(2)
display_df['Mvis/kgCO2'] = display_df['Mvis/kgCO2'].round(2)
display_df['Mvis/$'] = display_df['Mvis/$'].round(2)
display_df['Mvis'] = display_df['Mvis'].round(2)
display_df['Energy (Wh)'] = display_df['Energy (Wh)'].round(2)
display_df['Time (s)'] = display_df['Time (s)'].round(2)
display_df['Total Carbon (g CO2)'] = display_df['Total Carbon (g CO2)'].round(2)
display_df['Total Cost ($)'] = display_df['Total Cost ($)'].round(2)
display_df['Power (W)'] = display_df['Power (W)'].round(2)

# Create the final display table with the desired columns
final_display_df = display_df[[
    'Image Size',
    'Timesteps',
    'Channels',
    # 'Machine',
    'Location',
    'Mvis',
    'Mvis/h',
    'Mvis/kWh',
    'Mvis/kgCO2',
    'Mvis/$',
    'Power (W)',
    'Time (s)',
    'Energy (Wh)',
    # 'Dynamic Energy (Wh)',
    # 'Static Energy (Wh)',
    'Dynamic Power (%)',
    'Static Power (%)',
    'Operational Carbon (%)',
    'Embodied Carbon (%)',
    'Total Carbon (g CO2)',
    'Operational Cost (%)',
    'Capital Cost (%)',
    'Total Cost ($)',
]].copy()

print("=" * 150)
print("COMPREHENSIVE BENCHMARKS TABLE")
print("=" * 150)
print(final_display_df.to_string(index=False))
print()

# Function to convert DataFrame to LaTeX table with location separators
def dataframe_to_latex(df, caption, label, index=True, separate_by_location=False):
    """Convert DataFrame to LaTeX table format
    
    Args:
        df: DataFrame to convert
        caption: Table caption
        label: Table label for references
        index: Whether to include index as first column
        separate_by_location: Whether to add horizontal lines separating locations (for comprehensive table)
    """
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    
    # Get column names
    if index:
        columns = list(df.index.names) + list(df.columns)
    else:
        columns = list(df.columns)
    
    # Build table column specification with vertical separators
    # All carbon columns first, then all cost columns
    num_cols = len(columns)
    col_spec = ''
    
    if separate_by_location:
        # For comprehensive table: c|c|c|c|c | c|c|c | c|c|c
        # Benchmark|Machine|Location|Time|Energy | OpCarb%|CapCarb%|TotCarb | OpCost%|CapCost%|TotCost
        col_spec = 'c' * 5 + '|' + 'c' * 3 + '|' + 'c' * 3
    else:
        # For summary table: c | c|c|c | c|c|c
        # Benchmark | OpCarb%|CapCarb%|TotCarb | OpCost%|CapCost%|TotCost
        col_spec = 'c|' + 'c' * 3 + '|' + 'c' * 3
    
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\hline')
    lines.append(' & '.join(columns) + r' \\')
    lines.append(r'\hline')
    
    # Add data rows
    if separate_by_location:
        # Add horizontal lines separating locations
        prev_location = None
        if index:
            for idx, row in df.iterrows():
                current_location = row['Location']
                if prev_location is not None and current_location != prev_location:
                    lines.append(r'\hline')
                row_data = [str(idx)] + [f'{v:.2f}' if isinstance(v, (int, float)) else str(v) for v in row.values]
                lines.append(' & '.join(row_data) + r' \\')
                prev_location = current_location
        else:
            for _, row in df.iterrows():
                current_location = row['Location']
                if prev_location is not None and current_location != prev_location:
                    lines.append(r'\hline')
                row_data = [f'{v:.2f}' if isinstance(v, (int, float)) else str(v) for v in row.values]
                lines.append(' & '.join(row_data) + r' \\')
                prev_location = current_location
    else:
        # No location separation for summary table
        if index:
            for idx, row in df.iterrows():
                row_data = [str(idx)] + [f'{v:.2f}' if isinstance(v, (int, float)) else str(v) for v in row.values]
                lines.append(' & '.join(row_data) + r' \\')
        else:
            for _, row in df.iterrows():
                row_data = [f'{v:.2f}' if isinstance(v, (int, float)) else str(v) for v in row.values]
                lines.append(' & '.join(row_data) + r' \\')
    
    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\caption{' + caption + '}')
    lines.append(r'\label{' + label + '}')
    lines.append(r'\end{table}')
    
    return '\n'.join(lines)

# Generate LaTeX table for comprehensive results
latex_table_comprehensive = dataframe_to_latex(
    final_display_df,
    caption='Benchmarks with Operational and Capital Expenditures (Percentages and Totals)',
    label='tab:benchmarks_comprehensive',
    index=False,
    separate_by_location=True
)

# print("=" * 150)
# print("LATEX TABLE (Comprehensive)")
# print("=" * 150)
# print(latex_table_comprehensive)
# print()

# Save LaTeX tables to files
with open('benchmarks_comprehensive.tex', 'w') as f:
    f.write(latex_table_comprehensive)

print("LaTeX tables saved to:")
print("  - benchmarks_comprehensive.tex")
print("  - benchmarks_summary.tex")
