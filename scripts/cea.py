import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Carbon and Economic analysis.')
parser.add_argument('-l', '--lifetime', type=int, default=5, help='Lifetime in years (default: 5)')
args = parser.parse_args()

print(f"Using lifetime of {args.lifetime} years for all machines.")

# Parameters
Lifetime = args.lifetime * 365 * 24  # Lifetime in hours

# Read the three CSV files
machines_df = pd.read_csv('machines.csv').set_index('machine')
locations_df = pd.read_csv('locations.csv')
benchmarks_df = pd.read_csv('benchmarks.csv')

# Create a comprehensive table with benchmarks in rows
# and operational and capital expenditures (carbon and costs) in columns
results = []

for _, benchmark in benchmarks_df.iterrows():
    benchmark_name = benchmark['benchmark']
    machine_name = benchmark['machine']  # Machine is now specified in the benchmark
    time = benchmark['time'] / 3600 # from seconds to hours
    energy = benchmark['energy'] / 3.6e6 # from J to kWh (1 kWh = 3.6e6 J)
    
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
        operational_energy_cost = energy * ep
        operational_carbon = energy * ci
        capital_cost = machine_cost * (time / Lifetime)
        capital_carbon = machine_embodied * (time / Lifetime)
        
        results.append({
            'Benchmark': benchmark_name,
            'Machine': machine_name,
            'Location': location_id,
            'Time (h)': time,
            'Energy (kWh)': energy,
            'Operational Carbon (kg CO2)': operational_carbon,
            'Embodied Carbon (kg CO2)': capital_carbon,
            'Total Carbon (kg CO2)': operational_carbon + capital_carbon,
            'Operational Cost ($)': operational_energy_cost,
            'Capital Cost ($)': capital_cost,
            'Total Cost ($)': operational_energy_cost + capital_cost
        })

# Create DataFrame from results
results_df = pd.DataFrame(results)

# Sort by Location, then Benchmark, then Machine
results_df = results_df.sort_values(by=['Location', 'Benchmark', 'Machine']).reset_index(drop=True)

# Create a display DataFrame with percentages for operational and capital, absolute for totals
display_df = results_df.copy()
display_df['Operational Carbon (%)'] = (display_df['Operational Carbon (kg CO2)'] / display_df['Total Carbon (kg CO2)'] * 100).round(1)
display_df['Embodied Carbon (%)'] = (display_df['Embodied Carbon (kg CO2)'] / display_df['Total Carbon (kg CO2)'] * 100).round(1)
display_df['Operational Cost (%)'] = (display_df['Operational Cost ($)'] / display_df['Total Cost ($)'] * 100).round(1)
display_df['Capital Cost (%)'] = (display_df['Capital Cost ($)'] / display_df['Total Cost ($)'] * 100).round(1)

# Create the final display table with the desired columns
final_display_df = display_df[[
    'Benchmark', 'Machine', 'Location', 'Time (h)', 'Energy (kWh)',
    'Operational Carbon (%)', 'Embodied Carbon (%)', 'Total Carbon (kg CO2)',
    'Operational Cost (%)', 'Capital Cost (%)', 'Total Cost ($)'
]].copy()
print("=" * 150)
print("COMPREHENSIVE BENCHMARKS TABLE")
print("=" * 150)
print(final_display_df.to_string(index=False))
print()

# Create a summary table grouped by benchmark
summary_by_benchmark = results_df.groupby('Benchmark').agg({
    'Operational Carbon (kg CO2)': 'mean',
    'Embodied Carbon (kg CO2)': 'first',
    'Total Carbon (kg CO2)': 'mean',
    'Operational Cost ($)': 'mean',
    'Capital Cost ($)': 'first',
    'Total Cost ($)': 'mean'
}).round(2)

# Create summary display with percentages
summary_display = summary_by_benchmark.copy()
summary_display['Operational Carbon (%)'] = (summary_display['Operational Carbon (kg CO2)'] / summary_display['Total Carbon (kg CO2)'] * 100).round(1)
summary_display['Embodied Carbon (%)'] = (summary_display['Embodied Carbon (kg CO2)'] / summary_display['Total Carbon (kg CO2)'] * 100).round(1)
summary_display['Operational Cost (%)'] = (summary_display['Operational Cost ($)'] / summary_display['Total Cost ($)'] * 100).round(1)
summary_display['Capital Cost (%)'] = (summary_display['Capital Cost ($)'] / summary_display['Total Cost ($)'] * 100).round(1)

# Reorder summary display columns
summary_display = summary_display[[
    'Operational Carbon (%)', 'Embodied Carbon (%)', 'Total Carbon (kg CO2)',
    'Operational Cost (%)', 'Capital Cost (%)', 'Total Cost ($)'
]]

print("=" * 120)
print("SUMMARY BY BENCHMARK (Averaged across machines and locations)")
print("=" * 120)
print(summary_display.to_string())
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

print("=" * 150)
print("LATEX TABLE (Comprehensive)")
print("=" * 150)
print(latex_table_comprehensive)
print()

# Generate LaTeX table for summary by benchmark
latex_table_summary = dataframe_to_latex(
    summary_display,
    caption='Summary of Expenditures by Benchmark (Percentages and Totals)',
    label='tab:benchmarks_summary',
    index=True
)

print("=" * 150)
print("LATEX TABLE (Summary by Benchmark)")
print("=" * 150)
print(latex_table_summary)
print()

# Save LaTeX tables to files
with open('benchmarks_comprehensive.tex', 'w') as f:
    f.write(latex_table_comprehensive)

with open('benchmarks_summary.tex', 'w') as f:
    f.write(latex_table_summary)

print("LaTeX tables saved to:")
print("  - benchmarks_comprehensive.tex")
print("  - benchmarks_summary.tex")
