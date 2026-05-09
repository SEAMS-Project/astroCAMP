#!/usr/bin/env python3
"""
Helper module to generate results DataFrame using cea.py logic.
This ensures consistent calculations across all plotting scripts.
"""

import pandas as pd
from pathlib import Path


def generate_results_dataframe(benchmarks_csv_path, machines_csv_path, locations_csv_path, 
                               lifetime_years=5, location_ids=None):
    """
    Generate results DataFrame with all derived metrics following cea.py logic.
    
    Parameters:
    -----------
    benchmarks_csv_path : str or Path
        Path to benchmarks.csv
    machines_csv_path : str or Path
        Path to machines.csv
    locations_csv_path : str or Path
        Path to locations.csv
    lifetime_years : int
        Lifetime in years (default: 5)
    location_ids : list, optional
        List of location IDs to include (default: ['WA', 'SA'])
    
    Returns:
    --------
    pd.DataFrame
        Results dataframe with all calculated metrics
    """
    
    if location_ids is None:
        location_ids = ['WA', 'SA']
    
    # Convert paths to Path objects if needed
    benchmarks_csv_path = Path(benchmarks_csv_path)
    machines_csv_path = Path(machines_csv_path)
    locations_csv_path = Path(locations_csv_path)
    
    # Load benchmarks
    benchmarks_df = pd.read_csv(
        benchmarks_csv_path,
        header=None,
        names=[
            "im_size", "n_times", "n_chans", "wall_time", "wall_time_sec", "n_rows", "n_vis",
            "n_idg",
            "idg_h_sec", "idg_h_watt", "idg_h_jou",
            "idg_d_sec", "idg_d_watt", "idg_d_jou",
            "idg_grid_mvs",
            "cpu_j", "cpu_bsl_j", "cpu_bsl_std_j",
            "gpu0_j", "gpu1_j", "gpu2_j", "gpu3_j",
            "gpu_j", "gpu_bsl_j", "gpu_bsl_std_j",
            "tot_sys_j",
            "tot_pdu_j", "pdu_bsl_j", "pdu_bsl_std_j",
            "abs_cpu_j", "abs_gpu_j", "abs_pdu_j"
        ],
    )
    
    # Load machines and locations
    machines_df = pd.read_csv(machines_csv_path).set_index('machine')
    locations_df = pd.read_csv(locations_csv_path).set_index('id').reset_index()
    locations_df = locations_df[locations_df['id'].isin(location_ids)]
    
    # Constants
    Lifetime = lifetime_years * 365 * 24  # Lifetime in hours
    machine_name = 'R675 V3 + 4xH100 96GB'
    idle_pdu_watt = benchmarks_df['pdu_bsl_j'].mean()  # Baseline power
    
    # Assign machine to all benchmarks
    benchmarks_df['machine'] = machine_name
    benchmarks_df['mvis'] = benchmarks_df['n_vis'] / 1e6
    
    # Get machine parameters
    machine_cost = machines_df.loc[machine_name, 'cost']
    machine_embodied = machines_df.loc[machine_name, 'embodied']
    
    # Build results list
    results = []
    
    for _, benchmark in benchmarks_df.iterrows():
        benchmark_name = f"{benchmark['im_size']}_{benchmark['n_times']}_{benchmark['n_chans']}"
        time_hours = benchmark['wall_time_sec'] / 3600  # Convert to hours
        
        # Energy calculations (matching cea.py)
        energy_dynamic = benchmark['tot_sys_j'] / 3.6e6  # Dynamic energy in kWh
        energy_static = idle_pdu_watt / 4 * time_hours / 1000  # Static energy in kWh
        energy = energy_dynamic + energy_static  # Total energy in kWh
        
        mvis = benchmark['mvis']
        
        # Iterate over locations
        for _, location in locations_df.iterrows():
            location_id = location['id']
            location_name = location['location']
            ci = location['ci']  # Carbon intensity in kg CO2/kWh
            ep = location['ep']  # Electricity price in $/kWh
            
            # Calculate operational and capital expenditures
            operational_energy_cost = energy * ep  # in $
            operational_carbon = energy * ci  # in kg CO2
            capital_cost = machine_cost * (time_hours / Lifetime)
            capital_carbon = machine_embodied * (time_hours / Lifetime)
            
            results.append({
                'Image Size': benchmark['im_size'],
                'Timesteps': benchmark['n_times'],
                'Channels': benchmark['n_chans'],
                'Machine': machine_name,
                'Location': location_id,
                'Mvis': mvis,
                'Time (s)': benchmark['wall_time_sec'],
                'Dynamic Energy (Wh)': energy_dynamic * 1e3,
                'Static Energy (Wh)': energy_static * 1e3,
                'Energy (Wh)': energy * 1e3,
                'Power (W)': energy * 1e3 / time_hours,
                'Operational Carbon (g CO2)': operational_carbon * 1e3,
                'Embodied Carbon (g CO2)': capital_carbon * 1e3,
                'Total Carbon (g CO2)': (operational_carbon + capital_carbon) * 1e3,
                'Operational Cost ($)': operational_energy_cost,
                'Capital Cost ($)': capital_cost,
                'Total Cost ($)': operational_energy_cost + capital_cost,
                'Mvis/h': mvis / time_hours,
                'Mvis/kWh': mvis / energy,
                'Mvis/kgCO2': mvis / (operational_carbon + capital_carbon),
                'Mvis/$': mvis / (operational_energy_cost + capital_cost),
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Location, Image Size, Mvis
    results_df = results_df.sort_values(by=[
        'Location',
        'Image Size',
        'Mvis',
        'Channels',
        'Timesteps',
    ]).reset_index(drop=True)
    
    return results_df
