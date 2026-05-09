#!/usr/bin/env python3
"""
Master script to regenerate all plots using the updated benchmarks.csv data structure.
Runs all visualization scripts in sequence and reports status.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Define all plotting scripts to run
PLOT_SCRIPTS = [
    "plot_key_takeaway.py",
    "plot_largest_bars_and_lines.py",
    "plot_largest_image_metrics.py",
    "plot_largest_image_metrics_comparison.py",
    "plot10b_carbon_cost_efficiency_facets_conference.py",
    "plot_performance_vs_workload.py",
    "plot_regime_heatmaps.py",
    "plot_lifetime_breakdown.py",
    "plot_lifetime_breakdown_comparison.py",
    "plot_paper_views.py",
    "plot17b_kernel_time_breakdown_with_disk.py",
    "plot25_cpu_roofline_stacking.py",
    "plot25b_cpu_roofline_scaling_gap.py",
    "plot25c_cpu_roofline_scalability_comparison.py",
    "plot25d_cpu_scalability_only.py",
    "plot25e_cpu_scalability_with_gpu_speedup.py",
    "plot25f_cpu_roofline_with_scaling_gap.py",
    "plot26_gpu_cpu_roofline_comparison.py",
    "plot26b_cpu_gpu_execution_comparison.py",
    "plot26b_cpu_execution_comparison_conference.py",
    "plot26_gpu_cpu_roofline_comparison_conference.py",
    "plot26d_cpu_gpu_execution_comparison_gibj.py",
    "plot26c_cpu_gpu_speedup_utilization.py",
    "plot27_hotspots_summary.py",
    "plot28_darshan_io_summary.py",
    "plot29_gpu_bytes_per_joule.py",
    "plot32_problem_footprint_gib_per_joule.py",
    "plot32b_idg_data_movement_gb_per_joule.py",
    "plot33_problem_sizes_memory_io_overview.py",
    "plot33b_problem_sizes_memory_io_heatmap.py",
    "plot34_problem_size_stacked_linear.py",
    "plot34b_large_problem_size_stacked_linear.py",
    "plot34c_large_problem_size_with_gbj.py",
    "plot34d_large_problem_size_with_gbj_by_c.py",
    "plot34e_large_problem_size_with_gbj_by_c_single_column.py",
    "plot19d_largest_image_kernel_wall_saturation_signals_active_by_c.py",
    "plot30_rebuttal_gpu_roofline_hotspots.py",
    "plot31_cpu_scaling_phase_limits.py",
    "plot31b_cpu_scaling_gap_only.py",
    "table_problem_sizes_memory.py",
    "table_problem_sizes_memory_darshan.py",
    "cea.py",
]

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Create a log file for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = RESULTS_DIR / f"regeneration_log_{timestamp}.txt"

def log_message(msg, print_to_console=True):
    """Log message to file and optionally print to console."""
    with open(log_file, 'a') as f:
        f.write(msg + '\n')
    if print_to_console:
        print(msg)

def run_script(script_name):
    """Run a single plotting script and return success status."""
    script_path = BASE_DIR / script_name
    
    if not script_path.exists():
        log_message(f"❌ {script_name}: Script not found at {script_path}")
        return False
    
    log_message(f"\n{'='*80}")
    log_message(f"Running: {script_name}")
    log_message(f"{'='*80}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per script
        )
        
        if result.returncode == 0:
            log_message(f"✓ {script_name}: SUCCESS")
            # Log the last few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')[-5:]
                for line in lines:
                    log_message(f"  {line}")
            return True
        else:
            log_message(f"❌ {script_name}: FAILED (exit code {result.returncode})")
            if result.stderr:
                log_message("STDERR output:")
                for line in result.stderr.strip().split('\n')[-10:]:
                    log_message(f"  {line}")
            if result.stdout:
                log_message("STDOUT output (last 10 lines):")
                for line in result.stdout.strip().split('\n')[-10:]:
                    log_message(f"  {line}")
            return False
            
    except subprocess.TimeoutExpired:
        log_message(f"⏱ {script_name}: TIMEOUT (exceeded 5 minutes)")
        return False
    except Exception as e:
        log_message(f"❌ {script_name}: ERROR - {str(e)}")
        return False

def main():
    """Run all plotting scripts."""
    log_message("=" * 80)
    log_message("ASTROCAMP PLOT REGENERATION - STARTED")
    log_message("=" * 80)
    log_message(f"Timestamp: {datetime.now().isoformat()}")
    log_message(f"Base directory: {BASE_DIR}")
    log_message(f"Results directory: {RESULTS_DIR}")
    log_message(f"Log file: {log_file}")
    log_message("")
    
    results = {}
    start_time = datetime.now()
    
    # Run all scripts
    for i, script_name in enumerate(PLOT_SCRIPTS, 1):
        log_message(f"\n[{i}/{len(PLOT_SCRIPTS)}] ", print_to_console=False)
        success = run_script(script_name)
        results[script_name] = success
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    log_message("\n" + "=" * 80)
    log_message("REGENERATION SUMMARY")
    log_message("=" * 80)
    log_message(f"Total scripts: {len(PLOT_SCRIPTS)}")
    log_message(f"Successful: {successful}")
    log_message(f"Failed: {failed}")
    log_message(f"Duration: {duration}")
    log_message("")
    
    log_message("Results by script:")
    for script_name, success in results.items():
        status = "✓ SUCCESS" if success else "❌ FAILED"
        log_message(f"  {status:15s} {script_name}")
    
    log_message("")
    log_message("=" * 80)
    
    if failed == 0:
        log_message("All plots regenerated successfully!")
        print(f"\n✓ All plots regenerated successfully!")
        print(f"Log saved to: {log_file}")
        return 0
    else:
        log_message(f"WARNING: {failed} script(s) failed. Check log for details.")
        print(f"\n⚠ {failed} script(s) failed. See log for details:")
        print(f"Log saved to: {log_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
