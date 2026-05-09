#!/bin/bash

#-----------------------------------------------------------------------------
# Minimal Bash script to run a SKA-Low simulation with sources from the GLEAM
# catalogue. See README for pre-requisites.
#
# Load your computing environments here
# (e.g. modules, Spack environment, Python virtual env, ...)
#
# module load ...
# source VENV/bin/activate
#
#-----------------------------------------------------------------------------

set -e


# OSKAR SKA-Low telescope model (TM)
# ----------------------------------
TM="skalow.tm"
[ -d "$TM" ] || { echo "Error: directory '$TM' does not exist" >&2; exit 1; }
echo "-I- OSKAR TM: $TM"


# Read telescope's position (lat/long)
# ------------------------------------
[ -f "$TM/position.txt" ] || { echo "Error: directory '$TM/position.txt' does not exist" >&2; exit 1; }
read -r latlong < ${TM}/position.txt
IFS=' '
read -a TEL_LATLONG <<< "$latlong"


# Python simulation script
# ------------------------
SIM_PY="oskar_gleam_simulation.py"
[ -f "$SIM_PY" ] || { echo "Error: Python simulation script '$SIM_PY' does not exist" >&2; exit 1; }
echo "-I- Simulation script: $SIM_PY"


# Simulation parameters
# ---------------------
PHASE_CENTER_RA_DEG=25.0
PHASE_CENTER_DEC_DEG=-30.0
FP_PRECISION='single'
LENGTH_SEC=3600
START_FREQUENCY_HZ=151000000
START_FREQUENCY_MHZ=$((START_FREQUENCY_HZ / 1000000))
FOV_DEG=40


for N_TIMES in 1 8; do #64 128 256; do
    
    for N_CHANS in 1 8; do #64 128 256; do

        MS_BASENAME="oskar_skalow_gleam_${N_TIMES}t_${START_FREQUENCY_MHZ}MHz_${N_CHANS}c_${FP_PRECISION}"

        CMD="python $SIM_PY"
        CMD+=" --telescope_lon ${TEL_LATLONG[0]}"
        CMD+=" --telescope_lat ${TEL_LATLONG[1]}"
        CMD+=" --num_time_steps ${N_TIMES}"
        CMD+=" --input_directory ${TM}"
        CMD+=" --phase_centre_ra_deg ${PHASE_CENTER_RA_DEG}"
        CMD+=" --phase_centre_dec_deg ${PHASE_CENTER_DEC_DEG}"
        CMD+=" --fov_deg ${FOV_DEG}"
        CMD+=" --precision $FP_PRECISION"
        CMD+=" --start_frequency_hz $START_FREQUENCY_HZ"
        CMD+=" --frequency_inc_hz 1000"
        CMD+=" --num_channels ${N_CHANS}"
        #CMD+=" --use_gpus"                 # enable if you have access to GPUs
        CMD+=" --length_sec $LENGTH_SEC"
        CMD+=" --out_name $MS_BASENAME"
        echo "-I- CMD = $CMD"

        time $CMD
        
    done
done
