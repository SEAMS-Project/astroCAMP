# OSKAR Synthetic Datasets

## Overview

This directory contains synthetic visibility datasets for SKA-Low generated using OSKAR (Oxford SKA Simulator & Kolmogorov Analyser for Radio astronomy). These datasets provide realistic simulations of radio interferometric observations for algorithm development and benchmarking.

## Creating Synthetic Datasets for SKA Low with OSKAR

### Prerequisites

- OSKAR software installed ([https://github.com/OxfordSKA/OSKAR](https://github.com/OxfordSKA/OSKAR))
- SKA-Low telescope configuration files
- Sky model (sources or FITS cube)

### Dataset Generation Process

#### 1. Telescope Configuration

Configure OSKAR with SKA-Low specifications:
- **Array layout**: SKA-Low core stations (typically 512 stations in Western Australia configuration)
- **Station beam model**: Aperture array beam patterns
- **Frequency range**: 50-350 MHz (SKA-Low operational band)
- **Channel bandwidth**: Configurable, typically 100 kHz - 1 MHz
- **Polarization**: Full Stokes (XX, XY, YX, YY)

#### 2. Observation Settings

Set observation parameters in OSKAR:
```
Start time: [MJD or date/time]
Duration: [hours]
Time integration: [seconds, typically 1-10s]
Number of channels: [frequency channels]
Phase center: [RA, Dec coordinates]
```

#### 3. Sky Model Input

Provide a sky model in one of the following formats:
- **Point sources**: ASCII text file with RA, Dec, flux, spectral index
- **FITS cube**: 3D data cube for extended emission
- **OSKAR sky model**: Native OSKAR format with multiple components

#### 4. Run OSKAR Simulation

Execute the simulator:
```bash
oskar_sim_interferometer settings_file.ini
```

#### 5. Output Products

OSKAR generates visibility data in Measurement Set (MS) format containing:
- Complex visibilities (u,v,w coordinates)
- Baseline metadata
- Frequency and time axis information
- Antenna configuration

### Key Parameters for SKA-Low Simulations

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Frequency range | 50-350 MHz | SKA-Low band |
| Number of stations | 512 | Full SKA-Low core |
| Integration time | 1-10 s | Balance between data volume and time resolution |
| Bandwidth | 100 kHz - 1 MHz | Per channel |
| Observation duration | 4-12 hours | Track synthesis imaging |

### Example Configuration

A typical OSKAR settings file for SKA-Low might include:
```ini
[telescope]
input_directory = ska_low_config/

[observation]
start_frequency_hz = 50e6
num_channels = 256
frequency_inc_hz = 1e6
start_time_utc = 2025-01-01 00:00:00
length = 4h
num_time_steps = 1440

[interferometer]
channel_bandwidth_hz = 1e6
time_average_sec = 10
oskar_vis_filename = ska_low_synthetic.ms
```

### Dataset Validation

After generation, validate the dataset:
- Check visibility amplitude distributions
- Verify u,v coverage completeness
- Inspect frequency and time sampling
- Validate metadata consistency

## References

- OSKAR Documentation: [https://ska-telescope.gitlab.io/sim/oskar/](https://ska-telescope.gitlab.io/sim/oskar/)
- SKA-Low Configuration: SKA Observatory technical documentation
- Synthetic datasets derived from realistic telescope configurations for algorithm benchmarking
