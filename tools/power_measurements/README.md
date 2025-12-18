# tools/power_measurements/

Scripts and utilities to measure power and energy during a benchmark run.

Possible tools:

- Intel RAPL wrappers (CPU package power)
- NVIDIA NVML polling scripts (GPU power)
- IPMI or PDU readers for node-level measurements

Each script should document:

- required privileges (e.g. sudo)
- sampling interval
- output format (e.g. timestamp, power in watts)

These power logs are later ingested by `metrics/energy_carbon.py` to compute
energy to solution (Ec) and carbon to solution (Cc).

## Power measurement on KRIA kr260 board
- required privilege : none.
- sampling interval : 1 second.
- output format : written once in console. All measures are displayed in mW or µW, the unit is always given.
The script `measure_power.py` measures the average power consumed by the board for a given timespan.
We used the standard instantaneous power measurement tool provided with Linux. We first confirmed its accuracy by using Xilinx's PowerStat tool.
The script first measures the base power consumption for 30 seconds. It is expected that the user will not start the computation load during this phase. The script then starts a second phase of measurement, during which the user must launch the computation load. During this phase, power is measured every second. Once the user stops the measurement, the average power is displayed alongside the power overhead associated with the load. 
