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
