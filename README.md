# uvs_simulator_dashboard01
Primer dashboard sin obstáculos (claude)

## Dashboards

Each dashboard lives in its own folder, uses the simulator modules of this root
folder and has its own README and `requirements.txt`.

| Folder | Purpose | Launch (from this folder) |
| --- | --- | --- |
| `dashboard/` | Trajectory tracking under disturbances (SMC or simple feedback) | `python -m dashboard` |
| `dashboard02_smc/` | Sliding mode control lab: surge and yaw SMC parameters, phase plane with the boundary layer, error and chattering indices, current run vs previous run | `python -m dashboard02_smc` |
