# Benchmark study

Written 2026-08-27 01:02.

## Machine

- hostname: aspenrds
- platform: Windows-2016Server-10.0.14393-SP0
- python: 3.12.2
- cpus_physical: 48
- cpus_logical: 48
- memory_total_gb: 878.9
- started: 2026-08-26T10:13:11
- gurobi: 13.0.0

## Runs

- 147 runs in total, 147 of `four_node`
- model size from 13,116 to 2,961,285 variables
- wall time from 23.1 s to 7482.7 s
- peak memory from 280 MB to 14072 MB

### Termination

- optimal: 144
- maxTimeLimit: 3

### Slowest runs

| run | variables | binaries | gurobi [s] | B&B nodes | peak memory [MB] |
|---|---|---|---|---|---|
| `four_node_td0_pf_df` | 2,961,285 | 48 | 7201.2 | 1 | 14058 |
| `four_node_td0_pv0` | 2,908,715 | 48 | 7201.1 | 1 | 14072 |
| `four_node_td32_bp1_sm250` | 318,381 | 18,480 | 7201.1 | 11 | 6788 |
| `four_node_td32_sm250` | 299,949 | 48 | 4539.6 | 28 | 6374 |
| `four_node_td32_bp1` | 318,381 | 18,480 | 3049.8 | 1 | 7013 |
| `four_node_td16_bp1_sm250` | 181,293 | 9,264 | 2144.2 | 20 | 5128 |
| `four_node_td16_sm250` | 172,077 | 48 | 1509.5 | 22 | 4994 |
| `four_node_td0_pv0_st0` | 2,295,410 | 48 | 1415.7 | 1 | 9683 |
| `four_node_td4_bp1_df_cxfix_sm250` | 78,477 | 2,352 | 283.7 | 11 | 2110 |
| `four_node_td16_bp1` | 181,293 | 9,264 | 283.3 | 1 | 4962 |

## Figures

- `figures/four_node_cpu_utilisation.png`
- `figures/four_node_phase_durations.png`
- `figures/four_node_phase_scaling.png`
- `figures/four_node_resource_curves_cpu_percent.png`
- `figures/four_node_resource_curves_rss_mb.png`
- `figures/four_node_resources_vs_difficulty.png`
- `figures/four_node_scaling_by_complexity_gurobi_runtime_s.png`
- `figures/four_node_scaling_by_complexity_parallelism_solve.png`
- `figures/four_node_scaling_by_complexity_rss_peak_os_mb.png`
- `figures/four_node_scaling_by_complexity_wall_total_s.png`
- `figures/four_node_size_versus_difficulty.png`
- `figures/four_node_total_time.png`
- `figures/network_cpu_utilisation.png`
- `figures/network_phase_durations.png`
- `figures/network_phase_scaling.png`
- `figures/network_resource_curves_cpu_percent.png`
- `figures/network_resource_curves_rss_mb.png`
- `figures/network_scaling_by_complexity_gurobi_runtime_s.png`
- `figures/network_scaling_by_complexity_rss_peak_os_mb.png`
- `figures/network_scaling_by_complexity_wall_total_s.png`
- `figures/network_size_versus_difficulty.png`
- `figures/network_total_time.png`

## Files

- `benchmark_dataset.csv`: one row per run, the dataset of the study
- `results/<run>/profile_timeseries.csv`: the full resource curve
- `results/<run>/profile_phases.csv`: average and peak per phase
- `study.log`: what ran and how long it took
