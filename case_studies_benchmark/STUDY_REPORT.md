# Benchmark study

Written 2026-08-25 17:06.

## Machine

- hostname: UU200005
- platform: Windows-11-10.0.26200-SP0
- python: 3.12.7
- cpus_physical: 12
- cpus_logical: 14
- memory_total_gb: 15.5
- started: 2026-08-25T17:05:42
- gurobi: 13.0.0

## Runs

- 27 runs in total, 21 of `four_node`
- model size from 12,818 to 106,931 variables
- wall time from 5.6 s to 526.6 s
- peak memory from 353 MB to 2094 MB

### Termination

- optimal: 21

### Slowest runs

| run | variables | binaries | gurobi [s] | B&B nodes | peak memory [MB] |
|---|---|---|---|---|---|
| `four_node_td4_pf_df_cxfix_sm250` | 76,173 | 48 | 497.6 | 35 | 1792 |
| `four_node_td4_bp1_pf_df_cxfix_sm250` | 78,477 | 2,352 | 378.5 | 1 | 2094 |
| `four_node_td4_pf_df_cxfix` | 76,173 | 48 | 67.2 | 1 | 1213 |
| `four_node_td4_pf_df` | 76,125 | 0 | 38.8 | 0 | 976 |
| `four_node_td4_bp1_pf_df_cxfix_sm250_sp1` | 79,437 | 3,312 | 30.6 | 1 | 1423 |
| `four_node_td4_df_pv0` | 75,539 | 0 | 28.4 | 0 | 979 |
| `four_node_td4_pf_df_pv0` | 75,539 | 0 | 28.1 | 0 | 979 |
| `four_node_td4_pf_pv0` | 75,539 | 0 | 26.1 | 0 | 1012 |
| `four_node_td2_pf_df_pv0` | 59,843 | 0 | 24.1 | 0 | 730 |
| `four_node_td4_pv0` | 75,539 | 0 | 23.5 | 0 | 970 |

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
