# Case study benchmark

Runs ADOPT case studies with resource profiling switched on and collects the
profiles into a single dataset. The goal is to relate the size of a model to
the resources a run needs, so that the resource usage of a case study can be
predicted from its size before it is submitted to a cluster.

## Layout

| Path | Content | Tracked |
|---|---|---|
| `case_studies/` | One module per case study, defining its input data | yes |
| `run_benchmark.py` | Runner and dataset collector | yes |
| `plot_benchmark.py` | Figures of the collected profiles | yes |
| `benchmark_dataset.csv` | One row per run, the dataset of the study | yes |
| `figures/` | Figures of the study | yes |
| `inputData/` | Generated input data of the case studies | no |
| `results/` | Result folders of the runs, including the raw profiles | no |

## Case studies

| Name | Description |
|---|---|
| `network` | Two-nodal system, city and rural, from the documentation notebook |
| `four_node` | Four-nodal hydrogen system with two pressure levels, ported from the `four_node_configuration` study of AdOpT-NET0-RegToNation, with its Latin hypercube parameters frozen at the middle of their ranges |

## Usage

```bash
# A single run
python run_benchmark.py run --case network --typicaldays 30

# A sweep over the model size
python run_benchmark.py sweep --case network --typicaldays 5 10 30 60

# A single run with complexity knobs set
python run_benchmark.py run --case four_node --typicaldays 4 \
    --set bidirectional_precise=1 --set pipeline_capex=fixed_plus_linear

# Every combination of the knobs, each in its own process
python run_benchmark.py matrix --case four_node --typicaldays 2 4

# Only some knobs
python run_benchmark.py matrix --case four_node --typicaldays 4 \
    --knobs pipeline_capex bidirectional_precise

# Collect all runs done so far into benchmark_dataset.csv
python run_benchmark.py collect
```

## Complexity knobs of four_node

| Knob | Levels | What it changes |
|---|---|---|
| `typicaldays` | any | Number of typical days, 0 for full resolution |
| `typicaldays_method` | 1, 2 | 1 clusters model and data, 2 clusters only the model and keeps the data at 8760 steps |
| `electricity_price` | constant, fluctuating | Data only, no structural change |
| `hydrogen_demand` | constant, fluctuating | Data only, no structural change |
| `pipeline_capex` | linear, fixed_plus_linear | A fixed CAPEX term makes the arc CAPEX need a big-M disjunction |
| `bidirectional_precise` | 0, 1 | Precise bidirectionality adds binaries per arc **and per timestep** |
| `pipeline_size_min` | 0, 250 | A minimum size makes an arc either zero or at least size_min, one disjunction per arc |

`measure_complexity.py` builds the model for each knob without solving it and
reports how much each one adds:

```bash
python measure_complexity.py --case four_node --typicaldays 4
python measure_complexity.py --case four_node --typicaldays 4 --knobs bidirectional_precise
```

## One process per run

The `matrix` command starts a separate process for every configuration. This
is not optional bookkeeping, it is needed for the numbers to mean anything:

- `rss_peak_os_mb` is a high water mark of the whole process that never goes
  down. Two runs in one process would give the second one the peak of the
  first.
- Memory left behind by a previous run slows the next one down, which shows up
  as construction time that is not real.

`sweep` and repeated `run` calls in one process do not have this protection, so
use `matrix` whenever several configurations are compared.

## Figures

```bash
# Writes the three figures to figures/
python plot_benchmark.py

# Plot another metric of the timeseries over time
python plot_benchmark.py --metric cpu_percent

# Only some runs
python plot_benchmark.py --runs network_td5 network_td0
```

- `resource_curves_<metric>.png` — the metric over time for every run, with
  the phases shaded in their own colour
- `phase_durations.png` — duration per phase and run, absolute and relative
- `phase_scaling.png` — duration and peak memory per phase against the number
  of variables, log-log, with the fitted exponent in the legend

The phase colours are fixed in `PHASE_COLORS` and shared between the figures.
Plotting needs `matplotlib`, which is not a dependency of `adopt_net0` itself.

The input data of a case study is created once and then reused, so repeated
runs only rewrite the configuration. Climate data is downloaded from an
external api the first time a case study is set up, which needs an internet
connection.

## Output per run

Every run writes three csv files into its result folder, next to
`optimization_results.h5` and `solver_log.txt`:

- `profile_timeseries.csv` — one row per sample, the full resource curve
- `profile_phases.csv` — one row per phase, average and peak values
- `profile_summary.csv` — one row for the run, model size next to resources

`benchmark_dataset.csv` is the concatenation of all `profile_summary.csv`
files. The raw files are never modified or removed by the collector.

The phases are `read_data`, `preprocessing_checks`, `construct_model`,
`construct_balances`, `solver_setup`, `solve` and `write_results`. They are
stored as time intervals, so the timeseries can be split differently in
post-processing with `adopt_net0.diagnostics.load_run_profile`.

## Reading the numbers

- Use `cpu_user_s`, not `cpu_percent`. The sampling thread and the phase
  boundaries both call `cpu_percent()`, which shortens its measuring intervals
  and makes the percentages noisy. The `cpu_user_s` deltas are exact, and
  `parallelism_avg` in `profile_phases.csv` is derived from them.
- `solve_includes_translation` is 1 for every solver except
  `gurobi_persistent`. In that case the `solve` phase also holds the
  translation of the pyomo model into the gurobi model, which can be a large
  share of the phase. Keep it as a dummy in a regression, or filter on it.
- `rss_peak_os_mb` is the peak reported by the operating system and is exact.
  `rss_peak_sampled_mb` comes from the samples and can miss short spikes.
- Network counters are system wide, not specific to the process.
- On a cluster the `slurm_*` columns are filled from the environment. Validate
  the first run against `seff <jobid>`.