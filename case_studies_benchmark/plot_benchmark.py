"""
Plots the resource profiles collected by run_benchmark.py.

Three figures are written to the figures folder:

- resource_curves: the resource curve of every run over time, with the phases
  of the run shaded in different colours
- phase_durations: the duration of every phase per run, as a stacked bar
- phase_scaling: duration and peak memory per phase against the number of
  variables, on a log-log scale
- size_versus_difficulty: the size of the problem next to how hard it was to
  solve, which are not the same thing
- total_time: total wall time against the number of variables, with a fitted
  power law
- resources_vs_difficulty: cpu, memory and parallelism of the solve phase
  against runtime, simplex iterations and branch and bound nodes, for runs of
  a comparable size

The scaling with complexity is drawn as four figures rather than one. A single
figure with one line per knob combination needs sixty six lines, which cannot
be read, so the combinations are summarised instead:

- scaling_envelope: the median resource against the model size, with the
  spread over all modelling choices shaded around it. Answers how much of a
  resource is set by the size and how much is still open at that size
- scaling_by_knob: the same median and spread, split into the runs that have
  a knob on and the runs that have it off, one panel per knob. Answers which
  modelling choice separates the two bands
- knob_impact: the paired ratio of every run against the run that differs only
  in one knob. Answers what a modelling choice multiplies a resource by
- scaling_ladder: the eight configurations of the scaling stage drawn as
  explicit lines. The concrete trajectories behind the bands

Examples::

    python plot_benchmark.py
    python plot_benchmark.py --metric cpu_percent
    python plot_benchmark.py --metric rss_mb --runs network_td5 network_td40
    python plot_benchmark.py --case four_node --no-profile-check
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adopt_net0.diagnostics import load_run_profile

BASE = Path(__file__).parent
RESULTS_PATH = BASE / "results"
DATASET_FILE = BASE / "benchmark_dataset.csv"
FIGURES_PATH = BASE / "figures"

# The phases are always drawn in this order and with these colours, so that
# they can be compared between figures
PHASE_COLORS = {
    "read_data": "#4c72b0",
    "preprocessing_checks": "#8172b2",
    "construct_model": "#dd8452",
    "construct_balances": "#c44e52",
    "solver_setup": "#937860",
    "solve": "#55a868",
    "write_results": "#da8bc3",
}

# Resources that can go on the y axis of the complexity scaling figure
RESOURCE_LABELS = {
    "rss_peak_os_mb": "Peak memory of the run [MB]",
    "rss_peak_solve_mb": "Peak memory of the solve phase [MB]",
    "wall_total_s": "Total wall time [s]",
    "t_solve_s": "Solve phase [s]",
    "gurobi_runtime_s": "Gurobi runtime [s]",
    "cpu_user_s": "User CPU time of the run [s]",
    "cpu_user_solve_s": "User CPU time of the solve phase [s]",
    "parallelism_avg": "Cores kept busy [-]",
    "parallelism_solve": "Cores kept busy during the solve [-]",
    "t_construct_model_s": "Model construction [s]",
}

METRIC_LABELS = {
    "rss_mb": "Memory (RSS) [MB]",
    "vms_mb": "Virtual memory [MB]",
    "cpu_percent": "CPU [%]",
    "cpu_user_s": "Cumulative user CPU time [s]",
    "num_threads": "Threads [-]",
    "io_read_mb": "Cumulative disk read [MB]",
    "io_write_mb": "Cumulative disk write [MB]",
    "sys_mem_available_mb": "Available system memory [MB]",
}


def load_dataset(runs: list = None, case: str = None):
    """
    Loads the collected dataset and keeps only runs whose profile still exists

    :param list runs: optional list of case names to keep
    :param str case: optional case study to keep, matched on the case name
    :return: pandas DataFrame with one row per run, sorted by model size
    """
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"{DATASET_FILE} not found, run 'python run_benchmark.py collect' first"
        )

    dataset = pd.read_csv(DATASET_FILE)

    if case:
        dataset = dataset[dataset["case_name"].str.startswith(case)]
        if dataset.empty:
            raise ValueError(f"No runs of case study '{case}' in {DATASET_FILE}")

    if runs:
        dataset = dataset[dataset["case_name"].isin(runs)]
        if dataset.empty:
            raise ValueError(f"None of the runs {runs} found in {DATASET_FILE}")

    available = dataset["run_folder"].apply(
        lambda folder: (Path(folder) / "profile_timeseries.csv").exists()
    )
    if not available.all():
        missing = (~available).sum()
        print(f"Skipping {missing} run(s) whose raw profile is no longer on disk")

    return dataset[available].sort_values("n_vars").reset_index(drop=True)


def plot_resource_curves(dataset: pd.DataFrame, metric: str = "rss_mb"):
    """
    Plots the resource curve of every run, with the phases shaded

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :param str metric: column of profile_timeseries.csv that is plotted
    :return: Path of the figure
    """
    n_runs = len(dataset)
    n_columns = min(3, n_runs)
    n_rows = int(np.ceil(n_runs / n_columns))

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(5.2 * n_columns, 3.4 * n_rows),
        squeeze=False,
    )

    used_phases = []
    for index, run in dataset.iterrows():
        axis = axes[index // n_columns][index % n_columns]
        timeseries = load_run_profile(run["run_folder"])
        phases = pd.read_csv(Path(run["run_folder"]) / "profile_phases.csv")

        if metric not in timeseries.columns:
            axis.set_visible(False)
            continue

        # The phases are shaded in the background, the curve is drawn on top
        for _, phase in phases.iterrows():
            color = _phase_color(phase["phase"])
            axis.axvspan(
                phase["t_start"], phase["t_end"], color=color, alpha=0.25, lw=0
            )
            if phase["phase"] not in used_phases:
                used_phases.append(phase["phase"])

        axis.plot(timeseries["t"], timeseries[metric], color="black", lw=1.2)

        axis.set_title(
            f"{_short_label(run).replace(chr(10), ' ')}\n"
            f"{run['n_vars']:,} vars, {run['n_constrs']:,} constrs",
            fontsize=9,
        )
        axis.set_xlabel("Time [s]")
        axis.set_ylabel(METRIC_LABELS.get(metric, metric))
        axis.margins(x=0)

    for index in range(n_runs, n_rows * n_columns):
        axes[index // n_columns][index % n_columns].set_visible(False)

    handles = [
        Patch(facecolor=_phase_color(phase), alpha=0.25, label=phase)
        for phase in _sorted_phases(used_phases)
    ]
    handles.append(Line2D([0], [0], color="black", lw=1.2, label=metric))
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(f"Resource usage per phase: {METRIC_LABELS.get(metric, metric)}")
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))

    return _save(figure, f"resource_curves_{metric}.png")


def plot_phase_durations(dataset: pd.DataFrame):
    """
    Plots the duration of every phase per run as a stacked bar

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :return: Path of the figure
    """
    phases = _phases_in_dataset(dataset, prefix="t_", suffix="_s")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    labels = [_short_label(run) for _, run in dataset.iterrows()]
    positions = np.arange(len(dataset))

    # Absolute durations
    bottom = np.zeros(len(dataset))
    for phase in phases:
        values = dataset[f"t_{phase}_s"].fillna(0).to_numpy()
        axes[0].bar(
            positions, values, bottom=bottom, color=_phase_color(phase), label=phase
        )
        bottom += values
    axes[0].set_ylabel("Duration [s]")
    axes[0].set_title("Wall time per phase")

    # Relative durations
    totals = np.where(bottom > 0, bottom, 1)
    bottom_relative = np.zeros(len(dataset))
    for phase in phases:
        values = 100 * dataset[f"t_{phase}_s"].fillna(0).to_numpy() / totals
        axes[1].bar(
            positions, values, bottom=bottom_relative, color=_phase_color(phase)
        )
        bottom_relative += values
    axes[1].set_ylabel("Share of wall time [%]")
    axes[1].set_title("Relative wall time per phase")
    axes[1].set_ylim(0, 100)

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=8)

    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Runs sorted by number of variables")
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    return _save(figure, "phase_durations.png")


def plot_phase_scaling(dataset: pd.DataFrame):
    """
    Plots duration and peak memory per phase against the number of variables

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :return: Path of the figure
    """
    if len(dataset) < 2:
        print("Skipping phase_scaling, it needs at least two runs")
        return None

    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    n_vars = dataset["n_vars"].to_numpy()

    # The phases are taken from the duration columns only. Deriving them from
    # the memory columns as well would pick up rss_peak_os_mb and
    # rss_peak_sampled_mb, which are run totals and not phases.
    phases = _phases_in_dataset(dataset, prefix="t_", suffix="_s")

    for axis, (prefix, suffix, label) in zip(
        axes,
        [
            ("t_", "_s", "Wall time of the phase [s]"),
            ("cpu_user_", "_s", "User CPU time of the phase [s]"),
            ("rss_peak_", "_mb", "Peak memory of the phase [MB]"),
        ],
    ):
        for phase in phases:
            column = f"{prefix}{phase}{suffix}"
            if column not in dataset.columns:
                continue

            values = dataset[column].to_numpy(dtype=float)
            valid = np.isfinite(values) & (values > 0)
            if valid.sum() < 2:
                continue

            axis.plot(
                n_vars[valid],
                values[valid],
                "o-",
                color=_phase_color(phase),
                label=_scaling_label(phase, n_vars[valid], values[valid]),
            )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Number of variables [-]")
        axis.set_ylabel(label)
        axis.legend(frameon=False, fontsize=8)
        axis.grid(True, which="both", alpha=0.2)

    figure.suptitle("Scaling per phase, exponent a of y ~ n_vars^a in the legend")
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    return _save(figure, "phase_scaling.png")


def _scaling_label(phase: str, n_vars: np.ndarray, values: np.ndarray):
    """
    Builds a legend label holding the fitted scaling exponent

    :param str phase: name of the phase
    :param np.ndarray n_vars: number of variables per run
    :param np.ndarray values: metric per run
    :return: str legend label
    """
    exponent = np.polyfit(np.log(n_vars), np.log(values), 1)[0]
    return f"{phase} (a={exponent:.2f})"


def _phases_in_dataset(dataset: pd.DataFrame, prefix: str, suffix: str):
    """
    Returns the phases present in the dataset, in the order they are run

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :param str prefix: prefix of the columns holding the phases
    :param str suffix: suffix of the columns holding the phases
    :return: list of phase names
    """
    phases = [
        column[len(prefix) : -len(suffix)]
        for column in dataset.columns
        if column.startswith(prefix) and column.endswith(suffix)
    ]
    return _sorted_phases(phases)


def _sorted_phases(phases: list):
    """
    Sorts phases in the order in which they are run

    :param list phases: phase names
    :return: list of phase names
    """
    order = list(PHASE_COLORS)
    return sorted(phases, key=lambda phase: (_phase_index(phase, order), phase))


def _phase_index(phase: str, order: list):
    """
    Returns the position of a phase in the order in which phases are run

    :param str phase: name of the phase, possibly with a numbered suffix
    :param list order: phases in the order in which they are run
    :return: int position
    """
    base = _phase_base(phase, order)
    return order.index(base) if base in order else len(order)


def _phase_base(phase: str, order: list = None):
    """
    Strips the numbered suffix that repeated phases get

    :param str phase: name of the phase
    :param list order: phases in the order in which they are run
    :return: str phase name without suffix
    """
    order = order if order is not None else list(PHASE_COLORS)
    if phase in order:
        return phase
    return phase.rsplit("_", 1)[0]


def plot_cpu_utilisation(dataset: pd.DataFrame):
    """
    Plots how many cores each phase actually keeps busy.

    The parallelism is the cpu time of a phase divided by its wall time, so a
    value of 1 means one core was busy and a value of 4 means four cores were.
    It is read from profile_phases.csv, which holds both user and system cpu
    time. This is the number to look at when sizing the cores of a job.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :return: Path of the figure
    """
    parallelism = {}
    user_share = {}

    for _, run in dataset.iterrows():
        phases = pd.read_csv(Path(run["run_folder"]) / "profile_phases.csv")
        label = _short_label(run).replace("\n", " ")
        parallelism[label] = phases.set_index("phase")["parallelism_avg"]
        cpu_total = phases["cpu_user_s"] + phases["cpu_system_s"]
        # The values are taken as an array, as passing a Series to pd.Series
        # with an index reindexes it instead of relabelling it
        user_share[label] = pd.Series(
            (100 * phases["cpu_user_s"] / cpu_total.where(cpu_total > 0)).to_numpy(),
            index=phases["phase"],
        )

    parallelism = pd.DataFrame(parallelism)
    user_share = pd.DataFrame(user_share)
    order = _sorted_phases(list(parallelism.index))
    parallelism = parallelism.reindex(order)
    user_share = user_share.reindex(order)

    figure, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    positions = np.arange(len(parallelism.columns))
    width = 0.8 / max(len(order), 1)

    for offset, phase in enumerate(order):
        axes[0].bar(
            positions + offset * width,
            parallelism.loc[phase].to_numpy(dtype=float),
            width=width,
            color=_phase_color(phase),
            label=phase,
        )
        axes[1].bar(
            positions + offset * width,
            user_share.loc[phase].to_numpy(dtype=float),
            width=width,
            color=_phase_color(phase),
        )

    axes[0].axhline(1, color="black", ls="--", lw=0.8)
    axes[0].set_ylabel("Cores kept busy (cpu time / wall time) [-]")
    axes[0].set_title("Parallelism per phase")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].set_ylabel("User share of cpu time [%]")
    axes[1].set_title("User versus system cpu time per phase")
    axes[1].set_ylim(0, 100)

    for axis in axes:
        axis.set_xticks(positions + 0.4 - width / 2)
        axis.set_xticklabels(parallelism.columns, fontsize=8)
        axis.grid(True, axis="y", alpha=0.2)

    figure.suptitle("CPU usage per phase, runs sorted by number of variables")
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    return _save(figure, "cpu_utilisation.png")


def plot_resources_versus_difficulty(
    dataset: pd.DataFrame, size_tolerance: float = 0.2
):
    """
    Plots the resources of the solve phase against how hard the run was.

    Only runs of a comparable size are kept, so that what is left is the effect
    of the difficulty and not of the size. The question the figure answers is
    whether the runs that come out as harder, meaning more simplex iterations,
    more branch and bound nodes and more time, also need more resources.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :param float size_tolerance: runs are kept if their number of variables is
        within this fraction of the largest one
    :return: Path of the figure
    """
    needed = ["gurobi_runtime_s", "rss_peak_solve_mb", "cpu_user_solve_s"]
    if any(column not in dataset.columns for column in needed):
        print("Skipping resources_vs_difficulty, the dataset lacks solve metrics")
        return None

    # Keep the runs that sit in the largest cluster of comparable sizes
    largest = dataset["n_vars"].max()
    subset = dataset[dataset["n_vars"] >= (1 - size_tolerance) * largest].copy()
    if len(subset) < 3:
        print("Skipping resources_vs_difficulty, too few runs of a comparable size")
        return None

    subset["cpu_total_solve_s"] = subset["cpu_user_solve_s"] + subset.get(
        "cpu_system_solve_s", 0
    )

    resources = [
        ("cpu_total_solve_s", "CPU time of the solve phase [s]", True),
        ("rss_peak_solve_mb", "Peak memory of the solve phase [MB]", False),
        ("parallelism_solve", "Cores kept busy during the solve [-]", False),
    ]
    difficulties = [
        ("gurobi_runtime_s", "Gurobi runtime [s]"),
        ("gurobi_itercount", "Simplex iterations [-]"),
        ("gurobi_nodecount", "Branch and bound nodes [-]"),
    ]

    figure, axes = plt.subplots(
        len(resources), len(difficulties), figsize=(14, 11), squeeze=False
    )

    for row, (resource, resource_label, log_y) in enumerate(resources):
        for column, (difficulty, difficulty_label) in enumerate(difficulties):
            axis = axes[row][column]
            if resource not in subset.columns or difficulty not in subset.columns:
                axis.set_visible(False)
                continue

            x = subset[difficulty].to_numpy(dtype=float)
            y = subset[resource].to_numpy(dtype=float)
            axis.plot(x, y, "o", color="#4c72b0", markersize=8)

            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() >= 3 and np.std(x[finite]) > 0:
                r = np.corrcoef(x[finite], y[finite])[0, 1]
                axis.set_title(f"r = {r:+.3f}", fontsize=10)

            axis.set_xlabel(difficulty_label, fontsize=9)
            if column == 0:
                axis.set_ylabel(resource_label, fontsize=9)
            if log_y:
                axis.set_yscale("log")
            axis.grid(True, alpha=0.2)

    figure.suptitle(
        f"Do harder runs use more resources? "
        f"{len(subset)} runs between {subset['n_vars'].min():,} and "
        f"{subset['n_vars'].max():,} variables"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    return _save(figure, "resources_vs_difficulty.png")


# The modelling choices, as they can be read back out of a case name. The code
# is the token run_benchmark puts into the name. Two of the knobs are named
# after their cheap setting, so for those the code being absent is the
# expensive setting, which is what on_when_code_absent says.
KNOB_SPEC = {
    "storage": {
        "code": "st0",
        "on_when_code_absent": True,
        "label": "storage",
        "off_label": "no storage",
        "on_label": "storage",
    },
    "pipeline_size_min": {
        "code": "sm250",
        "on_when_code_absent": False,
        "label": "pipeline size_min",
        "off_label": "no size_min",
        "on_label": "size_min = 250",
    },
    "bidirectional_precise": {
        "code": "bp1",
        "on_when_code_absent": False,
        "label": "bidirectional precise",
        "off_label": "bidirectional simple",
        "on_label": "bidirectional precise",
    },
    "pipeline_capex": {
        "code": "cxfix",
        "on_when_code_absent": False,
        "label": "capex fixed + linear",
        "off_label": "capex linear",
        "on_label": "capex fixed + linear",
    },
    "electricity_price": {
        "code": "pf",
        "on_when_code_absent": False,
        "label": "price fluctuating",
        "off_label": "price constant",
        "on_label": "price fluctuating",
    },
    "hydrogen_demand": {
        "code": "df",
        "on_when_code_absent": False,
        "label": "demand fluctuating",
        "off_label": "demand constant",
        "on_label": "demand fluctuating",
    },
    "pv": {
        "code": "pv0",
        "on_when_code_absent": True,
        "label": "PV",
        "off_label": "no PV",
        "on_label": "PV",
    },
}

# The four resources the study is about, drawn as the panels of every summary
# figure so that they can be compared between figures
SUMMARY_METRICS = [
    "gurobi_runtime_s",
    "rss_peak_os_mb",
    "wall_total_s",
    "parallelism_solve",
]

# Resources that get their own scaling_by_knob grid. The two cpu measures
# answer different questions and are both kept: the parallelism is how many
# cores a job should ask for, the user cpu time is how much compute it burns
KNOB_PANEL_METRICS = [
    "gurobi_runtime_s",
    "rss_peak_os_mb",
    "parallelism_solve",
    "cpu_user_s",
]

# Parallelism is a ratio of order one, everything else spans decades
LINEAR_METRICS = {"parallelism_solve", "parallelism_avg"}


def add_knob_columns(dataset: pd.DataFrame):
    """
    Reads the modelling choices back out of the case names.

    The knobs are not columns of the dataset, they only survive as short codes
    in the case name. The codes are matched against the tokens of the name
    rather than against the name itself, so that one code cannot match inside
    another.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :return: the dataset with a boolean knob_<name> column per knob, True when
        the expensive setting of that knob was used
    """
    dataset = dataset.copy()
    tokens = dataset["case_name"].str.split("_")

    for knob, spec in KNOB_SPEC.items():
        present = tokens.apply(lambda parts: spec["code"] in parts)
        dataset[f"knob_{knob}"] = ~present if spec["on_when_code_absent"] else present

    # A run stopped by the time limit did not measure how long the solve takes,
    # it measured the time limit, so it has to be drawn differently
    dataset["censored"] = dataset.get(
        "termination_condition", pd.Series("", index=dataset.index)
    ).eq("maxTimeLimit")

    return dataset


def _label_log_axis(axis, values: np.ndarray):
    """
    Puts readable numbers on a log x axis whatever the range of the data.

    A ratio that stays inside one decade gets only the 10^0 tick from the
    default formatter, and a ratio spanning several decades gets an unreadable
    row of overlapping minor labels if every minor tick is written out. Which
    of the two is happening is decided from the data.

    :param axis: matplotlib axis
    :param np.ndarray values: the values drawn on the axis
    """
    finite = values[np.isfinite(values) & (values > 0)]
    if not len(finite):
        return

    decades = np.log10(finite.max()) - np.log10(finite.min())
    axis.xaxis.set_major_formatter(FuncFormatter(_format_ratio))
    if decades <= 1.5:
        axis.xaxis.set_minor_formatter(FuncFormatter(_format_ratio))
        axis.tick_params(axis="x", which="minor", labelsize=6.5)
    else:
        axis.xaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis="x", which="major", labelsize=8)


def _format_ratio(value: float, _position=None):
    """
    Writes a tick of a ratio axis without the trailing zeros of a float

    :param float value: value of the tick
    :param _position: position of the tick, required by matplotlib
    :return: str tick label
    """
    if value <= 0:
        return ""
    if value >= 1:
        return f"{value:.0f}"
    # Below one the number of decimals has to follow the magnitude, otherwise
    # 0.1 is rounded away to 0
    return f"{value:.{max(1, int(-np.floor(np.log10(value))))}f}"


def _ratio_label(metric: str):
    """
    Turns a resource label into the label of a ratio of that resource

    :param str metric: column of the dataset
    :return: str label without the unit, as a ratio has none
    """
    label = RESOURCE_LABELS.get(metric, metric)
    return label.split(" [")[0]


def _usable(dataset: pd.DataFrame, metric: str):
    """
    Drops the rows that cannot carry a metric on a log axis

    :param pd.DataFrame dataset: dataset with the knob columns added
    :param str metric: column of the dataset
    :return: pandas DataFrame with only the usable rows
    """
    if metric not in dataset.columns:
        return dataset.iloc[0:0]
    return dataset[dataset[metric].notna() & (dataset[metric] > 0)]


def _bin_edges(sizes: np.ndarray, n_bins: int = 7):
    """
    Cuts the size axis into equally wide bins of log size.

    Two subsets that are compared in the same panel have to be binned on the
    same edges, otherwise their medians land at different sizes and the two
    lines cross where the data does not.

    :param np.ndarray sizes: number of variables of every run in the panel
    :param int n_bins: number of bins
    :return: np.ndarray of edges in log10 space
    """
    log_sizes = np.log10(sizes)
    if log_sizes.max() == log_sizes.min():
        return np.array([log_sizes.min(), log_sizes.max() + 1])
    return np.linspace(log_sizes.min(), log_sizes.max(), n_bins + 1)


def _band(
    sizes: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray = None,
    min_runs: int = 3,
):
    """
    Summarises a cloud of runs into a median and two bands of spread.

    The runs sit at a handful of model sizes rather than on a continuum, so the
    size axis is cut into bins and every bin is reduced to its median and its
    percentiles. A bin holding one or two runs has no spread worth drawing and
    would put a spike in the median line, so it is dropped. The runs are still
    visible as the scatter underneath.

    :param np.ndarray sizes: number of variables per run
    :param np.ndarray values: metric per run
    :param np.ndarray edges: bin edges in log10 space, derived from the runs
        themselves when not given
    :param int min_runs: a bin needs at least this many runs to be kept
    :return: pandas DataFrame with one row per kept bin
    """
    edges = _bin_edges(sizes) if edges is None else edges
    log_sizes = np.log10(sizes)

    # digitize on the inner edges puts everything into 0..len(edges)-2
    index = np.clip(np.digitize(log_sizes, edges[1:-1]), 0, len(edges) - 2)

    rows = []
    for bin_number in range(len(edges) - 1):
        inside = index == bin_number
        if inside.sum() < min_runs:
            continue
        rows.append(
            {
                "bin": bin_number,
                "n_vars": np.median(sizes[inside]),
                "low": np.percentile(values[inside], 10),
                "q1": np.percentile(values[inside], 25),
                "median": np.median(values[inside]),
                "q3": np.percentile(values[inside], 75),
                "high": np.percentile(values[inside], 90),
                "n_runs": int(inside.sum()),
            }
        )

    return pd.DataFrame(rows, columns=BAND_COLUMNS).sort_values("n_vars")


BAND_COLUMNS = ["bin", "n_vars", "low", "q1", "median", "q3", "high", "n_runs"]


def _contiguous(band: pd.DataFrame):
    """
    Splits a band wherever a bin in between it was dropped.

    Drawing straight through a gap would shade a wedge across sizes at which
    nothing was measured, which reads as a measurement and is not one.

    :param pd.DataFrame band: summary as returned by _band
    :return: list of DataFrames, one per stretch of adjacent bins
    """
    if band.empty:
        return []
    breaks = band["bin"].diff().ne(1).cumsum()
    return [segment for _, segment in band.groupby(breaks)]


def _draw_band(band: pd.DataFrame, axis, color: str, label: str, alpha: float = 1.0):
    """
    Draws a median line with its interquartile and its ten to ninety band

    :param pd.DataFrame band: summary as returned by _band
    :param axis: matplotlib axis
    :param str color: colour of the line and of the shading
    :param str label: legend entry of the median line
    :param float alpha: multiplies the opacity of the shading
    """
    for position, segment in enumerate(_contiguous(band)):
        axis.fill_between(
            segment["n_vars"],
            segment["low"],
            segment["high"],
            color=color,
            alpha=0.12 * alpha,
            lw=0,
        )
        axis.fill_between(
            segment["n_vars"],
            segment["q1"],
            segment["q3"],
            color=color,
            alpha=0.28 * alpha,
            lw=0,
        )
        axis.plot(
            segment["n_vars"],
            segment["median"],
            "o-",
            color=color,
            markersize=5,
            lw=2,
            # Only the first segment carries the label, so that a band broken
            # by a gap does not appear twice in the legend
            label=label if position == 0 else None,
        )


def _finish_size_axis(axis, metric: str, show_xlabel: bool = True):
    """
    Applies the axis settings shared by every figure drawn against model size

    :param axis: matplotlib axis
    :param str metric: column plotted on the y axis
    :param bool show_xlabel: whether the x axis carries its label
    """
    axis.set_xscale("log")
    if metric not in LINEAR_METRICS:
        axis.set_yscale("log")
    if show_xlabel:
        axis.set_xlabel("Number of variables [-]")
    axis.set_ylabel(RESOURCE_LABELS.get(metric, metric), fontsize=9)
    axis.grid(True, which="both", alpha=0.2)


def plot_scaling_envelope(dataset: pd.DataFrame, metrics: list = None):
    """
    Plots the median resource against the model size with the spread shaded.

    Every run of the study is in this figure. The line is the median over all
    modelling choices at that size, the dark band holds the middle half of them
    and the light band holds eight out of ten. The width of the band is the
    point of the figure: it is how much of the resource is still open once the
    size of the model is known, which is the question the study asks.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :param list metrics: columns plotted, one panel each
    :return: Path of the figure
    """
    dataset = add_knob_columns(dataset)
    metrics = metrics or SUMMARY_METRICS

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)

    for position, metric in enumerate(metrics[:4]):
        axis = axes[position // 2][position % 2]
        usable = _usable(dataset, metric)
        if usable.empty:
            axis.set_visible(False)
            continue

        sizes = usable["n_vars"].to_numpy(dtype=float)
        values = usable[metric].to_numpy(dtype=float)

        # Every run behind the band, so that the raw spread stays visible
        censored = usable["censored"].to_numpy()
        axis.plot(
            sizes[~censored],
            values[~censored],
            "o",
            color="#4c72b0",
            markersize=3.5,
            alpha=0.30,
            zorder=1,
        )
        if censored.any():
            axis.plot(
                sizes[censored],
                values[censored],
                "v",
                color="#c44e52",
                markersize=8,
                mfc="none",
                mew=1.6,
                zorder=4,
                label="stopped by the 2 h limit, a lower bound",
            )

        band = _band(sizes, values)
        _draw_band(band, axis=axis, color="#c44e52", label="median")

        # How many runs sit behind each point of the median, so that a bin
        # carried by three runs is not read as if it were carried by thirty
        for _, point in band.iterrows():
            axis.annotate(
                f"{point['n_runs']:.0f}",
                (point["n_vars"], point["median"]),
                textcoords="offset points",
                xytext=(0, -13),
                ha="center",
                fontsize=6.5,
                color="#c44e52",
            )

        _finish_size_axis(axis, metric, show_xlabel=position >= 2)
        axis.legend(frameon=False, fontsize=8, loc="upper left")

    figure.suptitle(
        f"Resource against model size over all {len(dataset)} runs. "
        "Line is the median, dark band the middle half, light band 10 to 90 %",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    return _save(figure, "scaling_envelope.png")


def plot_scaling_by_knob(
    dataset: pd.DataFrame,
    metric: str = "gurobi_runtime_s",
    min_runs_per_side: int = 8,
    condition_on: str = "storage",
):
    """
    Plots the median and the spread separately for each setting of each knob.

    One panel per modelling choice, two bands in it: the runs that have the
    choice at its cheap setting and the runs that have it at its expensive one.
    A knob whose two bands lie on top of each other does not matter for that
    resource, a knob whose bands separate is a knob worth knowing about before
    a job is sized.

    Every panel but the one of the conditioning knob is restricted to the runs
    that have that knob on. Without this the panels of the other knobs mix two
    populations that are two orders of magnitude apart, and since the cheap
    population stops existing above a certain size, the median jumps by a
    factor of thirty where the resource itself grows by a factor of four. The
    jump would be a property of which runs were done, not of the model.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :param str metric: column of the dataset plotted on the y axis
    :param int min_runs_per_side: a knob needs this many runs with the choice
        on and this many with it off to get a panel
    :param str condition_on: knob that every other panel is conditioned on,
        None to draw every panel over all runs
    :return: Path of the figure
    """
    dataset = add_knob_columns(dataset)
    usable = _usable(dataset, metric)
    if usable.empty:
        print(f"Skipping scaling_by_knob, no run has a usable '{metric}'")
        return None

    # A knob needs enough runs on both sides to be worth a panel. pv was only
    # varied in the full resolution stage, so it has two runs and is left out
    varied = [
        knob
        for knob in KNOB_SPEC
        if min(
            (usable[f"knob_{knob}"]).sum(), (~usable[f"knob_{knob}"]).sum()
        )
        >= min_runs_per_side
    ]
    if not varied:
        print("Skipping scaling_by_knob, no knob was varied enough")
        return None

    # Both sides of every panel share the bin edges of the whole dataset, so
    # that the two medians are compared at the same model size
    edges = _bin_edges(usable["n_vars"].to_numpy(dtype=float))

    n_columns = 3
    n_rows = int(np.ceil(len(varied) / n_columns))
    figure, axes = plt.subplots(
        n_rows, n_columns, figsize=(4.8 * n_columns, 3.9 * n_rows), squeeze=False
    )

    for position, knob in enumerate(varied):
        axis = axes[position // n_columns][position % n_columns]
        spec = KNOB_SPEC[knob]

        # The conditioning knob is the one panel that has to see everything,
        # as conditioning it on itself would leave one band
        panel = usable
        conditioned = condition_on is not None and knob != condition_on
        if conditioned:
            panel = usable[usable[f"knob_{condition_on}"]]

        bands = {}
        for is_on, color in [(False, "#4c72b0"), (True, "#c44e52")]:
            subset = panel[panel[f"knob_{knob}"] == is_on]
            if subset.empty:
                continue
            sizes = subset["n_vars"].to_numpy(dtype=float)
            values = subset[metric].to_numpy(dtype=float)
            axis.plot(sizes, values, "o", color=color, markersize=3, alpha=0.25)
            bands[is_on] = _band(sizes, values, edges=edges)
            _draw_band(
                bands[is_on],
                axis=axis,
                color=color,
                label=f"{spec['on_label'] if is_on else spec['off_label']} "
                f"(n={len(subset)})",
            )

        # Past the last bin that both bands reach there is nothing to compare:
        # one band stops and a reader can mistake the surviving one for the
        # cheaper option. Taking the largest size of each side instead would
        # be fooled by a single far out run that no band was built from
        shared = (
            set(bands[True]["bin"]) & set(bands[False]["bin"])
            if len(bands) == 2
            else set()
        )
        if shared:
            both_up_to = max(
                bands[True].loc[bands[True]["bin"].isin(shared), "n_vars"].max(),
                bands[False].loc[bands[False]["bin"].isin(shared), "n_vars"].max(),
            )
            if panel["n_vars"].max() > both_up_to:
                axis.axvspan(
                    both_up_to,
                    panel["n_vars"].max() * 1.15,
                    color="#999999",
                    alpha=0.13,
                    lw=0,
                    zorder=0,
                )
                axis.axvline(both_up_to, color="#999999", ls=":", lw=1, zorder=0)

        title = spec["label"]
        if conditioned:
            title += f"\n({KNOB_SPEC[condition_on]['on_label']} runs only)"
        else:
            title += "\n(all runs)"
        axis.set_title(title, fontsize=10)
        _finish_size_axis(axis, metric, show_xlabel=position >= len(varied) - n_columns)
        # A resource on a linear axis fills the top left corner, where the
        # legend would sit, so the placement is left to matplotlib
        axis.legend(frameon=True, framealpha=0.85, edgecolor="none", fontsize=7.5,
                    loc="best")

    for position in range(len(varied), n_rows * n_columns):
        axes[position // n_columns][position % n_columns].set_visible(False)

    conditioning = (
        f". Every other panel is restricted to the "
        f"{KNOB_SPEC[condition_on]['on_label']} runs, as mixing them with the "
        f"{KNOB_SPEC[condition_on]['off_label']} runs makes the median jump "
        f"where the population changes rather than where the resource does"
        if condition_on is not None
        else ""
    )
    figure.suptitle(
        f"{RESOURCE_LABELS.get(metric, metric)}: one panel per modelling choice, "
        f"median and spread with the choice off and on{conditioning}. "
        f"Grey: only one of the two settings was run at that size",
        fontsize=10,
        wrap=True,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    return _save(figure, f"scaling_by_knob_{metric}.png")


def paired_ratios(dataset: pd.DataFrame, knob: str, metric: str):
    """
    Collects the effect of one knob, holding everything else fixed.

    Two runs form a pair when they agree on the number of typical days and on
    every other knob and differ only in this one. The ratio of the expensive
    run to the cheap one is then the cost of that modelling choice, free of the
    size effect and free of whatever the other choices were doing. Runs stopped
    by the time limit are left out, as their metric is a lower bound and would
    understate the ratio.

    :param pd.DataFrame dataset: dataset with the knob columns added
    :param str knob: knob whose effect is measured
    :param str metric: column of the dataset the ratio is taken of
    :return: np.ndarray of ratios, one per pair
    """
    usable = _usable(dataset, metric)
    usable = usable[~usable["censored"]]
    if usable.empty:
        return np.array([])

    others = [f"knob_{other}" for other in KNOB_SPEC if other != knob]
    keys = ["typicaldays_n"] + [column for column in others if column in usable.columns]

    ratios = []
    for _, group in usable.groupby(keys, dropna=False):
        cheap = group[~group[f"knob_{knob}"]]
        expensive = group[group[f"knob_{knob}"]]
        # More than one run on a side means the pair is not identified by the
        # keys, which would make the ratio meaningless
        if len(cheap) != 1 or len(expensive) != 1:
            continue
        ratios.append(expensive[metric].iloc[0] / cheap[metric].iloc[0])

    return np.array(ratios, dtype=float)


def plot_knob_impact(
    dataset: pd.DataFrame, metrics: list = None, min_pairs: int = 3
):
    """
    Plots what each modelling choice multiplies a resource by.

    Every point is one pair of runs that differ in a single knob, so the size
    of the model is already divided out and what is left is the cost of the
    choice itself. The box is the middle half of the pairs and the line inside
    it is the median, which is the number to quote. A knob whose box sits on
    the dashed line at one is free.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :param list metrics: columns the ratios are taken of, one panel each
    :param int min_pairs: a knob needs at least this many pairs to be drawn
    :return: Path of the figure
    """
    dataset = add_knob_columns(dataset)
    metrics = metrics or ["gurobi_runtime_s", "rss_peak_os_mb", "wall_total_s"]

    # A knob is drawn when at least one of the metrics can pair it up
    collected = {}
    for knob in KNOB_SPEC:
        per_metric = {
            metric: paired_ratios(dataset, knob, metric) for metric in metrics
        }
        if max((len(values) for values in per_metric.values()), default=0) >= min_pairs:
            collected[knob] = per_metric

    if not collected:
        print("Skipping knob_impact, no knob has enough pairs")
        return None

    # Worst knob at the top, ranked on the first metric
    order = sorted(
        collected,
        key=lambda knob: (
            np.median(collected[knob][metrics[0]])
            if len(collected[knob][metrics[0]])
            else 1.0
        ),
    )

    figure, axes = plt.subplots(
        1, len(metrics), figsize=(5.2 * len(metrics), 0.62 * len(order) + 3.2),
        squeeze=False,
    )

    for column, metric in enumerate(metrics):
        axis = axes[0][column]
        positions = np.arange(len(order))

        data = [
            collected[knob][metric]
            if len(collected[knob][metric]) >= min_pairs
            else np.array([np.nan])
            for knob in order
        ]

        axis.axvline(1, color="black", ls="--", lw=0.9, zorder=1)
        boxes = axis.boxplot(
            [values[np.isfinite(values)] for values in data],
            positions=positions,
            vert=False,
            widths=0.62,
            showfliers=False,
            patch_artist=True,
            medianprops={"color": "black", "lw": 1.8},
        )
        for box in boxes["boxes"]:
            box.set(facecolor="#4c72b0", alpha=0.55, lw=0.8)

        # The pairs themselves behind the box, jittered so they do not overlap
        generator = np.random.default_rng(0)
        for position, values in zip(positions, data):
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            axis.plot(
                finite,
                position + generator.uniform(-0.18, 0.18, len(finite)),
                "o",
                color="#333333",
                markersize=2.6,
                alpha=0.35,
                zorder=3,
            )

        axis.set_xscale("log")
        _label_log_axis(axis, np.concatenate([values for values in data]))

        axis.set_yticks(positions)
        axis.set_yticklabels(
            [
                f"{KNOB_SPEC[knob]['label']}\n({len(collected[knob][metric])} pairs)"
                for knob in order
            ]
            if column == 0
            else [""] * len(order),
            fontsize=8.5,
        )
        axis.set_xlabel(
            f"{_ratio_label(metric)}\nrelative to the same run with the choice off",
            fontsize=9,
        )
        axis.grid(True, axis="x", which="both", alpha=0.2)
        axis.set_ylim(-0.6, len(order) - 0.4)

        # The median is the number to quote, so it is written next to the box
        for position, knob in zip(positions, order):
            values = collected[knob][metric]
            if len(values) < min_pairs:
                continue
            axis.annotate(
                f"x{np.median(values):.2f}",
                (np.median(values), position + 0.34),
                ha="center",
                fontsize=7.5,
                color="#c44e52",
                fontweight="bold",
            )

    figure.suptitle(
        "What does each modelling choice cost? Every point is a pair of runs "
        "that differ in that choice alone",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))

    return _save(figure, "knob_impact.png")


# The knobs of the scaling stage, which is the subset that was run at every
# number of typical days and therefore the subset that makes readable lines
LADDER_KNOBS = ["storage", "bidirectional_precise", "pipeline_size_min"]

LADDER_COLORS = {
    (False, False): "#4c72b0",
    (True, False): "#55a868",
    (False, True): "#dd8452",
    (True, True): "#c44e52",
}

# Kept away from the four ladder colours, as the full resolution runs are not
# a continuation of any of the lines
FULL_RESOLUTION_COLORS = ["#8172b2", "#937860", "#da8bc3"]


def plot_scaling_ladder(dataset: pd.DataFrame, metrics: list = None):
    """
    Plots the configurations of the scaling stage as explicit lines.

    The bands of the other figures hide which configuration is where, so the
    eight combinations that were run at every number of typical days are drawn
    on their own. Colour is the pair of formulation knobs and the line style is
    whether storage is in the model, which keeps eight lines down to four
    colours. The runs at full time resolution are added as stars, as they were
    run with different knobs and are not part of any of the lines.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :param list metrics: columns plotted, one panel each
    :return: Path of the figure
    """
    dataset = add_knob_columns(dataset)
    metrics = metrics or SUMMARY_METRICS

    # The scaling stage left the remaining knobs at their cheap setting
    fixed = ["pipeline_capex", "electricity_price", "hydrogen_demand"]
    ladder = dataset[~dataset[[f"knob_{knob}" for knob in fixed]].any(axis=1)]
    ladder = ladder[ladder["typicaldays_n"] > 0]
    full = dataset[dataset["typicaldays_n"] == 0]

    if ladder.empty:
        print("Skipping scaling_ladder, no run of the scaling stage in the dataset")
        return None

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), squeeze=False)

    for position, metric in enumerate(metrics[:4]):
        axis = axes[position // 2][position % 2]
        usable = _usable(ladder, metric)
        if usable.empty:
            axis.set_visible(False)
            continue

        for keys, group in usable.groupby(
            [f"knob_{knob}" for knob in LADDER_KNOBS], dropna=False
        ):
            has_storage, precise, size_min = keys
            group = group.sort_values("n_vars")
            axis.plot(
                group["n_vars"],
                group[metric],
                marker="o" if has_storage else "s",
                ls="-" if has_storage else "--",
                color=LADDER_COLORS[(precise, size_min)],
                markersize=6,
                lw=1.9,
                label=_ladder_label(has_storage, precise, size_min),
            )
            for _, run in group[group["censored"]].iterrows():
                axis.plot(
                    run["n_vars"], run[metric], "v", color="black",
                    markersize=9, mfc="none", mew=1.6, zorder=5,
                )

        # Two of the full resolution runs were both stopped by the time limit,
        # so they share an x and a y and no annotation can separate them. They
        # go into the legend instead, one entry each
        ordered = _usable(full, metric).sort_values("n_vars")
        for number, (_, run) in enumerate(ordered.iterrows()):
            axis.plot(
                run["n_vars"],
                run[metric],
                "*",
                color=FULL_RESOLUTION_COLORS[number % len(FULL_RESOLUTION_COLORS)],
                markersize=15,
                ls="none",
                mec="black",
                mew=0.6,
                label=f"full res.: {_full_resolution_label(run)}",
                zorder=4,
            )
            if run["censored"]:
                axis.plot(
                    run["n_vars"], run[metric], "v", color="black",
                    markersize=9, mfc="none", mew=1.6, zorder=5,
                )

        _finish_size_axis(axis, metric, show_xlabel=position >= 2)
        if position == 0:
            axis.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=1)

    figure.suptitle(
        "The eight configurations of the scaling stage across typical days. "
        "Open triangles were stopped by the 2 h limit",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))

    return _save(figure, "scaling_ladder.png")


def plot_simple_what_costs(dataset: pd.DataFrame, metric: str = "gurobi_runtime_s"):
    """
    One bar per modelling choice: what it multiplies the solve time by.

    The simplest reading of the study. Every bar is the median over the pairs
    of runs that differ in that choice alone, so the size of the model is
    already divided out, and the whisker is the middle half of those pairs.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :param str metric: column the ratios are taken of
    :return: Path of the figure
    """
    dataset = add_knob_columns(dataset)

    collected = {
        knob: paired_ratios(dataset, knob, metric)
        for knob in KNOB_SPEC
    }
    collected = {knob: values for knob, values in collected.items() if len(values) >= 3}
    if not collected:
        print("Skipping simple_what_costs, no knob has enough pairs")
        return None

    order = sorted(collected, key=lambda knob: np.median(collected[knob]))
    medians = np.array([np.median(collected[knob]) for knob in order])
    q1 = np.array([np.percentile(collected[knob], 25) for knob in order])
    q3 = np.array([np.percentile(collected[knob], 75) for knob in order])
    positions = np.arange(len(order))

    figure, axis = plt.subplots(figsize=(10, 0.72 * len(order) + 2.8))

    # A ratio below one is a choice that made the model easier, so the bars
    # grow away from one rather than from zero
    axis.barh(
        positions,
        medians - 1,
        left=1,
        height=0.6,
        color=["#c44e52" if value >= 2 else "#a8b6c8" for value in medians],
    )
    axis.hlines(positions, q1, q3, color="black", lw=1.6)
    axis.axvline(1, color="black", lw=1.1)

    # Placed past the end of the whisker, otherwise the text sits on top of it
    for position, knob in enumerate(order):
        value = medians[position]
        axis.annotate(
            f"  x{value:.0f}" if value >= 10 else f"  x{value:.2f}",
            (max(q3[position], value), position),
            va="center",
            fontsize=11,
            fontweight="bold" if value >= 2 else "normal",
            color="#c44e52" if value >= 2 else "#333333",
        )

    axis.set_xscale("log")
    _label_log_axis(axis, np.concatenate(list(collected.values())))
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [
            f"{KNOB_SPEC[knob]['label']}\n({len(collected[knob])} pairs)"
            for knob in order
        ],
        fontsize=10.5,
    )
    axis.set_xlabel(
        f"{_ratio_label(metric)} relative to the same run with the choice off",
        fontsize=10,
    )
    axis.set_xlim(min(q1.min(), 0.5) * 0.7, max(q3.max(), 2) * 4)
    axis.grid(True, axis="x", which="major", alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_title(
        "What does each modelling choice cost?\n"
        "Median over pairs of runs that differ in that choice alone, "
        "bar to the middle half",
        fontsize=12,
    )

    figure.tight_layout()
    return _save(figure, "simple_what_costs.png")


def plot_simple_predictability(dataset: pd.DataFrame):
    """
    Memory can be predicted from the model size, time cannot.

    Both panels hold every run and the same kind of fit. What separates them is
    the width of the band around it: the memory of a run stays inside a factor
    of about two of the fit, the solve time is spread over more than a factor
    of thirty. That is the practical answer to whether resources can be sized
    from the model alone.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :return: Path of the figure
    """
    dataset = add_knob_columns(dataset)

    panels = [
        ("rss_peak_os_mb", "Peak memory [MB]", "#4c72b0"),
        ("gurobi_runtime_s", "Gurobi runtime [s]", "#c44e52"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), squeeze=False)

    for column, (metric, label, color) in enumerate(panels):
        axis = axes[0][column]
        usable = _usable(dataset, metric)

        sizes = usable["n_vars"].to_numpy(dtype=float)
        values = usable[metric].to_numpy(dtype=float)
        censored = usable["censored"].to_numpy()

        exponent, intercept = np.polyfit(np.log(sizes), np.log(values), 1)
        r2 = np.corrcoef(np.log(sizes), np.log(values))[0, 1] ** 2
        residual = values / np.exp(intercept + exponent * np.log(sizes))
        low, high = np.percentile(residual, [10, 90])

        fit_x = np.array([sizes.min(), sizes.max()])
        fit_y = np.exp(intercept) * fit_x**exponent
        axis.fill_between(fit_x, fit_y * low, fit_y * high, color=color,
                          alpha=0.16, lw=0,
                          label=f"80 % of the runs: x{low:.2f} to x{high:.2f}")
        axis.plot(fit_x, fit_y, "-", color=color, lw=2.2,
                  label=f"fit ~ n$^{{{exponent:.2f}}}$  (R$^2$={r2:.2f})")

        axis.plot(sizes[~censored], values[~censored], "o", color=color,
                  markersize=5, alpha=0.5, mec="none")
        if censored.any():
            axis.plot(sizes[censored], values[censored], "v", color="black",
                      markersize=9, mfc="none", mew=1.5,
                      label="stopped by the 2 h limit")

        axis.set_xscale("log")
        axis.set_yscale("log")
        # The band and the fit run past the largest run, so the limits follow
        # the measurements rather than the extrapolation
        axis.set_ylim(values.min() / 2.5, values.max() * 4)
        axis.set_xlabel("Number of variables [-]", fontsize=11)
        axis.set_ylabel(label, fontsize=11)
        axis.grid(True, which="major", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, fontsize=9, loc="upper left")
        axis.set_title(
            f"{'Memory: predictable' if column == 0 else 'Time: not predictable'}"
            f"\nspread around the fit is a factor {high / low:.0f}",
            fontsize=12,
        )

    figure.suptitle(
        "Can the resources of a run be guessed from the size of the model?",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(figure, "simple_predictability.png")


def plot_simple_cores(dataset: pd.DataFrame):
    """
    How many cores a run actually keeps busy, against how many it was given.

    The solver was left free to use every core of the machine. It did not. The
    figure is the answer to how many cores a job should ask for, which is the
    one resource the study says not to spend.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :return: Path of the figure
    """
    dataset = add_knob_columns(dataset)
    usable = _usable(dataset, "parallelism_solve")
    if usable.empty:
        print("Skipping simple_cores, no run has a usable parallelism")
        return None

    cores = usable["parallelism_solve"].to_numpy(dtype=float)
    available = usable["n_cpus_physical"].dropna()
    available = float(available.iloc[0]) if len(available) else np.nan

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), squeeze=False)

    # Left: the distribution, which is the number to act on
    axis = axes[0][0]
    axis.hist(cores, bins=np.arange(0, np.ceil(cores.max()) + 0.5, 0.5),
              color="#4c72b0", alpha=0.85)
    axis.axvline(np.median(cores), color="#c44e52", lw=2.2,
                 label=f"median {np.median(cores):.1f} cores")
    axis.axvline(np.percentile(cores, 95), color="#dd8452", lw=2.2, ls="--",
                 label=f"95 % of runs below {np.percentile(cores, 95):.1f} cores")
    axis.set_xlabel("Cores kept busy during the solve [-]", fontsize=11)
    axis.set_ylabel("Number of runs [-]", fontsize=11)
    axis.legend(frameon=False, fontsize=10)
    axis.grid(True, axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    title = f"{len(usable)} runs"
    if np.isfinite(available):
        title += f", each free to use all {available:.0f} cores"
    axis.set_title(title, fontsize=12)

    # Right: and it does not grow with the size of the model
    axis = axes[0][1]
    axis.plot(usable["n_vars"], cores, "o", color="#4c72b0", markersize=5,
              alpha=0.5, mec="none")
    if np.isfinite(available):
        axis.axhline(available, color="#c44e52", lw=2,
                     label=f"{available:.0f} cores available")
    axis.axhline(np.median(cores), color="#55a868", lw=2, ls="--",
                 label=f"median actually used: {np.median(cores):.1f}")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Number of variables [-]", fontsize=11)
    axis.set_ylabel("Cores kept busy during the solve [-]", fontsize=11)
    axis.legend(frameon=False, fontsize=10, loc="lower right")
    axis.grid(True, which="major", alpha=0.25)
    axis.set_axisbelow(True)
    axis.set_title("Bigger models do not use more cores", fontsize=12)

    figure.suptitle("How many cores should a job ask for?", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(figure, "simple_cores.png")


def _full_resolution_label(run: pd.Series):
    """
    Names a full resolution run by the knobs that are not at their default

    :param pd.Series run: row of the dataset with the knob columns added
    :return: str short label
    """
    parts = ["storage" if run["knob_storage"] else "no storage"]
    if not run["knob_pv"]:
        parts.append("no PV")
    if run["knob_electricity_price"] or run["knob_hydrogen_demand"]:
        parts.append("fluct.")
    return ", ".join(parts)


def _ladder_label(has_storage: bool, precise: bool, size_min: bool):
    """
    Builds the legend entry of one line of the scaling ladder

    :param bool has_storage: whether storage is in the model
    :param bool precise: whether bidirectional_precise is on
    :param bool size_min: whether pipeline size_min is set
    :return: str legend label
    """
    formulation = [
        name
        for name, active in [("bidir. precise", precise), ("size_min", size_min)]
        if active
    ]
    return (
        f"{'storage' if has_storage else 'no storage'}"
        f"{', ' + ' + '.join(formulation) if formulation else ', plain'}"
    )


def plot_total_time(dataset: pd.DataFrame):
    """
    Plots the total wall time of every run against the number of variables.

    A power law is fitted through the points when there are enough of them.
    Points that sit far above the fit at the same number of variables are the
    configurations whose difficulty does not come from their size.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :return: Path of the figure
    """
    figure, axis = plt.subplots(figsize=(9, 6))

    n_vars = dataset["n_vars"].to_numpy(dtype=float)
    wall = dataset["wall_total_s"].to_numpy(dtype=float)

    axis.plot(n_vars, wall, "o", color="#4c72b0", markersize=9, zorder=3)

    for (_, run), x, y in zip(dataset.iterrows(), n_vars, wall):
        axis.annotate(
            _short_label(run).replace("\n", " "),
            (x, y),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=7,
        )

    valid = (n_vars > 0) & (wall > 0)
    if valid.sum() >= 3 and len(np.unique(n_vars[valid])) >= 2:
        exponent, intercept = np.polyfit(np.log(n_vars[valid]), np.log(wall[valid]), 1)
        r2 = np.corrcoef(np.log(n_vars[valid]), np.log(wall[valid]))[0, 1] ** 2
        fit_x = np.linspace(n_vars[valid].min(), n_vars[valid].max(), 100)
        axis.plot(
            fit_x,
            np.exp(intercept) * fit_x**exponent,
            "--",
            color="#c44e52",
            label=f"fit: wall ~ vars^{exponent:.2f}  (R2={r2:.3f})",
        )
        axis.legend(frameon=False, fontsize=9)

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Number of variables [-]")
    axis.set_ylabel("Total wall time [s]")
    axis.set_title("Total time against model size")
    axis.grid(True, which="both", alpha=0.2)

    figure.tight_layout()

    return _save(figure, "total_time.png")


def plot_size_versus_difficulty(dataset: pd.DataFrame):
    """
    Plots the size of the problem next to how hard it was to solve.

    Size and difficulty are not the same thing. Switching a price or a demand
    from constant to fluctuating leaves the number of variables untouched but
    can change the solve time by a lot, so the two are drawn on the same axis
    to make that visible.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :return: Path of the figure
    """
    dataset = dataset.reset_index(drop=True)
    positions = np.arange(len(dataset))
    labels = [_short_label(run) for _, run in dataset.iterrows()]

    figure, axes = plt.subplots(3, 1, figsize=(max(9, 1.5 * len(dataset)), 11))

    # Size
    axes[0].bar(positions, dataset["n_vars"], color="#4c72b0", label="variables")
    axes[0].set_ylabel("Variables [-]")
    axes[0].set_title("Size of the problem")
    binaries = axes[0].twinx()
    binaries.plot(
        positions, dataset["n_binvars"], "o-", color="#c44e52", label="binaries"
    )
    binaries.set_ylabel("Binary variables [-]", color="#c44e52")
    binaries.tick_params(axis="y", labelcolor="#c44e52")

    # Difficulty
    axes[1].bar(
        positions,
        dataset["gurobi_runtime_s"],
        color="#55a868",
        label="gurobi runtime",
    )
    axes[1].bar(
        positions,
        dataset["t_solve_s"] - dataset["gurobi_runtime_s"],
        bottom=dataset["gurobi_runtime_s"],
        color="#dd8452",
        label="pyomo to gurobi translation",
    )
    axes[1].set_ylabel("Solve phase [s]")
    axes[1].set_title("How hard it was to solve")
    axes[1].legend(frameon=False, fontsize=8)

    # Work done by the solver, which is what difficulty really is
    if "gurobi_itercount" in dataset.columns:
        axes[2].bar(positions, dataset["gurobi_itercount"], color="#8172b2")
        axes[2].set_ylabel("Simplex iterations [-]")
        axes[2].set_title("Work done by the solver")
        nodes = axes[2].twinx()
        nodes.plot(positions, dataset["gurobi_nodecount"], "o-", color="#937860")
        nodes.set_ylabel("Branch and bound nodes [-]", color="#937860")
        nodes.tick_params(axis="y", labelcolor="#937860")

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=7)
        axis.grid(True, axis="y", alpha=0.2)

    figure.suptitle("Size is not difficulty, runs sorted by number of variables")
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    return _save(figure, "size_versus_difficulty.png")


# The short knob codes that run_benchmark puts into a case name, spelled out
# for the axis labels
KNOB_LABELS = {
    "pv0": "no PV",
    "st0": "no storage",
    "sp1": "storage precise",
    "pf": "price fluct.",
    "df": "demand fluct.",
    "cxfix": "capex fixed",
    "cxlin": "capex linear",
    "bp1": "bidir. precise",
    "m2": "td method 2",
}


def _short_label(run: pd.Series):
    """
    Builds a short label for a run from its case name.

    The part of the case name after the typical days holds the knobs that were
    varied, which is what distinguishes runs that have the same size.

    :param pd.Series run: row of the dataset
    :return: str label
    """
    name = str(run["case_name"])

    typicaldays = run.get("typicaldays_n", 0)
    if pd.isna(typicaldays) or typicaldays == 0:
        lines = ["full res."]
        marker = "_td0_"
    else:
        lines = [f"td{int(typicaldays)}"]
        marker = f"_td{int(typicaldays)}_"

    if marker in name:
        codes = name.split(marker, 1)[1].split("_")
    elif name.endswith(marker.rstrip("_")):
        codes = []
    else:
        codes = []

    # Settings that are the same for every run carry no information here
    codes = [code for code in codes if not code.startswith("gap")]
    if not codes:
        lines.append("baseline")
    else:
        lines += [KNOB_LABELS.get(code, code) for code in codes]

    return "\n".join(lines)


def _wrap(text: str, width: int = 34):
    """
    Wraps a long case name over several lines, so that titles do not overlap

    :param str text: text to wrap
    :param int width: maximum number of characters per line
    :return: str wrapped text
    """
    return "\n".join(textwrap.wrap(text, width=width)) or text


def _phase_color(phase: str):
    """
    Returns the colour of a phase

    :param str phase: name of the phase
    :return: str colour
    """
    return PHASE_COLORS.get(_phase_base(phase), "#999999")


# Set from the command line, so that figures of different case studies do not
# overwrite each other
FIGURE_PREFIX = ""


def _save(figure, filename: str):
    """
    Saves a figure to the figures folder

    :param figure: matplotlib figure
    :param str filename: name of the file
    :return: Path of the figure
    """
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    path = FIGURES_PATH / f"{FIGURE_PREFIX}{filename}"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Written {path}")
    return path


def main():
    """
    Command line interface of the plotting
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        default="rss_mb",
        help="column of profile_timeseries.csv plotted over time",
    )
    parser.add_argument("--runs", nargs="+", help="case names to plot, default all")
    parser.add_argument("--case", help="only plot runs of this case study")
    parser.add_argument(
        "--resources",
        nargs="+",
        default=SUMMARY_METRICS,
        help="resources plotted against the model size, panels of the summary "
        "figures and one scaling_by_knob figure each",
    )
    args = parser.parse_args()

    if args.case:
        global FIGURE_PREFIX
        FIGURE_PREFIX = f"{args.case}_"

    dataset = load_dataset(args.runs, case=args.case)
    print(f"Plotting {len(dataset)} runs")

    plot_resource_curves(dataset, metric=args.metric)
    if args.metric == "rss_mb":
        # Memory and cpu are the two metrics of interest, so the cpu curve is
        # drawn as well unless another metric was asked for explicitly
        plot_resource_curves(dataset, metric="cpu_percent")
    plot_phase_durations(dataset)
    plot_phase_scaling(dataset)
    plot_cpu_utilisation(dataset)
    plot_size_versus_difficulty(dataset)
    plot_total_time(dataset)
    plot_resources_versus_difficulty(dataset)

    plot_scaling_envelope(dataset, metrics=args.resources)
    plot_scaling_ladder(dataset, metrics=args.resources)
    plot_knob_impact(dataset)
    # One panel grid per resource, as a knob that matters for the time does not
    # have to matter for the memory or for the cores
    for resource in KNOB_PANEL_METRICS:
        plot_scaling_by_knob(dataset, metric=resource)

    # One figure per conclusion, for reading rather than for digging
    plot_simple_what_costs(dataset)
    plot_simple_predictability(dataset)
    plot_simple_cores(dataset)


if __name__ == "__main__":
    main()
