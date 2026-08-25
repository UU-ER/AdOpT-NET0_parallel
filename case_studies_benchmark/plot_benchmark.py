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
- scaling_by_complexity: a resource against the model size, one line per
  complexity setting and one point per number of typical days

Examples::

    python plot_benchmark.py
    python plot_benchmark.py --metric cpu_percent
    python plot_benchmark.py --metric rss_mb --runs network_td5 network_td40
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
    "cpu_user_s": "User CPU time [s]",
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


def plot_scaling_by_complexity(dataset: pd.DataFrame, metric: str = "rss_peak_os_mb"):
    """
    Plots a resource against the model size, one line per complexity setting.

    Every line is one combination of the complexity knobs, and the points along
    it are the different numbers of typical days. Reading the figure vertically
    compares configurations that have the same number of variables but a
    different complexity, which is the comparison the benchmark is about.

    :param pd.DataFrame dataset: dataset as returned by load_dataset
    :param str metric: column of the dataset plotted on the y axis
    :return: Path of the figure
    """
    if metric not in dataset.columns:
        print(f"Skipping scaling_by_complexity, '{metric}' is not in the dataset")
        return None

    # Older runs can lack a column that was added to the profiler later, and a
    # resource of zero cannot go on a log axis
    dataset = dataset[dataset[metric].notna() & (dataset[metric] > 0)].copy()
    if dataset.empty:
        print(f"Skipping scaling_by_complexity, no run has a usable '{metric}'")
        return None

    # The knobs are what remains of the case name once the typical days are
    # taken out, so runs of one configuration across typical days group together
    dataset["complexity"] = dataset["case_name"].str.replace(r"_td\d+", "", regex=True)

    groups = [
        (name, group.sort_values("n_vars"))
        for name, group in dataset.groupby("complexity")
    ]
    # Cheapest configuration first, so the legend reads like a ladder
    groups.sort(key=lambda item: item[1][metric].min())

    figure, axis = plt.subplots(figsize=(10, 6.5))
    colors = plt.colormaps["viridis"](np.linspace(0, 0.9, max(len(groups), 1)))

    single_point = 0
    for (name, group), color in zip(groups, colors):
        label = _complexity_label(name)
        if len(group) == 1:
            single_point += 1
        axis.plot(
            group["n_vars"],
            group[metric],
            "o-",
            color=color,
            markersize=7,
            label=label,
        )

    if single_point:
        print(
            f"{single_point} configuration(s) have a single point. Run the same "
            f"knobs for several numbers of typical days to turn them into lines."
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Number of variables [-]")
    axis.set_ylabel(RESOURCE_LABELS.get(metric, metric))
    axis.set_title(
        "One line per complexity setting, one point per number of typical days"
    )
    axis.grid(True, which="both", alpha=0.2)
    axis.legend(frameon=False, fontsize=7, ncol=1, loc="best")

    figure.tight_layout()

    return _save(figure, f"scaling_by_complexity_{metric}.png")


def _complexity_label(name: str):
    """
    Turns the knob part of a case name into a readable legend entry

    :param str name: case name with the typical days removed
    :return: str legend label
    """
    codes = [
        code
        for code in name.split("_")
        if code not in ["four", "node", "network"] and not code.startswith("gap")
    ]
    if not codes:
        return "baseline"
    return ", ".join(KNOB_LABELS.get(code, code) for code in codes)


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
        default=["rss_peak_os_mb", "wall_total_s", "parallelism_solve"],
        help="resources plotted against the model size, one figure each",
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
    for resource in args.resources:
        plot_scaling_by_complexity(dataset, metric=resource)


if __name__ == "__main__":
    main()
