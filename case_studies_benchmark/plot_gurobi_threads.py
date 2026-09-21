"""
Draws whether a gurobi option and the thread count interact.

The runs were not made to find the best option, they were made to find out
whether any combination of an option and a thread count beats the default. That
is an interaction, and an interaction is read off a picture: the response
against the thread count, one line per option level. Parallel lines mean the
two choices can be made independently, crossing lines mean they cannot.

Six figures, each answering one question:

1. interaction     does the option behave differently at different thread counts
2. phases          which part of the solve the option and the threads move
3. effect          is the effect bigger than the spread across configurations
4. cores           does the option use the cores the default leaves idle
5. memory          what the thread count costs in memory, per option level
6. cost            cpu seconds per simplex iteration, the chaos-free axis

Three responses are drawn, not one. A combination that is no faster but wants
half the memory is a win, because memory is what a cluster job has to ask for.

Only runs made after the clustering was seeded are used. Before that, two runs
of one configuration were not the same model, and a run from the older sweeps
can carry the same case name as a cell of this study.

Figures go to ``figures/<study>/`` so that one study does not overwrite another.

Examples::

    python plot_gurobi_threads.py
    python plot_gurobi_threads.py --option gurobi_cuts --study stage8_cuts
    python plot_gurobi_threads.py --dataset benchmark_dataset.csv --since 20260911
"""

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATASET_FILE = BASE / "benchmark_dataset.csv"
FIGURES_PATH = BASE / "figures"
RESULTS_PATH = BASE / "results"

BLUE, ORANGE, RED = "#3b6ea5", "#d1731f", "#c0392b"
GREEN, DARK, GREY = "#4a8c5f", "#2b2b2b", "#9a9a9a"

LEVEL_COLORS = [BLUE, ORANGE, GREEN, RED, DARK]

# The clustering was seeded on 2026-09-11. Runs before that rebuilt the model
# differently every time, so they are a different problem and cannot share a
# panel with these, even when the case name matches
SEEDED_FROM = "20260911"

# Tokens a case name can carry that are not part of the complexity
# configuration. Stripping them leaves the configuration itself, which is what
# the panels are grouped by
SETTING_TOKENS = re.compile(
    r"_(thr|aff|mth|xov|bh|scl|cm|nm|pre|cut|cp|mf|heu|nrh|nf|lpw|bd)-?[\d.]+"
)

OPTION_LABELS = {
    "gurobi_method": "Method",
    "gurobi_cuts": "Cuts",
    "gurobi_cutpasses": "CutPasses",
    "gurobi_mipfocus": "MIPFocus",
    "gurobi_lpwarmstart": "LPWarmStart",
    "gurobi_presolve": "Presolve",
    "gurobi_crossover": "Crossover",
    "gurobi_nodemethod": "NodeMethod",
}

LEVEL_NAMES = {
    "gurobi_method": {
        -1: "auto",
        0: "primal simplex",
        1: "dual simplex",
        2: "barrier",
        3: "concurrent",
    },
    "gurobi_cuts": {
        -1: "auto",
        0: "off",
        1: "moderate",
        2: "aggressive",
        3: "very aggressive",
    },
    "gurobi_lpwarmstart": {
        -1: "auto (gurobi default)",
        0: "off (adopt default)",
        1: "on",
        2: "basis",
    },
    "gurobi_mipfocus": {0: "balanced", 1: "feasibility", 2: "optimality", 3: "bound"},
    # -1 is the gurobi default, which on this model family means twenty-five to
    # fifty-seven passes. The finite levels are counts, so they label themselves
    "gurobi_cutpasses": {-1: "auto", 1: "1 pass", 2: "2 passes", 5: "5 passes"},
}

RESPONSES = [
    ("gurobi_runtime_s", "solver runtime [s]"),
    ("rss_peak_os_mb", "peak memory [MB]"),
    ("cpu_user_s", "cpu time [s]"),
]

# The phases of a solve, in the order they happen, with the colour each keeps
# across every figure
PHASES = [
    ("phase_presolve_s", "presolve", GREY),
    ("phase_root_lp_s", "root relaxation", BLUE),
    ("phase_root_cuts_s", "root cut loop", ORANGE),
    ("phase_tree_s", "tree", GREEN),
]

# A figure narrower than this clips its own title, and the titles here are two
# lines because they say how to read the picture, not only what it shows
MIN_WIDTH = 8.4

# Inches to keep free for the title and for the legend under it. Reserving a
# fraction instead, which is what tight_layout takes, leaves a gap that grows
# with the figure: the same 10 % is one inch on a short figure and three on a
# tall one
TITLE_INCHES = 0.85
LEGEND_INCHES = 0.62


# Measured on the three configurations that were run twice at four threads,
# before the clustering was seeded: the geometric mean of the run to run spread.
# Runtime is chaotic, memory is not, which is why an effect on memory means
# something at a much smaller size than an effect on runtime
# Multiples a reader thinks in. A ratio axis is labelled from this ladder,
# never from computed reciprocals, which produce things like 1.0101
RATIO_TICKS = [
    0.1,
    0.2,
    0.25,
    0.33,
    0.4,
    0.5,
    0.6,
    0.67,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.05,
    1.1,
    1.2,
    1.25,
    1.33,
    1.5,
    1.75,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    10.0,
]

NOISE = {"gurobi_runtime_s": 3.9, "rss_peak_os_mb": 1.09, "cpu_user_s": 3.9}


def load(dataset_file: Path, since: str = SEEDED_FROM, results_path: Path = None):
    """
    Reads the dataset and keeps the runs this study can use

    :param Path dataset_file: benchmark_dataset.csv to read
    :param str since: earliest run timestamp to keep, as it appears in the
        result folder name
    :param Path results_path: results folder the solver logs are looked up in,
        for runs made before the root node column existed
    :return: pandas DataFrame of the seeded four_node runs
    """
    dataset = pd.read_csv(dataset_file)
    dataset = dataset[dataset["case_name"].str.startswith("four_node")].copy()

    # The run folder is named <timestamp>_<case name>, and the timestamp is the
    # only thing that tells a seeded run from an unseeded one with the same
    # name. run_folder is added by the collector, result_folder_path is written
    # by the run itself, so a single profile summary can be read too
    folder_column = next(
        (name for name in ("run_folder", "result_folder_path") if name in dataset),
        None,
    )
    if folder_column is None:
        raise KeyError(
            "The dataset carries neither run_folder nor result_folder_path, so "
            "seeded and unseeded runs cannot be told apart"
        )

    stamps = (
        dataset[folder_column]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.rstrip("/")
        .str.rsplit("/", n=1)
        .str[-1]
        .str.split("_")
        .str[0]
    )
    before = len(dataset)
    dataset = dataset[stamps >= since].copy()
    dropped = before - len(dataset)
    if dropped:
        print(f"Dropped {dropped} runs from before {since}, made without a seed")

    dataset["folder"] = dataset[folder_column]

    # 0 threads means every core of the machine, which on this server is 48
    dataset["requested"] = dataset["gurobi_threads"].replace(0, 48)
    dataset["config"] = dataset["case_name"].str.replace(SETTING_TOKENS, "", regex=True)

    _backfill_root_node_end(dataset, Path(results_path or RESULTS_PATH))
    _add_phases(dataset)

    return dataset


def _backfill_root_node_end(dataset: pd.DataFrame, results_path: Path):
    """
    Fills root_node_end_s from the solver logs where the column is missing.

    Runs made before the column existed still have their solver log on disk, so
    nothing is lost, it only has to be read back.

    A run records the absolute path it was written to, which is a drive letter
    on the machine that ran it and means nothing anywhere else. So the recorded
    path is tried first, and the folder name under the results directory is
    tried after it, which is what makes this work from a different machine.

    A finished sweep is archived into a subfolder, and the stages of one study
    into a subfolder each, so the name is looked for at any depth below the
    results directory and not only directly under it. The index is built once,
    because walking a network share per run is slow.

    :param pd.DataFrame dataset: dataset to fill in place
    :param Path results_path: results directory to fall back to
    """
    from adopt_net0.diagnostics.profiling import collect_solver_log_metrics

    if "root_node_end_s" not in dataset:
        dataset["root_node_end_s"] = np.nan

    missing = dataset[dataset["root_node_end_s"].isna()]
    if missing.empty:
        return

    archived = {
        log.parent.name: log for log in Path(results_path).rglob("*/solver_log.txt")
    }

    filled = 0
    unreachable = 0
    for index, row in missing.iterrows():
        recorded = str(row["folder"]).replace("\\", "/").rstrip("/")
        name = recorded.rsplit("/", 1)[-1]
        candidates = [
            Path(recorded) / "solver_log.txt",
            results_path / name / "solver_log.txt",
            archived.get(name, results_path / "does not exist"),
        ]
        log_path = next((path for path in candidates if path.is_file()), None)
        if log_path is None:
            unreachable += 1
            continue
        value = collect_solver_log_metrics(log_path).get("root_node_end_s", "")
        if value != "":
            dataset.at[index, "root_node_end_s"] = value
            filled += 1

    if filled:
        print(f"Read the end of the root node out of {filled} solver logs")
    if unreachable:
        print(
            f"{unreachable} solver logs were not found. Point --results at the "
            "results folder of the machine that ran them"
        )


def _add_phases(dataset: pd.DataFrame):
    """
    Splits the solve into presolve, root relaxation, root cut loop and tree.

    The root cut loop is the stretch between the root relaxation finishing and
    the solver leaving the root node, and on these models it is usually the
    largest of the four. A run whose log did not carry the boundary gets NaN
    rather than a guess.

    :param pd.DataFrame dataset: dataset to fill in place
    """
    dataset["phase_presolve_s"] = dataset.get("presolve_s", np.nan)
    dataset["phase_root_lp_s"] = dataset["root_relaxation_s"] - dataset[
        "phase_presolve_s"
    ].fillna(0)
    dataset["phase_root_cuts_s"] = (
        dataset["root_node_end_s"] - dataset["root_relaxation_s"]
    )
    dataset["phase_tree_s"] = dataset["gurobi_runtime_s"] - dataset["root_node_end_s"]

    # A phase cannot be negative. Rounding in the log, which reports whole
    # seconds in the node table and two decimals for the relaxation, can push a
    # short phase just below zero
    for column, _, _ in PHASES:
        dataset[column] = dataset[column].clip(lower=0)


def level_name(option: str, value):
    """
    Spells a level of an option for a legend

    :param str option: dataset column of the option
    :param value: the level
    :return: str
    """
    names = LEVEL_NAMES.get(option, {})
    try:
        key = int(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{key} {names[key]}" if key in names else str(key)


def _panels(dataset: pd.DataFrame, option: str):
    """
    Configurations worth a panel, those holding a comparison

    :param pd.DataFrame dataset: dataset to look at
    :param str option: dataset column of the option
    :return: list of configuration names
    """
    return [
        name
        for name, group in dataset.groupby("config")
        if group[option].nunique() > 1 and group["requested"].nunique() > 1
    ]


def _thread_axis(axis, ticks):
    """
    Sets up a thread count axis, which is only readable on a log scale

    :param axis: matplotlib axis
    :param ticks: thread counts present
    """
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(int(tick)) for tick in ticks])
    axis.grid(True, which="major", ls=":", lw=0.6, color=GREY, alpha=0.6)


def plot_interaction(dataset: pd.DataFrame, option: str, output: Path):
    """
    Figure 1. Response against thread count, one line per option level

    :param pd.DataFrame dataset: DataFrame returned by load
    :param str option: dataset column of the option
    :param Path output: png file to write
    :return: Path written, or None
    """
    configs = _panels(dataset, option)
    if not configs:
        return None

    levels = sorted(dataset[option].dropna().unique())
    figure, axes = plt.subplots(
        len(RESPONSES),
        len(configs),
        figsize=(max(MIN_WIDTH, 1 + 4.6 * len(configs)), 3.0 * len(RESPONSES)),
        squeeze=False,
    )

    for row, (column, label) in enumerate(RESPONSES):
        for col, config in enumerate(configs):
            axis = axes[row][col]
            group = dataset[dataset["config"] == config]

            for index, level in enumerate(levels):
                line = group[group[option] == level].sort_values("requested")
                if line.empty:
                    continue
                axis.plot(
                    line["requested"],
                    line[column],
                    marker="o",
                    ms=5,
                    lw=1.6,
                    color=LEVEL_COLORS[index % len(LEVEL_COLORS)],
                    label=level_name(option, level) if row == 0 and col == 0 else None,
                )

            axis.set_yscale("log")
            _thread_axis(axis, sorted(group["requested"].unique()))
            if row == 0:
                axis.set_title(config.replace("four_node_", ""), fontsize=11)
            if row == len(RESPONSES) - 1:
                axis.set_xlabel("threads requested")
            if col == 0:
                axis.set_ylabel(label)

    _finish(
        figure,
        axes[0][0],
        option,
        f"Does {OPTION_LABELS.get(option, option)} interact with the thread count?\n"
        "parallel lines: choose the two independently. crossing lines: you cannot",
        output,
    )
    return output


def plot_phases(dataset: pd.DataFrame, option: str, output: Path):
    """
    Figure 2. Where the solve goes, stacked, per cell

    Turns "this option is 20 % faster" into "this option is 20 % faster because
    it shortened the cut loop", which is the difference between a number and a
    finding.

    :param pd.DataFrame dataset: DataFrame returned by load
    :param str option: dataset column of the option
    :param Path output: png file to write
    :return: Path written, or None
    """
    usable = dataset[dataset["phase_root_cuts_s"].notna()]
    configs = _panels(usable, option)
    if not configs:
        return None

    figure, axes = plt.subplots(
        1,
        len(configs),
        figsize=(max(MIN_WIDTH, 1 + 5.8 * len(configs)), 5.0),
        squeeze=False,
    )

    for col, config in enumerate(configs):
        axis = axes[0][col]
        group = usable[usable["config"] == config].sort_values([option, "requested"])

        labels = [
            f"{level_name(option, row[option]).split()[0]}\n{int(row['requested'])}t"
            for _, row in group.iterrows()
        ]
        positions = np.arange(len(group))
        bottom = np.zeros(len(group))

        for column, name, colour in PHASES:
            values = group[column].fillna(0).to_numpy()
            axis.bar(
                positions,
                values,
                bottom=bottom,
                color=colour,
                width=0.72,
                label=name if col == 0 else None,
            )
            bottom += values

        axis.set_xticks(positions)
        axis.set_xticklabels(labels, fontsize=8)
        axis.set_ylabel("seconds" if col == 0 else "")
        axis.set_title(config.replace("four_node_", ""), fontsize=11)
        axis.grid(True, axis="y", ls=":", lw=0.6, color=GREY, alpha=0.6)
        axis.set_axisbelow(True)

    _finish(
        figure,
        axes[0][0],
        f"{OPTION_LABELS.get(option, option)} and threads",
        "Where the solve goes, per cell\n"
        "the bar that grows is the one the option or the thread count moved",
        output,
    )
    return output


def plot_effect(dataset: pd.DataFrame, option: str, output: Path):
    """
    Figure 3. Effect size against the spread it has to clear

    For every level, the geometric mean of its ratio to the default level, over
    all configurations and thread counts, with the spread across those cells as
    the error bar. A bar that does not clear its own spread is not an effect.

    :param pd.DataFrame dataset: DataFrame returned by load
    :param str option: dataset column of the option
    :param Path output: png file to write
    :return: Path written, or None
    """
    levels = sorted(dataset[option].dropna().unique())
    if len(levels) < 2:
        return None

    # The default is the level the template ships, which is the one the study
    # is asking whether to move away from
    default = -1 if -1 in levels else levels[0]
    others = [level for level in levels if level != default]
    if not others:
        return None

    figure, axes = plt.subplots(
        1,
        len(RESPONSES),
        figsize=(max(MIN_WIDTH, 4.7 * len(RESPONSES)), 4.6),
        squeeze=False,
    )

    for index, (column, label) in enumerate(RESPONSES):
        axis = axes[0][index]
        base = dataset[dataset[option] == default].set_index(["config", "requested"])[
            column
        ]

        centres, spreads, names = [], [], []
        for level in others:
            arm = dataset[dataset[option] == level].set_index(["config", "requested"])[
                column
            ]
            paired = (arm / base).replace([np.inf, -np.inf], np.nan).dropna()
            if paired.empty:
                continue
            logs = np.log(paired)
            centres.append(float(np.exp(logs.mean())))
            spreads.append(float(np.exp(logs.std(ddof=0))) if len(logs) > 1 else 1.0)
            names.append(level_name(option, level).split()[0])

        if not centres:
            continue

        positions = np.arange(len(centres))
        colours = [RED if value > 1 else GREEN for value in centres]
        axis.bar(positions, centres, color=colours, width=0.6, zorder=3)
        axis.errorbar(
            positions,
            centres,
            yerr=[
                [c - c / s for c, s in zip(centres, spreads)],
                [c * s - c for c, s in zip(centres, spreads)],
            ],
            fmt="none",
            ecolor=DARK,
            capsize=4,
            lw=1.2,
            zorder=4,
        )
        axis.axhline(1.0, color=DARK, lw=1.2, zorder=2)

        # The run to run noise of the earlier sweeps, as a reminder of how big
        # an effect on this response has to be before it means anything
        noise = NOISE.get(column)
        if noise:
            axis.axhspan(1 / noise, noise, color=GREY, alpha=0.18, zorder=1)

        axis.set_xticks(positions)
        axis.set_xticklabels(names)
        # A single bar would otherwise be drawn edge to edge
        axis.set_xlim(-0.7, len(centres) - 0.3)

        # A log axis of ratios labels itself in powers of ten, which on a range
        # of a few per cent means no labels at all. The ticks are placed by hand
        # and spelled as the multiples they are
        axis.set_yscale("log")
        _ratio_axis(axis, centres, spreads, noise=NOISE.get(column))

        axis.set_ylabel(f"ratio to {level_name(option, default).split()[0]}")
        axis.set_title(label, fontsize=11)
        axis.grid(True, axis="y", ls=":", lw=0.6, color=GREY, alpha=0.6)

    figure.suptitle(
        f"How big is the {OPTION_LABELS.get(option, option)} effect, "
        "against the spread it has to clear\n"
        "error bar: spread across configurations and thread counts. "
        "shading: run to run noise of the earlier sweeps",
        fontsize=12,
        y=1 - 0.22 / figure.get_size_inches()[1],
    )
    figure.tight_layout(rect=(0, 0, 1, 1 - TITLE_INCHES / figure.get_size_inches()[1]))
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote {output}")
    return output


def _ratio_axis(axis, centres, spreads, noise=None):
    """
    Labels a log axis of ratios with the multiples a reader thinks in.

    Matplotlib labels a log axis in powers of ten. A ratio axis spanning a few
    per cent then carries no labels at all, which is how a figure ends up
    unreadable while looking finished. Computing the reciprocals instead gives
    labels like 1.0101, which is worse.

    :param axis: matplotlib axis
    :param centres: the ratios drawn
    :param spreads: the spread factor of each
    :param noise: run to run noise factor drawn behind, if any
    """
    lows = [centre / spread for centre, spread in zip(centres, spreads)]
    highs = [centre * spread for centre, spread in zip(centres, spreads)]
    if noise:
        lows.append(1 / noise)
        highs.append(noise)

    low, high = min(lows + [1.0]), max(highs + [1.0])
    margin = (high / low) ** 0.08
    low, high = low / margin, high * margin
    axis.set_ylim(low, high)

    inside = [value for value in RATIO_TICKS if low <= value <= high]
    if not inside:
        return

    # Thin them out until no two labels sit on top of each other. The axis is
    # logarithmic, so closeness is a ratio and not a difference
    span = math.log(high / low)
    kept = []
    for value in inside:
        if value == 1.0 or not kept:
            kept.append(value)
            continue
        if math.log(value / kept[-1]) >= 0.055 * span:
            kept.append(value)
    if 1.0 in inside and 1.0 not in kept:
        kept.append(1.0)

    axis.yaxis.set_major_locator(FixedLocator(sorted(kept)))
    axis.yaxis.set_minor_locator(FixedLocator([]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def plot_cores(dataset: pd.DataFrame, option: str, output: Path):
    """
    Figure 4. Cores actually used against cores asked for, per option level

    The earlier study found the solver keeping a median of 2.1 cores of 48 busy.
    This asks whether that was a property of the model or of the algorithm: if
    one level sits above the others, it is using cores the default leaves idle.

    :param pd.DataFrame dataset: DataFrame returned by load
    :param str option: dataset column of the option
    :param Path output: png file to write
    :return: Path written, or None
    """
    if "parallelism_solve" not in dataset:
        return None
    usable = dataset[dataset["parallelism_solve"].notna()]
    if usable.empty or usable[option].nunique() < 1:
        return None

    figure, axis = plt.subplots(figsize=(8.2, 5.6))
    limits = [0.8, 60]
    axis.plot(
        limits, limits, ls="--", color=DARK, lw=1.2, zorder=2, label="all of them"
    )

    for index, level in enumerate(sorted(usable[option].dropna().unique())):
        group = usable[usable[option] == level].sort_values("requested")
        middle = group.groupby("requested")["parallelism_solve"].median()
        colour = LEVEL_COLORS[index % len(LEVEL_COLORS)]
        axis.scatter(
            group["requested"],
            group["parallelism_solve"],
            s=26,
            color=colour,
            alpha=0.55,
            zorder=3,
        )
        axis.plot(
            middle.index,
            middle.to_numpy(),
            marker="o",
            ms=6,
            lw=1.8,
            color=colour,
            zorder=4,
            label=level_name(option, level),
        )

    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlim(*limits)
    axis.set_ylim(0.8, 60)
    ticks = sorted(usable["requested"].unique())
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(int(tick)) for tick in ticks])
    axis.set_xlabel("cores asked for")
    axis.set_ylabel("cores actually busy during the solve")
    axis.grid(True, which="major", ls=":", lw=0.6, color=GREY, alpha=0.6)
    axis.legend(frameon=False, title=OPTION_LABELS.get(option, option))
    axis.set_title(
        "Does the option use the cores the default leaves idle?\n"
        "distance below the dashed line is what a job would be paying for and "
        "not using",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote {output}")
    return output


def plot_memory(dataset: pd.DataFrame, option: str, output: Path):
    """
    Figure 5. What the thread count costs in memory, per option level

    Memory is the least noisy of the three responses and the one a cluster job
    has to commit to in advance, so it gets a figure of its own, normalised to
    the cheapest thread count so that the cost of asking for more is readable
    straight off the axis.

    :param pd.DataFrame dataset: DataFrame returned by load
    :param str option: dataset column of the option
    :param Path output: png file to write
    :return: Path written, or None
    """
    configs = _panels(dataset, option)
    if not configs:
        return None

    figure, axes = plt.subplots(
        1,
        len(configs),
        figsize=(max(MIN_WIDTH, 1 + 5.0 * len(configs)), 4.8),
        squeeze=False,
    )

    for col, config in enumerate(configs):
        axis = axes[0][col]
        group = dataset[dataset["config"] == config]
        fewest = group["requested"].min()

        for index, level in enumerate(sorted(group[option].dropna().unique())):
            line = group[group[option] == level].sort_values("requested")
            anchor = line[line["requested"] == fewest]["rss_peak_os_mb"]
            if anchor.empty:
                continue
            axis.plot(
                line["requested"],
                line["rss_peak_os_mb"] / anchor.iloc[0],
                marker="o",
                ms=5,
                lw=1.6,
                color=LEVEL_COLORS[index % len(LEVEL_COLORS)],
                label=level_name(option, level) if col == 0 else None,
            )

        axis.axhline(1.0, color=DARK, lw=1.0)
        _thread_axis(axis, sorted(group["requested"].unique()))
        axis.set_xlabel("threads requested")
        if col == 0:
            axis.set_ylabel(f"peak memory, relative to {int(fewest)} thread")
        axis.set_title(config.replace("four_node_", ""), fontsize=11)

    _finish(
        figure,
        axes[0][0],
        option,
        "What asking for more cores costs in memory\n"
        "a line that stays flat is a combination a cluster job can size cheaply",
        output,
    )
    return output


def plot_cost_per_iteration(dataset: pd.DataFrame, option: str, output: Path):
    """
    Figure 6. Cpu seconds per simplex iteration against the thread count

    Runtime is work done times cost per unit of work. The first is chaotic under
    any perturbation, the second is physical, so dividing it out is what makes a
    comparison between two runs of a MIP mean something.

    :param pd.DataFrame dataset: DataFrame returned by load
    :param str option: dataset column of the option
    :param Path output: png file to write
    :return: Path written, or None
    """
    usable = dataset[dataset["gurobi_itercount"] > 0].copy()
    configs = _panels(usable, option)
    if not configs:
        return None

    usable["cost"] = usable["cpu_user_s"] / usable["gurobi_itercount"]

    figure, axes = plt.subplots(
        1,
        len(configs),
        figsize=(max(MIN_WIDTH, 1 + 5.0 * len(configs)), 4.8),
        squeeze=False,
    )

    for col, config in enumerate(configs):
        axis = axes[0][col]
        group = usable[usable["config"] == config]
        for index, level in enumerate(sorted(group[option].dropna().unique())):
            line = group[group[option] == level].sort_values("requested")
            axis.plot(
                line["requested"],
                line["cost"] * 1000,
                marker="o",
                ms=5,
                lw=1.6,
                color=LEVEL_COLORS[index % len(LEVEL_COLORS)],
                label=level_name(option, level) if col == 0 else None,
            )
        axis.set_yscale("log")
        _thread_axis(axis, sorted(group["requested"].unique()))
        axis.set_xlabel("threads requested")
        if col == 0:
            axis.set_ylabel("cpu ms per simplex iteration")
        axis.set_title(config.replace("four_node_", ""), fontsize=11)

    _finish(
        figure,
        axes[0][0],
        option,
        "Cost per unit of work, with the chaotic factor divided out\n"
        "this axis survived every retraction in this project so far",
        output,
    )
    return output


def _finish(figure, legend_axis, option: str, title: str, output: Path):
    """
    Puts the shared legend and title on a figure and writes it

    :param figure: matplotlib figure
    :param legend_axis: axis whose handles make the legend, or None
    :param str option: option the legend is titled with
    :param str title: figure title
    :param Path output: png file to write
    """
    handles, labels = (
        legend_axis.get_legend_handles_labels() if legend_axis is not None else ([], [])
    )
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 5),
            frameon=False,
            title=OPTION_LABELS.get(option, option),
        )

    height = figure.get_size_inches()[1]
    figure.suptitle(title, fontsize=12, y=1 - 0.22 / height)
    figure.tight_layout(
        rect=(
            0,
            (LEGEND_INCHES / height) if handles else 0,
            1,
            1 - TITLE_INCHES / height,
        )
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote {output}")


def report_coverage(dataset: pd.DataFrame, option: str):
    """
    Prints the cells that are on disk, so a half finished stage is readable

    :param pd.DataFrame dataset: DataFrame returned by load
    :param str option: dataset column of the option
    """
    usable = dataset[dataset[option].notna()]
    if usable.empty:
        print(f"No runs carry {option}")
        return

    label = OPTION_LABELS.get(option, option)
    tables = [(column, name) for column, name in RESPONSES if column in usable]
    tables.append(("phase_root_cuts_s", "root cut loop [s]"))

    for column, name in tables:
        table = usable.pivot_table(
            index=["config", option], columns="requested", values=column, aggfunc="min"
        )
        print()
        print(f"{name}, {label} against threads:")
        print(table.to_string(float_format=lambda value: f"{value:9.1f}"))


def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument(
        "--option",
        default="gurobi_method",
        help="dataset column of the option to split the lines by",
    )
    parser.add_argument(
        "--study",
        default=None,
        help="subfolder of figures/ to write into, default the option name",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=FIGURES_PATH,
        help="folder the study subfolder is created in",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="results folder the solver logs are read from, when the path a run "
        "recorded belongs to another machine",
    )
    parser.add_argument(
        "--since",
        default=SEEDED_FROM,
        help="earliest run timestamp to keep, as it appears in the folder name",
    )
    args = parser.parse_args()

    dataset = load(args.dataset, args.since, args.results)
    if dataset.empty:
        print("No runs left after filtering")
        return

    report_coverage(dataset, args.option)

    study = args.study or args.option.replace("gurobi_", "") + "_by_threads"
    output_path = Path(args.figures) / study
    output_path.mkdir(parents=True, exist_ok=True)

    drawn = 0
    for name, draw in [
        ("1_interaction", plot_interaction),
        ("2_phases", plot_phases),
        ("3_effect_size", plot_effect),
        ("4_cores_used", plot_cores),
        ("5_memory_cost", plot_memory),
        ("6_cost_per_iteration", plot_cost_per_iteration),
    ]:
        if draw(dataset, args.option, output_path / f"{name}.png"):
            drawn += 1

    print()
    if drawn:
        print(f"{drawn} of 6 figures written to {output_path}")
    else:
        print(
            f"Nothing drawn yet: {output_path} needs at least two levels of "
            f"{args.option} and two thread counts in one configuration"
        )


if __name__ == "__main__":
    main()
