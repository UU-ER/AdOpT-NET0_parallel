"""
Draws what a thread count costs and what it buys, at the defaults.

Stage 10 fills in the thread counts between the levels stages 6 to 9 used, so
for the first time there is a ladder fine enough to find the edges rather than
interpolate across them. No option is varied: stages 6 to 9 found nothing
interacting with the thread count, so this is a question about the machine.

The question it answers is not "how fast is one run". It is **how many runs an
hour a fixed pile of cores delivers**, which is what a campaign is made of. A
job that asks for k threads occupies k cores, so a budget of C cores runs
floor(C / k) of them at once and the throughput is floor(C / k) / runtime. A
thread count that halves the runtime and quadruples the request is a loss.

Four figures, each answering one question:

1. ladder       what runtime and peak memory do as the thread count rises
2. throughput   how many runs an hour a core budget delivers, and the memory
                it has to have to run them side by side
3. efficiency   how far the speedup is from the cores it was given
4. regime       whether the solver changes strategy rather than scaling

Peak memory is drawn beside the throughput on purpose. The two point opposite
ways: thin jobs give the most throughput and need the most memory in total,
because memory per job falls more slowly than the job count rises. Which of the
two binds is what picks the thread count, and it depends on the machine.

Only runs made after the clustering was seeded are used, and only cells at the
solver defaults, so that an option of another stage cannot enter the ladder.

Examples::

    python plot_thread_resolution.py
    python plot_thread_resolution.py --cores 128 --study stage10_threads
    python plot_thread_resolution.py --dataset benchmark_dataset.csv --cores 48
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATASET_FILE = BASE / "benchmark_dataset.csv"
FIGURES_PATH = BASE / "figures"

BLUE, ORANGE, RED = "#3b6ea5", "#d1731f", "#c0392b"
GREEN, DARK, GREY = "#4a8c5f", "#2b2b2b", "#9a9a9a"

# One hue per configuration, in a fixed order, so a configuration keeps its
# colour across all four figures. Checked for colour vision deficiency
# separation rather than chosen by eye
CONFIG_COLORS = [BLUE, ORANGE, GREEN, RED]

# The clustering was seeded on 2026-09-11. Runs before that rebuilt the model
# differently every time, so they are a different problem
SEEDED_FROM = "20260911"

# A cell of this stage carries no solver option, so any of these tokens in the
# case name means the run belongs to another stage and is not on the ladder
OPTION_TOKENS = re.compile(r"_(mth|cut|cp|mf|lpw|xov|bh|scl|cm|nm|pre|heu|nrh|nf|bd)")

# Tokens that are settings rather than complexity, stripped to leave the
# configuration the panels are grouped by
SETTING_TOKENS = re.compile(r"_(thr|aff)-?[\d.]+")

# Cores a campaign has to spend. 100 is the question as it was asked: is it
# better to run 100 jobs of one thread or 50 of two
DEFAULT_CORES = 100

TITLE_INCHES = 0.95
LEGEND_INCHES = 0.62


def load(dataset_file: Path, since: str = SEEDED_FROM):
    """
    Reads the dataset and keeps the cells that are on the ladder

    :param Path dataset_file: benchmark_dataset.csv to read
    :param str since: earliest run timestamp to keep, as it appears in the
        result folder name
    :return: pandas DataFrame with one row per thread count and configuration
    """
    dataset = pd.read_csv(dataset_file)
    dataset = dataset[dataset["case_name"].str.startswith("four_node")].copy()

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
    if before - len(dataset):
        print(f"Dropped {before - len(dataset)} runs from before {since}, unseeded")

    before = len(dataset)
    dataset = dataset[~dataset["case_name"].str.contains(OPTION_TOKENS)].copy()
    if before - len(dataset):
        print(f"Dropped {before - len(dataset)} runs that carry a solver option")

    # 0 threads means every core of the machine, which on this server is 48
    dataset["threads"] = dataset["gurobi_threads"].replace(0, 48)
    dataset["config"] = (
        dataset["case_name"].str.replace(SETTING_TOKENS, "", regex=True)
    ).str.replace("four_node_", "", regex=False)

    # A cell run more than once is one point, and the median is the honest
    # summary of a quantity as chaotic as a MIP runtime
    columns = {
        "wall_total_s": "median",
        "gurobi_runtime_s": "median",
        "rss_peak_os_mb": "median",
        "cpu_user_s": "median",
        "gurobi_itercount": "median",
    }
    columns = {name: how for name, how in columns.items() if name in dataset}
    ladder = dataset.groupby(["config", "threads"]).agg(columns).reset_index()

    return ladder.sort_values(["config", "threads"])


def ladders(dataset: pd.DataFrame, minimum: int = 6):
    """
    The configurations with enough thread counts to draw a line through.

    Stages 6 to 9 measured four thread counts and stage 10 nine. A line through
    the four crosses the step the ninth was run to find, without showing it, so
    a configuration is drawn only if it carries most of the ladder.

    :param pd.DataFrame dataset: the ladder dataset
    :param int minimum: fewest thread counts a configuration needs
    :return: list of (name, group) pairs, cheapest configuration first
    """
    groups = [
        (name, group)
        for name, group in dataset.groupby("config")
        if len(group) >= minimum
    ]
    return sorted(groups, key=lambda pair: pair[1]["wall_total_s"].median())


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
    axis.set_axisbelow(True)


def _finish(figure, legend_axis, title: str, output: Path):
    """
    Puts the shared legend and title on a figure and writes it

    :param figure: matplotlib figure
    :param legend_axis: axis whose handles make the legend, or None
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
            ncol=min(len(labels), 4),
            frameon=False,
        )

    height = figure.get_size_inches()[1]
    figure.suptitle(title, fontsize=12, y=1 - 0.12 / height, va="top")
    figure.tight_layout(
        rect=(
            0,
            (LEGEND_INCHES / height) if handles else 0,
            1,
            1 - TITLE_INCHES / height,
        )
    )
    figure.subplots_adjust(top=1 - TITLE_INCHES / height)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Wrote {output}")


def _mark_cliff(axis, group, column):
    """
    Marks the step where a response jumps rather than rises.

    The ladder was run to find an edge, so the edge is drawn rather than left
    to the reader. A step is marked when it is at least half again the value
    before it, which no smooth scaling produces between two neighbouring
    thread counts.

    :param axis: matplotlib axis
    :param pd.DataFrame group: rows of one configuration, sorted by threads
    :param str column: response to look at
    :return: the thread count the jump lands on, or None
    """
    values = group[column].to_numpy()
    threads = group["threads"].to_numpy()
    for index in range(1, len(values)):
        if values[index] >= 1.5 * values[index - 1]:
            axis.axvspan(
                threads[index - 1], threads[index], color=RED, alpha=0.07, lw=0
            )
            return threads[index]
    return None


def plot_ladder(dataset: pd.DataFrame, cores: int, output: Path):
    """
    Runtime and peak memory against the thread count, one panel per response
    and configuration, because the two responses share no scale.

    :param pd.DataFrame dataset: the ladder dataset
    :param int cores: core budget, unused here, kept for a uniform signature
    :param Path output: png file to write
    """
    groups = ladders(dataset)
    responses = [
        ("wall_total_s", "wall time [s]"),
        ("rss_peak_os_mb", "peak memory [MB]"),
    ]

    figure, axes = plt.subplots(
        len(responses),
        len(groups),
        figsize=(max(8.4, 4.2 * len(groups)), 3.1 * len(responses)),
        squeeze=False,
    )

    for row, (column, label) in enumerate(responses):
        for col, (name, group) in enumerate(groups):
            axis = axes[row][col]
            colour = CONFIG_COLORS[col % len(CONFIG_COLORS)]

            cliff = _mark_cliff(axis, group, column)
            axis.plot(
                group["threads"],
                group[column],
                marker="o",
                ms=5,
                lw=2,
                color=colour,
            )

            best = group.loc[group[column].idxmin()]
            axis.annotate(
                f"{best[column]:.0f}",
                (best["threads"], best[column]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8,
                color=DARK,
            )
            if cliff is not None and row == 1:
                axis.annotate(
                    f"jumps at {int(cliff)}",
                    (cliff, group[column].max()),
                    textcoords="offset points",
                    xytext=(-6, -10),
                    ha="right",
                    fontsize=8,
                    color=RED,
                )

            _thread_axis(axis, group["threads"])
            axis.set_yscale("log")
            axis.set_ylabel(label if col == 0 else "")
            axis.set_xlabel("threads requested" if row == len(responses) - 1 else "")
            if row == 0:
                axis.set_title(name, fontsize=11)

    _finish(
        figure,
        None,
        "What a thread count costs and what it buys\n"
        "the shaded step is where the response jumps rather than rises",
        output,
    )


def plot_throughput(dataset: pd.DataFrame, cores: int, output: Path):
    """
    Runs an hour a core budget delivers, and the memory that costs.

    The two panels are the whole decision: the top one is what a campaign
    gets, the bottom one is what the machine has to have for it. They point
    opposite ways, so neither is drawn without the other.

    :param pd.DataFrame dataset: the ladder dataset
    :param int cores: cores the campaign has to spend
    :param Path output: png file to write
    """
    groups = ladders(dataset)

    figure, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)

    for index, (name, group) in enumerate(groups):
        colour = CONFIG_COLORS[index % len(CONFIG_COLORS)]
        jobs = np.floor(cores / group["threads"].to_numpy())
        hourly = jobs / group["wall_total_s"].to_numpy() * 3600
        total_gb = jobs * group["rss_peak_os_mb"].to_numpy() / 1024

        axes[0].plot(
            group["threads"], hourly, marker="o", ms=5, lw=2, color=colour, label=name
        )
        axes[1].plot(group["threads"], total_gb, marker="o", ms=5, lw=2, color=colour)

        best = int(np.argmax(hourly))
        axes[0].annotate(
            f"{name}: {hourly[best]:.0f}/h at {int(group['threads'].iloc[best])}",
            (group["threads"].iloc[best], hourly[best]),
            textcoords="offset points",
            xytext=(10, -3 + 11 * index),
            fontsize=8,
            color=colour,
        )
        # The bump: the thread count where packing the cores needs more
        # memory in total than the one below it, which is the step of figure 1
        # arriving in the only units a cluster request is written in
        threads = group["threads"].to_numpy()
        bump = next(
            (
                position
                for position in range(1, len(total_gb))
                if total_gb[position] > total_gb[position - 1]
            ),
            None,
        )
        if bump is not None:
            axes[1].annotate(
                f"{total_gb[bump]:.0f} GB at {int(threads[bump])}, "
                f"worse than {total_gb[bump - 1]:.0f} at {int(threads[bump - 1])}",
                (threads[bump], total_gb[bump]),
                textcoords="offset points",
                xytext=(10, 6 if index == 0 else -14),
                fontsize=8,
                color=colour,
            )

    axes[0].set_ylabel("runs per hour")
    axes[1].set_ylabel("memory for all of them [GB]")
    for axis in axes:
        axis.yaxis.set_label_coords(-0.085, 0.5)
    axes[1].set_xlabel("threads per job")
    for axis in axes:
        _thread_axis(axis, sorted(dataset["threads"].unique()))
        axis.set_yscale("log")

    _finish(
        figure,
        axes[0],
        f"How to spend {cores} cores\n"
        "top: what the campaign gets. bottom: what the machine must have",
        output,
    )


def plot_efficiency(dataset: pd.DataFrame, cores: int, output: Path):
    """
    Speedup against the cores it was given.

    The ideal line is there to be missed: a MIP whose root is nearly serial
    cannot follow it, and the gap is the argument for asking for few cores.

    :param pd.DataFrame dataset: the ladder dataset
    :param int cores: core budget, unused here
    :param Path output: png file to write
    """
    groups = ladders(dataset)

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    ticks = sorted(dataset["threads"].unique())
    ideal = np.array(ticks, dtype=float)
    axes[0].plot(ticks, ideal, ls="--", lw=1.4, color=GREY, label="ideal, k times")

    for index, (name, group) in enumerate(groups):
        colour = CONFIG_COLORS[index % len(CONFIG_COLORS)]
        base = group.loc[group["threads"].idxmin(), "wall_total_s"]
        speedup = base / group["wall_total_s"].to_numpy()
        axes[0].plot(
            group["threads"], speedup, marker="o", ms=5, lw=2, color=colour, label=name
        )
        axes[1].plot(
            group["threads"],
            speedup / group["threads"].to_numpy() * 100,
            marker="o",
            ms=5,
            lw=2,
            color=colour,
        )

    axes[0].set_ylabel("speedup against 1 thread")
    axes[0].set_yscale("log", base=2)
    axes[0].set_yticks([0.5, 1, 2, 4, 8, 16, 32])
    axes[0].set_yticklabels(["0.5", "1", "2", "4", "8", "16", "32"])
    axes[1].set_ylabel("cores actually paying off [%]")
    axes[1].axhline(100, ls="--", lw=1.4, color=GREY)

    for axis in axes:
        _thread_axis(axis, ticks)
        axis.set_xlabel("threads requested")

    _finish(
        figure,
        axes[0],
        "How much of the machine is doing anything\n"
        "the distance below the dashed line is the part that is wasted",
        output,
    )


def plot_regime(dataset: pd.DataFrame, cores: int, output: Path):
    """
    Whether the solver scales or switches strategy.

    Scaling moves the responses smoothly. A switch moves them once and then
    holds them: identical iteration counts and identical peak memory across
    several thread counts are not a solver using more of the machine, they are
    a solver doing the same different thing.

    :param pd.DataFrame dataset: the ladder dataset
    :param int cores: core budget, unused here
    :param Path output: png file to write
    """
    groups = ladders(dataset)
    responses = [
        ("gurobi_itercount", "simplex iterations"),
        ("rss_peak_os_mb", "peak memory [MB]"),
    ]
    responses = [pair for pair in responses if pair[0] in dataset]

    figure, axes = plt.subplots(1, len(responses), figsize=(9.6, 4.0), squeeze=False)

    for col, (column, label) in enumerate(responses):
        axis = axes[0][col]
        for index, (name, group) in enumerate(groups):
            colour = CONFIG_COLORS[index % len(CONFIG_COLORS)]
            values = group[column].to_numpy()
            axis.plot(
                group["threads"],
                values,
                marker="o",
                ms=5,
                lw=2,
                color=colour,
                label=name if col == 0 else None,
            )

            # A plateau is the evidence, so it is drawn rather than described
            plateau = [
                position
                for position in range(1, len(values))
                if values[position] == values[position - 1]
            ]
            if len(plateau) >= 2:
                first = group["threads"].iloc[plateau[0] - 1]
                axis.axvspan(first, group["threads"].max(), color=colour, alpha=0.06, lw=0)
                if col == 0:
                    axis.annotate(
                        f"identical from {int(first)} on",
                        (first, values[plateau[0]]),
                        textcoords="offset points",
                        xytext=(6, 10),
                        fontsize=8,
                        color=colour,
                    )

        _thread_axis(axis, sorted(dataset["threads"].unique()))
        axis.set_yscale("log")
        axis.set_ylabel(label)
        axis.set_xlabel("threads requested")

    _finish(
        figure,
        axes[0][0],
        "Does the solver scale, or does it switch strategy\n"
        "a shaded stretch is where more threads changed nothing at all",
        output,
    )


def report(dataset: pd.DataFrame, cores: int):
    """
    Prints the ladder and the throughput it implies

    :param pd.DataFrame dataset: the ladder dataset
    :param int cores: cores the campaign has to spend
    """
    for name, group in ladders(dataset):
        print(f"\n{name}, {cores} cores, jobs = floor({cores} / k):")
        print(
            f"  {'k':>4} {'wall s':>9} {'mem MB':>9} "
            f"{'jobs':>6} {'runs/h':>9} {'total GB':>9}"
        )
        for _, row in group.iterrows():
            jobs = cores // int(row["threads"])
            hourly = jobs / row["wall_total_s"] * 3600
            print(
                f"  {int(row['threads']):4d} {row['wall_total_s']:9.0f} "
                f"{row['rss_peak_os_mb']:9.0f} {jobs:6d} {hourly:9.1f} "
                f"{jobs * row['rss_peak_os_mb'] / 1024:9.1f}"
            )


def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument(
        "--figures",
        type=Path,
        default=FIGURES_PATH,
        help="folder the study subfolder is created in",
    )
    parser.add_argument(
        "--study",
        default="stage10_threads",
        help="subfolder of figures/ to write into",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=DEFAULT_CORES,
        help="cores a campaign has to spend, which sets how many jobs of k "
        "threads run side by side",
    )
    parser.add_argument(
        "--since",
        default=SEEDED_FROM,
        help="earliest run timestamp to keep, as it appears in the folder name",
    )
    args = parser.parse_args()

    dataset = load(args.dataset, args.since)
    if dataset.empty:
        print("No runs left after filtering")
        return

    kept = ladders(dataset)
    if not kept:
        print("No configuration has enough thread counts to draw a ladder")
        return
    for name, group in kept:
        print(f"{name}: {len(group)} thread counts, {sorted(group['threads'])}")

    report(dataset, args.cores)

    output_path = Path(args.figures) / args.study
    output_path.mkdir(parents=True, exist_ok=True)

    drawn = 0
    for name, draw in [
        ("1_ladder", plot_ladder),
        ("2_throughput", plot_throughput),
        ("3_efficiency", plot_efficiency),
        ("4_regime", plot_regime),
    ]:
        try:
            draw(dataset, args.cores, output_path / f"{name}.png")
            drawn += 1
        except (KeyError, ValueError) as error:
            print(f"Could not draw {name}: {type(error).__name__}: {error}")

    print(f"\n{drawn} of 4 figures written to {output_path}")


if __name__ == "__main__":
    main()
