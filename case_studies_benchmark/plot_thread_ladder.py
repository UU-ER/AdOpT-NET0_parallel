"""
Draws the thread ladder of stage 5.

The paired 4 against 48 thread sweep showed that the cost of a single simplex
iteration, not the number of iterations, is what changes with the thread count.
Three mechanisms produce that signature and a two point comparison cannot tell
them apart:

- memory bandwidth, which the extra threads compete for without adding any
- synchronisation, paid once per iteration and growing with the thread count
- cores spread across sockets, which makes part of the memory remote

The ladder separates them. Cost per iteration that climbs smoothly with the
thread count is bandwidth or synchronisation. A step where the ladder crosses a
socket boundary is the memory topology, and the pinned runs confirm it: the
same number of threads confined to cores that sit together.

Examples::

    python plot_thread_ladder.py
    python plot_thread_ladder.py --dataset benchmark_dataset.csv
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
DATASET_FILE = BASE / "benchmark_dataset.csv"
FIGURES_PATH = BASE / "figures"

# One colour per configuration, in the order they come out of the dataset
COLOURS = ["#3b6ea5", "#d1731f", "#4a8c5f", "#8c4a7d", "#a5433b"]
GREY = "#8a8a8a"

# What is plotted against the thread count, and whether lower is better
PANELS = [
    ("ms_cpu_per_iter", "CPU time per simplex iteration [ms]", True),
    ("ms_wall_per_iter", "Wall time per simplex iteration [ms]", True),
    ("gurobi_runtime_s", "Gurobi runtime [s]", True),
    ("parallelism_solve", "Cores busy during the solve", False),
]


def load(dataset_file: Path, min_thread_counts: int = 3):
    """
    Reads the dataset and picks out the runs of the thread ladder.

    A ladder run is recognised by its case name carrying a ``thr`` code, and
    the configuration it belongs to is what is left of the name once the thread
    and the affinity codes are taken off. A whole sweep can be run at a single
    thread count, and that is not a ladder, so a configuration only counts once
    it has been measured at several of them.

    :param Path dataset_file: benchmark_dataset.csv to read
    :param int min_thread_counts: how many distinct thread counts a
        configuration needs before it is treated as a ladder
    :return: pandas DataFrame of the ladder runs, with the thread count, the
        pinning and the configuration split out of the case name
    """
    dataset = pd.read_csv(dataset_file)

    threads = dataset["case_name"].str.extract(r"_thr(\d+)", expand=False)
    ladder = dataset[threads.notna()].copy()

    if ladder.empty:
        return ladder

    ladder["threads"] = threads[threads.notna()].astype(int)

    affinity = ladder["case_name"].str.extract(r"_aff(\d+)", expand=False)
    ladder["affinity_cores"] = affinity.fillna(0).astype(int)
    ladder["pinned"] = ladder["affinity_cores"] > 0

    ladder["config"] = (
        ladder["case_name"]
        .str.replace(r"_thr\d+", "", regex=True)
        .str.replace(r"_aff\d+", "", regex=True)
    )

    measured_at = ladder.groupby("config")["threads"].transform("nunique")
    ladder = ladder[measured_at >= min_thread_counts].copy()

    if ladder.empty:
        return ladder

    # A run that hit the time limit has a runtime set by the limit and not by
    # the machine, so it cannot sit on a cost curve
    ladder["censored"] = ladder["termination_condition"] != "optimal"

    iterations = ladder["gurobi_itercount"].replace(0, np.nan)
    cpu = ladder["cpu_user_solve_s"] + ladder["cpu_system_solve_s"]
    ladder["ms_cpu_per_iter"] = 1000 * cpu / iterations
    ladder["ms_wall_per_iter"] = 1000 * ladder["gurobi_runtime_s"] / iterations

    return ladder.sort_values(["config", "pinned", "threads"])


def repeats(ladder: pd.DataFrame):
    """
    Finds configurations run more than once at the same thread count.

    Two runs that agree on every setting should be the same measurement. Where
    they are not, the spread between them is the noise floor, and no effect
    smaller than it can be read off the ladder.

    :param ladder: DataFrame returned by load
    :return: DataFrame with one row per repeated configuration and thread count
    """
    rows = []
    grouped = ladder[~ladder["pinned"]].groupby(["config", "threads"])

    for (config, threads), runs in grouped:
        if len(runs) < 2:
            continue
        runtimes = runs["gurobi_runtime_s"]
        rows.append(
            {
                "config": config.replace("four_node_", ""),
                "threads": threads,
                "runs": len(runs),
                "fastest_s": runtimes.min(),
                "slowest_s": runtimes.max(),
                "spread": runtimes.max() / runtimes.min(),
                "n_nnz_unique": runs["n_nnz"].nunique(),
                "objval_unique": runs["gurobi_objval"].round(0).nunique(),
            }
        )

    return pd.DataFrame(rows)


def plot_ladder(ladder: pd.DataFrame, output: Path):
    """
    Draws the four panels of the ladder, one line per configuration

    :param ladder: DataFrame returned by load
    :param Path output: png file to write
    """
    configs = list(dict.fromkeys(ladder["config"]))

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))

    for axis, (column, label, lower_is_better) in zip(axes.ravel(), PANELS):
        for number, config in enumerate(configs):
            colour = COLOURS[number % len(COLOURS)]
            runs = ladder[ladder["config"] == config]

            # Every run is drawn, and the line goes through the median at each
            # thread count. Where a configuration was run twice at the same
            # count the two markers show the noise directly, and a line through
            # one of them would hide it
            free = runs[~runs["pinned"] & ~runs["censored"]]
            axis.scatter(free["threads"], free[column], s=26, color=colour, zorder=3)
            middle = free.groupby("threads")[column].median()
            axis.plot(
                middle.index,
                middle.values,
                lw=1.6,
                color=colour,
                label=config.replace("four_node_", ""),
            )

            # Censored runs are drawn hollow, so that a curve that flattens
            # because of the time limit cannot be read as a real plateau
            stopped = runs[~runs["pinned"] & runs["censored"]]
            axis.scatter(
                stopped["threads"],
                stopped[column],
                s=44,
                facecolors="none",
                edgecolors=colour,
                linewidths=1.4,
                zorder=4,
            )

            pinned = runs[runs["pinned"] & ~runs["censored"]]
            axis.scatter(
                pinned["threads"],
                pinned[column],
                s=70,
                marker="*",
                color=colour,
                edgecolors="k",
                linewidths=0.5,
                zorder=5,
            )

        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(ladder["threads"].unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("solver threads")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25, which="both", lw=0.5)
        if lower_is_better:
            axis.set_yscale("log")
        axis.set_title(label, fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [
        plt.Line2D(
            [],
            [],
            marker="*",
            ms=11,
            color=GREY,
            ls="none",
            label="pinned to an equal number of cores",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ms=7,
            mfc="none",
            mec=GREY,
            ls="none",
            label="hit the time limit, not a real value",
        ),
    ]
    labels += [handle.get_label() for handle in handles[len(labels) :]]

    figure.legend(
        handles,
        labels,
        fontsize=8.5,
        frameon=False,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.suptitle(
        "Cost of solver work against the thread count\n"
        "a smooth climb is bandwidth or synchronisation, a step is a socket "
        "boundary",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


def report(ladder: pd.DataFrame):
    """
    Prints the numbers behind the figure, including the pinned comparison

    :param ladder: DataFrame returned by load
    """
    columns = [
        "config",
        "threads",
        "pinned",
        "gurobi_runtime_s",
        "gurobi_itercount",
        "gurobi_nodecount",
        "ms_cpu_per_iter",
        "ms_wall_per_iter",
        "parallelism_solve",
        "rss_peak_os_mb",
        "censored",
    ]
    print("\n=== every ladder run ===")
    print(ladder[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    repeated = repeats(ladder)
    if not repeated.empty:
        print("\n=== the noise floor: same configuration, same thread count ===")
        print("These runs differ in nothing that was asked for, so the spread")
        print("between them is the smallest effect the ladder can resolve. A")
        print("differing n_nnz or objective means the model itself was rebuilt")
        print("differently and the two runs are not the same problem.\n")
        print(repeated.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    pinned = ladder[ladder["pinned"]]
    if pinned.empty:
        return

    print("\n=== pinned against free, at the same thread count ===")
    print("A pinned run that is clearly faster means the free run was paying")
    print("for cores spread across sockets rather than for the parallelism.\n")

    rows = []
    for _, run in pinned.iterrows():
        free = ladder[
            (ladder["config"] == run["config"])
            & (ladder["threads"] == run["threads"])
            & (~ladder["pinned"])
        ]
        if free.empty:
            continue
        # Against the median, so that a repeated free run does not decide the
        # comparison by which of its two values happens to come first
        free = free.loc[free["gurobi_runtime_s"].sort_values().index[len(free) // 2]]
        rows.append(
            {
                "config": run["config"].replace("four_node_", ""),
                "threads": run["threads"],
                "free_s": free["gurobi_runtime_s"],
                "pinned_s": run["gurobi_runtime_s"],
                "ratio": run["gurobi_runtime_s"] / free["gurobi_runtime_s"],
                "free_ms_cpu_iter": free["ms_cpu_per_iter"],
                "pinned_ms_cpu_iter": run["ms_cpu_per_iter"],
            }
        )

    if rows:
        print(
            pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )


def main():
    """
    Command line interface of the ladder plot
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument(
        "--output", type=Path, default=FIGURES_PATH / "four_node_thread_ladder.png"
    )
    parser.add_argument(
        "--min-thread-counts",
        dest="min_thread_counts",
        type=int,
        default=3,
        help="how many distinct thread counts a configuration needs before it "
        "is treated as a ladder, so that a whole sweep run at one thread count "
        "is not mistaken for one",
    )
    args = parser.parse_args()

    ladder = load(args.dataset, min_thread_counts=args.min_thread_counts)
    if ladder.empty:
        print(f"No runs with a thread count in the case name found in {args.dataset}")
        return

    print(f"{len(ladder)} ladder runs over {ladder['config'].nunique()} configurations")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_ladder(ladder, args.output)
    report(ladder)


if __name__ == "__main__":
    main()
