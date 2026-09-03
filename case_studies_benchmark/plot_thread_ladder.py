"""
Draws the thread ladder of stage 5.

The paired 4 against 48 thread sweep showed that the cost of a single simplex
iteration, not the number of iterations, is what changes with the thread count.
Three mechanisms produce that signature and a two point comparison cannot tell
them apart:

- memory bandwidth, which the extra threads compete for without adding any
- synchronisation, paid once per iteration and growing with the thread count
- cores spread across sockets, which makes part of the memory remote

It answered none of them. Three configurations were run a second time at four
threads with nothing asked for that differed, and the two runs came out up to a
factor 5.9 apart.

The cause is not that the models are far apart. The typical-day clustering is an
unseeded k-means, so it does come out slightly differently each time, but only
slightly: the two matrices differ by **0.0003 to 0.003 % of their non-zeros**,
and in one of the three pairs by a single non-zero. That is enough. Branch and
bound is chaotic under a perturbation of any size, so a coefficient that moves
flips a branching decision and the tree diverges. This is ordinary MIP
performance variability rather than anything peculiar to ADOPT.

The objective is not used as evidence here. The runs stop at a 2 % MIP gap, so
two of them can report objectives a percent apart while solving exactly the same
problem.

The control is in the same dataset: full resolution does no clustering, its
models come out identical to every digit, and there the thread count moves the
runtime by 3 %.

So this script draws two figures. The ladder itself, each curve against the
spread of the runs that were repeated, and the repeats on their own next to the
full resolution control.

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
from matplotlib.patches import Patch

BASE = Path(__file__).parent
DATASET_FILE = BASE / "benchmark_dataset.csv"
FIGURES_PATH = BASE / "figures"

RED, BLUE, GREEN, DARK, GREY = "#c0392b", "#3b6ea5", "#4a8c5f", "#2b2b2b", "#9a9a9a"

# One colour per configuration in the scaling figure
COLOURS = ["#3b6ea5", "#d1731f", "#4a8c5f", "#8c4a7d", "#a5433b"]


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
    Draws the runtime of each configuration against the thread count.

    One panel per configuration, because they sit decades apart and sharing an
    axis makes all three unreadable. Behind each curve is the spread between
    the runs of that configuration that were repeated at one thread count. That
    band is the smallest difference the ladder could resolve, and a curve that
    does not leave it says nothing about threads.

    :param ladder: DataFrame returned by load
    :param Path output: png file to write
    """
    configs = sorted(
        dict.fromkeys(ladder["config"]),
        key=lambda name: ladder[ladder["config"] == name]["gurobi_runtime_s"].median(),
    )
    ticks = sorted(ladder["threads"].unique())

    figure, axes = plt.subplots(
        len(configs), 1, figsize=(9.6, 3.2 * len(configs)), sharex=True, squeeze=False
    )
    axes = axes.ravel()

    for axis, config in zip(axes, configs):
        runs = ladder[ladder["config"] == config]
        free = runs[~runs["pinned"]]
        finished = free[~free["censored"]]

        band = _noise_band(free)
        if band:
            low, high = band
            axis.axhspan(low, high, color=RED, alpha=0.13, zorder=0)
            for edge in band:
                axis.axhline(edge, color=RED, lw=0.9, ls="--", alpha=0.55, zorder=1)
            axis.annotate(
                "run-to-run noise:\nsame configuration,\nsame thread count",
                xy=(ticks[-1], np.sqrt(low * high)),
                xytext=(-6, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                color=RED,
            )

        axis.scatter(
            finished["threads"],
            finished["gurobi_runtime_s"],
            s=46,
            color=BLUE,
            edgecolors="white",
            linewidths=1,
            zorder=4,
        )
        middle = finished.groupby("threads")["gurobi_runtime_s"].median()
        axis.plot(middle.index, middle.values, lw=1.7, color=BLUE, zorder=3)

        # Censored runs are drawn hollow, so that a curve flattening against the
        # time limit cannot be read as a real plateau
        stopped = free[free["censored"]]
        axis.scatter(
            stopped["threads"],
            stopped["gurobi_runtime_s"],
            s=60,
            facecolors="none",
            edgecolors=BLUE,
            linewidths=1.5,
            zorder=4,
        )

        pinned = runs[runs["pinned"]]
        axis.scatter(
            pinned["threads"],
            pinned["gurobi_runtime_s"],
            s=120,
            marker="*",
            color="white",
            edgecolors=DARK,
            linewidths=1.1,
            zorder=5,
        )

        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks)
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_yscale("log")
        axis.grid(alpha=0.22, which="both", lw=0.5)
        axis.set_ylabel("gurobi runtime [s]")

        title = config.replace("four_node_", "")
        if len(finished) > 1:
            span = (
                finished["gurobi_runtime_s"].max() / finished["gurobi_runtime_s"].min()
            )
            title += f"   -   the whole ladder spans x{span:.1f}"
            if band:
                title += f", the noise alone is x{band[1] / band[0]:.1f}"
        axis.set_title(title, fontsize=10.5, loc="left", pad=6)

    axes[-1].set_xlabel("solver threads")
    axes[0].legend(
        handles=[
            plt.Line2D([], [], marker="o", ms=7, color=BLUE, lw=1.7, label="one run"),
            plt.Line2D(
                [],
                [],
                marker="o",
                ms=8,
                mfc="none",
                mec=BLUE,
                ls="none",
                label="hit the time limit",
            ),
            plt.Line2D(
                [],
                [],
                marker="*",
                ms=12,
                mfc="white",
                mec=DARK,
                ls="none",
                label="pinned to an equal number of cores",
            ),
            Patch(fc=RED, alpha=0.13, ec=RED, ls="--", label="run-to-run noise"),
        ],
        fontsize=8.5,
        frameon=False,
        ncol=2,
        loc="upper left",
    )

    figure.suptitle(
        "The thread ladder cannot separate the thread count from the noise\n"
        "and the band is only two runs, so the true noise is wider",
        fontsize=12,
        y=0.997,
    )
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.075 / len(configs) * 3))
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


SCALED = [
    ("gurobi_runtime_s", "Runtime", "how long it took", "chaotic: no usable trend"),
    (
        "ms_cpu_per_iter",
        "CPU per simplex iteration",
        "what a unit of work costs",
        "roughly doubles",
    ),
    ("rss_peak_os_mb", "Peak memory", "what the solver held", "roughly quadruples"),
]


def plot_what_scales(ladder: pd.DataFrame, output: Path):
    """
    Draws what the thread count does move, next to what it does not.

    Runtime is the product of how much work the search happened to do, which is
    chaotic, and what a unit of that work costs, which is physical. Dividing the
    chaotic factor out leaves two quantities that do rise cleanly with the
    thread count and that the noise does not swamp.

    Runs that closed at the root are marked, since for those the tree cannot be
    the explanation and whatever is left is the thread count.

    :param ladder: DataFrame returned by load
    :param Path output: png file to write
    """
    runs = ladder[~ladder["pinned"] & ~ladder["censored"]]
    configs = sorted(
        dict.fromkeys(runs["config"]),
        key=lambda name: runs[runs["config"] == name]["gurobi_runtime_s"].median(),
    )
    ticks = sorted(runs["threads"].unique())

    figure, axes = plt.subplots(1, len(SCALED), figsize=(14.5, 5.0), sharex=True)

    for axis, (column, title, subtitle, verdict) in zip(axes, SCALED):
        for number, config in enumerate(configs):
            group = runs[runs["config"] == config].sort_values("threads")
            # Each configuration is normalised by its own cheapest thread count,
            # which is not always 1: a run that hit the time limit is out
            base = group[group["threads"] == group["threads"].min()][column].median()
            scaled = group[column] / base

            at_root = group["gurobi_nodecount"] == 1
            axis.scatter(
                group["threads"][~at_root],
                scaled[~at_root],
                s=40,
                color=COLOURS[number % len(COLOURS)],
                edgecolors="white",
                linewidths=0.9,
                zorder=3,
            )
            axis.scatter(
                group["threads"][at_root],
                scaled[at_root],
                s=64,
                marker="D",
                color=COLOURS[number % len(COLOURS)],
                edgecolors="white",
                linewidths=0.9,
                zorder=4,
            )
            middle = scaled.groupby(group["threads"]).median()
            axis.plot(
                middle.index,
                middle.values,
                lw=1.7,
                color=COLOURS[number % len(COLOURS)],
            )

        axis.axhline(1, color=DARK, lw=1, ls="--", zorder=1)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(ticks)
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_yticks([0.25, 0.5, 1, 2, 4, 8])
        axis.set_yticklabels(["0.25x", "0.5x", "1x", "2x", "4x", "8x"])
        axis.set_ylim(0.16, 9)
        axis.set_xlabel("solver threads")
        axis.grid(alpha=0.22, which="both", lw=0.5)
        axis.set_title(f"{title}\n{subtitle}", fontsize=11)
        axis.annotate(
            verdict,
            xy=(0.5, 0.965),
            xycoords="axes fraction",
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=DARK,
        )

    axes[0].set_ylabel(
        "relative to the same configuration\nat its fewest measured threads"
    )

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ms=7,
            color=COLOURS[number % len(COLOURS)],
            lw=1.7,
            label=config.replace("four_node_", ""),
        )
        for number, config in enumerate(configs)
    ]
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="D",
            ms=7,
            color=GREY,
            ls="none",
            label="closed at the root (1 node): the search cannot explain it",
        )
    )
    figure.legend(
        handles=handles,
        fontsize=9,
        frameon=False,
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    figure.suptitle(
        "Same configuration, more threads: runtime tells you nothing,\n"
        "memory and cost per unit of work tell you a lot",
        fontsize=12.5,
        y=1.0,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.98))
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Wrote {output}")


def _noise_band(free: pd.DataFrame):
    """
    Returns the widest spread between runs repeated at one thread count

    :param free: the unpinned runs of one configuration
    :return: (fastest, slowest) tuple, or None if nothing was repeated
    """
    widest = None
    for _, runs in free.groupby("threads"):
        if len(runs) < 2:
            continue
        low = runs["gurobi_runtime_s"].min()
        high = runs["gurobi_runtime_s"].max()
        if widest is None or high / low > widest[1] / widest[0]:
            widest = (low, high)

    return widest


def plot_noise(ladder: pd.DataFrame, dataset_file: Path, output: Path):
    """
    Draws the repeated runs against a run whose model is provably identical.

    Runs with typical days rebuild the model with an unseeded k-means. The
    clustering is nearly stable, so the two models are all but the same, and
    that is the point: a difference of a few thousandths of a percent in the
    matrix moves the runtime by a factor of several. Full resolution does no
    clustering at all and is the control.

    The objective is deliberately not used as evidence. The runs stop at a 2 %
    MIP gap, so two of them can report objectives that far apart while solving
    exactly the same problem.

    :param ladder: DataFrame returned by load
    :param Path dataset_file: the dataset, re-read to reach the control runs
    :param Path output: png file to write
    """
    rows = []
    for (config, _), runs in ladder[~ladder["pinned"]].groupby(["config", "threads"]):
        if len(runs) < 2:
            continue
        rows.append(_pair(config, runs, clustered=True))

    dataset = pd.read_csv(dataset_file)
    dataset["stem"] = dataset["case_name"].str.replace(r"_thr\d+$", "", regex=True)
    for stem, runs in dataset[dataset["typicaldays_n"] == 0].groupby("stem"):
        if len(runs) > 1 and (runs["termination_condition"] == "optimal").all():
            rows.append(_pair(stem, runs, clustered=False))

    if not rows:
        print("Nothing was run twice, so there is no noise to show")
        return

    pairs = pd.DataFrame(rows).sort_values("ratio")

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.6, 1.1 * len(pairs) + 2.4),
        gridspec_kw={"width_ratios": [1.75, 1]},
    )
    positions = np.arange(len(pairs))

    left = axes[0]
    for position, pair in enumerate(pairs.itertuples()):
        colour = RED if pair.clustered else GREEN
        left.plot(
            [pair.low, pair.high],
            [position, position],
            lw=3.4,
            color=colour,
            alpha=0.55,
            solid_capstyle="round",
            zorder=2,
        )
        left.scatter(
            [pair.low, pair.high],
            [position, position],
            s=68,
            color=colour,
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )
        left.annotate(
            f"x{pair.ratio:.2f}",
            xy=(pair.high, position),
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=colour,
        )

    left.set_yticks(positions)
    left.set_yticklabels(
        [
            f"{pair.label}\n{'clustered' if pair.clustered else 'full resolution, no clustering'}"
            for pair in pairs.itertuples()
        ],
        fontsize=9,
    )
    left.set_xscale("log")
    left.set_ylim(-0.65, len(pairs) - 0.35)
    left.set_xlabel("gurobi runtime [s], log scale")
    left.set_title(
        "Two runs of the same configuration,\nnothing asked for that differs",
        fontsize=11,
        pad=10,
    )
    left.set_xlim(pairs["low"].min() * 0.6, pairs["high"].max() * 2.6)
    left.grid(alpha=0.25, axis="x", which="both", lw=0.5)
    left.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                color=RED,
                lw=3.4,
                alpha=0.55,
                marker="o",
                ms=8,
                markerfacecolor=RED,
                label="typical days: k-means rebuilds the model",
            ),
            plt.Line2D(
                [],
                [],
                color=GREEN,
                lw=3.4,
                alpha=0.55,
                marker="o",
                ms=8,
                markerfacecolor=GREEN,
                label="full resolution: identical model, the control",
            ),
        ],
        fontsize=8.5,
        frameon=False,
        loc="upper left",
    )

    right = axes[1]
    right.barh(
        positions,
        pairs["model_pct"],
        height=0.55,
        color=[RED if pair.clustered else GREEN for pair in pairs.itertuples()],
        alpha=0.75,
        edgecolor="k",
        linewidth=0.5,
    )
    for position, pair in enumerate(pairs.itertuples()):
        identical = pair.model_pct == 0
        right.annotate(
            "the same matrix" if identical else f"{pair.model_pct:.4f} %",
            xy=(pair.model_pct, position),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=GREEN if identical else DARK,
            fontweight="bold" if identical else "normal",
        )
    right.set_yticks(positions)
    right.set_yticklabels([])
    right.set_ylim(-0.65, len(pairs) - 0.35)
    right.set_xlim(0, max(pairs["model_pct"].max() * 1.9, 0.001))
    right.set_xlabel("difference in the number of non-zeros [%]")
    right.set_title("and the two models are all but identical", fontsize=11, pad=10)
    right.grid(alpha=0.25, axis="x", lw=0.5)

    figure.suptitle(
        "A few thousandths of a percent of the matrix moves the runtime by a "
        "factor of several:\nMIP solve time is chaotic, so the benchmark "
        "cannot resolve anything below it",
        fontsize=12,
        y=1.04,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    print(f"Wrote {output}")


def _pair(label: str, runs: pd.DataFrame, clustered: bool):
    """
    Summarises two runs of one configuration into a row of the noise figure

    :param str label: name of the configuration
    :param runs: the runs, two or more
    :param bool clustered: False for a full resolution run, which is the control
    :return: dict with the runtimes, their ratio and the objective difference
    """
    ordered = runs.sort_values("gurobi_runtime_s")
    low = ordered["gurobi_runtime_s"].iloc[0]
    high = ordered["gurobi_runtime_s"].iloc[-1]
    objectives = ordered["gurobi_objval"]

    nonzeros = ordered["n_nnz"]

    return {
        "label": label.replace("four_node_", "").replace("_thr4", ""),
        "low": low,
        "high": high,
        "ratio": high / low,
        # How far apart the two matrices are. The objective is not used: the
        # runs stop at a 2 % MIP gap, so it can differ by that much on one and
        # the same problem
        "model_pct": 100 * (nonzeros.max() - nonzeros.min()) / nonzeros.min(),
        "objective_pct": 100
        * abs(objectives.max() - objectives.min())
        / abs(objectives.iloc[0]),
        "clustered": clustered,
    }


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
    plot_noise(
        ladder, args.dataset, args.output.with_name("four_node_run_to_run_noise.png")
    )
    plot_what_scales(ladder, args.output.with_name("four_node_threads_what_scales.png"))
    report(ladder)


if __name__ == "__main__":
    main()
