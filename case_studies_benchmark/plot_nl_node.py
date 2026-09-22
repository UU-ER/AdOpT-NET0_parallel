"""
Draws the figures of the nine node case study, stages 12 and 13.

Four figures, one per finding:

1. ``core_seconds``   what a run costs in reserved core seconds at each thread
                      count. The cheapest cell is one thread on td4 and two on
                      td15, which is the result the stage exists for
2. ``root_algorithm`` why. Below two threads gurobi cannot run its concurrent
                      root method, so the single thread run pays four times the
                      simplex iterations on the larger model
3. ``memory``         peak memory against the thread count, and the step
                      between four and six threads
4. ``cuts``           stage 12, cuts = 0 against the default, paired

Run it from the benchmark folder::

    python plot_nl_node.py
    python plot_nl_node.py --dataset benchmark_dataset.csv --figures figures

Every figure is drawn from whatever cells are on disk, so it degrades to the
resolutions that have been run rather than failing while a stage is half done.
Each one is written as a png for a slide and a pdf for the paper.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import (
    PALETTE,
    annotate_bars,
    apply_publication_style,
    faint_grid,
    finalize,
)

BASE = Path(__file__).parent
DATASET_FILE = BASE / "benchmark_dataset.csv"
FIGURES_PATH = BASE / "figures" / "nl_node"

CASE = "nl_node"

# Resolutions, in the order they are drawn, with the label and the colour they
# carry. Blue is the small model, red the large one, kept the same in every
# figure so a reader learns the mapping once
RESOLUTIONS = [
    (4, "4 typical days", PALETTE["blue_main"]),
    (15, "15 typical days", PALETTE["red_strong"]),
]


def load(dataset: Path):
    """
    Reads the dataset and keeps the runs of this case study.

    :param Path dataset: benchmark_dataset.csv
    :return: DataFrame with one row per run, with the thread count and the
        resolution as columns rather than as pieces of the case name
    """
    data = pd.read_csv(dataset)
    data = data[data["case_name"].str.startswith(f"{CASE}_")].copy()

    data["typicaldays"] = data["case_name"].str.extract(r"_td(\d+)").astype(int)
    data["threads"] = data["case_name"].str.extract(r"_thr(\d+)").astype(int)
    data["cuts_off"] = data["case_name"].str.contains("_cut0")

    # Reserved core seconds, which is what a scheduler bills and therefore
    # what sets the throughput of a campaign. The CPU a run actually burns is
    # a different and smaller number, and it is not what a job is charged
    data["core_seconds"] = data["threads"] * data["wall_total_s"]

    return data.sort_values(["typicaldays", "cuts_off", "threads"])


def _ladder(data: pd.DataFrame, typicaldays: int):
    """
    The thread ladder of one resolution, at the solver defaults

    :param DataFrame data: every run of the case study
    :param int typicaldays: resolution to pick
    :return: DataFrame sorted by thread count
    """
    ladder = data[(data["typicaldays"] == typicaldays) & ~data["cuts_off"]]
    return ladder.sort_values("threads")


def figure_core_seconds(data: pd.DataFrame, output: Path):
    """
    Reserved core seconds against the thread count, one panel per resolution.

    Each panel is normalised to its own one thread cell and carries its own y
    axis: the two resolutions differ by a factor twenty in absolute cost, and
    sharing the axis would flatten the td15 panel, which is the one holding
    the result. The cheapest cell is filled, the others are hollow.

    :param DataFrame data: every run of the case study
    :param Path output: file to write, without a suffix
    """
    figure, axes = plt.subplots(1, len(RESOLUTIONS), figsize=(11, 4.2))

    for axis, (typicaldays, label, colour) in zip(axes, RESOLUTIONS):
        ladder = _ladder(data, typicaldays)
        if ladder.empty:
            continue

        relative = (ladder["core_seconds"] / ladder["core_seconds"].iloc[0]).values
        best = int(relative.argmin())
        positions = range(len(ladder))

        bars = axis.bar(
            positions,
            relative,
            color=[colour if i == best else "white" for i in positions],
            edgecolor=colour,
            linewidth=1.8,
            zorder=3,
        )
        annotate_bars(axis, bars, fmt="{:.2f}", fontsize=11)

        axis.axhline(1, color=PALETTE["grey_text"], linewidth=1.2, zorder=4)
        axis.set_xticks(list(positions))
        axis.set_xticklabels(ladder["threads"])
        axis.set_xlabel("threads asked for")
        axis.set_ylim(0, relative.max() * 1.18)

        cheapest = int(ladder["threads"].iloc[best])
        axis.set_title(
            f"{label}: cheapest at {cheapest} thread{'s' if cheapest > 1 else ''}"
        )
        faint_grid(axis)

    axes[0].set_ylabel("reserved core seconds,\nrelative to one thread")
    figure.suptitle(
        "The cheapest thread count moves with the size of the model",
        fontsize=16,
    )
    return finalize(figure, output)


def figure_root_algorithm(data: pd.DataFrame, output: Path):
    """
    Simplex iterations and wall time against the thread count.

    The mechanism behind the first figure. The step from one thread to two is
    not parallelism: it is gurobi switching to its concurrent root method,
    which closes the relaxation in a quarter of the iterations. Above two
    threads both panels are flat, and on td4 the cells at six, sixteen and
    forty eight threads are the same run to the iteration.

    :param DataFrame data: every run of the case study
    :param Path output: file to write, without a suffix
    """
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for typicaldays, label, colour in RESOLUTIONS:
        ladder = _ladder(data, typicaldays)
        if ladder.empty:
            continue

        # Relative to the single thread cell of the same resolution. The two
        # resolutions are a factor thirty apart in both quantities, so on
        # absolute axes the small one is a flat line at the bottom and its
        # shape, which is the control, cannot be read
        for axis, column in zip(axes, ["gurobi_itercount", "wall_total_s"]):
            axis.plot(
                ladder["threads"],
                ladder[column] / ladder[column].iloc[0],
                marker="o",
                color=colour,
                label=label,
            )

    ticks = sorted(data["threads"].unique())
    for axis, ylabel, title in zip(
        axes,
        [
            "simplex iterations,\nrelative to one thread",
            "wall time,\nrelative to one thread",
        ],
        ["the work the solver does", "the time it takes"],
    ):
        axis.axhline(1, color=PALETTE["grey_text"], linewidth=1.0, zorder=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks)
        axis.set_xticklabels(ticks)
        axis.minorticks_off()
        axis.set_xlabel("threads asked for")
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.set_title(title)
        axis.legend()
        faint_grid(axis)

    figure.suptitle(
        "A single thread cannot run the concurrent root method", fontsize=16
    )
    return finalize(figure, output)


def figure_memory(data: pd.DataFrame, output: Path):
    """
    Peak memory against the thread count, both resolutions on one axis.

    Memory per job is what decides how many jobs fit on a machine, and it
    steps rather than scales: flat to four threads, a factor 1.6 to 2.5 in
    the single step to six, flat again above it.

    :param DataFrame data: every run of the case study
    :param Path output: file to write, without a suffix
    """
    figure, axis = plt.subplots(figsize=(7.2, 4.4))

    for typicaldays, label, colour in RESOLUTIONS:
        ladder = _ladder(data, typicaldays)
        if ladder.empty:
            continue
        axis.plot(
            ladder["threads"],
            ladder["rss_peak_os_mb"] / 1024,
            marker="o",
            color=colour,
            label=label,
        )

    axis.axvspan(4, 6, color=PALETTE["neutral"], alpha=0.45, zorder=0)
    axis.annotate(
        "the step",
        (4.9, 0.5),
        xycoords=("data", "axes fraction"),
        ha="center",
        fontsize=12,
        color=PALETTE["grey_text"],
        rotation=90,
    )

    ticks = sorted(data["threads"].unique())
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.set_xticklabels(ticks)
    axis.minorticks_off()
    axis.set_xlabel("threads asked for")
    axis.set_ylabel("peak memory, GB")
    axis.set_ylim(bottom=0)
    axis.set_title("Memory per job steps between four and six threads")
    axis.legend()
    faint_grid(axis)

    return finalize(figure, output)


def figure_cuts(data: pd.DataFrame, output: Path):
    """
    Stage 12, cuts = 0 against the default, paired cell by cell.

    A ratio above one is a loss. Everything except the wall time at four
    threads is a loss, and the memory at fifteen typical days is a factor two
    and a half, which is the axis that decides how many jobs fit on a machine.

    :param DataFrame data: every run of the case study
    :param Path output: file to write, without a suffix
    """
    responses = [
        ("gurobi_runtime_s", "solver time", PALETTE["blue_main"]),
        ("cpu_user_s", "CPU seconds", PALETTE["teal"]),
        ("rss_peak_os_mb", "peak memory", PALETTE["red_strong"]),
    ]

    cells = []
    for typicaldays, _, _ in RESOLUTIONS:
        for threads in [1, 4]:
            subset = data[
                (data["typicaldays"] == typicaldays) & (data["threads"] == threads)
            ]
            default = subset[~subset["cuts_off"]]
            cuts_off = subset[subset["cuts_off"]]
            if default.empty or cuts_off.empty:
                continue
            cells.append(
                (
                    f"td{typicaldays}\n{threads} thread"
                    f"{'s' if threads > 1 else ''}",
                    [
                        cuts_off[column].iloc[0] / default[column].iloc[0]
                        for column, _, _ in responses
                    ],
                )
            )

    if not cells:
        return []

    figure, axis = plt.subplots(figsize=(8.4, 4.4))
    width = 0.26

    for offset, (_, label, colour) in enumerate(responses):
        bars = axis.bar(
            [position + (offset - 1) * width for position in range(len(cells))],
            [values[offset] for _, values in cells],
            width=width,
            label=label,
            color=colour,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        annotate_bars(axis, bars, fmt="{:.2f}", fontsize=10)

    axis.axhline(1, color=PALETTE["grey_text"], linewidth=1.4, zorder=4)
    # Outside the data area, so it cannot land on a bar or on its annotation
    axis.set_xlim(-0.55, len(cells) - 0.45)
    axis.annotate(
        "the default",
        (len(cells) - 0.42, 1.0),
        ha="left",
        va="center",
        fontsize=11,
        color=PALETTE["grey_text"],
        annotation_clip=False,
    )
    axis.set_xticks(range(len(cells)))
    axis.set_xticklabels([name for name, _ in cells])
    axis.set_ylabel("cuts = 0, relative to the default")
    axis.set_title("Turning the cuts off costs more than it saves")
    axis.legend(ncol=3, loc="upper left")
    faint_grid(axis)

    return finalize(figure, output)


def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--figures", type=Path, default=FIGURES_PATH)
    args = parser.parse_args()

    data = load(args.dataset)
    if data.empty:
        raise SystemExit(f"No {CASE} runs in {args.dataset}")

    apply_publication_style()

    threads_stage = args.figures / "stage13_threads"
    cuts_stage = args.figures / "stage12_cuts"

    written = []
    written += figure_core_seconds(data, threads_stage / "nl_node_core_seconds")
    written += figure_root_algorithm(data, threads_stage / "nl_node_root_algorithm")
    written += figure_memory(data, threads_stage / "nl_node_memory")
    written += figure_cuts(data, cuts_stage / "nl_node_cuts")

    print(f"Wrote {len(written)} files:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
