"""
Slide figures on the gurobi option study, for the meeting of 2026-09-23.

Four figures, in the style of plot_meeting_slides.py and numbered after its
ten, so that both scripts can write into one meeting folder:

11. ``options``        every option level of stages 6 to 11 against the
                       default, on solver time, peak memory and CPU seconds
12. ``cuts_four_node`` cuts off on four_node, pair by pair: the factor two it
                       looked like
13. ``cuts_quality``   why it looked like that: at a 2 % gap it stops on the
                       first incumbent under tolerance, a looser bound and in
                       most pairs a worse solution
14. ``cuts_tree``      cuts off on nl_node at 0.5 %, where the tree has to
                       close the gap: the tree grows and the win is gone

The four_node runs are picked by the manifests of stages 6 to 11, which is what
separates them from every other run with the same case name. The nl_node runs
are the solo runs of stage 12.

Run it from the benchmark folder::

    python plot_meeting_options.py --output <folder of the meeting>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import PALETTE, annotate_bars, apply_publication_style, faint_grid, finalize

BASE = Path(__file__).parent
DATASET_FILE = BASE / "benchmark_dataset.csv"
RESULTS_PATH = BASE / "results"
OUTPUT_PATH = BASE / "figures" / "meetings"

# The stages of the option study on four_node
OPTION_STAGES = [6, 7, 8, 9, 10, 11]

# Each option with the level the adopt template ships, which is what every
# other level is paired against, and the levels in the order a slide lists
# them. CutPasses has no column value at the default, the gurobi default -1
OPTIONS = [
    ("gurobi_method", "Method", -1, {1: "dual simplex", 2: "barrier", 3: "concurrent"}),
    ("gurobi_cuts", "Cuts", -1, {0: "off", 3: "very aggressive"}),
    ("gurobi_cutpasses", "CutPasses", -1, {1: "1 pass", 2: "2 passes", 5: "5 passes"}),
    ("gurobi_mipfocus", "MIPFocus", 0, {3: "bound"}),
    ("gurobi_lpwarmstart", "LPWarmStart", 0, {-1: "gurobi default"}),
]

RESPONSES = [
    ("gurobi_runtime_s", "solver time"),
    ("rss_peak_os_mb", "peak memory"),
    ("cpu_user_s", "CPU seconds"),
]

# Names a slide uses for the four configurations of four_node
CONFIG_NAMES = {
    "td4": "td4",
    "td16": "td16",
    "td16_bp1": "td16\nbidirectional",
    "td4_st0": "td4\nno storage",
}

THREAD_COLOURS = {4: PALETTE["blue_main"], 48: PALETTE["red_strong"]}


def _run_names(data: pd.DataFrame):
    """
    The run folder name of every row, which is what a manifest lists.

    :param DataFrame data: rows of the dataset
    :return: Series of folder names
    """
    return (
        data["result_folder_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.rstrip("/")
        .str.rsplit("/", n=1)
        .str[-1]
    )


def load_options(dataset: Path, results: Path):
    """
    The four_node runs of the option study, one row per distinct cell.

    Stages 7 and 9 share cells with 6 and 8, and the default cell of stage 11
    is the default cell of stage 8, so a run can be listed by two manifests.
    It is kept once.

    :param Path dataset: benchmark_dataset.csv
    :param Path results: folder searched for manifest_stage<n>.txt
    :return: DataFrame with a config column and every option filled in
    """
    wanted = set()
    for stage in OPTION_STAGES:
        for manifest in results.glob(f"**/manifest_stage{stage}.txt"):
            wanted |= {line.strip() for line in manifest.read_text().splitlines()}
    wanted.discard("")

    data = pd.read_csv(dataset, low_memory=False)
    data = data[_run_names(data).isin(wanted)].copy()
    data["config"] = (
        data["case_name"]
        .str.replace(r"_(thr|mth|cut|mf|lpw|cp)-?[\d.]+", "", regex=True)
        .str.replace("four_node_", "", regex=False)
    )
    data["gurobi_cutpasses"] = data["gurobi_cutpasses"].fillna(-1)

    columns = [column for column, _, _, _ in OPTIONS]
    return data.drop_duplicates(subset=["config", "gurobi_threads"] + columns)


def paired(data: pd.DataFrame, option: str, default, level):
    """
    Every pair of runs that differ in one option alone.

    :param DataFrame data: rows returned by load_options
    :param str option: column of the option
    :param default: level every other level is compared against
    :param level: level to compare
    :return: DataFrame with one row per pair, the level's columns and the
        default's with a _default suffix
    """
    key = ["config", "gurobi_threads"] + [
        column for column, _, _, _ in OPTIONS if column != option
    ]
    base = data[data[option] == default].set_index(key)
    arm = data[data[option] == level].set_index(key)
    return arm.join(base, rsuffix="_default", how="inner").reset_index()


def _geometric(ratios: pd.Series):
    """
    Geometric mean and geometric spread of a set of ratios.

    :param Series ratios: positive ratios
    :return: (mean, spread), the spread as a multiplicative factor
    """
    logs = np.log(ratios.dropna())
    return float(np.exp(logs.mean())), float(np.exp(logs.std(ddof=0)))


def figure_options(data: pd.DataFrame, output: Path):
    """
    Every option level against the default, on the three responses.

    One row per level, grouped by option. The point is the geometric mean of
    its paired ratios, the bar their geometric spread across configurations
    and thread counts. A point is coloured only where the whole bar sits on
    one side of one: green if better, red if worse.

    :param DataFrame data: rows returned by load_options
    :param Path output: file to write, without a suffix
    """
    rows = []
    for option, name, default, levels in OPTIONS:
        for level, label in levels.items():
            pairs = paired(data, option, default, level)
            if pairs.empty:
                continue
            rows.append((name, label, pairs))
    if not rows:
        return []

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 8.2), sharey=True)
    positions = list(range(len(rows)))[::-1]

    for axis, (column, title) in zip(axes, RESPONSES):
        for y, (_, _, pairs) in zip(positions, rows):
            mean, spread = _geometric(pairs[column] / pairs[f"{column}_default"])
            low, high = mean / spread, mean * spread
            if high < 1:
                colour = PALETTE["green_strong"]
            elif low > 1:
                colour = PALETTE["red_strong"]
            else:
                colour = PALETTE["neutral"]
            axis.plot([low, high], [y, y], color=PALETTE["grey_text"], lw=2.2, zorder=2)
            axis.scatter(
                mean, y, s=190, color=colour, edgecolor="black", lw=1.2, zorder=3
            )
            axis.annotate(
                f"x{mean:.2f}",
                (mean, y),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=13,
            )

        axis.axvline(1, color="black", lw=1.6, zorder=1)
        axis.set_xscale("log")
        axis.set_xlim(0.25, 4)
        axis.set_xticks([0.25, 0.5, 1, 2, 4])
        axis.set_xticklabels(["x0.25", "x0.5", "x1", "x2", "x4"])
        axis.set_xlabel("relative to the default")
        axis.set_title(title)
        faint_grid(axis, which="x")

    # Separators between the options, so the rows read as groups
    names = [name for name, _, _ in rows]
    for index in range(1, len(rows)):
        if names[index] != names[index - 1]:
            for axis in axes:
                axis.axhline(positions[index] + 0.5, color=PALETTE["neutral"], lw=1.2)

    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([f"{name}: {label}" for name, label, _ in rows])
    axes[0].set_ylim(-0.7, len(rows) - 0.3)

    runs = len(data)
    figure.suptitle(
        "No gurobi option beats the default on all three: "
        "cuts off is faster, and needs more memory",
        fontsize=21,
    )
    figure.text(
        0.5,
        0.005,
        f"four_node, {runs} runs at a 2 % gap: 4 configurations, threads 1 to 48, "
        "every level paired with the default it differs from in that option alone. "
        "Point: geometric mean. Bar: spread across the pairs",
        ha="center",
        fontsize=13,
        color=PALETTE["grey_text"],
    )

    return finalize(figure, output, pad=2.4)


def figure_cuts_four_node(data: pd.DataFrame, output: Path):
    """
    Cuts off on four_node, pair by pair, by configuration.

    :param DataFrame data: rows returned by load_options
    :param Path output: file to write, without a suffix
    """
    pairs = paired(data, "gurobi_cuts", -1, 0)
    if pairs.empty:
        return []

    configs = [name for name in CONFIG_NAMES if name in set(pairs["config"])]
    figure, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    panels = [("gurobi_runtime_s", "solver time"), ("rss_peak_os_mb", "peak memory")]

    for axis, (column, title) in zip(axes, panels):
        ratio = pairs[column] / pairs[f"{column}_default"]
        for x, config in enumerate(configs):
            for threads, colour in THREAD_COLOURS.items():
                chosen = (pairs["config"] == config) & (pairs["gurobi_threads"] == threads)
                offset = -0.14 if threads == 4 else 0.14
                jitter = np.linspace(-0.06, 0.06, chosen.sum())
                axis.scatter(
                    x + offset + jitter,
                    ratio[chosen],
                    s=90,
                    color=colour,
                    edgecolor="black",
                    lw=0.6,
                    label=f"{threads} threads" if x == 0 else None,
                    zorder=3,
                )

        mean, _ = _geometric(ratio)
        axis.axhline(1, color="black", lw=1.6, zorder=1)
        axis.axhline(mean, color=PALETTE["grey_text"], lw=2.0, ls="--", zorder=2)
        axis.annotate(
            f"all pairs: x{mean:.2f}",
            (len(configs) - 0.5, mean),
            xytext=(0, 8),
            textcoords="offset points",
            ha="right",
            fontsize=15,
            color=PALETTE["grey_text"],
        )
        axis.set_yscale("log")
        axis.set_ylim(0.1, 4)
        axis.set_yticks([0.125, 0.25, 0.5, 1, 2, 4])
        axis.set_yticklabels(["x0.125", "x0.25", "x0.5", "x1", "x2", "x4"])
        axis.set_xticks(range(len(configs)))
        axis.set_xticklabels([CONFIG_NAMES[config] for config in configs])
        axis.set_title(title)
        faint_grid(axis)

    axes[0].set_ylabel("cuts off, relative to the default")
    axes[0].legend(loc="lower left", fontsize=14)
    figure.suptitle(
        f"On four_node at a 2 % gap, cuts off looks like a factor two: "
        f"{len(pairs)} pairs",
        fontsize=21,
    )

    return finalize(figure, output)


def figure_cuts_quality(data: pd.DataFrame, output: Path):
    """
    What cuts off gives up for its speed on four_node.

    Left: each pair's time ratio against how much worse the objective it stops
    on is. Right: the gap each run actually stops at, default against cuts
    off, one line per pair. A run stops at the first incumbent under 2 %, so a
    faster arm can simply be stopping earlier.

    :param DataFrame data: rows returned by load_options
    :param Path output: file to write, without a suffix
    """
    pairs = paired(data, "gurobi_cuts", -1, 0)
    if pairs.empty:
        return []

    time = pairs["gurobi_runtime_s"] / pairs["gurobi_runtime_s_default"]
    worse = (pairs["gurobi_objval"] / pairs["gurobi_objval_default"] - 1) * 100
    configs = [name for name in CONFIG_NAMES if name in set(pairs["config"])]
    colours = [
        PALETTE["blue_main"],
        PALETTE["red_strong"],
        PALETTE["teal"],
        PALETTE["violet"],
    ]

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(15.5, 6.8), gridspec_kw={"width_ratios": [3, 2]}
    )
    for config, colour in zip(configs, colours):
        chosen = pairs["config"] == config
        left.scatter(
            time[chosen],
            worse[chosen],
            s=110,
            color=colour,
            edgecolor="black",
            lw=0.6,
            label=CONFIG_NAMES[config].replace("\n", " "),
            zorder=3,
        )
    left.axhline(0, color="black", lw=1.6)
    left.axvline(1, color="black", lw=1.6)
    left.set_xscale("log")
    left.set_xlim(0.08, 2.5)
    left.set_xticks([0.125, 0.25, 0.5, 1, 2])
    left.set_xticklabels(["x0.125", "x0.25", "x0.5", "x1", "x2"])
    left.set_xlabel("solver time, cuts off relative to the default")
    left.set_ylabel("objective, cuts off vs default [%]")
    left.annotate(
        "faster, worse solution",
        (0.1, worse.max() * 0.92),
        fontsize=14,
        color=PALETTE["grey_text"],
    )
    left.legend(loc="lower right", fontsize=12)
    left.set_title(
        f"worse in {(worse > 0.05).sum()} of {len(pairs)} pairs, "
        f"by up to {worse.max():.1f} %"
    )
    faint_grid(left, which="both")

    gaps = pairs[["gurobi_mipgap_default", "gurobi_mipgap"]] * 100
    for _, row in gaps.iterrows():
        right.plot([0, 1], row.values, color=PALETTE["neutral"], lw=1.4, zorder=2)
    for x, column in enumerate(gaps.columns):
        right.scatter(
            [x] * len(gaps),
            gaps[column],
            s=70,
            color=PALETTE["blue_main"] if x == 0 else PALETTE["red_strong"],
            edgecolor="black",
            lw=0.5,
            zorder=3,
        )
        right.annotate(
            f"mean {gaps[column].mean():.2f} %",
            (x, 2.08),
            ha="center",
            fontsize=14,
        )
    right.axhline(2, color=PALETTE["red_strong"], lw=1.6, ls=":")
    right.set_xticks([0, 1])
    right.set_xticklabels(["default", "cuts off"])
    right.set_xlim(-0.5, 1.5)
    right.set_ylim(0, 2.35)
    right.set_ylabel("gap the run stopped at [%]")
    right.set_title("tolerance 2 %")
    faint_grid(right)

    figure.suptitle(
        "It is faster because it stops sooner: a looser bound, often a worse solution",
        fontsize=21,
    )

    return finalize(figure, output)


def figure_cuts_tree(options: pd.DataFrame, nl_data: pd.DataFrame, output: Path):
    """
    Cuts off on nl_node at 0.5 %, next to what the tree did on four_node.

    Left: nl_node, cuts off against the default, cell by cell, on solver time,
    CPU seconds and memory. Right: the branch and bound nodes of the default
    and of cuts off, in both case studies. On four_node the tree moves both
    ways without cuts, and a 2 % gap lets a run stop before it matters; on
    nl_node at 0.5 % it grows in every cell, and the cuts earn their time
    back. The two case studies also differ in the gap, so the panel shows
    where the win disappears, not which of the two changes removes it.

    :param DataFrame options: four_node rows returned by load_options
    :param DataFrame nl_data: nl_node rows of the dataset
    :param Path output: file to write, without a suffix
    """
    nl = nl_data[
        nl_data["case_name"].str.startswith("nl_node_td")
        & nl_data["gurobi_threads"].isin([1, 4])
    ].copy()
    nl["typicaldays"] = nl["case_name"].str.extract(r"_td(\d+)")[0].astype(int)
    nl["cut0"] = nl["case_name"].str.contains("_cut0")
    off = nl[nl["cut0"]].set_index(["typicaldays", "gurobi_threads"])
    base = nl[~nl["cut0"]].set_index(["typicaldays", "gurobi_threads"])
    cells = off.join(base, rsuffix="_default", how="inner").sort_index()
    four = paired(options, "gurobi_cuts", -1, 0)
    if cells.empty or four.empty:
        return []

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(16.5, 6.8), gridspec_kw={"width_ratios": [3, 2]}
    )

    width = 0.26
    measures = [
        ("gurobi_runtime_s", "solver time", PALETTE["blue_main"]),
        ("cpu_user_s", "CPU seconds", PALETTE["teal"]),
        ("rss_peak_os_mb", "peak memory", PALETTE["red_strong"]),
    ]
    for offset, (column, label, colour) in enumerate(measures):
        ratios = cells[column] / cells[f"{column}_default"]
        bars = left.bar(
            np.arange(len(cells)) + (offset - 1) * width,
            ratios,
            width=width,
            color=colour,
            edgecolor="black",
            lw=1.0,
            label=label,
            zorder=3,
        )
        annotate_bars(left, bars, fmt="{:.2f}", fontsize=13)
    left.axhline(1, color="black", lw=1.6)
    left.set_xticks(range(len(cells)))
    left.set_xticklabels(
        [f"td{days}\n{threads} thread{'s' if threads > 1 else ''}" for days, threads in cells.index]
    )
    left.set_ylabel("cuts off, relative to the default")
    left.set_ylim(0, 2.8)
    left.legend(loc="upper left", fontsize=13, ncol=3)
    left.set_title("nl_node at 0.5 %: every cell pays for it")
    faint_grid(left)

    groups = [
        ("four_node\n2 % gap", four["gurobi_nodecount_default"], four["gurobi_nodecount"]),
        ("nl_node\n0.5 % gap", cells["gurobi_nodecount_default"], cells["gurobi_nodecount"]),
    ]
    for x, (_, default, without) in enumerate(groups):
        for d, w in zip(default, without):
            right.plot(
                [x - 0.17, x + 0.17], [d + 1, w + 1], color=PALETTE["neutral"], lw=1.2, zorder=2
            )
        right.scatter([x - 0.17] * len(default), default + 1, s=70,
                      color=PALETTE["blue_main"], edgecolor="black", lw=0.5, zorder=3,
                      label="default" if x == 0 else None)
        right.scatter([x + 0.17] * len(without), without + 1, s=70,
                      color=PALETTE["red_strong"], edgecolor="black", lw=0.5, zorder=3,
                      label="cuts off" if x == 0 else None)
        ratio = float(np.exp(np.log((without + 1) / (default + 1)).mean()))
        right.annotate(
            f"tree x{ratio:.1f}",
            (x, 260),
            ha="center",
            fontsize=15,
            fontweight="bold",
        )
    right.set_yscale("log")
    right.set_ylim(0.8, 450)
    right.set_xticks(range(len(groups)))
    right.set_xticklabels([name for name, _, _ in groups])
    right.set_xlim(-0.6, 1.6)
    right.set_ylabel("branch and bound nodes + 1")
    right.set_title("tree without cuts: mixed on four_node,\nalways larger on nl_node")
    right.legend(loc="lower right", fontsize=13)
    faint_grid(right)

    figure.suptitle(
        "Where the tree has to close the gap, the cuts pay for themselves",
        fontsize=21,
    )

    return finalize(figure, output)


def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULTS_PATH,
        help="folder searched for the manifests of stages 6 to 11",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    options = load_options(args.dataset, args.results)
    if options.empty:
        raise SystemExit(f"No runs of stages {OPTION_STAGES} in {args.dataset}")
    nl_data = pd.read_csv(args.dataset, low_memory=False)
    nl_data = nl_data[nl_data["case_name"].str.startswith("nl_node_")]

    # Larger than the paper preset: a slide is read from the back of a room
    apply_publication_style(font_size=19)

    written = []
    written += figure_options(options, args.output / "11_gurobi_options")
    written += figure_cuts_four_node(options, args.output / "12_cuts_off_four_node")
    written += figure_cuts_quality(options, args.output / "13_cuts_off_quality")
    written += figure_cuts_tree(options, nl_data, args.output / "14_cuts_off_with_a_tree")

    print(f"Wrote {len(written)} files to {args.output}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
