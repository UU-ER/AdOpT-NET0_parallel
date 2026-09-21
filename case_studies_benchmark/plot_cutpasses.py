"""
Draws what the root cut loop costs and what it buys, pass by pass.

Stage 11 tests a premise read out of the node logs of stages 6 to 9: the first
cut pass moves the bound by 0.33 to 0.39 %, the twenty-five to fifty-seven
passes after it by 0.03 to 0.23 %, and those later passes cost a quarter to a
half of the whole solve. If that is the whole story then ``CutPasses = 1``
should match ``Cuts = 0`` on time without giving away the gap, and would be a
better recommendation than turning cuts off.

The stage runs five arms over a pass budget, cheapest first::

    0     cuts off entirely, the arm to beat
    1     one root cut pass
    2     two
    5     five
    inf   the gurobi default, which is unlimited passes

crossed with 4 and 48 threads on td4 and td16. Twenty cells, one run each.

Four figures, each answering one question:

1. ladder       what the pass budget does to wall time and peak memory
2. phases       where the seconds go, presolve, root LP, cut loop and tree
3. buys         bound gained against cut loop seconds spent, pass by pass
4. quality      the thing that actually decides it: every run stops at the
                first incumbent inside the 2 % tolerance, so a tighter bound
                does not buy time, it buys a better solution

Figure 4 is the point of the stage. The arms are not racing to the same answer.

Every cell is a single run and a MIP tree at 48 threads is not deterministic,
so the ordering of two arms within a few per cent means nothing. The effects
drawn here are factors, not per cent.

Examples::

    python plot_cutpasses.py
    python plot_cutpasses.py --dataset "//server/share/benchmark_dataset.csv" \\
        --manifest "//server/share/results/manifest_stage11.txt"
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
MANIFEST_FILE = BASE / "results" / "manifest_stage11.txt"
FIGURES_PATH = BASE / "figures"

BLUE, ORANGE, RED = "#3b6ea5", "#d1731f", "#c0392b"
GREEN, DARK, GREY = "#4a8c5f", "#2b2b2b", "#9a9a9a"
PURPLE = "#7d5ba6"

# One hue per thread count, fixed across all four figures
THREAD_COLORS = {4: BLUE, 48: ORANGE}

# The phases of a solve, in the order they happen, with the colour each keeps.
# The cut loop is the one the stage is about, so it is the one that stands out
PHASES = [
    ("presolve_s", "presolve", GREY),
    ("root_lp_s", "root LP", BLUE),
    ("cut_loop_s", "root cut loop", ORANGE),
    ("tree_s", "tree", GREEN),
]

# The pass budget of the default arm. Gurobi's default is unlimited passes,
# spelled -1, and it has to sort to the right of every finite budget. The arms
# are drawn on evenly spaced positions rather than on their own values: the
# budgets are a designed ladder, not a measured quantity, and on any numeric
# axis 5 and unlimited collide
DEFAULT_ARM = 1e9
DEFAULT_LABEL = "default"

# The mipgap every run of the study was given. A run stops at the first
# incumbent that puts the gap under this, so it is the reason the arms do not
# end at the same objective
TOLERANCE_PCT = 2.0

# The clustering was seeded on 2026-09-11. Runs before that rebuilt the model
# differently every time, so they are a different problem
SEEDED_FROM = "20260911"

TITLE_INCHES = 0.95
LEGEND_INCHES = 0.62


def load(dataset_file: Path, manifest_file: Path, since: str = SEEDED_FROM):
    """
    Reads the dataset and keeps the cells of stage 11.

    The manifest is what makes this stage separable. Its arms are spelled in
    two different columns, ``gurobi_cuts`` for the cuts off arm and
    ``gurobi_cutpasses`` for the rest, and the default cell is spelled exactly
    like the default cell of stage 8, so filtering on the options alone pulls
    in runs of other stages.

    :param Path dataset_file: benchmark_dataset.csv to read
    :param Path manifest_file: manifest_stage11.txt listing the run folders
    :param str since: earliest run timestamp to keep, as it appears in the
        result folder name
    :return: pandas DataFrame with one row per cell
    """
    dataset = pd.read_csv(dataset_file)

    folder_column = next(
        (name for name in ("run_folder", "result_folder_path") if name in dataset),
        None,
    )
    if folder_column is None:
        raise KeyError(
            "The dataset carries neither run_folder nor result_folder_path, so "
            "the runs of the stage cannot be looked up"
        )

    folders = (
        dataset[folder_column]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.rstrip("/")
        .str.rsplit("/", n=1)
        .str[-1]
    )

    wanted = [
        line.strip() for line in manifest_file.read_text().splitlines() if line.strip()
    ]
    stage = dataset[folders.isin(wanted)].copy()
    stage["run"] = folders[folders.isin(wanted)]

    missing = set(wanted) - set(stage["run"])
    if missing:
        print(f"{len(missing)} of {len(wanted)} runs of the manifest are not in the dataset")

    before = len(stage)
    stage = stage[stage["run"].str.split("_").str[0] >= since].copy()
    if before - len(stage):
        print(f"Dropped {before - len(stage)} runs from before {since}, unseeded")

    return derive(stage)


def derive(stage: pd.DataFrame):
    """
    Adds the columns the figures are drawn from.

    The pass budget is one axis built out of two columns, and the solve is cut
    into four phases out of the three timestamps the node log gives:
    ``presolve_s``, ``root_relaxation_s`` and ``root_node_end_s`` are all
    measured from the start of the solve, so the phases are their differences.

    :param pd.DataFrame stage: the runs of the stage
    :return: the same frame with the derived columns
    """
    stage = stage.copy()

    passes = stage["gurobi_cutpasses"].fillna(-1).replace(-1, DEFAULT_ARM)
    stage["arm"] = np.where(stage["gurobi_cuts"] == 0, 0, passes).astype(float)
    stage["label"] = np.where(
        stage["arm"] == DEFAULT_ARM,
        DEFAULT_LABEL,
        stage["arm"].astype(int, errors="ignore").astype(str),
    )
    stage["label"] = np.where(stage["arm"] == 0, "cuts off", stage["label"])

    # Evenly spaced position of each arm on the ladder, shared by every panel
    ladder = sorted(stage["arm"].unique())
    stage["position"] = stage["arm"].map({arm: index for index, arm in enumerate(ladder)})

    # 0 threads means every core of the machine, which on this server is 48
    stage["threads"] = stage["gurobi_threads"].replace(0, 48)
    stage["config"] = "td" + stage["typicaldays_n"].astype(int).astype(str)

    stage["root_lp_s"] = stage["root_relaxation_s"] - stage["presolve_s"]
    stage["cut_loop_s"] = stage["root_node_end_s"] - stage["root_relaxation_s"]
    stage["tree_s"] = stage["gurobi_runtime_s"] - stage["root_node_end_s"]
    stage["gap_pct"] = stage["gurobi_mipgap"] * 100

    # What every arm of a configuration is compared against on quality: the
    # best objective any arm of that configuration found. The runs stop at the
    # tolerance rather than at optimality, so this is the best known, not the
    # optimum
    best = stage.groupby("config")["gurobi_objval"].transform("min")
    stage["excess_pct"] = (stage["gurobi_objval"] / best - 1) * 100

    return stage.sort_values(["config", "threads", "arm"])


def cells(stage: pd.DataFrame):
    """
    The configuration and thread count pairs, cheapest configuration first and
    the thread counts of a configuration side by side

    :param pd.DataFrame stage: the stage dataset
    :return: list of ((config, threads), group) pairs
    """
    order = {name: rank for rank, (name, _) in enumerate(configs(stage))}
    groups = list(stage.groupby(["config", "threads"]))
    return sorted(groups, key=lambda pair: (order[pair[0][0]], pair[0][1]))


def configs(stage: pd.DataFrame):
    """
    The configurations, cheapest first

    :param pd.DataFrame stage: the stage dataset
    :return: list of (name, group) pairs
    """
    groups = list(stage.groupby("config"))
    return sorted(groups, key=lambda pair: pair[1]["wall_total_s"].median())


def _arm_axis(axis, group):
    """
    Sets up a pass budget axis, evenly spaced and labelled by the budget

    :param axis: matplotlib axis
    :param pd.DataFrame group: rows carrying position and label
    """
    ticks = group.drop_duplicates("position").sort_values("position")
    axis.set_xticks(ticks["position"].to_numpy())
    axis.set_xticklabels(ticks["label"], fontsize=8.5)
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


def plot_ladder(stage: pd.DataFrame, output: Path):
    """
    Wall time and peak memory against the pass budget.

    Cuts off is drawn as a horizontal line rather than only as the leftmost
    point, because every other arm is being asked to beat it and a line makes
    that readable without arithmetic.

    :param pd.DataFrame stage: the stage dataset
    :param Path output: png file to write
    """
    groups = configs(stage)
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

    for row, (column, ylabel) in enumerate(responses):
        for col, (name, group) in enumerate(groups):
            axis = axes[row][col]

            for threads, line in group.groupby("threads"):
                line = line.sort_values("arm")
                colour = THREAD_COLORS.get(threads, DARK)
                axis.plot(
                    line["position"],
                    line[column],
                    "o-",
                    color=colour,
                    lw=1.6,
                    ms=5,
                    label=f"{threads} threads",
                )

                off = line[line["arm"] == 0]
                if not off.empty:
                    axis.axhline(
                        off[column].iloc[0], color=colour, ls="--", lw=1.0, alpha=0.55
                    )

            _arm_axis(axis, group)
            axis.set_ylabel(ylabel if col == 0 else "")
            axis.set_xlabel("root cut passes allowed" if row == len(responses) - 1 else "")
            if row == 0:
                axis.set_title(name, fontsize=11)

    _finish(
        figure,
        axes[0][0],
        "Stage 11: what a budget of root cut passes costs\n"
        "dashed line is the same cell with cuts off, the arm to beat",
        output,
    )


def plot_phases(stage: pd.DataFrame, output: Path):
    """
    Where the seconds of a solve go, arm by arm.

    This is the mechanism figure. The pass budget is a knob on one of the four
    bands and nothing else, so anything that moves in the other three is the
    bound being paid back somewhere later.

    :param pd.DataFrame stage: the stage dataset
    :param Path output: png file to write
    """
    groups = cells(stage)

    figure, axes = plt.subplots(
        1,
        len(groups),
        figsize=(max(8.4, 3.3 * len(groups)), 3.9),
        squeeze=False,
    )

    for col, ((config, threads), group) in enumerate(groups):
        axis = axes[0][col]
        group = group.sort_values("position")
        positions = group["position"].to_numpy()
        bottom = np.zeros(len(group))

        for column, label, colour in PHASES:
            values = group[column].to_numpy(dtype=float)
            axis.bar(
                positions,
                values,
                bottom=bottom,
                color=colour,
                width=0.68,
                label=label,
                edgecolor="white",
                lw=0.6,
            )
            bottom += values

        axis.set_xticks(positions)
        axis.set_xticklabels(group["label"], rotation=30, ha="right", fontsize=8)
        axis.set_xlabel("root cut passes allowed", fontsize=9)
        axis.set_title(f"{config}, {threads} threads", fontsize=10)
        axis.set_ylabel("solver time [s]" if col == 0 else "")
        axis.grid(True, axis="y", ls=":", lw=0.6, color=GREY, alpha=0.6)
        axis.set_axisbelow(True)

    _finish(
        figure,
        axes[0][0],
        "Stage 11: the cut loop is the band the budget moves\n"
        "and the tree is where a loose bound is paid back",
        output,
    )


def plot_buys(stage: pd.DataFrame, output: Path):
    """
    The bound gained against the cut loop seconds it cost.

    Both axes are measured against the cuts off cell of the same configuration
    and thread count, so the curve starts at the origin and every point is
    "what this budget bought over turning cuts off". A curve that bends early
    is the premise: the first pass is worth more per second than the rest.

    :param pd.DataFrame stage: the stage dataset
    :param Path output: png file to write
    """
    groups = configs(stage)

    figure, axes = plt.subplots(
        1,
        len(groups),
        figsize=(max(8.4, 4.4 * len(groups)), 3.9),
        squeeze=False,
    )

    for col, (name, group) in enumerate(groups):
        axis = axes[0][col]

        for threads, line in group.groupby("threads"):
            line = line.sort_values("position")
            off = line[line["arm"] == 0]
            if off.empty:
                continue
            base = off.iloc[0]

            seconds = line["cut_loop_s"] - base["cut_loop_s"]
            bound = (line["gurobi_objbound"] / base["gurobi_objbound"] - 1) * 100
            colour = THREAD_COLORS.get(threads, DARK)

            axis.plot(
                seconds,
                bound,
                "o-",
                color=colour,
                lw=1.6,
                ms=5,
                label=f"{threads} threads",
            )
            for x, y, label in zip(seconds, bound, line["label"]):
                if label == "cuts off":
                    continue
                axis.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, -9),
                    fontsize=7.5,
                    color=colour,
                )

        axis.axhline(0, color=GREY, lw=0.8)
        axis.axvline(0, color=GREY, lw=0.8)
        axis.set_title(name, fontsize=11)
        axis.set_xlabel("cut loop seconds over cuts off [s]")
        axis.set_ylabel("final bound over cuts off [%]" if col == 0 else "")
        axis.grid(True, ls=":", lw=0.6, color=GREY, alpha=0.6)
        axis.set_axisbelow(True)
        axis.margins(x=0.14, y=0.16)

    _finish(
        figure,
        axes[0][0],
        "Stage 11: what a pass buys, against what it costs\n"
        "the first pass is the efficient one, and it is not the cheap one",
        output,
    )


def plot_quality(stage: pd.DataFrame, output: Path):
    """
    The figure the stage is decided on: time against the answer it returns.

    Every run stops at the first incumbent that puts the gap under 2 %, so the
    arms do not race to the same objective. Turning cuts off is fast because it
    stops at a looser bound, and a looser bound means it stops on a worse
    incumbent. That is the cost the runtime figure hides.

    :param pd.DataFrame stage: the stage dataset
    :param Path output: png file to write
    """
    groups = configs(stage)

    figure, axes = plt.subplots(
        1,
        len(groups),
        figsize=(max(8.4, 4.4 * len(groups)), 3.9),
        squeeze=False,
    )

    for col, (name, group) in enumerate(groups):
        axis = axes[0][col]

        for threads, line in group.groupby("threads"):
            line = line.sort_values("position")
            off = line[line["arm"] == 0]
            if off.empty:
                continue
            relative = line["wall_total_s"] / off["wall_total_s"].iloc[0]
            colour = THREAD_COLORS.get(threads, DARK)

            # No line through the arms. The pass budget is ordered and the
            # runtime is roughly monotone in it, but the quality is a lottery:
            # which incumbent a run happens to hold when the gap crosses the
            # tolerance. A line would draw that lottery as a trajectory
            axis.scatter(
                relative,
                line["excess_pct"],
                s=46,
                color=colour,
                alpha=0.85,
                zorder=3,
                label=f"{threads} threads",
            )
            for x, y, label in zip(relative, line["excess_pct"], line["label"]):
                axis.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7.5,
                    color=colour,
                )

        axis.axhline(0, color=GREEN, lw=1.0, ls="--")
        axis.axhline(
            TOLERANCE_PCT, color=RED, lw=1.0, ls=":", label=f"{TOLERANCE_PCT:.0f} % tolerance"
        )
        axis.set_title(name, fontsize=11)
        axis.set_xlabel("wall time, relative to cuts off")
        axis.set_ylabel(
            "objective above the best any arm found [%]" if col == 0 else ""
        )
        axis.grid(True, ls=":", lw=0.6, color=GREY, alpha=0.6)
        axis.set_axisbelow(True)
        axis.margins(x=0.14, y=0.16)

    _finish(
        figure,
        axes[0][0],
        "Stage 11: cuts off is the fastest arm in all four cells,\n"
        "and stops on a 1.2 % worse solution in three of them",
        output,
    )


def report(stage: pd.DataFrame):
    """
    Prints the table the figures are drawn from, cell by cell

    :param pd.DataFrame stage: the stage dataset
    """
    print(f"\n{len(stage)} runs, {stage['config'].nunique()} configurations\n")

    for (config, threads), group in cells(stage):
        group = group.sort_values("arm")
        off = group[group["arm"] == 0]
        if off.empty:
            print(f"{config}, {threads} threads: no cuts off cell, skipped")
            continue
        base = off.iloc[0]

        print(
            f"--- {config}, {threads} threads. Cuts off: {base['wall_total_s']:.0f} s "
            f"wall, cut loop {base['cut_loop_s']:.0f} s, gap {base['gap_pct']:.2f} %"
        )
        for _, row in group.iterrows():
            print(
                f"    {row['label']:>9}  wall {row['wall_total_s']:7.1f} s "
                f"({row['wall_total_s'] / base['wall_total_s']:4.2f}x)  "
                f"cut loop +{row['cut_loop_s'] - base['cut_loop_s']:6.0f} s  "
                f"tree {row['tree_s']:6.0f} s  "
                f"bound +{(row['gurobi_objbound'] / base['gurobi_objbound'] - 1) * 100:6.3f} %  "
                f"gap {row['gap_pct']:5.2f} %  "
                f"excess {row['excess_pct']:5.2f} %  "
                f"mem {row['rss_peak_os_mb']:6.0f} MB"
            )
        print()

    best = stage.loc[stage.groupby("config")["wall_total_s"].idxmin()]
    print("Fastest arm per configuration:")
    for _, row in best.iterrows():
        print(
            f"    {row['config']}: {row['label']} at {row['threads']} threads, "
            f"{row['wall_total_s']:.0f} s, {row['excess_pct']:.2f} % above the best "
            f"objective found"
        )

    clean = stage[stage["excess_pct"] < 1e-9]
    if not clean.empty:
        fastest = clean.loc[clean.groupby("config")["wall_total_s"].idxmin()]
        print("\nFastest arm that still lands on the best objective found:")
        for _, row in fastest.iterrows():
            print(
                f"    {row['config']}: {row['label']} at {row['threads']} threads, "
                f"{row['wall_total_s']:.0f} s"
            )


def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_FILE,
        help="manifest_stage11.txt, which is what separates this stage from "
        "stage 8, whose default cell is spelled identically",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=FIGURES_PATH,
        help="folder the study subfolder is created in",
    )
    parser.add_argument(
        "--study",
        default="stage11_cutpasses",
        help="subfolder of figures/ to write into",
    )
    parser.add_argument(
        "--since",
        default=SEEDED_FROM,
        help="earliest run timestamp to keep, as it appears in the folder name",
    )
    args = parser.parse_args()

    stage = load(args.dataset, args.manifest, args.since)
    if stage.empty:
        print("No runs left after filtering")
        return

    report(stage)

    output_path = Path(args.figures) / args.study
    output_path.mkdir(parents=True, exist_ok=True)

    drawn = 0
    for name, draw in [
        ("1_ladder", plot_ladder),
        ("2_phases", plot_phases),
        ("3_what_a_pass_buys", plot_buys),
        ("4_time_versus_quality", plot_quality),
    ]:
        try:
            draw(stage, output_path / f"{name}.png")
            drawn += 1
        except (KeyError, ValueError) as error:
            print(f"Could not draw {name}: {type(error).__name__}: {error}")

    print(f"\n{drawn} of 4 figures written to {output_path}")


if __name__ == "__main__":
    main()
