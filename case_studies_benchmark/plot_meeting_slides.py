"""
Slide figures for the parallel programming meeting.

Four figures, sized and weighted for a projector rather than for a page:
fewer elements, larger type, one message each.

1. ``throughput``   how many runs an hour a fixed core budget delivers at
                    one, two and four threads per run. The answer flips with
                    the size of the model, which is the headline
2. ``memory``       memory per core against the thread count, against the
                    two gigabytes a core a cluster partition gives. On the
                    large model a single thread per run does not fit
3. ``mechanism``    why two threads wins on the large model: below two
                    gurobi cannot run its concurrent root method and pays
                    four times the simplex iterations
4. ``cuts``         the option that looked like a factor two win and is not
                    one once the model has a branch and bound tree

Run it from the benchmark folder::

    python plot_meeting_slides.py --output <folder of the meeting>

The core budget of the first figure is a command line argument, since the
answer is stated per machine.
"""

import argparse
import re
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
OUTPUT_PATH = BASE / "figures" / "meetings"

CASE = "nl_node"

# The two models, with the words a slide uses for them rather than the
# resolution they happen to be run at
MODELS = [
    (4, "Small model\n143k variables, 8k binaries", PALETTE["blue_main"]),
    (15, "Large model\n318k variables, 30k binaries", PALETTE["red_strong"]),
]

# The packings worth showing. Three and six threads are measured and left out
# of the slide: three is dominated everywhere and six is past the memory step
PACKINGS = [1, 2, 4]

# Gigabytes of memory a core gets on a thin cluster partition, which is what
# makes the memory figure a constraint rather than an observation
MEMORY_PER_CORE_BUDGET = 2.0

# The resolution and the machine the contention packs were run at. Those
# figures are about one configuration on one machine, so they are stated for
# it rather than for the core budget the first figure takes
CONTENTION_TYPICALDAYS = 4
CONTENTION_CORES = 48


def load(dataset: Path):
    """
    Reads the dataset and keeps the default arm of the case study.

    :param Path dataset: benchmark_dataset.csv
    :return: DataFrame with the resolution, the thread count and the cut arm
        as columns
    """
    data = pd.read_csv(dataset)
    data = data[data["case_name"].str.startswith(f"{CASE}_")].copy()

    # A pack run is named after its packing rather than after its settings,
    # so it carries neither. It is filtered out of the ladder figures by the
    # -1 and picked out by name where it is wanted
    for column, pattern in [
        ("typicaldays", r"_td(\d+)"),
        ("threads", r"_thr(\d+)"),
    ]:
        data[column] = (
            data["case_name"].str.extract(pattern)[0].fillna(-1).astype(int)
        )
    data["cuts_off"] = data["case_name"].str.contains("_cut0")
    data["memory_gb"] = data["rss_peak_os_mb"] / 1024

    return data


def _cell(data: pd.DataFrame, typicaldays: int, threads: int, cuts_off=False):
    """
    One run, or None if that cell has not been run

    :param DataFrame data: every run of the case study
    :param int typicaldays: resolution
    :param int threads: thread count
    :param bool cuts_off: whether to take the cuts = 0 arm
    :return: Series of the run, or None
    """
    subset = data[
        (data["typicaldays"] == typicaldays)
        & (data["threads"] == threads)
        & (data["cuts_off"] == cuts_off)
    ]
    return None if subset.empty else subset.iloc[0]


def figure_throughput(data: pd.DataFrame, output: Path, cores: int):
    """
    Runs an hour a fixed core budget delivers, by packing.

    The bars are what a campaign actually gets: the budget divided into as
    many workers as the thread count allows, each worker taking the wall time
    measured for that thread count.

    :param DataFrame data: every run of the case study
    :param Path output: file to write, without a suffix
    :param int cores: size of the core budget
    """
    # A panel per model, each on its own linear axis. The two differ by a
    # factor twenty in throughput, so one shared axis, log or not, hides the
    # thing the figure is about: which bar is the tallest changes
    figure, axes = plt.subplots(1, len(MODELS), figsize=(13.5, 6.4))
    shades = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["neutral"]]

    for axis, (typicaldays, label, _) in zip(axes, MODELS):
        heights, names = [], []
        for threads in PACKINGS:
            run = _cell(data, typicaldays, threads)
            workers = cores // threads
            heights.append(
                0 if run is None else workers / (run["wall_total_s"] / 3600)
            )
            names.append(f"{workers} x {threads}")

        best = max(range(len(heights)), key=lambda index: heights[index])
        bars = axis.bar(
            range(len(heights)),
            heights,
            width=0.62,
            color=shades,
            edgecolor="black",
            linewidth=[2.6 if i == best else 1.0 for i in range(len(heights))],
            zorder=3,
        )
        annotate_bars(axis, bars, fmt="{:.0f}", fontsize=16)

        axis.set_xticks(range(len(heights)))
        axis.set_xticklabels(names)
        axis.set_xlabel("workers x threads each")
        axis.set_ylim(0, max(heights) * 1.2)
        axis.set_title(label)
        faint_grid(axis)

    axes[0].set_ylabel("runs completed per hour")
    figure.suptitle(
        f"The best way to spend {cores} cores flips with the size of the model",
        fontsize=21,
    )
    return finalize(figure, output)


def figure_memory(data: pd.DataFrame, output: Path):
    """
    Memory per core against the thread count, against the cluster budget.

    Memory per job is flat up to four threads, so memory per *core* falls as
    the threads rise. On the large model a single thread per run asks for two
    gigabytes a core, which is the whole of a thin partition's budget.

    :param DataFrame data: every run of the case study
    :param Path output: file to write, without a suffix
    """
    figure, axis = plt.subplots(figsize=(12.5, 6.4))

    width = 0.26
    shades = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["neutral"]]

    for offset, threads in enumerate(PACKINGS):
        heights = []
        for typicaldays, _, _ in MODELS:
            run = _cell(data, typicaldays, threads)
            heights.append(0 if run is None else run["memory_gb"] / threads)

        bars = axis.bar(
            [position + (offset - 1) * width for position in range(len(MODELS))],
            heights,
            width=width,
            label=f"{threads} thread{'s' if threads > 1 else ''} per run",
            color=shades[offset],
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        annotate_bars(axis, bars, fmt="{:.2f}", fontsize=15)

    axis.axhline(
        MEMORY_PER_CORE_BUDGET,
        color=PALETTE["red_strong"],
        linewidth=2.6,
        linestyle="--",
        zorder=4,
    )
    # On the left, where the small model's bars leave the space free: the
    # tall bar this line is about is the right hand one
    axis.annotate(
        "what a thin cluster partition gives: 2 GB per core",
        (-0.42, MEMORY_PER_CORE_BUDGET + 0.07),
        ha="left",
        fontsize=15,
        color=PALETTE["red_strong"],
    )

    axis.set_xticks(range(len(MODELS)))
    axis.set_xticklabels([label for _, label, _ in MODELS])
    axis.set_ylabel("memory needed per core, GB")
    axis.set_ylim(0, MEMORY_PER_CORE_BUDGET * 1.35)
    axis.set_title("On the large model, one thread per run barely fits")
    axis.legend(ncol=3, loc="upper right")
    faint_grid(axis)

    return finalize(figure, output)


def figure_mechanism(data: pd.DataFrame, output: Path):
    """
    Why two threads wins on the large model.

    Two panels, both relative to the single thread run of the same model, so
    that the small model and the large one can be read on one scale. The left
    panel is the work the solver does, the right one the time it takes.

    :param DataFrame data: every run of the case study
    :param Path output: file to write, without a suffix
    """
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))

    for typicaldays, label, colour in MODELS:
        ladder = data[
            (data["typicaldays"] == typicaldays) & ~data["cuts_off"]
        ].sort_values("threads")
        if ladder.empty:
            continue

        short = label.split("\n")[0]
        for axis, column in zip(axes, ["gurobi_itercount", "wall_total_s"]):
            axis.plot(
                ladder["threads"],
                ladder[column] / ladder[column].iloc[0],
                marker="o",
                color=colour,
                label=short,
            )

    ticks = sorted(data["threads"].unique())
    for axis, title in zip(
        axes, ["Work: simplex iterations", "Time: wall clock"]
    ):
        axis.axhline(1, color=PALETTE["grey_text"], linewidth=1.2, zorder=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ticks)
        axis.set_xticklabels(ticks)
        axis.minorticks_off()
        axis.set_xlabel("threads per run")
        axis.set_ylabel("relative to one thread")
        axis.set_ylim(0, 1.25)
        axis.set_title(title)
        faint_grid(axis)

    axes[0].legend(loc="lower right")
    figure.suptitle(
        "Below two threads gurobi cannot run its concurrent root method",
        fontsize=21,
    )
    return finalize(figure, output)


def figure_cuts(data: pd.DataFrame, output: Path):
    """
    cuts = 0 against the default, paired.

    The option that was worth a factor two on a model closing at the root,
    measured again on a model with a tree. A ratio above one is a loss.

    :param DataFrame data: every run of the case study
    :param Path output: file to write, without a suffix
    """
    responses = [
        ("gurobi_runtime_s", "solver time", PALETTE["blue_main"]),
        ("cpu_user_s", "CPU seconds", PALETTE["teal"]),
        ("rss_peak_os_mb", "peak memory", PALETTE["red_strong"]),
    ]

    cells = []
    for typicaldays, label, _ in MODELS:
        for threads in [1, 4]:
            default = _cell(data, typicaldays, threads, cuts_off=False)
            cuts_off = _cell(data, typicaldays, threads, cuts_off=True)
            if default is None or cuts_off is None:
                continue
            cells.append(
                (
                    f"{label.split(chr(10))[0].replace(' model', '')}\n"
                    f"{threads} thread{'s' if threads > 1 else ''}",
                    [cuts_off[column] / default[column] for column, _, _ in responses],
                )
            )

    if not cells:
        return []

    figure, axis = plt.subplots(figsize=(12.5, 6.4))
    width = 0.26

    for offset, (_, label, colour) in enumerate(responses):
        bars = axis.bar(
            [position + (offset - 1) * width for position in range(len(cells))],
            [values[offset] for _, values in cells],
            width=width,
            label=label,
            color=colour,
            edgecolor="black",
            linewidth=1.0,
            zorder=3,
        )
        annotate_bars(axis, bars, fmt="{:.2f}", fontsize=14)

    axis.axhline(1, color=PALETTE["grey_text"], linewidth=2.0, zorder=4)
    axis.set_xlim(-0.55, len(cells) - 0.45)
    axis.annotate(
        "the gurobi default",
        (len(cells) - 0.42, 1.0),
        ha="left",
        va="center",
        fontsize=14,
        color=PALETTE["grey_text"],
        annotation_clip=False,
    )
    axis.set_xticks(range(len(cells)))
    axis.set_xticklabels([name for name, _ in cells])
    axis.set_ylabel("cuts = 0, relative to the default")
    axis.set_title("Turning the cuts off stops paying once the model has a tree")
    axis.legend(ncol=3, loc="upper left")
    faint_grid(axis)

    return finalize(figure, output)


def read_packs(log: Path):
    """
    Reads what each pack of the contention stage actually delivered.

    The makespan of a pack is a property of the pack, not of a run, so it
    lives in the study log rather than in the dataset.

    :param Path log: study.log
    :return: list of dicts with the threads, the jobs, the makespan and the
        measured throughput, in the order they ran
    """
    if not log.is_file():
        return []

    pattern = re.compile(
        r"\[CONTENTION\] pack(\d+)x(\d+): makespan ([\d.]+) min, "
        r"([\d.]+) jobs per hour, (\d+) of (\d+) failed"
    )

    packs = []
    for line in log.read_text(errors="ignore").splitlines():
        found = pattern.search(line)
        if found:
            threads, jobs, makespan, rate, failed, total = found.groups()
            packs.append(
                {
                    "threads": int(threads),
                    "jobs": int(jobs),
                    "makespan_min": float(makespan),
                    "measured": float(rate),
                    "failed": int(failed),
                    "total": int(total),
                }
            )
    return packs


def figure_measured_throughput(data: pd.DataFrame, packs: list, output: Path):
    """
    What a full machine delivers, measured, against what the arithmetic said.

    Every other figure of this study is built from runs that had the machine
    to themselves, and the throughput of a packing is then arithmetic:
    workers divided by the wall time one run takes alone. This is the same
    quantity measured by running the pack.

    :param DataFrame data: every run of the case study
    :param list packs: packs as read from the study log
    :param Path output: file to write, without a suffix
    """
    usable = [pack for pack in packs if pack["failed"] < pack["total"]]
    if not usable:
        return []

    figure, axis = plt.subplots(figsize=(12.5, 6.4))
    width = 0.38

    names, predicted, measured = [], [], []
    for pack in usable:
        run = _cell(data, CONTENTION_TYPICALDAYS, pack["threads"])
        if run is None:
            continue
        names.append(f"{pack['jobs']} x {pack['threads']}")
        predicted.append(pack["jobs"] / (run["wall_total_s"] / 3600))
        measured.append(pack["measured"])

    positions = range(len(names))
    bars = axis.bar(
        [position - width / 2 for position in positions],
        predicted,
        width=width,
        label="predicted from runs measured alone",
        color=PALETTE["neutral"],
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )
    annotate_bars(axis, bars, fmt="{:.0f}", fontsize=15)

    best = max(range(len(measured)), key=lambda index: measured[index])
    bars = axis.bar(
        [position + width / 2 for position in positions],
        measured,
        width=width,
        label="measured on a full machine",
        color=[
            PALETTE["blue_main"] if i == best else PALETTE["blue_secondary"]
            for i in positions
        ],
        edgecolor="black",
        linewidth=[2.6 if i == best else 1.0 for i in positions],
        zorder=3,
    )
    annotate_bars(axis, bars, fmt="{:.0f}", fontsize=15)

    for position, (prediction, actual) in enumerate(zip(predicted, measured)):
        # Well clear of the bar labels, which sit just above their own bars
        axis.annotate(
            f"x{prediction / actual:.1f} too optimistic",
            (position, max(prediction, actual) * 1.13),
            ha="center",
            va="bottom",
            fontsize=14,
            color=PALETTE["red_strong"],
        )

    axis.set_xticks(list(positions))
    axis.set_xticklabels(names)
    axis.set_xlabel(f"workers x threads each, on {CONTENTION_CORES} cores")
    axis.set_ylabel("runs completed per hour")
    axis.set_ylim(0, max(predicted) * 1.34)
    axis.set_title(
        "Measured, the thinnest packing is the worst, not the best"
    )
    axis.legend(loc="upper right")
    faint_grid(axis)

    return finalize(figure, output)


def figure_contention_phases(data: pd.DataFrame, output: Path):
    """
    Where a run loses its time when 48 of them share one machine.

    Phase by phase, the median of the pack against the same run alone. The
    solver itself loses a factor five and a half, which is memory bandwidth,
    since a job at one thread does not compete for a core. Reading the input
    loses far more, and that part is ours: every job reads its own copy from
    a network share.

    :param DataFrame data: every run of the case study, packs included
    :param Path output: file to write, without a suffix
    """
    phases = [
        ("t_read_data_s", "reading\nthe input"),
        ("t_construct_model_s", "building\nthe model"),
        ("t_solve_s", "solving"),
        ("t_write_results_s", "writing\nresults"),
        ("rss_peak_os_mb", "peak\nmemory"),
    ]

    solo = _cell(data, CONTENTION_TYPICALDAYS, 1)
    pack = data[data["case_name"].str.contains("_pack1x48_")]
    if solo is None or pack.empty:
        return []

    ratios = [pack[column].median() / solo[column] for column, _ in phases]

    figure, axis = plt.subplots(figsize=(12.5, 6.4))
    colours = [
        PALETTE["red_strong"] if ratio >= 5 else PALETTE["teal"]
        for ratio in ratios
    ]
    bars = axis.bar(
        range(len(phases)),
        ratios,
        width=0.6,
        color=colours,
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )
    annotate_bars(axis, bars, fmt="x{:.1f}", fontsize=16)

    axis.axhline(1, color=PALETTE["grey_text"], linewidth=2.0, zorder=4)
    axis.set_xticks(range(len(phases)))
    axis.set_xticklabels([label for _, label in phases])
    axis.set_ylabel("48 jobs at once, relative to one job alone")
    axis.set_ylim(0, max(ratios) * 1.18)
    axis.set_title(
        "Where the time goes when 48 one thread jobs share one machine"
    )
    faint_grid(axis)

    return finalize(figure, output)


def main():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--cores",
        type=int,
        default=60,
        help="core budget the throughput figure is stated for",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=BASE / "study.log",
        help="study log, which is where the makespan of a pack lives",
    )
    args = parser.parse_args()

    data = load(args.dataset)
    if data.empty:
        raise SystemExit(f"No {CASE} runs in {args.dataset}")

    # Larger than the paper preset: a slide is read from the back of a room
    apply_publication_style(font_size=19)

    written = []
    written += figure_throughput(data, args.output / "01_throughput", args.cores)
    written += figure_memory(data, args.output / "02_memory_per_core")
    written += figure_mechanism(data, args.output / "03_why_two_threads")
    written += figure_cuts(data, args.output / "04_cuts_off")

    packs = read_packs(args.log)
    if packs:
        written += figure_measured_throughput(
            data, packs, args.output / "05_measured_throughput"
        )
        written += figure_contention_phases(
            data, args.output / "06_where_the_time_goes"
        )
    else:
        print(f"No contention packs in {args.log}, figures 5 and 6 skipped")

    print(f"Wrote {len(written)} files to {args.output}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
