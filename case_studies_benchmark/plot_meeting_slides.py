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
5. ``measured``     what a full machine delivers, against the arithmetic
6. ``phases``       where a job of a full pack loses its time
7. ``fill``         throughput against how many jobs run at once, at one
                    and at two threads, with repeated packs drawn hollow
8. ``phases``       the wall time of a job split into its phases, against
                    how many jobs run at once
9. ``slowdown``     reading against the solver: how much slower each runs,
                    and whether the solver gets the cores it asked for
10. ``timeline``    how many jobs of a pack are in each phase over time

Figures 7 to 10 read the phase timestamps of each job from ``--results``, and
``--figures`` draws a subset, so a folder of earlier figures is not redrawn.

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

# The packs the timeline figure draws: one thread at the peak and at a full
# machine, and two threads at a full machine
TIMELINE_PACKS = ["fill24x1", "fill48x1", "fill24x2"]


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
    loses far more, and that part is ours: it includes clustering the year
    into typical days, which alone already keeps about twelve cores busy, see
    figure_reading.

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


# The two thread counts stage 15 swept, with the colour and marker every
# stage 15 figure gives them
FILL_ARMS = [
    (1, "1 thread a job", PALETTE["blue_main"], "o"),
    (2, "2 threads a job", PALETTE["red_strong"], "s"),
]

# Phases of a run as the stage 15 figures group them. Checks, the model, the
# balances and the solver setup are all a few seconds alone and are one bar
PHASE_GROUPS = [
    ("reading the input", ["t_read_data_s"], PALETTE["red_light"]),
    (
        "building the model",
        [
            "t_preprocessing_checks_s",
            "t_construct_model_s",
            "t_construct_balances_s",
            "t_solver_setup_s",
        ],
        PALETTE["neutral"],
    ),
    ("solving", ["t_solve_s"], PALETTE["blue_secondary"]),
    ("writing results", ["t_write_results_s"], PALETTE["green_strong"]),
]


def read_fills(log: Path):
    """
    Reads what each pack of the worker count stages delivered.

    Stage 15 names a pack fill<jobs>x<threads> and stage 16, which repeats
    some of them, rep<jobs>x<threads>. Stage 14 names its packs the other way
    round, pack<threads>x<jobs>, and those are read too, since they are the
    same measurement at a full machine. A pack that was attempted more than
    once counts at its last attempt, because the early attempts of stage 15
    all failed on the shared summary file and were run again.

    :param Path log: study.log
    :return: list of dicts with the threads, the jobs, the measured
        throughput and whether the pack is a repeat
    """
    if not log.is_file():
        return []

    pattern = re.compile(
        r"\[(?:CONTENTION|WORKERS|REPEAT)\] (pack|fill|rep)(\d+)x(\d+): "
        r"makespan [\d.]+ min, ([\d.]+) jobs per hour, (\d+) of (\d+) failed"
    )

    last = {}
    for line in log.read_text(errors="ignore").splitlines():
        found = pattern.search(line)
        if not found:
            continue
        kind, first, second, rate, failed, total = found.groups()
        jobs, threads = (second, first) if kind == "pack" else (first, second)
        last[(kind, int(jobs), int(threads))] = {
            "threads": int(threads),
            "jobs": int(jobs),
            "measured": float(rate),
            "repeat": kind != "fill",
            "failed": int(failed),
            "total": int(total),
        }

    # A pack that lost every job has no makespan worth the name
    return [pack for pack in last.values() if pack["failed"] < pack["total"]]


def pack_runs(data: pd.DataFrame):
    """
    The runs of the fill and contention packs, with their packing as columns.

    :param DataFrame data: every run of the case study
    :return: DataFrame of the pack runs with jobs, pack_threads and label
        columns
    """
    names = data["case_name"].str.extract(r"_(fill|pack|rep)(\d+)x(\d+)_job")
    packs = data[names[0].notna()].copy()
    names = names[names[0].notna()]
    first, second = names[1].astype(int), names[2].astype(int)
    stage14 = names[0] == "pack"
    packs["jobs"] = second.where(stage14, first)
    packs["pack_threads"] = first.where(stage14, second)
    packs["label"] = names[0] + names[1] + "x" + names[2]
    return packs


def figure_fill(fills: list, output: Path):
    """
    Throughput against how many jobs run at once, at one and at two threads.

    The x axis is the number of jobs rather than the number of cores they ask
    for, because that is the axis the two curves line up on. A repeat of a
    pack is drawn hollow next to its first sample, so the spread between two
    runs of the same pack is on the slide next to the differences it has to
    be compared with.

    :param list fills: packs as read by read_fills
    :param Path output: file to write, without a suffix
    """
    if not any(not pack["repeat"] for pack in fills):
        return []

    figure, axis = plt.subplots(figsize=(12.5, 6.4))
    for threads, label, colour, marker in FILL_ARMS:
        first = sorted(
            (p for p in fills if p["threads"] == threads and not p["repeat"]),
            key=lambda p: p["jobs"],
        )
        again = [p for p in fills if p["threads"] == threads and p["repeat"]]
        axis.plot(
            [p["jobs"] for p in first],
            [p["measured"] for p in first],
            color=colour,
            marker=marker,
            markersize=11,
            linewidth=2.6,
            label=label,
            zorder=3,
        )
        axis.scatter(
            [p["jobs"] for p in again],
            [p["measured"] for p in again],
            s=170,
            marker=marker,
            facecolors="white",
            edgecolors=colour,
            linewidths=2.4,
            label=f"{label}, same pack on another night",
            zorder=4,
        )

    axis.axvline(
        CONTENTION_CORES,
        color=PALETTE["grey_text"],
        linestyle="--",
        linewidth=1.6,
        zorder=2,
    )
    axis.annotate(
        f"{CONTENTION_CORES} cores",
        (CONTENTION_CORES, 5),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=14,
        color=PALETTE["grey_text"],
    )
    axis.set_xlabel("jobs running at once")
    axis.set_ylabel("runs completed per hour")
    axis.set_ylim(0, max(p["measured"] for p in fills) * 1.15)
    axis.set_title("Both thread counts peak at the same 70 to 78 runs an hour")
    axis.legend(loc="lower left", fontsize=13)
    faint_grid(axis)

    return finalize(figure, output)


def figure_phases(data: pd.DataFrame, output: Path):
    """
    Where the wall time of a job goes, phase by phase, as the machine fills.

    Median of every pack, next to the same run alone, one panel per thread
    count. The height of a bar is the wall time of a job, so the figure
    answers directly whether the time a full machine costs is lost in the
    solver or around it.

    :param DataFrame data: every run of the case study, packs included
    :param Path output: file to write, without a suffix
    """
    packs = pack_runs(data)
    packs = packs[packs["label"].str.startswith("fill")]
    if packs.empty:
        return []

    columns = [column for _, group, _ in PHASE_GROUPS for column in group]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15.5, 6.6),
        sharey=True,
        gridspec_kw={"width_ratios": [8, 7]},
    )
    for axis, (threads, label, _, _) in zip(axes, FILL_ARMS):
        solo = _cell(data, CONTENTION_TYPICALDAYS, threads)
        medians = (
            packs[packs["pack_threads"] == threads]
            .groupby("jobs")[columns]
            .median()
        )
        rows = [("alone", solo[columns])] + [
            (str(jobs), medians.loc[jobs]) for jobs in medians.index
        ]

        bottom = [0.0] * len(rows)
        for name, group, colour in PHASE_GROUPS:
            heights = [row[group].sum() / 60 for _, row in rows]
            axis.bar(
                range(len(rows)),
                heights,
                bottom=bottom,
                width=0.7,
                color=colour,
                edgecolor="black",
                linewidth=0.8,
                label=name,
                zorder=3,
            )
            bottom = [b + h for b, h in zip(bottom, heights)]

        axis.set_xticks(range(len(rows)))
        axis.set_xticklabels([name for name, _ in rows])
        axis.set_xlabel("jobs running at once")
        axis.set_title(label)
        faint_grid(axis)

    axes[0].set_ylabel("wall time of one job, min (median)")
    axes[0].legend(loc="upper left", fontsize=13)
    figure.suptitle("Past 30 jobs the extra time is mostly reading the input")

    return finalize(figure, output)


def figure_slowdown(data: pd.DataFrame, output: Path):
    """
    How much slower each part of a job runs as the machine fills.

    Left: reading the input and the solver, each against the same run alone.
    The solver is measured per unit of gurobi work, which is deterministic and
    identical in every job of a pack, so the ratio is the slowdown of the
    solver alone, net of any change in the search.
    Right: the cores the solver keeps busy, against the same run alone. Not
    against the threads asked, because two threads alone keep only 1.3 cores
    busy. Below one, the solver is waiting for cores it had alone, which
    means something else on the machine is using more cores than the pack
    reserved.

    :param DataFrame data: every run of the case study, packs included
    :param Path output: file to write, without a suffix
    """
    packs = pack_runs(data)
    packs = packs[packs["label"].str.startswith("fill")].copy()
    if packs.empty:
        return []
    packs["solver_speed"] = packs["gurobi_runtime_s"] / packs["gurobi_work"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(15.5, 6.4))
    for threads, label, colour, marker in FILL_ARMS:
        solo = _cell(data, CONTENTION_TYPICALDAYS, threads)
        solo_speed = solo["gurobi_runtime_s"] / solo["gurobi_work"]
        arm = (
            packs[packs["pack_threads"] == threads]
            .groupby("jobs")
            .median(numeric_only=True)
        )
        left.plot(
            arm.index,
            arm["t_read_data_s"] / solo["t_read_data_s"],
            color=colour,
            marker=marker,
            markersize=10,
            linewidth=2.6,
            label=f"reading, {label}",
            zorder=3,
        )
        left.plot(
            arm.index,
            arm["solver_speed"] / solo_speed,
            color=colour,
            marker=marker,
            markersize=10,
            linewidth=2.6,
            linestyle="--",
            markerfacecolor="white",
            label=f"solver, {label}",
            zorder=3,
        )
        right.plot(
            arm.index,
            arm["parallelism_solve"] / solo["parallelism_solve"],
            color=colour,
            marker=marker,
            markersize=10,
            linewidth=2.6,
            label=label,
            zorder=3,
        )

    left.axhline(1, color=PALETTE["grey_text"], linewidth=1.6)
    left.set_yscale("log")
    left.set_yticks([1, 2, 5, 10, 20, 50, 100])
    left.set_yticklabels(["x1", "x2", "x5", "x10", "x20", "x50", "x100"])
    left.set_xlabel("jobs running at once")
    left.set_ylabel("slower than the same run alone")
    left.set_title("Reading degrades far more than the solver")
    left.legend(loc="upper left", fontsize=12)
    faint_grid(left)

    right.axhline(1, color=PALETTE["grey_text"], linewidth=1.6)
    right.set_ylim(0, 1.3)
    right.set_xlabel("jobs running at once")
    right.set_ylabel("cores the solver keeps busy, vs alone")
    right.set_title("The solver loses cores it had alone")
    right.legend(loc="lower left", fontsize=13)
    faint_grid(right)

    return finalize(figure, output)


def _phase_intervals(results: Path, label: str):
    """
    Start and end of every phase of every job of one pack, in wall clock time.

    :param Path results: folder the pack results were archived under
    :param str label: pack label, such as fill48x1
    :return: DataFrame with job, phase, start and end in seconds from the
        start of the first job
    """
    bounds = [
        ("reading the input", "start:read_data", "end:read_data"),
        ("building the model", "end:read_data", "start:solve"),
        ("solving", "start:solve", "end:solve"),
        ("writing results", "end:solve", "end:write_results"),
    ]
    rows = []
    for series in results.glob(f"**/{label}/job*/*/profile_timeseries.csv"):
        events = pd.read_csv(series, usecols=["timestamp", "event"]).dropna()
        stamps = dict(zip(events["event"], events["timestamp"]))
        job = series.parent.parent.name
        for phase, start, end in bounds:
            if start in stamps and end in stamps:
                rows.append((job, phase, stamps[start], stamps[end]))

    intervals = pd.DataFrame(rows, columns=["job", "phase", "start", "end"])
    if not intervals.empty:
        intervals[["start", "end"]] -= intervals["start"].min()
    return intervals


def figure_timeline(results: Path, labels: list, output: Path):
    """
    How many jobs of a pack are in each phase, half minute by half minute.

    Shows whether the phases of different jobs overlap: if every job reads at
    the same time, reading contends with reading and solving with solving; if
    they are staggered, one job's reading takes cores from another job's
    solver.

    :param Path results: folder the pack results were archived under
    :param list labels: pack labels to draw, one panel each
    :param Path output: file to write, without a suffix
    """
    panels = [(label, _phase_intervals(results, label)) for label in labels]
    panels = [(label, frame) for label, frame in panels if not frame.empty]
    if not panels:
        return []

    figure, axes = plt.subplots(
        len(panels), 1, figsize=(12.5, 3.4 * len(panels) + 0.8), sharex=True
    )
    axes = [axes] if len(panels) == 1 else list(axes)
    names = [name for name, _, _ in PHASE_GROUPS]
    colours = [colour for _, _, colour in PHASE_GROUPS]

    for axis, (label, frame) in zip(axes, panels):
        grid = list(range(0, int(frame["end"].max()) + 30, 30))
        counts = [
            [
                (
                    (frame["phase"] == name)
                    & (frame["start"] <= t)
                    & (frame["end"] > t)
                ).sum()
                for t in grid
            ]
            for name in names
        ]
        axis.stackplot(
            [t / 60 for t in grid],
            counts,
            labels=names,
            colors=colours,
            edgecolor="black",
            linewidth=0.4,
        )
        jobs, threads = re.match(r"\D+(\d+)x(\d+)", label).groups()
        axis.set_ylim(0, int(jobs) * 1.05)
        axis.set_ylabel("jobs")
        axis.set_title(
            f"{jobs} jobs of {threads} thread{'s' if threads != '1' else ''}",
            fontsize=16,
        )
        faint_grid(axis)

    axes[-1].set_xlabel("minutes from the start of the pack")
    axes[0].legend(loc="upper right", fontsize=12, ncol=2)

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
    parser.add_argument(
        "--results",
        type=Path,
        default=BASE / "results",
        help="results folder, searched for the phase timestamps of a pack",
    )
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        help="numbers of the figures to draw, all of them if left out",
    )
    args = parser.parse_args()

    data = load(args.dataset)
    if data.empty:
        raise SystemExit(f"No {CASE} runs in {args.dataset}")

    # Larger than the paper preset: a slide is read from the back of a room
    apply_publication_style(font_size=19)

    # Each figure with the inputs it needs, so that a subset can be drawn
    packs = read_packs(args.log)
    fills = read_fills(args.log)
    figures = {
        1: lambda: figure_throughput(
            data, args.output / "01_throughput", args.cores
        ),
        2: lambda: figure_memory(data, args.output / "02_memory_per_core"),
        3: lambda: figure_mechanism(data, args.output / "03_why_two_threads"),
        4: lambda: figure_cuts(data, args.output / "04_cuts_off"),
        5: lambda: figure_measured_throughput(
            data, packs, args.output / "05_measured_throughput"
        ),
        6: lambda: figure_contention_phases(
            data, args.output / "06_where_the_time_goes"
        ),
        7: lambda: figure_fill(fills, args.output / "07_throughput_against_jobs"),
        8: lambda: figure_phases(data, args.output / "08_phases_against_jobs"),
        9: lambda: figure_slowdown(data, args.output / "09_reading_or_solver"),
        10: lambda: figure_timeline(
            args.results, TIMELINE_PACKS, args.output / "10_timeline"
        ),
    }

    written = []
    for number in args.figures or sorted(figures):
        drawn = figures[number]()
        if not drawn:
            print(f"Figure {number} skipped, its runs are not in the inputs")
        written += drawn

    print(f"Wrote {len(written)} files to {args.output}")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
