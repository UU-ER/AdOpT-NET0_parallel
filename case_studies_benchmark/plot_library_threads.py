"""
Figures of stage 17 against stage 15: the same packs, with the numerical
libraries held to one thread and left free.

Stages 14 to 16 found the reading of the input slowing down by up to x125 when
many jobs ran at once, and the reading packs of 2026-09-24 found why: every
job's typical day clustering (sklearn KMeans over OpenMP, and numpy's BLAS)
started about 100 threads on the 48 core server, whatever gurobi was given.
Stage 17 ran the packs of stage 15 again with OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS and MKL_NUM_THREADS at 1. These figures put the two side
by side:

1. ``01_throughput_against_jobs``  runs an hour against jobs at once, one
                                   panel per thread count, stage 15 and 16
                                   against stage 17
2. ``02_phases_against_jobs``      where the wall time of one job goes, phase
                                   by phase, libraries free and at one thread
3. ``03_reading_and_solver``       reading time and solver speed against jobs
                                   at once: what the fix buys in each
4. ``04_timeline_<jobs>x1``        jobs in each phase over time, the same pack
                                   with the libraries free and at one thread

Examples::

    python plot_library_threads.py
    python plot_library_threads.py --share "//server/share/.../case_studies_benchmark"
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import PALETTE, apply_publication_style, faint_grid, finalize
from plot_meeting_slides import PHASE_GROUPS, _phase_intervals

BASE = Path(__file__).parent
OUTPUT_PATH = BASE / "figures" / "library_threads"

CORES = 48

# The two arms, by the label their packs carry. Stage 16 repeated four one
# thread packs of stage 15 with the libraries free, so it belongs to that arm
ARMS = {
    "free": {
        "name": "libraries free (stages 15, 16)",
        "labels": ["fill", "rep"],
        "colour": PALETTE["neutral"],
        "edge": PALETTE["grey_text"],
    },
    "one": {
        "name": "libraries at 1 thread (stage 17)",
        "labels": ["omp"],
        "colour": PALETTE["blue_main"],
        "edge": PALETTE["blue_main"],
    },
}

THREAD_ARMS = [(1, "1 thread a job", "o"), (2, "2 threads a job", "s")]


def arm_of(kind: str) -> str:
    """
    The arm a pack label belongs to

    :param str kind: fill, rep or omp
    :return: key of ARMS
    """
    return next(arm for arm, spec in ARMS.items() if kind in spec["labels"])


def read_throughput(log: Path) -> pd.DataFrame:
    """
    Runs an hour of every pack of stages 15 to 17, from study.log.

    A pack run more than once counts at its last attempt: the first attempts
    of stage 15 all failed on the shared summary file. The rate is the one the
    study logged, jobs of the pack over its makespan, failed jobs included,
    which is how every stage has been compared so far. Stage 17 lost one job
    in each of 48x1 and 64x1, both at import, from the virtual environment on
    the network share.

    :param Path log: study.log
    :return: one row per pack
    """
    pattern = re.compile(
        r"\[(?:WORKERS|REPEAT|WORKERS_OMP)\] (fill|rep|omp)(\d+)x(\d+): "
        r"makespan ([\d.]+) min, ([\d.]+) jobs per hour, (\d+) of (\d+) failed"
    )
    last = {}
    for line in log.read_text(errors="ignore").splitlines():
        found = pattern.search(line)
        if not found:
            continue
        kind, jobs, threads, makespan, rate, failed, total = found.groups()
        last[(kind, int(jobs), int(threads))] = {
            "kind": kind,
            "arm": arm_of(kind),
            "jobs": int(jobs),
            "threads": int(threads),
            "makespan_min": float(makespan),
            "rate": float(rate),
            "failed": int(failed),
            "total": int(total),
        }
    packs = pd.DataFrame(last.values())
    return packs[packs["failed"] < packs["total"]]


def read_runs(dataset: Path) -> pd.DataFrame:
    """
    The runs of the packs of stages 15 to 17, with their packing as columns

    :param Path dataset: benchmark_dataset.csv
    :return: one row per job
    """
    data = pd.read_csv(dataset, low_memory=False)
    names = data["case_name"].astype(str).str.extract(r"_(fill|rep|omp)(\d+)x(\d+)_job")
    runs = data[names[0].notna()].copy()
    names = names[names[0].notna()]
    runs["kind"] = names[0]
    runs["arm"] = names[0].map(arm_of)
    runs["jobs"] = names[1].astype(int)
    runs["pack_threads"] = names[2].astype(int)
    runs["label"] = names[0] + names[1] + "x" + names[2]
    runs["solver_s_per_work"] = runs["gurobi_runtime_s"] / runs["gurobi_work"]
    for _, group, _ in PHASE_GROUPS:
        for column in group:
            runs[column] = runs[column].fillna(0)
    return runs


def figure_throughput(packs: pd.DataFrame, output: Path):
    """
    Runs an hour against jobs at once, one panel per thread count
    """
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.4), sharey=True)

    for axis, (threads, title, marker) in zip(axes, THREAD_ARMS):
        for arm, spec in ARMS.items():
            data = packs[(packs["arm"] == arm) & (packs["threads"] == threads)]
            first = data[data["kind"] != "rep"].sort_values("jobs")
            again = data[data["kind"] == "rep"]
            axis.plot(first["jobs"], first["rate"], marker=marker, markersize=10,
                      linewidth=2.6, color=spec["edge"], label=spec["name"], zorder=3)
            if not again.empty:
                axis.scatter(again["jobs"], again["rate"], s=150, marker=marker,
                             facecolors="white", edgecolors=spec["edge"],
                             linewidths=2.2, label="same pack, run again", zorder=4)
            for _, pack in first.iterrows():
                axis.annotate(f"{pack['rate']:.0f}", (pack["jobs"], pack["rate"]),
                              textcoords="offset points", xytext=(0, 9),
                              ha="center", fontsize=11, color=spec["edge"])

        axis.axvline(CORES / threads, color=PALETTE["grey_text"], linestyle="--",
                     linewidth=1.4, zorder=2)
        axis.annotate(f"{CORES} cores full", (CORES / threads, 4), xytext=(6, 0),
                      textcoords="offset points", fontsize=12,
                      color=PALETTE["grey_text"])
        axis.set_title(title)
        axis.set_xlabel("jobs running at once")
        axis.set_ylim(0, packs["rate"].max() * 1.15)
        faint_grid(axis)

    axes[0].set_ylabel("runs completed per hour")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=12)
    figure.suptitle("Libraries held to one thread: x1.4 to x3.4 the runs an hour, and no cliff past 30 jobs")
    return finalize(figure, output)


def figure_phases(runs: pd.DataFrame, output: Path):
    """
    Median wall time of one job split by phase, one row per thread count and
    one column per arm, on a shared axis
    """
    figure, axes = plt.subplots(2, 2, figsize=(17, 11), sharey=True)

    for row, (threads, title, _) in enumerate(THREAD_ARMS):
        for column, (arm, spec) in enumerate(ARMS.items()):
            axis = axes[row][column]
            data = runs[
                (runs["arm"] == arm)
                & (runs["pack_threads"] == threads)
                & (runs["kind"] != "rep")
            ]
            columns = [c for _, group, _ in PHASE_GROUPS for c in group]
            medians = data.groupby("jobs")[columns].median().sort_index()
            positions = range(len(medians))
            bottom = [0.0] * len(medians)
            for name, group, colour in PHASE_GROUPS:
                heights = (medians[group].sum(axis=1) / 60).tolist()
                axis.bar(positions, heights, bottom=bottom, width=0.7,
                         color=colour, edgecolor="black", linewidth=0.8,
                         label=name, zorder=3)
                bottom = [b + h for b, h in zip(bottom, heights)]
            for position, total in zip(positions, bottom):
                axis.annotate(f"{total:.0f}", (position, total),
                              textcoords="offset points", xytext=(0, 3),
                              ha="center", fontsize=11, color=PALETTE["grey_text"])
            axis.set_xticks(list(positions), [str(j) for j in medians.index])
            axis.set_title(f"{title}, {spec['name']}", fontsize=14)
            faint_grid(axis)
            if row == 1:
                axis.set_xlabel("jobs running at once")
        axes[row][0].set_ylabel("wall time of one job, min (median)")

    axes[0][0].legend(loc="upper left", fontsize=12)
    figure.suptitle("With the libraries at one thread the reading stops growing, and a job is its solve")
    return finalize(figure, output)


def figure_reading_and_solver(runs: pd.DataFrame, output: Path):
    """
    Left: reading time of one job. Right: solver seconds per unit of gurobi
    work, which is deterministic, so what grows is the solver being slowed by
    the machine and not a different search
    """
    figure, (left, right) = plt.subplots(1, 2, figsize=(16, 6.4))

    for threads, title, marker in THREAD_ARMS:
        for arm, spec in ARMS.items():
            data = runs[(runs["arm"] == arm) & (runs["pack_threads"] == threads)
                        & (runs["kind"] != "rep")]
            medians = data.groupby("jobs").median(numeric_only=True).sort_index()
            style = dict(marker=marker, markersize=9, linewidth=2.4,
                         color=spec["edge"],
                         linestyle="-" if threads == 1 else "--",
                         label=f"{title}, {spec['name']}")
            left.plot(medians.index, medians["t_read_data_s"] / 60, **style)
            right.plot(medians.index, medians["solver_s_per_work"], **style)

    left.set_ylabel("reading time of one job, min (median)")
    left.set_title("Reading the input")
    right.set_ylabel("solver seconds per unit of gurobi work")
    right.set_title("Solver speed")
    for axis in (left, right):
        axis.set_xlabel("jobs running at once")
        axis.set_ylim(bottom=0)
        faint_grid(axis)
    left.legend(loc="upper left", fontsize=11)
    return finalize(figure, output)


def figure_timeline(results: Path, jobs: int, output: Path):
    """
    Jobs in each phase over time, the same one thread pack in both arms
    """
    panels = []
    for kind, arm in [("fill", "free"), ("omp", "one")]:
        frame = _phase_intervals(results, f"{kind}{jobs}x1")
        if not frame.empty:
            panels.append((arm, frame))
    if len(panels) < 2:
        return []

    end = max(frame["end"].max() for _, frame in panels)
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    names = [name for name, _, _ in PHASE_GROUPS]
    colours = [colour for _, _, colour in PHASE_GROUPS]

    for axis, (arm, frame) in zip(axes, panels):
        grid = list(range(0, int(end) + 30, 30))
        counts = [
            [((frame["phase"] == name) & (frame["start"] <= t) & (frame["end"] > t)).sum()
             for t in grid]
            for name in names
        ]
        axis.stackplot([t / 60 for t in grid], counts, labels=names, colors=colours,
                       edgecolor="black", linewidth=0.4)
        makespan = frame["end"].max() / 60
        axis.axvline(makespan, color="black", linestyle=":", linewidth=1.4)
        axis.annotate(f"last job done at {makespan:.0f} min", (makespan, jobs * 0.9),
                      xytext=(6, 0), textcoords="offset points", fontsize=12)
        axis.set_ylim(0, jobs * 1.05)
        axis.set_ylabel("jobs")
        axis.set_title(f"{jobs} jobs of 1 thread, {ARMS[arm]['name']}", fontsize=15)
        faint_grid(axis)

    axes[-1].set_xlabel("minutes from the start of the pack")
    axes[-1].legend(loc="center right", fontsize=12)
    return finalize(figure, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--share", type=Path, default=BASE,
        help="case_studies_benchmark folder holding study.log, the dataset and "
        "results, e.g. on the network share",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--timelines", type=int, nargs="*", default=[36, 48, 64])
    args = parser.parse_args()

    apply_publication_style()
    packs = read_throughput(args.share / "study.log")
    runs = read_runs(args.share / "benchmark_dataset.csv")

    table = packs.pivot_table(index=["threads", "jobs"], columns="kind", values="rate")
    print(table.round(1).to_string())

    figure_throughput(packs, args.output / "01_throughput_against_jobs")
    figure_phases(runs, args.output / "02_phases_against_jobs")
    figure_reading_and_solver(runs, args.output / "03_reading_and_solver")
    for jobs in args.timelines:
        figure_timeline(args.share / "results", jobs,
                        args.output / f"04_timeline_{jobs}x1")

    print(f"\nFigures written to {args.output}")


if __name__ == "__main__":
    main()
