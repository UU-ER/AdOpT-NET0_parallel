"""
Figures of the reading packs run by ``profile_reading.py pack``.

The question is why the reading of the input slows down so much more than the
solver when many jobs run at once. The packs read the input and nothing else,
N jobs starting at the same moment, and every job profiles its reading. These
figures split that reading into its steps and put the steps next to what the
machine was doing:

1. ``01_steps_against_jobs``   median time of every step of the reading,
                               against the number of jobs, one panel per arm
2. ``02_slowdown_by_step``     each step against the same step alone: which
                               one grows
3. ``03_threads_and_cpu``      threads one job starts, cpu it burns, and what
                               limiting its threads does to it
4. ``04_timeline_<arm>_<N>``   one pack over time: every job's steps, the
                               cores in use and the memory left on the machine

The steps are timed by the job itself (steps.csv), not taken from cProfile,
whose cumulative times are unreliable on Python 3.12. What falls between the
steps (topology, configuration, node locations) is "other".

Examples::

    python plot_reading_profiles.py
    python plot_reading_profiles.py --timelines default:6 default:20 omp1:12
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from figure_style import PALETTE, apply_publication_style, faint_grid, finalize

BASE = Path(__file__).parent
PROFILES_PATH = BASE / "reading_profiles"
OUTPUT_PATH = BASE / "figures" / "reading_profiles"

# Steps of DataHandle.read_data, in the order it runs them, with the method
# profile_reading.py times for each
STEPS = [
    ("other", None),
    ("time series", "_read_time_series"),
    ("technology data", "_read_technology_data"),
    ("network data", "_read_network_data"),
    ("clustering", "_cluster_data"),
]

STEP_COLORS = {
    "other": PALETTE["neutral"],
    "time series": PALETTE["blue_secondary"],
    "technology data": PALETTE["teal"],
    "network data": PALETTE["green_strong"],
    "clustering": PALETTE["red_strong"],
}

ARM_NAMES = {
    "default": "as the study runs",
    "omp1": "OMP_NUM_THREADS = 1",
}

ARM_STYLES = {"default": "-", "omp1": "--"}

# A pack is marked as short of memory when the machine had less than this left
# at some point of the reading. Below it Windows pages the jobs out to disk,
# and the pack measures the disk rather than the contention it is run for
MEMORY_TIGHT_MB = 500


def step_bounds(steps_file: Path) -> dict:
    """
    Start and end of the steps of one job's reading, as the job timed them

    :param Path steps_file: steps.csv written by the job
    :return: dict step -> (epoch start, epoch end)
    """
    steps = pd.read_csv(steps_file)
    return {row.step: (row.t_start, row.t_end) for row in steps.itertuples()}


def load_jobs() -> tuple:
    """
    Reads every job of every pack

    :return: (jobs, series): one row per job, and the resource curve of every
        job with its step at each sample
    """
    rows, curves = [], []

    for summary_file in sorted(PROFILES_PATH.glob("*/*/jobs*/job*/profile_summary.csv")):
        job_path = summary_file.parent
        pack_path = job_path.parent
        steps_file = job_path / "steps.csv"
        if not steps_file.is_file():
            continue

        key = {
            "arm": pack_path.parents[1].name,
            "case": pack_path.parent.name,
            "jobs": int(pack_path.name.removeprefix("jobs")),
            "job": job_path.name,
        }
        profile = pd.read_csv(summary_file).iloc[0]
        row = dict(key)
        for column in ["t_read_data_s", "parallelism_read_data", "threads_peak_read_data"]:
            row[column] = profile.get(column)
        row["cpu_read_data_s"] = profile.get("cpu_user_read_data_s", 0) + profile.get(
            "cpu_system_read_data_s", 0
        )

        curve = pd.read_csv(job_path / "profile_timeseries.csv")
        start = curve.loc[curve["event"] == "start:read_data", "timestamp"].iloc[0]
        end = curve.loc[curve["event"] == "end:read_data", "timestamp"].iloc[0]
        curve = curve[(curve["timestamp"] >= start) & (curve["timestamp"] <= end)].copy()

        # Every sample belongs to the step running at its time, and to "other"
        # between the steps (topology, configuration, node locations)
        bounds = step_bounds(steps_file)
        curve["step"] = "other"
        for step, (begin, finish) in bounds.items():
            inside = (curve["timestamp"] >= begin) & (curve["timestamp"] <= finish)
            curve.loc[inside, "step"] = step
        for step, _ in STEPS[1:]:
            begin, finish = bounds.get(step, (start, start))
            row[f"t_{step}_s"] = finish - begin
        row["t_other_s"] = max(
            0.0, row["t_read_data_s"] - sum(row[f"t_{s}_s"] for s, _ in STEPS[1:])
        )
        # Before the first step: topology, configuration
        bounds["other"] = (start, min(b for b, _ in bounds.values()))
        row["start"] = start
        row["bounds"] = bounds
        row["memory_min_mb"] = curve["sys_mem_available_mb"].min()

        # Cores the clustering kept busy: its cpu time over its duration,
        # from the samples that fall inside it
        clustering = curve[curve["step"] == "clustering"]
        if len(clustering) >= 2:
            cpu = (
                clustering["cpu_user_s"].iloc[-1]
                + clustering["cpu_system_s"].iloc[-1]
                - clustering["cpu_user_s"].iloc[0]
                - clustering["cpu_system_s"].iloc[0]
            )
            wall = clustering["timestamp"].iloc[-1] - clustering["timestamp"].iloc[0]
            row["clustering_cores"] = cpu / wall if wall > 0 else None
        rows.append(row)

        for column, value in key.items():
            curve[column] = value
        curves.append(curve)

    jobs = pd.DataFrame(rows)
    series = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return jobs, series


def pack_medians(jobs: pd.DataFrame) -> pd.DataFrame:
    """
    Median over the jobs of each pack, with the memory low of the pack

    :param jobs: one row per job
    :return: one row per arm, case and pack size
    """
    numeric = jobs.drop(columns=["bounds", "job"])
    medians = numeric.groupby(["arm", "case", "jobs"]).median(numeric_only=True)
    medians["memory_min_mb"] = numeric.groupby(["arm", "case", "jobs"])[
        "memory_min_mb"
    ].min()
    return medians.reset_index()


def figure_steps(medians: pd.DataFrame, output: Path):
    """
    Median time of every step against the number of jobs, one panel per arm
    """
    arms = [arm for arm in ARM_NAMES if arm in set(medians["arm"])]
    figure, axes = plt.subplots(
        1, len(arms), figsize=(7 * len(arms), 5.5), sharey=True, squeeze=False
    )

    for axis, arm in zip(axes[0], arms):
        data = medians[medians["arm"] == arm].sort_values("jobs")
        positions = range(len(data))
        bottom = [0.0] * len(data)
        tight = (data["memory_min_mb"] < MEMORY_TIGHT_MB).tolist()
        for step, _ in STEPS:
            values = data[f"t_{step}_s"].tolist()
            bars = axis.bar(
                positions, values, bottom=bottom, width=0.7,
                color=STEP_COLORS[step], edgecolor="black", linewidth=0.8, zorder=2,
            )
            for bar, is_tight in zip(bars, tight):
                if is_tight:
                    bar.set_hatch("///")
            bottom = [b + v for b, v in zip(bottom, values)]
        for position, total in zip(positions, bottom):
            axis.annotate(
                f"{total:.0f} s", (position, total), textcoords="offset points",
                xytext=(0, 3), ha="center", va="bottom", fontsize=11,
                color=PALETTE["grey_text"],
            )
        axis.set_xticks(list(positions), [str(j) for j in data["jobs"]])
        axis.set_xlabel("jobs reading at once")
        axis.set_title(ARM_NAMES[arm])
        faint_grid(axis)

    axes[0][0].set_ylabel("reading time of one job, s (median)")
    handles = [Patch(facecolor=STEP_COLORS[s], edgecolor="black", label=s) for s, _ in reversed(STEPS)]
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="///",
                         label=f"machine under {MEMORY_TIGHT_MB} MB free"))
    axes[0][0].legend(handles=handles, loc="upper left")
    figure.suptitle("Where the reading time of one job goes, laptop, 12 cores")
    finalize(figure, output)


def figure_slowdown(medians: pd.DataFrame, output: Path):
    """
    Every step against the same step alone, one panel per arm
    """
    arms = [a for a in ARM_NAMES if a in set(medians["arm"])]
    figure, axes = plt.subplots(
        1, len(arms), figsize=(7 * len(arms), 5.5), sharey=True, squeeze=False
    )

    for axis, arm in zip(axes[0], arms):
        data = medians[medians["arm"] == arm].sort_values("jobs")
        solo = data[data["jobs"] == 1]
        if solo.empty:
            continue
        lines = [(s, f"t_{s}_s", STEP_COLORS[s]) for s, _ in STEPS[1:] if s != "network data"]
        lines.append(("whole reading", "t_read_data_s", "black"))
        for name, column, color in lines:
            base = solo[column].iloc[0]
            if base <= 0:
                continue
            axis.plot(data["jobs"], data[column] / base, marker="o", color=color,
                      label=name, linewidth=3.2 if name == "whole reading" else 2.4)

        axis.axhline(1, color=PALETTE["grey_text"], linewidth=1)
        axis.set_xlabel("jobs reading at once")
        axis.set_title(ARM_NAMES[arm])
        faint_grid(axis)

    axes[0][0].set_ylabel("slower than the same step alone")
    axes[0][0].legend(loc="upper left")
    figure.suptitle("Which step of the reading grows with the number of jobs")
    finalize(figure, output)


def figure_clustering_cores(jobs: pd.DataFrame, medians: pd.DataFrame, output: Path):
    """
    What the threads of the clustering cost, both arms. Left: threads one job
    holds. Middle: cpu seconds one job spends on its reading, which is the
    same work in every job, so what grows is waste. Right: clustering time
    """
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(20, 5.5))

    for arm in [a for a in ARM_NAMES if a in set(medians["arm"])]:
        data = medians[medians["arm"] == arm].sort_values("jobs")
        color = PALETTE["red_strong"] if arm == "default" else PALETTE["blue_main"]
        style = dict(marker="o", linestyle=ARM_STYLES[arm], color=color, label=ARM_NAMES[arm])
        left.plot(data["jobs"], data["threads_peak_read_data"], **style)
        middle.plot(data["jobs"], data["cpu_read_data_s"], **style)
        right.plot(data["jobs"], data["t_clustering_s"], **style)

    left.set_ylabel("threads of one job (peak)")
    left.set_title("Threads one job starts")
    middle.set_ylabel("cpu seconds of one job's reading")
    middle.set_title("CPU spent on the same work")
    right.set_ylabel("clustering time of one job, s (median)")
    right.set_title("Clustering time")
    for axis in (left, middle, right):
        axis.set_xlabel("jobs reading at once")
        axis.set_ylim(bottom=0)
        faint_grid(axis)
    left.legend(loc="center right")
    finalize(figure, output)


def figure_timeline(jobs: pd.DataFrame, series: pd.DataFrame, arm: str,
                    n_jobs: int, cores: int, output: Path):
    """
    One pack over time: the steps of every job, the cores in use summed over
    the jobs, and the memory left on the machine
    """
    pack = jobs[(jobs["arm"] == arm) & (jobs["jobs"] == n_jobs)].sort_values("job")
    curves = series[(series["arm"] == arm) & (series["jobs"] == n_jobs)].copy()
    if pack.empty:
        return
    t_zero = pack["start"].min()

    figure, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [max(2, len(pack) / 4), 1.2, 1]},
    )

    for row_index, (_, job) in enumerate(pack.iterrows()):
        for step, (begin, end) in job["bounds"].items():
            top.barh(row_index, end - begin, left=begin - t_zero, height=0.8,
                     color=STEP_COLORS[step], edgecolor="black", linewidth=0.3)
    top.set_yticks(range(len(pack)), [j.removeprefix("job") for j in pack["job"]],
                   fontsize=9)
    top.set_ylabel("job")
    top.invert_yaxis()
    top.legend(handles=[Patch(facecolor=STEP_COLORS[s], edgecolor="black", label=s)
                        for s, _ in STEPS], loc="lower right", fontsize=10, ncols=5)
    top.set_title(f"{n_jobs} jobs reading at once, {ARM_NAMES[arm]}")

    # Cores in use: cpu percent of every job, on a common 1 s grid, summed
    curves["second"] = (curves["timestamp"] - t_zero).round()
    by_step = (
        curves.groupby(["second", "job", "step"])["cpu_percent"].mean().reset_index()
        .groupby(["second", "step"])["cpu_percent"].sum().unstack(fill_value=0) / 100
    )
    order = [s for s, _ in STEPS if s in by_step.columns]
    middle.stackplot(by_step.index, [by_step[s] for s in order],
                     colors=[STEP_COLORS[s] for s in order])
    middle.axhline(cores, color="black", linestyle="--", linewidth=1.2)
    middle.annotate(f"{cores} cores", (0, cores), textcoords="offset points",
                    xytext=(4, 3), fontsize=10)
    middle.set_ylabel("cores in use\n(all jobs)")
    faint_grid(middle)

    memory = curves.groupby("second")["sys_mem_available_mb"].min() / 1024
    bottom.plot(memory.index, memory.values, color=PALETTE["violet"])
    bottom.axhline(MEMORY_TIGHT_MB / 1024, color=PALETTE["grey_text"],
                   linestyle=":", linewidth=1.2)
    bottom.set_ylabel("memory free\nGB")
    bottom.set_ylim(bottom=0)
    bottom.set_xlabel("seconds since the jobs started reading")
    faint_grid(bottom)

    finalize(figure, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--timelines", nargs="*", default=None,
        help="packs to draw a timeline of, as arm:jobs; every pack by default",
    )
    parser.add_argument("--cores", type=int, default=12,
                        help="cores of the machine the packs ran on")
    args = parser.parse_args()

    apply_publication_style()
    jobs, series = load_jobs()
    if jobs.empty:
        print(f"No jobs found below {PROFILES_PATH}")
        return
    medians = pack_medians(jobs)

    columns = ["arm", "jobs", "t_read_data_s"]
    columns += [f"t_{s}_s" for s, _ in STEPS]
    columns += ["clustering_cores", "memory_min_mb"]
    print(medians[columns].round(1).to_string(index=False))

    figure_steps(medians, OUTPUT_PATH / "01_steps_against_jobs")
    figure_slowdown(medians, OUTPUT_PATH / "02_slowdown_by_step")
    figure_clustering_cores(jobs, medians, OUTPUT_PATH / "03_threads_and_cpu")

    if args.timelines is None:
        packs = medians[["arm", "jobs"]].itertuples(index=False)
    else:
        packs = [(p.split(":")[0], int(p.split(":")[1])) for p in args.timelines]
    for arm, n_jobs in packs:
        figure_timeline(jobs, series, arm, n_jobs, args.cores,
                        OUTPUT_PATH / f"04_timeline_{arm}_{n_jobs:02d}")

    print(f"\nFigures written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
