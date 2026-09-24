"""
Profiles the reading of the input, alone and in packs of concurrent jobs.

Stages 14 to 16 showed the reading phase of a job slowing down x8 at 24
concurrent jobs and x125 at 64, against x2 to x3 for the solver. This script
reproduces the reading part of such a pack and nothing else: N jobs start
``read_data`` at the same moment, each on its own copy of the input as in the
study, and no model is built or solved. A pack of 64 therefore costs the time
of its reading, minutes and not hours.

Every job runs its reading under cProfile and under the resource monitor, and
writes to its own folder:

- cprofile_read_data.prof: the functions the time goes into
- profile_summary.csv: t_read_data_s, cpu time, parallelism, peak threads, io
- profile_timeseries.csv: the resource curve of the job

``summary`` then splits the reading of every job into its steps (technology
data, clustering, time series, ...) and tabulates them, one row per job, and
``plot_reading_profiles.py`` draws them. The label is how two arms of the same
packs are told apart, e.g. the default against OMP_NUM_THREADS=1.

Nothing is written below results/, so ``run_benchmark.py collect`` does not
pick these jobs up as benchmark runs.

Examples::

    # One job, profile printed to the terminal
    python profile_reading.py one --case nl_node --typicaldays 4

    # Packs of 1, 24, 36, 48 and 64 jobs, one after the other
    python profile_reading.py pack --jobs 1 24 36 48 64 --label default

    # The same packs with the clustering held to one thread per job
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        python profile_reading.py pack --jobs 1 24 36 48 64 --label omp1

    # Tables, then figures
    python profile_reading.py summary
    python plot_reading_profiles.py
"""

import argparse
import cProfile
import os
import pstats
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

BASE = Path(__file__).parent
INPUT_DATA_PATH = BASE / "inputData"
OUTPUT_PATH = BASE / "reading_profiles"

# Seconds between launching the last job of a pack and the moment every job
# starts reading. Python and adopt take a few seconds to import, and on a full
# machine longer: a job that starts reading before the others have finished
# importing would read on an emptier machine than the pack it is part of
START_DELAY_S = 30

# The steps of DataHandle.read_data the reading is split into, by the function
# cProfile books them to. Their cumulative times do not overlap, and what is
# left of read_data is shown as "other"
STEPS = {
    "time series": "_read_time_series",
    "technology data": "_read_technology_data",
    "network data": "_read_network_data",
    "clustering": "_cluster_data",
}


def input_folder(case: str, typicaldays: int, job: int) -> Path:
    """
    Input folder of one job slot. Slots are reused by every pack and every
    label, since reading does not change the folder

    :param str case: name of the case study
    :param int typicaldays: number of typical days
    :param int job: index of the job in its pack
    :return: Path
    """
    return INPUT_DATA_PATH / f"{case}_reading_td{typicaldays}_{job}"


def prepare(case: str, typicaldays: int, n_jobs: int):
    """
    Writes the input folders of the first n job slots that do not exist yet

    :param str case: name of the case study
    :param int typicaldays: number of typical days
    :param int n_jobs: number of job slots needed
    """
    from case_studies import CASE_STUDIES

    results_path = OUTPUT_PATH / "setup"
    results_path.mkdir(parents=True, exist_ok=True)

    for job in range(n_jobs):
        folder = input_folder(case, typicaldays, job)
        if (folder / "ConfigModel.json").is_file():
            continue
        print(f"[PREPARE] {folder.name}")
        CASE_STUDIES[case].setup(
            folder,
            results_path,
            case_name=f"reading_td{typicaldays}_{job}",
            typicaldays=typicaldays,
        )


def read(input_path: Path, output_path: Path, start_at: float):
    """
    One job: waits for the start of the pack, reads the input, writes the
    profiles. Run in its own process

    :param Path input_path: input folder of the job
    :param Path output_path: folder the profiles are written to
    :param float start_at: epoch second at which reading starts
    """
    import adopt_net0 as adopt

    # The monitor runs read_data under cProfile when the phase is named here,
    # and ADOPT_CPROFILE_CPU, if set by the caller, is passed on as it is
    os.environ["ADOPT_CPROFILE"] = "read_data"

    steps = time_steps()
    model = adopt.ModelHub()

    wait = start_at - time.time()
    if wait > 0:
        time.sleep(wait)
    late_s = max(0.0, -wait)

    model.read_data(input_path)
    model.profiler.add_metadata(
        start_late_s=late_s,
        omp_num_threads=os.environ.get("OMP_NUM_THREADS", ""),
    )
    model.profiler.stop()
    model.profiler.write(output_path)

    with open(output_path / "steps.csv", "w") as file:
        file.write("step,t_start,t_end\n")
        for step, t_start, t_end in steps:
            file.write(f"{step},{t_start},{t_end}\n")


def time_steps() -> list:
    """
    Records the start and end of every step of DataHandle.read_data.

    The steps are timed here rather than taken from cProfile, whose cumulative
    times are unreliable on Python 3.12: in a pack of 12 one job reported its
    whole read_data as 0 s and another as half its length, while the own times
    of the functions added up correctly. The methods of the class are wrapped
    in this process only, the package is left untouched.

    :return: list that fills with (step, epoch start, epoch end) as read_data
        runs
    """
    import functools

    from adopt_net0.data_management.handle_input_data import DataHandle

    records = []

    def timed(step, method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            t_start = time.time()
            try:
                return method(*args, **kwargs)
            finally:
                records.append((step, t_start, time.time()))

        return wrapper

    for step, function in STEPS.items():
        setattr(DataHandle, function, timed(step, getattr(DataHandle, function)))

    return records


def pack(args):
    """
    Runs one pack per entry of --jobs, one after the other
    """
    prepare(args.case, args.typicaldays, max(args.jobs))

    for n_jobs in args.jobs:
        pack_path = (
            OUTPUT_PATH / args.label / f"{args.case}_td{args.typicaldays}" / f"jobs{n_jobs}"
        )
        if pack_path.exists() and not args.overwrite:
            print(f"[PACK] {pack_path} exists, skipped (--overwrite to redo)")
            continue

        start_at = time.time() + START_DELAY_S
        print(f"[PACK] {n_jobs} jobs, label {args.label}, reading at +{START_DELAY_S} s")

        processes = []
        for job in range(n_jobs):
            job_path = pack_path / f"job{job:02d}"
            job_path.mkdir(parents=True, exist_ok=True)
            log_file = open(job_path / "stdout.log", "w")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "read",
                "--input",
                str(input_folder(args.case, args.typicaldays, job)),
                "--output",
                str(job_path),
                "--start-at",
                str(start_at),
            ]
            processes.append(
                (
                    subprocess.Popen(
                        command, cwd=str(BASE), stdout=log_file, stderr=subprocess.STDOUT
                    ),
                    log_file,
                )
            )

        failed = 0
        for process, log_file in processes:
            failed += process.wait() != 0
            log_file.close()

        makespan = time.time() - start_at
        print(f"[PACK] {n_jobs} jobs done in {makespan:.0f} s, {failed} failed")


def step_times(steps_file: Path, total: float) -> dict:
    """
    Splits the reading of one job into its steps

    :param Path steps_file: steps.csv written by the job
    :param float total: duration of the whole reading
    :return: dict step -> seconds, "other" included
    """
    times = {step: 0.0 for step in STEPS}
    with open(steps_file) as file:
        next(file)
        for line in file:
            step, t_start, t_end = line.strip().split(",")
            times[step] += float(t_end) - float(t_start)
    times["other"] = max(0.0, total - sum(times.values()))
    return times


def summary(args):
    """
    Writes one row per job to reading_profiles/summary.csv and the medians
    per pack to summary_median.csv. The figures are plot_reading_profiles.py
    """
    import pandas as pd

    rows = []
    for summary_file in sorted(OUTPUT_PATH.glob("*/*/jobs*/job*/profile_summary.csv")):
        job_path = summary_file.parent
        pack_path = job_path.parent
        row = {
            "label": pack_path.parents[1].name,
            "case": pack_path.parent.name,
            "jobs": int(pack_path.name.removeprefix("jobs")),
            "job": job_path.name,
        }
        profile = pd.read_csv(summary_file).iloc[0]
        for column in [
            "t_read_data_s",
            "cpu_user_read_data_s",
            "cpu_system_read_data_s",
            "parallelism_read_data",
            "threads_peak_read_data",
            "start_late_s",
            "io_read_mb",
        ]:
            row[column] = profile.get(column, None)
        steps_file = job_path / "steps.csv"
        if steps_file.is_file():
            times = step_times(steps_file, row["t_read_data_s"])
            row.update({f"t_{step}_s": t for step, t in times.items()})
        rows.append(row)

    if not rows:
        print(f"No jobs found below {OUTPUT_PATH}")
        return

    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_PATH / "summary.csv", index=False)

    medians = (
        table.groupby(["label", "case", "jobs"])
        .median(numeric_only=True)
        .reset_index()
    )
    medians.to_csv(OUTPUT_PATH / "summary_median.csv", index=False)

    columns = ["label", "case", "jobs", "t_read_data_s"]
    columns += [f"t_{step}_s" for step in [*STEPS, "other"]]
    columns += ["parallelism_read_data", "threads_peak_read_data"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(medians[[c for c in columns if c in medians]].round(1).to_string(index=False))

    print("\nFigures: python plot_reading_profiles.py")


def one(args):
    """
    One job in this process, profile printed to the terminal
    """
    import adopt_net0 as adopt

    prepare(args.case, args.typicaldays, 1)
    input_path = input_folder(args.case, args.typicaldays, 0)

    timer = time.process_time if args.cpu else time.perf_counter
    profile = cProfile.Profile(timer)

    model = adopt.ModelHub()
    start = time.perf_counter()
    profile.enable()
    model.read_data(input_path)
    profile.disable()
    wall = time.perf_counter() - start
    model.profiler.stop()

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output = args.output or OUTPUT_PATH / "read_data.prof"
    profile.dump_stats(str(output))
    print(f"\nread_data took {wall:.1f} s wall, profile written to {output}\n")

    stats = pstats.Stats(profile).strip_dirs()
    stats.sort_stats("cumulative").print_stats(args.top)
    stats.sort_stats("tottime").print_stats(args.top)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def add_case(command):
        command.add_argument("--case", default="nl_node")
        command.add_argument("--typicaldays", type=int, default=4)

    command = commands.add_parser("one", help="one job, profile printed")
    add_case(command)
    command.add_argument("--output", type=Path, default=None)
    command.add_argument("--top", type=int, default=25)
    command.add_argument("--cpu", action="store_true",
                         help="time with the cpu clock instead of the wall clock")

    command = commands.add_parser("pack", help="packs of concurrent jobs")
    add_case(command)
    command.add_argument("--jobs", type=int, nargs="+", required=True)
    command.add_argument("--label", default="default")
    command.add_argument("--overwrite", action="store_true")

    command = commands.add_parser("read", help="one job of a pack, internal")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--start-at", type=float, required=True)

    commands.add_parser("summary", help="tables of all packs")

    args = parser.parse_args()
    if args.command == "one":
        one(args)
    elif args.command == "pack":
        pack(args)
    elif args.command == "read":
        read(args.input, args.output, args.start_at)
    elif args.command == "summary":
        summary(args)


if __name__ == "__main__":
    main()
