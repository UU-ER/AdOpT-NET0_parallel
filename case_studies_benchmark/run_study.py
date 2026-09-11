"""
Runs the whole benchmark study unattended and reports the results.

The study is split into stages that run one after the other. Every stage can
be skipped, and every run inside a stage is skipped if it was already done, so
the study can be interrupted and started again without losing work.

Stages:

0. sizes      how many variables, constraints and binaries each complexity
              knob adds. The models are built but never solved, so this is
              cheap and gives the size side of the study on its own
1. matrix     every combination of the complexity knobs at small numbers of
              typical days. This fills the cluster of runs that have a
              comparable size, which is where complexity can be compared
              without size getting in the way
2. scaling    a reduced set of knobs at larger numbers of typical days, which
              turns every configuration into a line across model sizes
3. full       full time resolution for a few configurations, which anchors the
              largest end of the scale
4. report     collects every run into one dataset, draws the figures and
              writes a summary
5. threads    a few configurations run again at a ladder of thread counts, to
              find out why a restricted solver is the faster one. Not part of
              the study proper, so it is not in the default set of stages
6. gurobi a1  the root relaxation algorithm crossed with the thread count, on
              a cheap configuration and one carrying binaries. Answers whether
              any combination of a solver option and a thread count beats the
              default at all
7. gurobi a2  the same factorial on two more configurations, which says whether
              the answer depends on the complexity of the model
8. cuts a1    the options that steer the cut loop at the root node, crossed
              with the thread count. That loop is where about 80 % of the
              solve goes on this model family, so these options have far more
              time in scope than the root algorithm does
9. cuts a2    the same on two more configurations

Examples::

    python run_study.py                     # the whole study
    python run_study.py --stages 0 1        # only the sizes and the matrix
    python run_study.py --stages 4          # only redo the figures and report
    python run_study.py --stages 5          # only the thread ladder
    python run_study.py --stages 6          # gurobi options crossed with threads
    python run_study.py --stages 6 --dry-run
"""

import argparse
import datetime
import re
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

BASE = Path(__file__).parent
REPO_ROOT = BASE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BASE))

RESULTS_PATH = BASE / "results"
FIGURES_PATH = BASE / "figures"
DATASET_FILE = BASE / "benchmark_dataset.csv"
REPORT_FILE = BASE / "STUDY_REPORT.md"
LOG_FILE = BASE / "study.log"
MACHINE_FILE = BASE / "machine.json"

CASE = "four_node"

# Hours. A run that hits this limit still gives a valid resource measurement,
# it just does not reach the optimum, and the sweep carries on
TIME_LIMIT = 2

# Threads the solver may use. 0 is every core of the machine, which is what the
# first sweep ran with. Setting it makes every run look like a cluster job that
# asked for that many cores, and it reaches the case name, so a sweep at a
# different thread count does not skip the runs of the first one as already
# done. Overridden from the command line with --threads
THREADS = 0

# Knobs swept in the matrix stage. typicaldays_method is left out on purpose:
# it multiplies the model by about forty, and at full resolution it has no
# effect at all, which would produce duplicate runs
MATRIX_KNOBS = [
    "storage",
    "electricity_price",
    "hydrogen_demand",
    "pipeline_capex",
    "bidirectional_precise",
    "pipeline_size_min",
]
MATRIX_TYPICALDAYS = [2, 4]

# A smaller set of knobs, run at larger sizes so that every configuration
# becomes a line rather than a point
SCALING_KNOBS = ["storage", "bidirectional_precise", "pipeline_size_min"]
SCALING_TYPICALDAYS = [2, 4, 16, 32]

# Full resolution is expensive, so only a few configurations are run. The
# model reaches about three million variables here.
FULL_RESOLUTION_RUNS = [
    ["--set", "pv=off", "--set", "storage=off"],
    ["--set", "pv=off", "--set", "storage=on"],
    [
        "--set",
        "storage=on",
        "--set",
        "electricity_price=fluctuating",
        "--set",
        "hydrogen_demand=fluctuating",
    ],
]

RESOURCES = [
    "rss_peak_os_mb",
    "wall_total_s",
    "gurobi_runtime_s",
    "parallelism_solve",
]

# --- Stage 5, the thread ladder -------------------------------------------
#
# The 4 thread sweep showed that a restricted solver is not slower and on the
# heavy MIPs is faster, and that the whole effect sits in the cost of a single
# simplex iteration rather than in the number of iterations. Three things could
# produce that signature and the paired sweep cannot tell them apart: memory
# bandwidth, per-iteration synchronisation, and cores spread across sockets.
#
# Running the same model along a ladder of thread counts separates the first
# two from the third. A cost per iteration that climbs smoothly with the thread
# count is bandwidth or synchronisation; a step where the ladder crosses a
# socket boundary is the memory topology. The pinned runs below settle it: same
# number of threads, but confined to cores that sit together.
#
# Configurations are named the way the case name builds them, so
# four_node_td32_bp1 is typicaldays 32 with bidirectional_precise on and every
# other knob at its default.
THREAD_LADDER_CONFIGS = [
    {
        # ~180 s at 48 threads and x0.35 at 4, the strongest effect among the
        # cheap runs, so it can afford the full ladder
        "typicaldays": 4,
        "knobs": {
            "bidirectional_precise": 1,
            "electricity_price": "fluctuating",
            "pipeline_capex": "fixed_plus_linear",
            "pipeline_size_min": 250,
        },
        "threads": [1, 2, 4, 8, 12, 16, 24, 32, 48],
    },
    {
        # ~280 s at 48 threads and x1.05 at 4, a configuration the thread count
        # did not move. It is the control: whatever the ladder shows here is
        # not the effect being chased
        "typicaldays": 16,
        "knobs": {"bidirectional_precise": 1},
        "threads": [1, 2, 4, 8, 12, 16, 24, 32, 48],
    },
    {
        # 3050 s at 48 threads against 732 s at 4, with the search path
        # provably unchanged. The run the whole question comes from, and the
        # expensive one, so it gets a coarser ladder
        "typicaldays": 32,
        "knobs": {"bidirectional_precise": 1},
        "threads": [1, 4, 16, 48],
    },
]

# Thread counts that are also run pinned to an equal number of cores. Kept to
# the two cheaper configurations, and to counts high enough that an unpinned
# run has a reason to spread across sockets

# Stage 6. Does any combination of a gurobi option and a thread count beat the
# default? An existence question, not a tuning exercise, so the design is a
# full factorial with the thread count crossed in: a one-factor-at-a-time
# screen excludes interactions by construction and cannot answer it.
#
# The responses are runtime, peak memory and cpu seconds. A combination that is
# no faster but wants half the memory is a win, because memory is what a
# cluster job has to ask for.
#
# threads = 1 is not decoration. It is the only level where barrier and
# concurrent cannot win through parallelism, so an option that still helps
# there is helping algorithmically, and the gain is not the threads'.

# Stage 8. The smoke test of 2026-09-11 on td4 showed where the time of this
# model family actually goes: presolve 0.9 s, the root relaxation 19 s, and
# then a hundred seconds of cut rounds at the root node, with the tree closing
# after eighteen nodes. So roughly 80 % of the solve is the cut loop at the
# root, not the root LP and not the search.
#
# Two things follow. The options that steer the cut loop have far more time in
# scope than Method has, and an almost empty tree has nothing to parallelise,
# which is the likely reason the solver was only ever seen using 2.1 cores of
# 48. This stage crosses the cut loop options with the thread count.
#
# lpwarmstart is in here because the adopt template forces it to 0 while the
# gurobi default is -1, and a model that re-solves its root LP once per cut
# round is exactly where that costs something.
GUROBI_CUT_LEVELS = [0, -1, 3]
GUROBI_MIPFOCUS_LEVELS = [0, 3]
GUROBI_LPWARMSTART_LEVELS = [0, -1]
GUROBI_CUT_THREAD_LEVELS = [4, 48]

GUROBI_THREAD_LEVELS = [1, 4, 16, 48]

# -1 auto, 1 dual simplex (serial), 2 barrier (parallel), 3 concurrent
GUROBI_METHOD_LEVELS = [-1, 1, 2, 3]

# Two configurations per stage, one cheap and one that carries the binaries.
# Sized from the measured wall times: td4 is 59 s, td16_bp1 339 s, td4_st0 34 s,
# td16 271 s. td0 is deliberately absent, at 7500 s a run a sixteen cell
# factorial over it is 33 hours on its own
GUROBI_CONFIGS = {
    "a1": [
        {"typicaldays": 4, "knobs": {}},
        {"typicaldays": 16, "knobs": {"bidirectional_precise": 1}},
    ],
    "a2": [
        {"typicaldays": 4, "knobs": {"storage": "off"}},
        {"typicaldays": 16, "knobs": {}},
    ],
}

THREAD_LADDER_PINNED = [16, 24]
THREAD_LADDER_PINNED_CONFIGS = [0, 1]


def log(message: str):
    """
    Prints a message and appends it to the study log

    :param str message: message to write
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(line + "\n")


def write_machine_info():
    """
    Writes what the study ran on, so that timings can be compared later

    :return: dict with the machine information
    """
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpus_physical": psutil.cpu_count(logical=False),
        "cpus_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    try:
        import gurobipy

        info["gurobi"] = ".".join(str(part) for part in gurobipy.gurobi.version())
    except Exception as error:
        info["gurobi"] = f"unavailable ({error})"

    with open(MACHINE_FILE, "w", encoding="utf-8") as file:
        json.dump(info, file, indent=2)

    log(
        f"Machine: {info['hostname']}, {info['cpus_physical']} physical cores, "
        f"{info['memory_total_gb']} GB, gurobi {info['gurobi']}"
    )
    return info


def run(command: list, dry_run: bool = False, produced: list = None):
    """
    Runs one command of the study and reports how long it took

    :param list command: command line to run
    :param bool dry_run: if True, the command is only printed
    :param list produced: if given, the name of the result folder the command
        created is appended to it. A run writes exactly one folder, so the
        difference before and after names it without having to parse anything
    :return: True if the command succeeded
    """
    printable = " ".join(str(part) for part in command)
    if dry_run:
        print(f"  would run: {printable}")
        return True

    before = _result_folders()

    log(f"START {printable}")
    start = time.time()
    result = subprocess.run(command, cwd=str(BASE))
    duration = time.time() - start

    if produced is not None:
        produced.extend(sorted(_result_folders() - before))

    if result.returncode == 0:
        log(f"DONE  in {duration / 60:.1f} min")
        return True

    log(f"FAILED with exit code {result.returncode} after {duration / 60:.1f} min")
    return False


def _result_folders():
    """
    Names of the result folders at the top level

    :return: set of folder names
    """
    if not RESULTS_PATH.is_dir():
        return set()
    return {folder.name for folder in RESULTS_PATH.iterdir() if folder.is_dir()}


def _cell_settings(options: dict, threads: int):
    """
    Settings of one cell of a factorial.

    The stage and the manifest both go through here, so that the case name a
    run is given and the case name the manifest looks for cannot drift apart.

    :param dict options: solver options of the cell
    :param int threads: number of threads
    :return: dict of settings
    """
    return {
        "mipgap": 0.02,
        "time_limit": TIME_LIMIT,
        "threads": threads,
        "affinity_cores": 0,
        "solver": "gurobi",
        "sampling_interval": 0.5,
        **options,
    }


# Which cells belong to which stage, so a manifest can be written for a stage
# that has already run as well as for one that is running now
STAGE_CELLS = {
    6: lambda: _gurobi_thread_runs("a1"),
    7: lambda: _gurobi_thread_runs("a2"),
    8: lambda: _gurobi_cut_runs("a1"),
    9: lambda: _gurobi_cut_runs("a2"),
}


def _folders_of(case_name: str):
    """
    Finished result folders of a case name

    :param str case_name: case name to look for
    :return: sorted list of folder names, oldest first
    """
    found = []
    for folder in RESULTS_PATH.glob(f"*_{case_name}*"):
        without_timestamp = folder.name.split("_", 1)[-1]
        without_counter = re.sub(r"-\d+$", "", without_timestamp)
        if without_counter != case_name:
            continue
        if (folder / "profile_summary.csv").exists():
            found.append(folder.name)
    return sorted(found)


def write_manifest(stage: int):
    """
    Writes the list of result folders that belong to a stage.

    Built from the cells of the stage rather than from what happened to be
    created while it ran, so it can be written for a stage that finished
    earlier, and so it survives a stage that was interrupted and resumed.

    A cell shared with another stage, such as the all default cell, is listed
    by both. That is correct: it is one run, and both stages use it.

    :param int stage: stage number, 6 to 9
    :return: Path of the manifest, or None if the stage has no cells
    """
    from run_benchmark import _build_case_name

    if stage not in STAGE_CELLS:
        return None

    lines = []
    missing = 0
    for typicaldays, knobs, options, threads in STAGE_CELLS[stage]():
        settings = _cell_settings(options, threads)
        case_name = _build_case_name(
            CASE, {"typicaldays": typicaldays, **knobs, **settings}
        )
        folders = _folders_of(case_name)
        if not folders:
            missing += 1
            continue
        # The newest, in case a cell was run more than once
        lines.append(folders[-1])

    manifest = RESULTS_PATH / f"manifest_stage{stage}.txt"
    manifest.write_text("\n".join(sorted(set(lines))) + "\n")
    log(
        f"[MANIFEST] stage {stage}: {len(set(lines))} folders written to "
        f"{manifest.name}" + (f", {missing} cells not on disk" if missing else "")
    )
    return manifest


def stage_sizes(dry_run: bool = False):
    """
    Measures how much each knob adds to the size, without solving anything

    :param bool dry_run: if True, the commands are only printed
    :return: True if the stage succeeded
    """
    log("=== Stage 0: size of every complexity knob ===")
    ok = True
    for typicaldays in [2, 4]:
        ok &= run(
            [
                sys.executable,
                "measure_complexity.py",
                "--case",
                CASE,
                "--typicaldays",
                str(typicaldays),
            ],
            dry_run,
        )
    return ok


def stage_matrix(dry_run: bool = False):
    """
    Runs every combination of the complexity knobs at small sizes

    :param bool dry_run: if True, the commands are only printed
    :return: True if the stage succeeded
    """
    combinations = 2 ** len(MATRIX_KNOBS)
    log(
        f"=== Stage 1: matrix, {combinations} combinations x "
        f"{len(MATRIX_TYPICALDAYS)} typical days = "
        f"{combinations * len(MATRIX_TYPICALDAYS)} runs ==="
    )
    return run(
        [
            sys.executable,
            "run_benchmark.py",
            "matrix",
            "--case",
            CASE,
            "--typicaldays",
            *[str(value) for value in MATRIX_TYPICALDAYS],
            "--knobs",
            *MATRIX_KNOBS,
            "--time-limit",
            str(TIME_LIMIT),
            "--threads",
            str(THREADS),
        ],
        dry_run,
    )


def stage_scaling(dry_run: bool = False):
    """
    Runs a reduced set of knobs across several model sizes

    :param bool dry_run: if True, the commands are only printed
    :return: True if the stage succeeded
    """
    combinations = 2 ** len(SCALING_KNOBS)
    log(
        f"=== Stage 2: scaling, {combinations} combinations x "
        f"{len(SCALING_TYPICALDAYS)} typical days = "
        f"{combinations * len(SCALING_TYPICALDAYS)} runs ==="
    )
    return run(
        [
            sys.executable,
            "run_benchmark.py",
            "matrix",
            "--case",
            CASE,
            "--typicaldays",
            *[str(value) for value in SCALING_TYPICALDAYS],
            "--knobs",
            *SCALING_KNOBS,
            "--time-limit",
            str(TIME_LIMIT),
            "--threads",
            str(THREADS),
        ],
        dry_run,
    )


def stage_full_resolution(dry_run: bool = False):
    """
    Runs a few configurations at full time resolution

    :param bool dry_run: if True, the commands are only printed
    :return: True if every run succeeded
    """
    log(
        f"=== Stage 3: full resolution, {len(FULL_RESOLUTION_RUNS)} runs, "
        f"about 3 million variables each ==="
    )
    ok = True
    for extra in FULL_RESOLUTION_RUNS:
        ok &= run(
            [
                sys.executable,
                "run_benchmark.py",
                "run",
                "--case",
                CASE,
                "--typicaldays",
                "0",
                "--time-limit",
                str(TIME_LIMIT),
                "--threads",
                str(THREADS),
                *extra,
            ],
            dry_run,
        )
    return ok


def _thread_ladder_runs():
    """
    Builds the list of runs of the thread ladder, cheapest configuration first

    :return: list of (typicaldays, knobs, threads, affinity_cores) tuples
    """
    runs = []

    for index, config in enumerate(THREAD_LADDER_CONFIGS):
        for threads in config["threads"]:
            runs.append((config["typicaldays"], config["knobs"], threads, 0))

        if index not in THREAD_LADDER_PINNED_CONFIGS:
            continue

        # The pinned run only means something next to the free run at the same
        # thread count, so it is never asked for on its own
        for threads in THREAD_LADDER_PINNED:
            if threads in config["threads"]:
                runs.append((config["typicaldays"], config["knobs"], threads, threads))

    return runs


def stage_thread_ladder(dry_run: bool = False):
    """
    Runs a few configurations along a ladder of thread counts.

    Every run is skipped if it is already on disk, so the stage can be
    interrupted and started again. A failing run does not stop the ladder.

    :param bool dry_run: if True, the commands are only printed
    :return: True if every run that was attempted succeeded
    """
    from run_benchmark import already_done

    runs = _thread_ladder_runs()
    log(
        f"=== Stage 5: thread ladder, {len(THREAD_LADDER_CONFIGS)} configurations, "
        f"{len(runs)} runs ==="
    )

    ok = True
    skipped = 0

    for number, (typicaldays, knobs, threads, affinity) in enumerate(runs, start=1):
        settings = {
            "mipgap": 0.02,
            "time_limit": TIME_LIMIT,
            "threads": threads,
            "affinity_cores": affinity,
            "solver": "gurobi",
            "sampling_interval": 0.5,
        }

        if not dry_run and already_done(CASE, typicaldays, knobs, settings):
            skipped += 1
            log(
                f"[LADDER] run {number}/{len(runs)}: td{typicaldays}, "
                f"{threads} threads, already done, skipped"
            )
            continue

        command = [
            sys.executable,
            "run_benchmark.py",
            "run",
            "--case",
            CASE,
            "--typicaldays",
            str(typicaldays),
            "--time-limit",
            str(TIME_LIMIT),
            "--threads",
            str(threads),
        ]
        if affinity:
            command += ["--affinity-cores", str(affinity)]
        for knob, value in knobs.items():
            command += ["--set", f"{knob}={value}"]

        log(f"[LADDER] run {number}/{len(runs)}: td{typicaldays}, {threads} threads")
        ok &= run(command, dry_run)

    if not dry_run:
        log(f"[LADDER] {skipped} of {len(runs)} runs were already done")
        run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    return ok


def _gurobi_thread_runs(part: str):
    """
    Builds the cells of the gurobi option by thread count factorial

    Cheapest configuration first, so an interrupted stage still leaves a
    complete block behind.

    :param str part: which half of the design, "a1" or "a2"
    :return: list of (typicaldays, knobs, options, threads) tuples
    """
    runs = []
    for config in GUROBI_CONFIGS[part]:
        for method in GUROBI_METHOD_LEVELS:
            for threads in GUROBI_THREAD_LEVELS:
                runs.append(
                    (
                        config["typicaldays"],
                        config["knobs"],
                        {"method": method},
                        threads,
                    )
                )
    return runs


def stage_gurobi_threads(dry_run: bool = False, part: str = "a1"):
    """
    Crosses the root relaxation algorithm with the thread count.

    Every run is skipped if it is already on disk, so the stage can be
    interrupted and started again. A failing run does not stop the factorial.

    :param bool dry_run: if True, the commands are only printed
    :param str part: which half of the design, "a1" or "a2"
    :return: True if every run that was attempted succeeded
    """
    from run_benchmark import already_done

    runs = _gurobi_thread_runs(part)
    log(
        f"=== Stage 6{part}: gurobi options by threads, "
        f"{len(GUROBI_CONFIGS[part])} configurations, {len(runs)} runs ==="
    )

    ok = True
    skipped = 0

    produced = []

    for number, (typicaldays, knobs, options, threads) in enumerate(runs, start=1):
        settings = _cell_settings(options, threads)
        method = options["method"]

        if not dry_run and already_done(CASE, typicaldays, knobs, settings):
            skipped += 1
            log(
                f"[GUROBI] run {number}/{len(runs)}: td{typicaldays}, "
                f"method {method}, {threads} threads, already done, skipped"
            )
            continue

        command = [
            sys.executable,
            "run_benchmark.py",
            "run",
            "--case",
            CASE,
            "--typicaldays",
            str(typicaldays),
            "--time-limit",
            str(TIME_LIMIT),
            "--threads",
            str(threads),
            "--set",
            f"method={method}",
        ]
        for knob, value in knobs.items():
            command += ["--set", f"{knob}={value}"]

        log(
            f"[GUROBI] run {number}/{len(runs)}: td{typicaldays}, "
            f"method {method}, {threads} threads"
        )
        ok &= run(command, dry_run, produced)

    if not dry_run:
        log(f"[GUROBI] {skipped} of {len(runs)} runs were already done")
        write_manifest(6 if part == "a1" else 7)
        run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    return ok


def _gurobi_cut_runs(part: str):
    """
    Builds the cells of the cut loop by thread count factorial

    :param str part: which set of configurations, "a1" or "a2"
    :return: list of (typicaldays, knobs, options, threads) tuples
    """
    runs = []
    for config in GUROBI_CONFIGS[part]:
        for cuts in GUROBI_CUT_LEVELS:
            for mipfocus in GUROBI_MIPFOCUS_LEVELS:
                for lpwarmstart in GUROBI_LPWARMSTART_LEVELS:
                    for threads in GUROBI_CUT_THREAD_LEVELS:
                        runs.append(
                            (
                                config["typicaldays"],
                                config["knobs"],
                                {
                                    "cuts": cuts,
                                    "mipfocus": mipfocus,
                                    "lpwarmstart": lpwarmstart,
                                },
                                threads,
                            )
                        )
    return runs


def stage_gurobi_cuts(dry_run: bool = False, part: str = "a1"):
    """
    Crosses the options that steer the root cut loop with the thread count.

    Every run is skipped if it is already on disk, so the stage can be
    interrupted and started again. A failing run does not stop the factorial.

    A cell at cuts = 0 can be much slower than the rest, since the cut loop is
    what closes this model, and can reach the time limit. That is not a failure
    and the run still carries a valid resource measurement, with
    termination_condition saying what happened.

    :param bool dry_run: if True, the commands are only printed
    :param str part: which set of configurations, "a1" or "a2"
    :return: True if every run that was attempted succeeded
    """
    from run_benchmark import already_done

    runs = _gurobi_cut_runs(part)
    log(
        f"=== Stage 8{part}: root cut loop by threads, "
        f"{len(GUROBI_CONFIGS[part])} configurations, {len(runs)} runs ==="
    )

    ok = True
    skipped = 0
    produced = []

    for number, (typicaldays, knobs, options, threads) in enumerate(runs, start=1):
        settings = _cell_settings(options, threads)
        spelled = ", ".join(f"{name} {value}" for name, value in options.items())

        if not dry_run and already_done(CASE, typicaldays, knobs, settings):
            skipped += 1
            log(
                f"[CUTS] run {number}/{len(runs)}: td{typicaldays}, {spelled}, "
                f"{threads} threads, already done, skipped"
            )
            continue

        command = [
            sys.executable,
            "run_benchmark.py",
            "run",
            "--case",
            CASE,
            "--typicaldays",
            str(typicaldays),
            "--time-limit",
            str(TIME_LIMIT),
            "--threads",
            str(threads),
        ]
        for name, value in options.items():
            command += ["--set", f"{name}={value}"]
        for knob, value in knobs.items():
            command += ["--set", f"{knob}={value}"]

        log(
            f"[CUTS] run {number}/{len(runs)}: td{typicaldays}, {spelled}, "
            f"{threads} threads"
        )
        ok &= run(command, dry_run, produced)

    if not dry_run:
        log(f"[CUTS] {skipped} of {len(runs)} runs were already done")
        write_manifest(8 if part == "a1" else 9)
        run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    return ok


def stage_report(dry_run: bool = False):
    """
    Collects the runs, draws the figures and writes the summary

    :param bool dry_run: if True, the commands are only printed
    :return: True if the stage succeeded
    """
    log("=== Stage 4: dataset, figures and report ===")

    ok = run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    for case in [CASE, "network"]:
        run(
            [
                sys.executable,
                "plot_benchmark.py",
                "--case",
                case,
                "--resources",
                *RESOURCES,
            ],
            dry_run,
        )

    if not dry_run:
        write_report()

    return ok


def write_report():
    """
    Writes a summary of the study to STUDY_REPORT.md
    """
    import pandas as pd

    if not DATASET_FILE.exists():
        log("No dataset to report on")
        return

    dataset = pd.read_csv(DATASET_FILE)
    case_runs = dataset[dataset["case_name"].str.startswith(CASE)].copy()

    machine = {}
    if MACHINE_FILE.exists():
        machine = json.loads(MACHINE_FILE.read_text(encoding="utf-8"))

    lines = [
        "# Benchmark study",
        "",
        f"Written {datetime.datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "## Machine",
        "",
    ]
    for key, value in machine.items():
        lines.append(f"- {key}: {value}")

    lines += [
        "",
        "## Runs",
        "",
        f"- {len(dataset)} runs in total, {len(case_runs)} of `{CASE}`",
    ]

    if not case_runs.empty:
        lines.append(
            f"- model size from {case_runs['n_vars'].min():,.0f} to "
            f"{case_runs['n_vars'].max():,.0f} variables"
        )
        lines.append(
            f"- wall time from {case_runs['wall_total_s'].min():.1f} s to "
            f"{case_runs['wall_total_s'].max():.1f} s"
        )
        lines.append(
            f"- peak memory from {case_runs['rss_peak_os_mb'].min():.0f} MB to "
            f"{case_runs['rss_peak_os_mb'].max():.0f} MB"
        )

        if "termination_condition" in case_runs.columns:
            counts = case_runs["termination_condition"].value_counts()
            lines += ["", "### Termination", ""]
            for condition, count in counts.items():
                lines.append(f"- {condition}: {count}")

        lines += _difficulty_section(case_runs)
        lines += _slowest_section(case_runs)

    lines += ["", "## Figures", ""]
    for figure in sorted(FIGURES_PATH.glob("*.png")):
        lines.append(f"- `figures/{figure.name}`")

    lines += [
        "",
        "## Files",
        "",
        "- `benchmark_dataset.csv`: one row per run, the dataset of the study",
        "- `results/<run>/profile_timeseries.csv`: the full resource curve",
        "- `results/<run>/profile_phases.csv`: average and peak per phase",
        "- `study.log`: what ran and how long it took",
        "",
    ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    log(f"Report written to {REPORT_FILE}")


def _difficulty_section(case_runs):
    """
    Builds the section on resources against difficulty at a comparable size

    :param case_runs: rows of the dataset belonging to the case study
    :return: list of markdown lines
    """
    import numpy as np

    lines = []
    largest = case_runs["n_vars"].max()
    subset = case_runs[case_runs["n_vars"] >= 0.8 * largest]

    if len(subset) < 4:
        return lines

    lines += [
        "",
        "### Do harder runs use more resources?",
        "",
        f"{len(subset)} runs between {subset['n_vars'].min():,.0f} and "
        f"{subset['n_vars'].max():,.0f} variables, so size is held roughly "
        "constant and what is left is the difficulty.",
        "",
        "| resource | vs gurobi runtime | vs simplex iterations | vs B&B nodes |",
        "|---|---|---|---|",
    ]

    difficulties = ["gurobi_runtime_s", "gurobi_itercount", "gurobi_nodecount"]
    resources = ["cpu_user_solve_s", "rss_peak_solve_mb", "parallelism_solve"]

    for resource in resources:
        if resource not in subset.columns:
            continue
        cells = []
        for difficulty in difficulties:
            if difficulty not in subset.columns:
                cells.append("n/a")
                continue
            x = subset[difficulty].to_numpy(dtype=float)
            y = subset[resource].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() < 3 or np.std(x[finite]) == 0:
                cells.append("n/a")
            else:
                cells.append(f"{np.corrcoef(x[finite], y[finite])[0, 1]:+.3f}")
        lines.append(f"| `{resource}` | " + " | ".join(cells) + " |")

    return lines


def _slowest_section(case_runs):
    """
    Builds the section listing the slowest and the heaviest runs

    :param case_runs: rows of the dataset belonging to the case study
    :return: list of markdown lines
    """
    lines = [
        "",
        "### Slowest runs",
        "",
        "| run | variables | binaries | gurobi [s] | B&B nodes | peak memory [MB] |",
        "|---|---|---|---|---|---|",
    ]

    columns = [
        "n_vars",
        "n_binvars",
        "gurobi_runtime_s",
        "gurobi_nodecount",
        "rss_peak_os_mb",
    ]
    if any(column not in case_runs.columns for column in columns):
        return []

    slowest = case_runs.nlargest(10, "gurobi_runtime_s")
    for _, run_row in slowest.iterrows():
        lines.append(
            f"| `{run_row['case_name']}` "
            f"| {run_row['n_vars']:,.0f} "
            f"| {run_row['n_binvars']:,.0f} "
            f"| {run_row['gurobi_runtime_s']:.1f} "
            f"| {run_row['gurobi_nodecount']:.0f} "
            f"| {run_row['rss_peak_os_mb']:.0f} |"
        )

    return lines


STAGES = {
    0: stage_sizes,
    1: stage_matrix,
    2: stage_scaling,
    3: stage_full_resolution,
    4: stage_report,
    5: stage_thread_ladder,
    6: lambda dry_run=False: stage_gurobi_threads(dry_run, part="a1"),
    7: lambda dry_run=False: stage_gurobi_threads(dry_run, part="a2"),
    8: lambda dry_run=False: stage_gurobi_cuts(dry_run, part="a1"),
    9: lambda dry_run=False: stage_gurobi_cuts(dry_run, part="a2"),
}

# The thread ladder answers a question about the machine rather than about the
# case study, and it re-runs configurations the study already covers, so asking
# for the study does not ask for it
DEFAULT_STAGES = [0, 1, 2, 3, 4]


def main():
    """
    Command line interface of the study
    """
    global THREADS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        choices=sorted(STAGES),
        default=DEFAULT_STAGES,
        help="stages to run, default every stage of the study proper. Stage 5, "
        "the thread ladder, has to be asked for by name",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=THREADS,
        help="threads the solver may use, 0 for every core of the machine. "
        "Runs at different thread counts live side by side in the results "
        "folder, as the thread count is part of the case name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would run without running it",
    )
    parser.add_argument(
        "--manifest",
        type=int,
        nargs="+",
        choices=sorted(STAGE_CELLS),
        help="write the list of result folders belonging to these stages and "
        "exit, without running anything. Built from the cells of the stage, so "
        "it also works for a stage that finished earlier",
    )
    args = parser.parse_args()

    if args.manifest:
        for stage in args.manifest:
            write_manifest(stage)
        return
    THREADS = args.threads

    started = time.time()
    if not args.dry_run:
        log("")
        log("#" * 70)
        log("Benchmark study starting")
        log(f"Solver threads: {THREADS if THREADS else 'all cores'}")
        write_machine_info()

    for stage in args.stages:
        try:
            STAGES[stage](args.dry_run)
        except Exception as error:
            log(f"Stage {stage} raised {type(error).__name__}: {error}")
            import traceback

            traceback.print_exc()

    if not args.dry_run:
        log(f"Study finished in {(time.time() - started) / 3600:.2f} hours")
        log(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
