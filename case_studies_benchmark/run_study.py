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
10. threads   the thread counts between the levels of stages 6 to 9. Runtime
              is bought between 1 and 4 threads and memory triples between 4
              and 16, and neither edge is located. The level that comes out of
              this stage is the one a cluster job asks for
11. cutpasses how many rounds of the root cut loop are worth running, against
              turning the cuts off altogether. Stages 8 and 9 found cuts = 0
              worth a factor two, and the node logs say why: the first pass
              carries the bound and the rest are nearly free of it
12. nl cuts   whether cuts = 0 is still the faster arm on the nine node case
              study. Everything the study recommends was measured on four_node
              at 2 %, where difficulty comes from the resolution and an arm
              could buy time by stopping at a looser bound. The cheap half of
              that case study, two resolutions, and it runs first
13. nl threads the thread ladder of the same case study, at the cut setting
              stage 12 favoured. Answers on a topology driven model what stage
              10 answered on four_node: how many cores a run should ask for
14. contention a fixed core budget filled with jobs of 1, 2, 3, 4 and 6
              threads, run concurrently. Every other stage measures runs that
              had the machine to themselves, so the throughput argument for
              thin jobs is arithmetic. This measures it
15. workers   how full to fill the machine, at one and at two threads a job:
              half the cores, three quarters, all of them and a third more
              than all of them. Stage 14 always spends the whole budget, this
              asks whether it should, since the jobs share one memory system
              rather than oversubscribing the cores
16. repeat    the one thread packs of stage 15 where the curve falls, run a
              second time. Two packs that stage 14 and 15 both ran differ by
              up to a quarter, so a single pack cannot place a cliff
17. workers omp the packs of stage 15 again, with the numerical libraries of
              every job held to one thread. Stage 15 ran with them free, and
              the typical day clustering of every job started about 100
              threads on 48 cores, which is what made the reading explode

Every job is launched with OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and
MKL_NUM_THREADS set to --omp-threads, 1 by default. Stages up to 16 ran
without them, as --omp-threads 0 still does.

Examples::

    python run_study.py                     # the whole study
    python run_study.py --stages 0 1        # only the sizes and the matrix
    python run_study.py --stages 4          # only redo the figures and report
    python run_study.py --stages 5          # only the thread ladder
    python run_study.py --stages 6          # gurobi options crossed with threads
    python run_study.py --stages 6 --dry-run
    python run_study.py --stages 10 11      # the follow-up of 2026-09-21
    python run_study.py --stages 12         # the nine node case study, cuts
    python run_study.py --stages 13         # the same case study, threads
    python run_study.py --manifest 13       # its manifest, after the fact
    python run_study.py --stages 14         # what a full machine delivers
    python run_study.py --stages 15         # how full to fill it
    python run_study.py --stages 16         # the falling part of it, again
    python run_study.py --stages 17         # stage 15 with the libraries at 1
"""

import argparse
import datetime
import re
import json
import platform
import shutil
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

# The optimality gap every cell is solved to. Owned by run_benchmark so that a
# run launched by hand and a run launched by a stage are solved to the same
# gap. 0.5 % since 2026-09-21, see the comment there for why 2 % was dropped
from run_benchmark import MIPGAP

# Hours. A run that hits this limit still gives a valid resource measurement,
# it just does not reach the optimum, and the sweep carries on. It reaches the
# case name, so raising it makes a cell a different run rather than the same
# one measured for longer, and already_done will not match the old cells.
# Overridden from the command line with --time-limit, which is how the nine
# node stages get their four hours without the four_node archive moving
TIME_LIMIT = 2

# Threads the solver may use. 0 is every core of the machine, which is what the
# first sweep ran with. Setting it makes every run look like a cluster job that
# asked for that many cores, and it reaches the case name, so a sweep at a
# different thread count does not skip the runs of the first one as already
# done. Overridden from the command line with --threads
THREADS = 0

# Threads the numerical libraries of the reading may use: sklearn's KMeans in
# the typical day clustering (OpenMP) and numpy's BLAS. Left alone they size
# their pools to every core of the machine, whatever gurobi is given, and on
# 2026-09-24 a job alone on the 48 core server started about 100 threads and
# burned 540 cpu seconds on a 54 s reading. 48 jobs doing that at once took
# 273 s each to read, against 43 s at one thread, and one thread was faster
# even alone. 1 by default; 0 leaves the libraries free, which is how every
# stage up to 16 ran. Overridden from the command line with --omp-threads
OMP_THREADS = 1

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

# Stage 10. The thread recommendation rests on four levels, 1, 4, 16 and 48,
# and both of the things worth knowing sit between them. Runtime is bought
# between 1 and 4, td4 244 s to 182 s and td16 1731 s to 1224 s, and nothing
# is bought above 4. Memory triples between 4 and 16, td16 1638 MB to 4476 MB.
# Neither edge is located, and the level that comes out of this stage is what
# a cluster job asks for.
THREAD_RESOLUTION_LEVELS = [2, 3, 6, 8, 12]

# Stage 11. Read out of the node log of every default cell of stages 6 to 9:
# the first cut pass moves the bound by 0.33 to 0.39 %, the twenty-five to
# fifty-seven passes after it by 0.03 to 0.23 %, and those cost a quarter to a
# half of the whole solve. That is gurobi's own stated trigger for CutPasses.
#
# It should beat cuts = 0, which wins the same time by skipping every pass
# including the one that pays, and therefore stops at a looser gap, 1.42 %
# against 0.97 % inside the same 2 % tolerance.
GUROBI_CUTPASSES_LEVELS = [1, 2, 5, -1]

# The arm cuts = 0 is compared against, run inside this stage rather than read
# out of stages 8 and 9. Those runs are archived, which is to say invisible to
# already_done, and a baseline measured in the same campaign is worth the two
# cells it costs
GUROBI_CUTPASSES_REFERENCES = [{"cuts": 0}]

GUROBI_CUTPASSES_THREAD_LEVELS = [4, 48]

# Stage 12, the nine node case study. Every number the study has is a
# four_node number, from a model whose difficulty comes from its resolution:
# the median tree is one node and the deepest ever seen is 248. nl_node puts
# 84 candidate arcs on a real corridor map, so its binaries scale with the
# topology instead. The stage asks the two decisions of stages 6 to 11 again
# there, at 0.5 % where the solver can no longer buy time by stopping early:
# does one thread per run still win on core seconds, and is cuts = 0 still the
# fastest arm
NL_CASE = "nl_node"

# The hard configuration of that case: a binary per arc from the minimum size,
# a binary per arc and timestep from the precise bidirectionality, and a big M
# on the arc capex
NL_KNOBS = {
    "bidirectional_precise": 1,
    "pipeline_size_min": 250,
    "pipeline_capex": "fixed_plus_linear",
}

# Stage 12, the cut question, which runs first and is the cheap one. cuts = 0
# was worth x0.43 on four_node at 2 %, and stage 11 then showed that part of
# that was a 1.2 % worse solution rather than a faster one. At 0.5 % that
# escape is closed. Two resolutions and two thread counts are enough to say
# whether it is still a win here
NL_CUT_TYPICALDAYS = [4, 15]
NL_CUT_ARMS = [{}, {"cuts": 0}]
NL_CUT_THREAD_LEVELS = [1, 4]

# Stage 13, the thread question, launched once stage 12 has been read. It is
# the expensive one, so it runs a single cut setting rather than both
NL_THREAD_TYPICALDAYS = [4, 15, 30]

# 1 against 2 is the comparison stage 10 could not separate, x1.41 on a single
# run against a noise floor of about x1.3. 4 is the old recommendation, 16 and
# 48 give the core second curve its upper end
# 3 and 6 are in because stage 10 put the memory cliff exactly between them,
# a factor 2.7 in one step from 4 to 6 threads, and because a 48 core budget
# packs into 16 jobs of 3 or 8 of 6. Those are candidate answers, so they need
# a measured core second cost rather than an interpolated one
NL_THREAD_LEVELS = [1, 2, 3, 4, 6, 16, 48]

# td15 drops the two levels that are already known to be bad and expensive,
# keeping the small counts the packing decision is actually between
NL_THREAD_LEVELS_MEDIUM = [1, 2, 3, 4, 6]

# td30 is the expensive end. td15 at 1 thread took 2.84 h, so a td30 cell at 1
# thread would sit on the time limit: the ladder there starts at 2
NL_THREAD_LEVELS_LARGE = [2, 4]

# Hours, per resolution, for the nine node stages. Every one of them is
# spelled out rather than falling back to TIME_LIMIT, because the limit is
# part of the case name: a cell looked up at a different limit from the one it
# ran at is a different run, which silently re-runs it and writes an empty
# manifest. Stages 12 and 13 ran td4 and td15 at four hours, so those stay at
# four. td30 gets eight: td15 took 1.3 h at four threads and 2.9 h at one, and
# four hours would censor exactly the cells the stage is for
NL_TIME_LIMITS = {4: 4, 15: 4, 30: 8}


def _nl_time_limit(typicaldays: int):
    """
    Hours a nine node cell of this resolution may take.

    :param int typicaldays: number of typical days of the cell
    :return: time limit in hours
    """
    if typicaldays not in NL_TIME_LIMITS:
        raise KeyError(
            f"No time limit for td{typicaldays} in NL_TIME_LIMITS. Add one "
            "rather than falling back, the limit is part of the case name"
        )
    return NL_TIME_LIMITS[typicaldays]

# Stage 14, the packing question, and the one thing the core second
# arithmetic cannot answer. Every run of this study so far had the machine to
# itself, so "one thread per job wins on core seconds" is arithmetic over solo
# runs: it assumes N jobs do not get in each other's way. This fills a fixed
# core budget with jobs of k threads, 48 of 1, 24 of 2, 16 of 3, 12 of 4 and 8
# of 6, and measures the throughput each packing actually delivers
CONTENTION_CORE_BUDGET = 48
CONTENTION_THREAD_LEVELS = [1, 2, 3, 4, 6]

# The cheap resolution, since the arms are what is being compared and a pack
# has to be run several times over. td4 solo is 6 to 8 minutes a run
CONTENTION_TYPICALDAYS = 4

# Stage 15, how full to fill the machine. Stage 14 asks how to shape the jobs
# at a budget that is always spent in full; this asks whether spending it in
# full is right at all.
#
# Fractions of the core budget rather than worker counts, so that the two
# thread counts are compared at the same amount of machine: 0.5 leaves half
# the cores idle, 1.0 fills them exactly and 1.33 oversubscribes by a third.
# Oversubscription is in on purpose, because a run is not solver all the way
# through: stage 14 measured the reading of the input and the building of the
# model slowing by x44 and x9 in a full pack, and those wait rather than
# compute, so they interleave.
#
# Both arms get a fine grid, but not at the same place, because the two
# curves are not expected to peak at the same fill. At two threads stage 14
# measured 72.1 runs an hour on a full machine, so the maximum is at or near
# full and the grid is refined between three quarters and full. At one thread
# the same machine gave 46.0, the worst of the five packings, so that curve is
# already falling at full and its maximum should sit lower: the grid is
# refined between half and three quarters instead. Putting the extra points
# above three quarters there would only measure the descent in detail
CONTENTION_FILL_LEVELS = {
    2: [0.5, 0.75, 0.83, 0.92, 1.0, 1.08, 1.33],
    1: [0.5, 0.625, 0.75, 0.875, 0.9375, 1.0, 1.33],
}

# Threads a job asks for, cheapest arm first. Two comes first because stage
# 14 chose it: 24 jobs of two threads delivered 72.1 runs an hour against
# 46.0 for 48 jobs of one, although the arithmetic over solo runs had one
# thread ahead by x1.8. One is still swept, to find out whether it loses at
# every fill or only at a full machine
CONTENTION_WORKER_THREADS = [2, 1]

# Stage 16, the one thread packs of stage 15 run a second time. Stage 15 put a
# drop from 72.3 to 41.5 runs an hour between 30 and 36 jobs, and a recovery
# to 57.3 at 45, but the same pack run twice has differed by up to a quarter:
# 24x2 gave 72.1 in stage 14 and 56.4 in stage 15, 48x1 gave 46.0 and 53.3.
# One more sample of each tells whether the drop is a cliff or noise
REPEAT_ONE_THREAD_JOBS = [30, 36, 42, 45]

# The arm stage 13 runs. The gurobi defaults until stage 12 says otherwise:
# set it to {"cuts": 0} if that stage finds cuts off is the faster arm here
# too. Every cell of stage 13 that shares a resolution and a thread count with
# stage 12 is then the same run, and already_done skips it
NL_THREAD_ARM = {}

# Configurations of stages 10 and 11, cheapest first. td16_bp1 is left out on
# purpose: at 3208 s a cell it costs more than these two together, and stages 8
# and 9 already cover it. td0 is out for the same reason as everywhere else
RESOLUTION_CONFIGS = [
    {"typicaldays": 4, "knobs": {}},
    {"typicaldays": 16, "knobs": {}},
]


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


def _job_environment():
    """
    Environment of a job the study launches.

    The thread limits of the numerical libraries are set here rather than
    inside adopt, because they are read once when a library is first loaded:
    set in the environment of the new process, they are in place before
    anything is imported.

    :return: dict, a copy of this process's environment with the limits set
    """
    import os

    environment = dict(os.environ)
    if OMP_THREADS:
        for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
            environment[variable] = str(OMP_THREADS)
    return environment


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
    result = subprocess.run(command, cwd=str(BASE), env=_job_environment())
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


def _cell_settings(options: dict, threads: int, time_limit: float = None):
    """
    Settings of one cell of a factorial.

    The stage and the manifest both go through here, so that the case name a
    run is given and the case name the manifest looks for cannot drift apart.

    :param dict options: solver options of the cell
    :param int threads: number of threads
    :param float time_limit: hours the cell may take, TIME_LIMIT if not given
    :return: dict of settings
    """
    return {
        "mipgap": MIPGAP,
        "time_limit": TIME_LIMIT if time_limit is None else time_limit,
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
    10: lambda: _thread_resolution_runs(),
    11: lambda: _cutpasses_runs(),
    12: lambda: _nl_cut_runs(),
    13: lambda: _nl_thread_runs(),
}

# The case study a stage runs, where it is not the one the study is built
# around. The manifest has to spell the case name exactly as the run did, so
# it cannot assume CASE
STAGE_CASES = {
    12: NL_CASE,
    13: NL_CASE,
}


def _folders_of(case_name: str):
    """
    Finished result folders of a case name.

    Archived runs are included, so a manifest can be written for a stage whose
    results have already been filed away, which is the usual case.

    :param str case_name: case name to look for
    :return: sorted list of folder names, oldest first
    """
    from run_benchmark import run_folders

    found = []
    for folder in run_folders(RESULTS_PATH):
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

    :param int stage: stage number, 6 to 11
    :return: Path of the manifest, or None if the stage has no cells
    """
    from run_benchmark import _build_case_name

    if stage not in STAGE_CELLS:
        return None

    lines = []
    missing = 0
    for typicaldays, knobs, options, threads in STAGE_CELLS[stage]():
        settings = _cell_settings(
            options,
            threads,
            _nl_time_limit(typicaldays) if stage in STAGE_CASES else None,
        )
        case_name = _build_case_name(
            STAGE_CASES.get(stage, CASE),
            {"typicaldays": typicaldays, **knobs, **settings},
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
            "mipgap": MIPGAP,
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


def _thread_resolution_runs():
    """
    Builds the cells of the thread resolution ladder

    :return: list of (typicaldays, knobs, options, threads) tuples
    """
    runs = []
    for config in RESOLUTION_CONFIGS:
        for threads in THREAD_RESOLUTION_LEVELS:
            runs.append((config["typicaldays"], config["knobs"], {}, threads))
    return runs


def stage_thread_resolution(dry_run: bool = False):
    """
    Fills in the thread counts between the levels the study already has.

    Everything is at the solver defaults: the question is about the machine
    and the thread count, not about an option, and stages 6 to 9 showed
    nothing interacting with the thread count anyway.

    :param bool dry_run: if True, the commands are only printed
    :return: True if every run that was attempted succeeded
    """
    from run_benchmark import already_done

    runs = _thread_resolution_runs()
    log(
        f"=== Stage 10: thread resolution, {len(RESOLUTION_CONFIGS)} "
        f"configurations, {len(runs)} runs ==="
    )

    ok = True
    skipped = 0
    produced = []

    for number, (typicaldays, knobs, options, threads) in enumerate(runs, start=1):
        settings = _cell_settings(options, threads)

        if not dry_run and already_done(CASE, typicaldays, knobs, settings):
            skipped += 1
            log(
                f"[RESOLUTION] run {number}/{len(runs)}: td{typicaldays}, "
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
        for knob, value in knobs.items():
            command += ["--set", f"{knob}={value}"]

        log(
            f"[RESOLUTION] run {number}/{len(runs)}: td{typicaldays}, "
            f"{threads} threads"
        )
        ok &= run(command, dry_run, produced)

    if not dry_run:
        log(f"[RESOLUTION] {skipped} of {len(runs)} runs were already done")
        write_manifest(10)
        run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    return ok


def _cutpasses_runs():
    """
    Builds the cells of the cut pass count by thread count factorial

    The reference arms are part of the factorial rather than a separate stage,
    so that cuts = 0, the default and every CutPasses level are measured in one
    campaign and can be compared without reaching into an archive.

    :return: list of (typicaldays, knobs, options, threads) tuples
    """
    arms = [{"cutpasses": level} for level in GUROBI_CUTPASSES_LEVELS]
    arms += GUROBI_CUTPASSES_REFERENCES

    runs = []
    for config in RESOLUTION_CONFIGS:
        for options in arms:
            for threads in GUROBI_CUTPASSES_THREAD_LEVELS:
                runs.append(
                    (
                        config["typicaldays"],
                        config["knobs"],
                        dict(options),
                        threads,
                    )
                )
    return runs


def stage_cutpasses(dry_run: bool = False):
    """
    Crosses the number of root cut passes with the thread count.

    cutpasses = -1 is the gurobi default and therefore also the baseline cell
    of the stage, spelled exactly like the all default cell of stage 8, so
    already_done shares it if it is still visible.

    :param bool dry_run: if True, the commands are only printed
    :return: True if every run that was attempted succeeded
    """
    from run_benchmark import already_done

    runs = _cutpasses_runs()
    log(
        f"=== Stage 11: root cut passes by threads, "
        f"{len(RESOLUTION_CONFIGS)} configurations, {len(runs)} runs ==="
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
                f"[CUTPASSES] run {number}/{len(runs)}: td{typicaldays}, "
                f"{spelled}, {threads} threads, already done, skipped"
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
            f"[CUTPASSES] run {number}/{len(runs)}: td{typicaldays}, "
            f"{spelled}, {threads} threads"
        )
        ok &= run(command, dry_run, produced)

    if not dry_run:
        log(f"[CUTPASSES] {skipped} of {len(runs)} runs were already done")
        write_manifest(11)
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


def _nl_cut_runs():
    """
    Builds the cells of the cut stage of the nine node case study.

    Cheapest first, so that a launch that does not finish leaves whole
    resolutions behind rather than half of each.

    :return: list of (typicaldays, knobs, options, threads) tuples
    """
    runs = []
    for typicaldays in NL_CUT_TYPICALDAYS:
        for options in NL_CUT_ARMS:
            for threads in NL_CUT_THREAD_LEVELS:
                runs.append((typicaldays, dict(NL_KNOBS), dict(options), threads))
    return runs


def _nl_thread_runs():
    """
    Builds the cells of the thread stage of the nine node case study.

    One cut arm, NL_THREAD_ARM, chosen once stage 12 has been read. td30
    carries only the thread counts the Snellius decision is between.

    :return: list of (typicaldays, knobs, options, threads) tuples
    """
    ladders = {
        min(NL_THREAD_TYPICALDAYS): NL_THREAD_LEVELS,
        max(NL_THREAD_TYPICALDAYS): NL_THREAD_LEVELS_LARGE,
    }

    runs = []
    for typicaldays in NL_THREAD_TYPICALDAYS:
        levels = ladders.get(typicaldays, NL_THREAD_LEVELS_MEDIUM)
        for threads in levels:
            runs.append((typicaldays, dict(NL_KNOBS), dict(NL_THREAD_ARM), threads))
    return runs


def _run_nl_cells(runs: list, stage: int, label: str, title: str, dry_run: bool):
    """
    Runs the cells of one nine node stage.

    Both stages of that case study are the same loop over a different
    factorial, so they share it rather than having it written twice.

    :param list runs: cells, as (typicaldays, knobs, options, threads) tuples
    :param int stage: stage number, which the manifest is written for
    :param str label: tag the log lines carry
    :param str title: headline of the stage
    :param bool dry_run: if True, the commands are only printed
    :return: True if every run that was attempted succeeded
    """
    from run_benchmark import already_done

    log(f"=== Stage {stage}: {title}, {len(runs)} runs ===")

    ok = True
    skipped = 0
    produced = []

    for number, (typicaldays, knobs, options, threads) in enumerate(runs, start=1):
        time_limit = _nl_time_limit(typicaldays)
        settings = _cell_settings(options, threads, time_limit)
        spelled = ", ".join(f"{name} {value}" for name, value in options.items())
        spelled = spelled or "defaults"

        if not dry_run and already_done(NL_CASE, typicaldays, knobs, settings):
            skipped += 1
            log(
                f"[{label}] run {number}/{len(runs)}: td{typicaldays}, "
                f"{spelled}, {threads} threads, already done, skipped"
            )
            continue

        command = [
            sys.executable,
            "run_benchmark.py",
            "run",
            "--case",
            NL_CASE,
            "--typicaldays",
            str(typicaldays),
            "--time-limit",
            str(time_limit),
            "--threads",
            str(threads),
        ]
        for name, value in options.items():
            command += ["--set", f"{name}={value}"]
        for knob, value in knobs.items():
            command += ["--set", f"{knob}={value}"]

        log(
            f"[{label}] run {number}/{len(runs)}: td{typicaldays}, "
            f"{spelled}, {threads} threads"
        )
        ok &= run(command, dry_run, produced)

    if not dry_run:
        log(f"[{label}] {skipped} of {len(runs)} runs were already done")
        write_manifest(stage)
        run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    return ok


def _contention_packs():
    """
    Builds the packings of the contention stage.

    One entry per thread count, each filling the core budget as evenly as the
    count divides it.

    :return: list of (threads, jobs) tuples
    """
    return [
        (threads, CONTENTION_CORE_BUDGET // threads)
        for threads in CONTENTION_THREAD_LEVELS
    ]


def _prepare_input_folders(jobs: int):
    """
    Gives every job of a pack its own clean copy of the input data.

    The jobs used to build their own copy, which meant as many concurrent
    copytree calls of nineteen megabytes from one source on a network share
    as there were jobs. The copies are made here instead, one after another,
    from a folder the case study has just built, and any copy left over from
    an earlier pack is replaced rather than trusted.

    This is not what corrupted the packs, which was the shared Summary.xlsx,
    but concurrent copying of the same tree is worth not doing anyway.

    :param int jobs: how many copies are needed
    """
    from run_benchmark import CASE_STUDIES, INPUT_DATA_PATH

    source = INPUT_DATA_PATH / NL_CASE
    if not (source / "Topology.json").exists():
        CASE_STUDIES[NL_CASE].setup(
            source, RESULTS_PATH, typicaldays=CONTENTION_TYPICALDAYS
        )

    log(f"[PREPARE] copying the input data into {jobs} job folders")
    for job in range(1, jobs + 1):
        target = INPUT_DATA_PATH / f"{NL_CASE}_job{job:02d}"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)


def _run_pack(threads: int, jobs: int, label: str, tag: str, dry_run: bool):
    """
    Runs one pack of concurrent jobs and logs what it delivered.

    Every job gets its own input data folder, since setup rewrites it on each
    call and two runs sharing one would read half written files, and no job
    collects: concurrent writers would race on the dataset, so the caller
    collects once afterwards.

    :param int threads: threads each job asks for
    :param int jobs: how many jobs run at the same time
    :param str label: name the runs and the log lines carry
    :param str tag: tag of the stage, for the log
    :param bool dry_run: if True, the commands are only printed
    :return: True if every job succeeded
    """
    log(
        f"[{tag}] {label}: {jobs} concurrent jobs of {threads} "
        f"thread{'s' if threads > 1 else ''}, td{CONTENTION_TYPICALDAYS}"
    )

    # Each job writes its own console to a file. Without this every job of a
    # pack writes to the same terminal, which is unreadable while they run and
    # gone afterwards: five jobs of stage 14 failed leaving an empty result
    # folder and no way to find out why
    log_path = BASE / "pack_logs"
    log_path.mkdir(exist_ok=True)

    if not dry_run:
        _prepare_input_folders(jobs)

    commands = []
    for job in range(1, jobs + 1):
        command = [
            sys.executable,
            "run_benchmark.py",
            "run",
            "--case",
            NL_CASE,
            "--typicaldays",
            str(CONTENTION_TYPICALDAYS),
            "--time-limit",
            str(_nl_time_limit(CONTENTION_TYPICALDAYS)),
            "--threads",
            str(threads),
            "--case-name",
            f"{NL_CASE}_{label}_job{job:02d}",
            "--input-suffix",
            f"job{job:02d}",
            "--results-subdir",
            f"{label}/job{job:02d}",
            "--no-collect",
        ]
        for knob, value in NL_KNOBS.items():
            command += ["--set", f"{knob}={value}"]
        commands.append(command)

    if dry_run:
        print(f"  would run {jobs} copies of:")
        print("    " + " ".join(str(part) for part in commands[0]))
        return True

    start = time.time()
    environment = _job_environment()
    processes, handles = [], []
    for job, command in enumerate(commands, start=1):
        handle = open(log_path / f"{label}_job{job:02d}.log", "w")
        handles.append(handle)
        processes.append(
            subprocess.Popen(
                command,
                cwd=str(BASE),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        )

    codes = [process.wait() for process in processes]
    for handle in handles:
        handle.close()
    makespan = time.time() - start

    failed = [job for job, code in enumerate(codes, start=1) if code != 0]
    log(
        f"[{tag}] {label}: makespan {makespan / 60:.1f} min, "
        f"{jobs / (makespan / 3600):.1f} jobs per hour, "
        f"{len(failed)} of {jobs} failed"
        + (f", jobs {failed}, see pack_logs/" if failed else "")
    )
    return not failed


def stage_contention(dry_run: bool = False):
    """
    Fills the machine with jobs and measures what each packing delivers.

    The study recommends one thread per job on the strength of core seconds
    measured on runs that each had the machine to themselves. That is an
    assumption, not a measurement: 48 jobs at one thread share the memory
    bandwidth and the last level cache of one machine, and 8 jobs at six
    threads do not share them in the same way. This runs each packing for
    real, on the same configuration, and reports jobs per hour.

    :param bool dry_run: if True, the commands are only printed
    :return: True if every pack that was attempted succeeded
    """
    packs = _contention_packs()
    log(
        f"=== Stage 14: {NL_CASE} contention, {CONTENTION_CORE_BUDGET} cores, "
        f"{len(packs)} packings ==="
    )

    ok = True
    for threads, jobs in packs:
        ok &= _run_pack(
            threads, jobs, f"pack{threads}x{jobs}", "CONTENTION", dry_run
        )

    if not dry_run:
        run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    return ok


def _worker_count_packs():
    """
    The packs of stage 15, and of stage 17 which repeats them

    :return: list of (threads, jobs) tuples, cheapest arm first
    """
    packs = []
    for threads in CONTENTION_WORKER_THREADS:
        for fill in CONTENTION_FILL_LEVELS[threads]:
            jobs = max(1, round(CONTENTION_CORE_BUDGET * fill / threads))
            # Two fills a third of a core apart round to the same pack
            if (threads, jobs) not in packs:
                packs.append((threads, jobs))
    return packs


def stage_worker_count(dry_run: bool = False):
    """
    How full to fill the machine, at one thread per job.

    Stage 14 asks how to shape the jobs at a budget that is always spent in
    full. This asks whether spending it in full is right, at both thread
    counts that are still candidates: half the machine, three quarters, all of
    it, and a third more than all of it.

    Throughput against the worker count is a curve with a maximum, and where
    that maximum sits is the number a campaign runner should use. If it sits
    below the core count, leaving cores idle is the faster choice, and that is
    a real possibility here: the jobs do not oversubscribe the cores, they
    share one memory system, and the first pack of stage 14 lost a factor two
    to it.

    :param bool dry_run: if True, the commands are only printed
    :return: True if every pack that was attempted succeeded
    """
    packs = _worker_count_packs()

    log(
        f"=== Stage 15: {NL_CASE} worker count on "
        f"{CONTENTION_CORE_BUDGET} cores, {len(packs)} packs ==="
    )

    ok = True
    for threads, jobs in packs:
        ok &= _run_pack(
            threads, jobs, f"fill{jobs}x{threads}", "WORKERS", dry_run
        )

    if not dry_run:
        run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    return ok


def stage_repeat_fill(dry_run: bool = False):
    """
    Runs the one thread packs of stage 15 where its curve falls a second time.

    The packs are labelled rep rather than fill, so that they write into
    results folders and pack logs of their own and enter the dataset as runs of
    their own, instead of overwriting the first sample.

    :param bool dry_run: if True, the commands are only printed
    :return: True if every pack that was attempted succeeded
    """
    log(
        f"=== Stage 16: {NL_CASE} one thread packs repeated on "
        f"{CONTENTION_CORE_BUDGET} cores, {len(REPEAT_ONE_THREAD_JOBS)} packs ==="
    )

    ok = True
    for jobs in REPEAT_ONE_THREAD_JOBS:
        ok &= _run_pack(1, jobs, f"rep{jobs}x1", "REPEAT", dry_run)

    if not dry_run:
        run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    return ok


def stage_worker_count_omp(dry_run: bool = False):
    """
    Stage 15 again, with the numerical libraries held to OMP_THREADS threads.

    Stage 15 ran with the libraries free, and every job's typical day
    clustering started about 100 threads on the 48 core server. Reading alone,
    that made the reading x5 slower at 48 jobs and one thread per library cut
    it back to x1.1. What that is worth in runs an hour, with the solvers of
    the other jobs sharing the machine, is what this measures: the same packs,
    labelled omp instead of fill, so that they sit next to stage 15 rather than
    on top of it.

    :param bool dry_run: if True, the commands are only printed
    :return: True if every pack that was attempted succeeded
    """
    if not OMP_THREADS:
        log("Stage 17 needs --omp-threads above 0, it would repeat stage 15")
        return False

    packs = _worker_count_packs()
    log(
        f"=== Stage 17: {NL_CASE} worker count on {CONTENTION_CORE_BUDGET} "
        f"cores, libraries at {OMP_THREADS} thread, {len(packs)} packs ==="
    )

    ok = True
    for threads, jobs in packs:
        ok &= _run_pack(threads, jobs, f"omp{jobs}x{threads}", "WORKERS_OMP", dry_run)

    if not dry_run:
        run([sys.executable, "run_benchmark.py", "collect"], dry_run)

    return ok


def stage_nl_cuts(dry_run: bool = False):
    """
    Asks whether cuts = 0 is still the faster arm on the nine node case study.

    The first of the two stages of that case study and the cheap one. Both
    recommendations the study makes were measured on four_node at a 2 % gap,
    where an arm could buy time by stopping at a looser bound. This runs the
    two arms at two resolutions and two thread counts, at 0.5 %.

    :param bool dry_run: if True, the commands are only printed
    :return: True if every run that was attempted succeeded
    """
    return _run_nl_cells(
        _nl_cut_runs(),
        12,
        "NL_CUTS",
        f"{NL_CASE}, cuts by threads, {len(NL_CUT_TYPICALDAYS)} resolutions",
        dry_run,
    )


def stage_nl_threads(dry_run: bool = False):
    """
    The thread ladder of the nine node case study.

    Launched once stage 12 has been read, with NL_THREAD_ARM set to whichever
    cut setting that stage favoured. The question is the one stage 10 answered
    on four_node: how many core seconds a run costs at each thread count, and
    therefore how many cores a cluster job should ask for.

    :param bool dry_run: if True, the commands are only printed
    :return: True if every run that was attempted succeeded
    """
    arm = ", ".join(f"{name} {value}" for name, value in NL_THREAD_ARM.items())
    return _run_nl_cells(
        _nl_thread_runs(),
        13,
        "NL_THREADS",
        f"{NL_CASE}, thread ladder at {arm or 'the defaults'}, "
        f"{len(NL_THREAD_TYPICALDAYS)} resolutions",
        dry_run,
    )


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
    10: stage_thread_resolution,
    11: stage_cutpasses,
    12: stage_nl_cuts,
    13: stage_nl_threads,
    14: stage_contention,
    15: stage_worker_count,
    16: stage_repeat_fill,
    17: stage_worker_count_omp,
}

# The thread ladder answers a question about the machine rather than about the
# case study, and it re-runs configurations the study already covers, so asking
# for the study does not ask for it
DEFAULT_STAGES = [0, 1, 2, 3, 4]


def main():
    """
    Command line interface of the study
    """
    global THREADS, TIME_LIMIT, OMP_THREADS

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
        "--time-limit",
        dest="time_limit",
        type=float,
        default=TIME_LIMIT,
        help="hours a single run may take before the solver gives up. It is "
        "part of the case name, so a stage run at a different limit does not "
        "skip the runs of the first one as already done",
    )
    parser.add_argument(
        "--omp-threads",
        dest="omp_threads",
        type=int,
        default=OMP_THREADS,
        help="threads the numerical libraries of every job may use (OpenMP "
        "and BLAS, which run the typical day clustering). 0 leaves them free, "
        "one per core, which is how stages up to 16 ran",
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
    TIME_LIMIT = args.time_limit
    OMP_THREADS = args.omp_threads

    started = time.time()
    if not args.dry_run:
        log("")
        log("#" * 70)
        log("Benchmark study starting")
        log(f"Solver threads: {THREADS if THREADS else 'all cores'}")
        log(f"Library threads: {OMP_THREADS if OMP_THREADS else 'all cores'}")
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
