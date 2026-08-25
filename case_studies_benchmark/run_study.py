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

Examples::

    python run_study.py                     # the whole study
    python run_study.py --stages 0 1        # only the sizes and the matrix
    python run_study.py --stages 4          # only redo the figures and report
    python run_study.py --dry-run           # print what would run
"""

import argparse
import datetime
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


def run(command: list, dry_run: bool = False):
    """
    Runs one command of the study and reports how long it took

    :param list command: command line to run
    :param bool dry_run: if True, the command is only printed
    :return: True if the command succeeded
    """
    printable = " ".join(str(part) for part in command)
    if dry_run:
        print(f"  would run: {printable}")
        return True

    log(f"START {printable}")
    start = time.time()
    result = subprocess.run(command, cwd=str(BASE))
    duration = time.time() - start

    if result.returncode == 0:
        log(f"DONE  in {duration / 60:.1f} min")
        return True

    log(f"FAILED with exit code {result.returncode} after {duration / 60:.1f} min")
    return False


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
                *extra,
            ],
            dry_run,
        )
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
}


def main():
    """
    Command line interface of the study
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        choices=sorted(STAGES),
        default=sorted(STAGES),
        help="stages to run, default all of them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would run without running it",
    )
    args = parser.parse_args()

    started = time.time()
    if not args.dry_run:
        log("")
        log("#" * 70)
        log("Benchmark study starting")
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
