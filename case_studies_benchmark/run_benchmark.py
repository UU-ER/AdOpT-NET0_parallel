"""
Runs case studies with resource profiling switched on and collects the
profiles of all runs into a single dataset.

The dataset relates the size of a model (number of variables, constraints,
non-zeros, ...) to the resources a run needed (wall time, cpu time, peak
memory, disk and network io), so that the resource usage of a case study can
be predicted from its size.

Examples::

    # One run of the network case study with 30 typical days
    python run_benchmark.py run --case network --typicaldays 30

    # A sweep over the number of typical days
    python run_benchmark.py sweep --case network --typicaldays 5 10 30 60

    # Collect all runs done so far into benchmark_dataset.csv
    python run_benchmark.py collect
"""

import argparse
import inspect
import itertools
import logging
import re
import subprocess
import sys
import traceback
from pathlib import Path

# The package is imported from the repository, so that the benchmark runs
# against the working tree and not against an installed version
REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

import adopt_net0 as adopt
from adopt_net0.diagnostics import collect_profiles

from case_studies import CASE_STUDIES

BASE = Path(__file__).parent
INPUT_DATA_PATH = BASE / "inputData"
RESULTS_PATH = BASE / "results"
DATASET_FILE = BASE / "benchmark_dataset.csv"

log = logging.getLogger(__name__)


def run_case(case: str, **settings):
    """
    Sets up and runs a single case study

    :param str case: name of the case study, a key of CASE_STUDIES
    :param settings: settings passed on to the setup function of the case
    :return: ModelHub of the run
    """
    if case not in CASE_STUDIES:
        raise KeyError(f"Unknown case study '{case}', available: {list(CASE_STUDIES)}")

    case_study = CASE_STUDIES[case]

    # Input data of a case study is reused across runs, the results of every
    # run go to their own folder
    input_data_path = INPUT_DATA_PATH / case
    case_name = settings.pop("case_name", None) or _build_case_name(case, settings)

    log_msg = f"--- Benchmark run: {case_name} ---"
    print(log_msg)
    log.info(log_msg)

    case_study.setup(input_data_path, RESULTS_PATH, case_name=case_name, **settings)

    model = adopt.ModelHub()
    model.read_data(input_data_path)
    model.quick_solve()

    return model


# Settings that are left out of the case name, as they end up as columns of
# the dataset anyway and would only make the results folder name unreadable
DEFAULT_SETTINGS = {
    "mipgap": 0.02,
    "time_limit": 2,
    # 0 is every core of the machine. A run with a different number of threads
    # is a different run, so it has to reach the case name, otherwise
    # already_done would recognise the unrestricted run as this one and skip it
    "threads": 0,
    "carbon_price": 0,
    "solver": "gurobi",
    "sampling_interval": 0.5,
    "typicaldays_method": 1,
    "electricity_price": "constant",
    "hydrogen_demand": "constant",
    "pipeline_capex": "linear",
    "bidirectional_precise": 0,
    "pipeline_size_min": 0,
    "storage": "on",
    "storage_precise": 0,
    "pv": "on",
}


# Short codes for the knobs. Windows refuses paths beyond 260 characters, and
# the results folder name is built from the case name, so spelling the knobs
# out in full breaks a sweep that varies several of them at once.
KNOB_CODES = {
    "typicaldays_method": "m",
    "electricity_price": "p",
    "hydrogen_demand": "d",
    "pipeline_capex": "cx",
    "bidirectional_precise": "bp",
    "pipeline_size_min": "sm",
    "storage": "st",
    "storage_precise": "sp",
    "pv": "pv",
    "carbon_price": "co2",
    "mipgap": "gap",
    "time_limit": "tl",
    "threads": "thr",
    "solver": "slv",
}

VALUE_CODES = {
    "constant": "c",
    "fluctuating": "f",
    "linear": "lin",
    "fixed_plus_linear": "fix",
    "on": "1",
    "off": "0",
    "gurobi": "grb",
    "gurobi_persistent": "grbp",
}


def _build_case_name(case: str, settings: dict):
    """
    Builds a short case name from the settings that deviate from the default

    :param str case: name of the case study
    :param dict settings: settings passed to the setup function
    :return: str case name
    """
    parts = [case]
    if "typicaldays" in settings:
        parts.append(f"td{settings['typicaldays']}")

    for key, value in sorted(settings.items()):
        if key == "typicaldays" or DEFAULT_SETTINGS.get(key) == value:
            continue
        knob = KNOB_CODES.get(key, key)
        parts.append(f"{knob}{VALUE_CODES.get(value, value)}")

    return "_".join(parts)


# The two settings of every complexity knob, cheapest first. A case study that
# is not listed here can still be run, but not swept with the matrix command.
KNOB_LEVELS = {
    "four_node": {
        "typicaldays_method": [1, 2],
        "electricity_price": ["constant", "fluctuating"],
        "hydrogen_demand": ["constant", "fluctuating"],
        "pipeline_capex": ["linear", "fixed_plus_linear"],
        "bidirectional_precise": [0, 1],
        "pipeline_size_min": [0, 250],
        "storage": ["off", "on"],
        "storage_precise": [0, 1],
        "pv": ["off", "on"],
    },
    "network": {
        "typicaldays_method": [1, 2],
        "carbon_price": [0, 100],
    },
}


def already_done(case: str, typicaldays: int, combination: dict, settings: dict):
    """
    Checks whether a configuration has already been run.

    A results folder is named after a timestamp followed by the case name, so
    a finished run can be recognised by its profile summary being on disk. This
    makes a long sweep resumable: interrupt it, start it again, and it picks up
    where it stopped.

    :param str case: name of the case study
    :param int typicaldays: number of typical days
    :param dict combination: knob settings of this run
    :param dict settings: settings shared by all runs
    :return: True if a finished run with these settings exists
    """
    case_name = _build_case_name(
        case, {"typicaldays": typicaldays, **combination, **settings}
    )

    for folder in RESULTS_PATH.glob(f"*_{case_name}*"):
        # A folder is named <timestamp>_<case name>-<counter>. The name has to
        # match exactly, not just as a prefix of a longer configuration.
        without_timestamp = folder.name.split("_", 1)[-1]
        without_counter = re.sub(r"-\d+$", "", without_timestamp)
        if without_counter != case_name:
            continue
        if (folder / "profile_summary.csv").exists():
            return True

    return False


def run_matrix(case: str, typicaldays: list, knobs: list = None, **settings):
    """
    Runs every combination of the complexity knobs.

    The runs are ordered from cheap to expensive, so that a sweep that is cut
    short still covers the small configurations. A failing run does not stop
    the matrix.

    :param str case: name of the case study
    :param list typicaldays: numbers of typical days to run
    :param list knobs: knobs to sweep, default all of the case study
    :param settings: settings passed on to the setup function
    :return: number of runs that failed
    """
    if case not in KNOB_LEVELS:
        raise KeyError(f"No knob levels defined for case study '{case}'")

    levels = KNOB_LEVELS[case]
    if knobs:
        unknown = [knob for knob in knobs if knob not in levels]
        if unknown:
            raise KeyError(f"Unknown knobs {unknown}, available: {sorted(levels)}")
        levels = {knob: levels[knob] for knob in knobs}

    combinations = [
        dict(zip(levels, values)) for values in itertools.product(*levels.values())
    ]
    # Cheapest first: the number of knobs that are not at their first level
    combinations.sort(
        key=lambda combination: sum(
            combination[knob] != levels[knob][0] for knob in levels
        )
    )

    total = len(combinations) * len(typicaldays)
    print(f"[MATRIX] {len(levels)} knobs, {len(combinations)} combinations")
    print(f"[MATRIX] {len(typicaldays)} typical day settings, {total} runs in total")

    failed = 0
    skipped = 0
    done = 0
    for days in sorted(typicaldays, key=lambda value: (value == 0, value)):
        for combination in combinations:
            done += 1

            # Skipping what is already on disk makes a long sweep resumable
            if already_done(case, days, combination, settings):
                skipped += 1
                print(f"[MATRIX] run {done}/{total}: already done, skipped")
                continue

            print(f"\n[MATRIX] run {done}/{total}: typicaldays={days}, {combination}")
            if not _run_isolated(case, days, combination, settings):
                failed += 1

    completed = total - failed - skipped
    print(
        f"\n[MATRIX] {completed} run, {skipped} already done, {failed} failed, "
        f"{total} in total"
    )
    return failed


def _run_isolated(case: str, typicaldays: int, combination: dict, settings: dict):
    """
    Runs one configuration in a separate process.

    Every run gets its own process for two reasons. The peak memory reported by
    the operating system is a high water mark of the whole process and never
    goes down, so a second run in the same process would inherit the peak of
    the first. And memory left behind by a previous run slows down the next
    one, which would show up as construction time that is not real.

    :param str case: name of the case study
    :param int typicaldays: number of typical days
    :param dict combination: knob settings of this run
    :param dict settings: settings shared by all runs
    :return: True if the run succeeded
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--case",
        case,
        "--typicaldays",
        str(typicaldays),
    ]

    for knob, value in combination.items():
        command += ["--set", f"{knob}={value}"]
    for key, value in settings.items():
        if key in ["mipgap", "solver", "sampling_interval", "time_limit", "threads"]:
            command += [f"--{key.replace('_', '-')}", str(value)]
        else:
            command += ["--set", f"{key}={value}"]

    # The collect at the end of the child run is redundant here, the matrix
    # collects once when it is done
    result = subprocess.run(command, cwd=str(Path(__file__).parent))

    if result.returncode != 0:
        log_msg = (
            f"Run failed with exit code {result.returncode}: "
            f"typicaldays={typicaldays}, {combination}"
        )
        print(log_msg)
        log.error(log_msg)
        return False

    return True


def collect(save: bool = True):
    """
    Collects the profiles of all runs into a single dataset

    :param bool save: if True, the dataset is written to benchmark_dataset.csv
    :return: pandas DataFrame with one row per run
    """
    profiles = collect_profiles(RESULTS_PATH, save_path=DATASET_FILE if save else None)

    if profiles.empty:
        print(f"No runs found in {RESULTS_PATH}")
        return profiles

    print(f"Collected {len(profiles)} runs")
    if save:
        print(f"Dataset written to {DATASET_FILE}")

    columns = [
        "case_name",
        "n_vars",
        "n_constrs",
        "n_nnz",
        "wall_total_s",
        "cpu_user_s",
        "rss_peak_os_mb",
        "t_construct_model_s",
        "t_solve_s",
        "gurobi_runtime_s",
    ]
    columns = [column for column in columns if column in profiles.columns]
    print(profiles[columns].to_string(index=False))

    return profiles


def main():
    """
    Command line interface of the benchmark
    """
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run a single case study")
    _add_case_arguments(run_parser)

    sweep_parser = subparsers.add_parser(
        "sweep", help="run a case study for several numbers of typical days"
    )
    _add_case_arguments(sweep_parser, sweep=True)

    matrix_parser = subparsers.add_parser(
        "matrix", help="run every combination of the complexity knobs"
    )
    _add_case_arguments(matrix_parser, sweep=True)
    matrix_parser.add_argument(
        "--knobs", nargs="+", help="knobs to sweep, default all of the case study"
    )

    subparsers.add_parser("collect", help="collect all runs into a dataset")

    args = parser.parse_args()

    if args.command == "collect":
        collect()
        return

    settings = {
        "mipgap": args.mipgap,
        "time_limit": args.time_limit,
        "threads": args.threads,
        "solver": args.solver,
        "sampling_interval": args.sampling_interval,
    }
    # carbon_price only exists on the case studies that have it as a knob
    if "carbon_price" in knobs_of(args.case):
        settings["carbon_price"] = args.carbon_price
    settings.update(parse_overrides(args.case, args.overrides))

    if args.command == "matrix":
        run_matrix(args.case, args.typicaldays, knobs=args.knobs, **settings)
        collect()
        return

    if args.command == "run":
        run_case(args.case, typicaldays=args.typicaldays, **settings)
    else:
        # A failing run should not stop the sweep, the runs that did work are
        # still collected at the end
        for typicaldays in args.typicaldays:
            try:
                run_case(args.case, typicaldays=typicaldays, **settings)
            except Exception:
                log_msg = f"Run with {typicaldays} typical days failed"
                print(log_msg)
                log.error(log_msg)
                traceback.print_exc()

    collect()


def _add_case_arguments(parser, sweep: bool = False):
    """
    Adds the arguments shared by the run and the sweep command

    :param parser: argparse parser to add the arguments to
    :param bool sweep: if True, several numbers of typical days are accepted
    """
    parser.add_argument("--case", default="network", choices=list(CASE_STUDIES))
    if sweep:
        parser.add_argument(
            "--typicaldays", type=int, nargs="+", default=[5, 10, 30, 60]
        )
    else:
        parser.add_argument("--typicaldays", type=int, default=30)
    parser.add_argument("--mipgap", type=float, default=0.02)
    parser.add_argument(
        "--time-limit",
        dest="time_limit",
        type=float,
        default=2,
        help="solver time limit in hours, so that a pathological run cannot "
        "hold an unattended sweep for the template default of 100 hours",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="number of threads the solver may use, 0 for every core of the "
        "machine. This is the number a cluster job would ask for, and it ends "
        "up in the case name so that sweeps at different thread counts do not "
        "recognise each other as already done",
    )
    parser.add_argument("--carbon-price", dest="carbon_price", type=float, default=0)
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument(
        "--sampling-interval", dest="sampling_interval", type=float, default=0.5
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KNOB=VALUE",
        help="set a knob of the case study, can be repeated",
    )


def knobs_of(case: str):
    """
    Returns the knobs of a case study and their default values.

    A knob is any parameter of its setup function beyond the ones the runner
    handles itself, which lets a case study add complexity knobs without the
    runner having to know about them.

    :param str case: name of the case study
    :return: dict of knob name to default value
    """
    handled = {
        "input_data_path",
        "results_path",
        "typicaldays",
        "mipgap",
        "solver",
        "sampling_interval",
        "case_name",
    }
    parameters = inspect.signature(CASE_STUDIES[case].setup).parameters

    return {
        name: parameter.default
        for name, parameter in parameters.items()
        if name not in handled
    }


def parse_overrides(case: str, overrides: list):
    """
    Parses KNOB=VALUE strings into settings of the right type

    :param str case: name of the case study
    :param list overrides: list of "knob=value" strings
    :return: dict with the parsed settings
    """
    knobs = knobs_of(case)
    settings = {}

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected KNOB=VALUE, got '{override}'")
        knob, value = override.split("=", 1)

        if knob not in knobs:
            raise KeyError(
                f"'{knob}' is not a knob of case study '{case}', "
                f"available: {sorted(knobs)}"
            )

        # The type of the default decides how the value is read
        default = knobs[knob]
        if isinstance(default, bool):
            settings[knob] = value.lower() in ["1", "true", "yes"]
        elif isinstance(default, int):
            settings[knob] = int(value)
        elif isinstance(default, float):
            settings[knob] = float(value)
        else:
            settings[knob] = value

    return settings


if __name__ == "__main__":
    main()
