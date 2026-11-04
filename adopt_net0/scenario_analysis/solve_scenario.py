"""
Solves one scenario. Called by the Snakefile as

    python -m adopt_net0.scenario_analysis.solve_scenario <input folder> <done file> --threads 1
"""

import argparse
import json
import logging
import time
from pathlib import Path

from threadpoolctl import threadpool_limits

log = logging.getLogger(__name__)


def solve_scenario(input_path: str | Path, done_file: str | Path, threads: int = None):
    """
    Solves the scenario in an input data folder and records the outcome. The
    record is written only once the solve has finished, also if infeasible

    :param str/Path input_path: input data folder of the scenario
    :param str/Path done_file: file the outcome is written to once solved
    :param int threads: threads gurobi may use, if given, else the configuration's
    :return: the record of the outcome
    :rtype: dict
    """
    import adopt_net0 as adopt

    input_path = Path(input_path)
    done_file = Path(done_file)

    if threads is not None:
        _set_threads(input_path, threads)

    start = time.time()

    # Numerical libraries at one thread (stage 17), gurobi threads are separate
    with threadpool_limits(limits=1):
        pyhub = adopt.ModelHub()
        pyhub.read_data(input_path)
        pyhub.quick_solve()

    termination = str(pyhub.solution.solver.termination_condition)
    record = {
        "scenario": input_path.name,
        "termination": termination,
        "feasible": termination
        not in ["infeasible", "infeasibleOrUnbounded", "unbounded"],
        "wall_s": round(time.time() - start, 1),
        "threads": threads,
    }

    done_file.parent.mkdir(parents=True, exist_ok=True)
    with open(done_file, "w") as json_file:
        json.dump(record, json_file, indent=4)

    log.info(f"Scenario {input_path.name} finished: {termination}")
    return record


def _set_threads(input_path: Path, threads: int):
    """
    Sets the threads gurobi may use in the configuration of the scenario

    :param Path input_path: input data folder of the scenario
    :param int threads: number of threads
    """
    config_file = input_path / "ConfigModel.json"
    with open(config_file, "r") as json_file:
        config = json.load(json_file)

    config["solveroptions"]["threads"]["value"] = threads

    with open(config_file, "w") as json_file:
        json.dump(config, json_file, indent=4)


def main():
    """
    Command line interface, which is how the scheduler calls it
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="input data folder")
    parser.add_argument("done_file", type=Path, help="record written once solved")
    parser.add_argument("--threads", type=int, help="threads gurobi may use")
    arguments = parser.parse_args()

    solve_scenario(arguments.input_path, arguments.done_file, arguments.threads)


if __name__ == "__main__":
    main()
