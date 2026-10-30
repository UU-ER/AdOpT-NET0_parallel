"""
Tries the scenario analysis on a case study that has already been built.

Does three things, in the order a user would:

1. writes the catalogue of the case study, which is every parameter a scenario
   can change, and the page for putting an analysis together by searching it
2. writes an example definition, if there is not one already, and reads it back
3. checks it against the case study and, for every derived parameter, solves its
   equations for the first scenario and prints what would be written

Nothing is run and nothing in the case study is changed, so it is safe to point
at any input data folder.

Examples::

    python main_scenario_analysis.py
    python main_scenario_analysis.py --case-study path/to/input/data
    python main_scenario_analysis.py --catalogue-only --search capex
"""

import argparse
import json
import logging
from pathlib import Path

from adopt_net0.data_preprocessing import create_scenario_analysis_template
from adopt_net0.scenario_analysis import (
    read_definition,
    solve_derived,
    validate_definition,
    write_builder,
    write_catalogue,
)

# The benchmark case study, which is built and can be scanned as it is
DEFAULT_CASE_STUDY = Path("case_studies_benchmark/inputData/four_node")


def show_catalogue(case_study: Path, search: str = None):
    """
    Writes the catalogue and the builder page of a case study, and prints a part
    of the catalogue

    :param Path case_study: path to the input data folder
    :param str search: only print the parameters whose name contains this
    """
    catalogue = write_catalogue(case_study)
    page = write_builder(case_study)

    by_target = {}
    for entry in catalogue:
        by_target.setdefault(entry["target"], []).append(entry)

    print(f"\n{len(catalogue)} parameters in {case_study}")
    for target in sorted(by_target):
        print(f"  {len(by_target[target]):4d}  {target}")

    print(f"\nCatalogue  {case_study / 'ParameterCatalogue.json'}")
    print(f"Builder    {page}")
    print(f"           open it to search the catalogue and build the analysis")

    if search is None:
        return

    print(f"\nParameters matching '{search}':\n")
    for entry in catalogue:
        if search.lower() not in entry["parameter"].lower():
            continue
        scope = ", ".join(
            f"{name}={len(members)}" for name, members in entry["scope"].items()
        )
        print(
            f"  {entry['parameter']:<52} {entry['target']:<13} "
            f"{str(entry['current'])[:28]:<30} {scope}"
        )


def check_definition(case_study: Path):
    """
    Reads the definition of the case study, checks it and solves what it derives

    :param Path case_study: path to the input data folder
    """
    scenario_file = case_study / "ScenarioAnalysis.json"
    if not scenario_file.exists():
        create_scenario_analysis_template(case_study)
        print(f"\nA template was written to {scenario_file}. Fill it in and run again")

    definition = read_definition(scenario_file)
    problems = validate_definition(definition, case_study)

    print(f"\n--- {scenario_file.name} ---")
    if problems:
        print(f"{len(problems)} problems:\n")
        for problem in problems:
            print(f"  - {problem}")
        return

    print("No problems")

    with open(case_study / "Topology.json", "r") as json_file:
        topology = json.load(json_file)

    for name, block in definition.get("derived", {}).items():
        # The first value of every symbol, which is one scenario of the analysis
        sample = {
            symbol: definition["values"][0]
            for symbol, definition in block["symbols"].items()
        }
        resolved = solve_derived(block, sample, topology)

        spelled = ", ".join(f"{key} = {value}" for key, value in sample.items())
        print(f"\n--- derived '{name}' at {spelled} ---\n")
        for reference, value in sorted(resolved.items()):
            print(f"  {reference:<46} {value:14.4f}")


def main():
    """
    Command line interface of the example
    """
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-study",
        type=Path,
        default=DEFAULT_CASE_STUDY,
        help="input data folder of a case study that has already been built",
    )
    parser.add_argument(
        "--search",
        help="print the parameters of the catalogue whose name contains this",
    )
    parser.add_argument(
        "--catalogue-only",
        action="store_true",
        help="only write the catalogue, without reading a definition",
    )
    arguments = parser.parse_args()

    show_catalogue(arguments.case_study, arguments.search)

    if not arguments.catalogue_only:
        check_definition(arguments.case_study)


if __name__ == "__main__":
    main()
