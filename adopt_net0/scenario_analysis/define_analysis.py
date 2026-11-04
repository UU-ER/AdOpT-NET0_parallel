"""
Reads and checks the definition of a scenario analysis (ScenarioAnalysis.json).
Parameters are direct (exist in the input data, checked against the catalogue),
derived (defined by equations, see derived_parameters.py) or functions (anything
no equation reaches).
"""

import json
import logging
import re
from pathlib import Path

from .build_analysis import scan_input_data
from .derived_parameters import parse_equations
from .utilities import read_topology

log = logging.getLogger(__name__)

# Targets a parameter of the definition can have. The first five are applied by
# writing to the input data, the last one by calling a registered function
TARGETS = ["config", "carrier_data", "technology", "network", "structural", "function"]

OPERATIONS = ["set", "multiply"]
AGGREGATES = ["mean", "sum"]
COMBINATIONS = ["grid", "list", "lhs"]
NAMINGS = ["short", "readable", "hashed"]
RESOURCES = ["cores", "memory_gb", "threads_per_scenario", "memory_per_scenario_mb"]


def read_definition(file_path: str | Path) -> dict:
    """
    Reads the definition of a scenario analysis from its json file

    :param str/Path file_path: path to the scenario analysis file
    :return: the definition
    :rtype: dict
    """
    file_path = Path(file_path)
    with open(file_path, "r") as json_file:
        definition = json.load(json_file)

    log.info(
        f"Scenario analysis '{definition.get('analysis', {}).get('name', {}).get('value')}' "
        f"read from {file_path}"
    )
    return definition


def validate_definition(definition: dict, folder_path: str | Path) -> list:
    """
    Checks a scenario analysis against the case study it will be run on.

    Everything that can be known without running anything is checked here: that
    the parameters exist, that the values are there, that the equations of a
    derived parameter parse, and that they determine what they write. The
    problems are collected rather than raised one at a time, so that a
    definition with several mistakes needs one pass to fix.

    :param dict definition: the definition, as read by read_definition
    :param str/Path folder_path: path to the folder containing the case study data
    :return: list of problems, empty if the definition is sound
    :rtype: list
    """
    folder_path = Path(folder_path)
    catalogue = scan_input_data(folder_path)

    topology = read_topology(folder_path)

    problems = []
    problems += _validate_analysis(definition.get("analysis", {}))
    problems += _validate_parameters(definition.get("parameters", {}), catalogue)
    problems += _validate_derived(definition.get("derived", {}), catalogue, topology)
    problems += _validate_short(definition)
    problems += _validate_resources(definition.get("resources", {}))

    if not definition.get("parameters") and not definition.get("derived"):
        problems.append("The analysis changes nothing: no parameters and no derived")

    if problems:
        log.warning(f"Scenario analysis has {len(problems)} problems")
    else:
        log.info("Scenario analysis is sound")

    return problems


def _validate_analysis(analysis: dict) -> list:
    """
    Checks the settings of the analysis itself

    :param dict analysis: the analysis block of the definition
    :return: list of problems
    :rtype: list
    """
    problems = []

    if not analysis:
        return ["No 'analysis' block, so the analysis has no name and no method"]

    combination = analysis.get("combination", {}).get("value")
    if combination not in COMBINATIONS:
        problems.append(
            f"analysis/combination is '{combination}', not one of {COMBINATIONS}"
        )

    if combination == "lhs" and not analysis.get("n_samples", {}).get("value"):
        problems.append("analysis/combination is 'lhs' but n_samples is not set")

    if not analysis.get("name", {}).get("value"):
        problems.append("analysis/name is not set, so the results have nowhere to go")

    naming = analysis.get("naming", {}).get("value", "short")
    if naming not in NAMINGS:
        problems.append(f"analysis/naming is '{naming}', not one of {NAMINGS}")

    return problems


def _validate_resources(resources: dict) -> list:
    """
    Checks that the resources given are positive numbers

    :param dict resources: the resources block of the definition
    :return: list of problems
    :rtype: list
    """
    problems = []
    for key in RESOURCES:
        value = resources.get(key, {}).get("value")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            problems.append(f"resources/{key} is '{value}', not a positive number")
    return problems


def _validate_short(definition: dict) -> list:
    """
    Checks the 'short' aliases the scenarios are named by: each spells one
    parameter only, and holds only what a folder name can

    :param dict definition: the definition
    :return: list of problems
    :rtype: list
    """
    aliases = {
        name: parameter.get("short")
        for name, parameter in definition.get("parameters", {}).items()
    }
    for block in definition.get("derived", {}).values():
        for symbol, symbol_definition in block.get("symbols", {}).items():
            aliases[symbol] = symbol_definition.get("short")

    # A parameter without an alias is spelled by its name, which can collide too
    spelled = {name: alias or name for name, alias in aliases.items()}

    problems = []
    for name, alias in aliases.items():
        if alias is not None and not re.fullmatch(r"[A-Za-z0-9._-]+", str(alias)):
            problems.append(
                f"{name}: short '{alias}' can only hold letters, digits, dots, "
                f"dashes and underscores"
            )

    taken = {}
    for name, alias in spelled.items():
        taken.setdefault(alias, []).append(name)
    for alias, names in taken.items():
        if len(names) > 1:
            problems.append(
                f"{', '.join(names)}: all spelled '{alias}' in the scenario names"
            )

    return problems


def _validate_parameters(parameters: dict, catalogue: list) -> list:
    """
    Checks the direct parameters against the catalogue of the case study

    :param dict parameters: the parameters block of the definition
    :param list catalogue: catalogue of the case study
    :return: list of problems
    :rtype: list
    """
    known = {}
    for entry in catalogue:
        known.setdefault(entry["parameter"], []).append(entry)

    problems = []
    for name, parameter in parameters.items():
        target = parameter.get("target")
        if target not in TARGETS:
            problems.append(f"{name}: target is '{target}', not one of {TARGETS}")

        values = parameter.get("values")
        if not isinstance(values, list) or not values:
            problems.append(f"{name}: values is missing or empty")

        operation = parameter.get("operation", "set")
        if operation not in OPERATIONS:
            problems.append(
                f"{name}: operation is '{operation}', not one of {OPERATIONS}"
            )

        # A function is not in the catalogue by definition, it is the escape
        if target == "function":
            if not parameter.get("function"):
                problems.append(f"{name}: target is 'function' but no function named")
            continue

        # A list of identifiers moves several parameters of the input data by
        # the same value, as one parameter of the analysis
        identifiers = parameter.get("parameter", name)
        if isinstance(identifiers, str):
            identifiers = [identifiers]

        for identifier in identifiers:
            if identifier not in known:
                problems.append(
                    f"{name}: '{identifier}' is not a parameter of this case study"
                    + _suggestion(identifier, known)
                )
                continue

            problems += _validate_scope(name, parameter, known[identifier])

    return problems


def _validate_scope(name: str, parameter: dict, entries: list) -> list:
    """
    Checks that the scope a parameter is narrowed to exists.

    A parameter that names no scope changes everything the catalogue offers,
    which is the convention the writers of the input data follow, so an absent
    scope is never a problem. A scope that names something the parameter does
    not have is.

    :param str name: name the parameter has in the definition
    :param dict parameter: the parameter of the definition
    :param list entries: catalogue entries the parameter refers to
    :return: list of problems
    :rtype: list
    """
    scope = parameter.get("scope", {})
    if not scope:
        return []

    available = {}
    for entry in entries:
        for set_name, members in entry["scope"].items():
            available.setdefault(set_name, set()).update(members)

    problems = []
    for set_name, members in scope.items():
        if set_name not in available:
            problems.append(
                f"{name}: cannot be narrowed by '{set_name}', only by "
                f"{sorted(available)}"
            )
            continue
        unknown = [member for member in members if member not in available[set_name]]
        if unknown:
            problems.append(
                f"{name}: {set_name} {unknown} are not among "
                f"{sorted(available[set_name])}"
            )

    return problems


def _validate_derived(derived: dict, catalogue: list, topology: dict) -> list:
    """
    Checks the derived parameters.

    The check that matters is the last one: the equations have to determine the
    parameters they write. A system with fewer equations than unknowns means the
    study is relying on an assumption it has not written down.

    :param dict derived: the derived block of the definition
    :param list catalogue: catalogue of the case study
    :param dict topology: contents of the topology file
    :return: list of problems
    :rtype: list
    """
    known = {entry["parameter"] for entry in catalogue}

    problems = []
    for name, block in derived.items():
        symbols = block.get("symbols", {})
        if not symbols:
            problems.append(f"{name}: no symbols, so nothing varies")

        for symbol, definition in symbols.items():
            values = definition.get("values")
            if not isinstance(values, list) or not values:
                problems.append(f"{name}/{symbol}: values is missing or empty")

        aggregate = block.get("aggregate", "mean")
        if aggregate not in AGGREGATES:
            problems.append(
                f"{name}: aggregate is '{aggregate}', not one of {AGGREGATES}"
            )

        equations = block.get("defines", [])
        if not equations:
            problems.append(f"{name}: no defines, so nothing is written")
            continue

        try:
            system = parse_equations(equations, topology, set(symbols))
        except ValueError as error:
            problems.append(f"{name}: {error}")
            continue

        for reference in system["references"].values():
            if reference["parameter"] not in known:
                problems.append(
                    f"{name}: '{reference['parameter']}' is not a parameter of this "
                    "case study"
                )

        problems += _validate_determinacy(name, system)

    return problems


def _validate_determinacy(name: str, system: dict) -> list:
    """
    Checks that a system of equations determines its unknowns.

    Fewer equations than unknowns is the interesting case: it means an
    assumption is missing, and the message says how many and which, because the
    unknown that is left free is usually the assumption that was never written.

    :param str name: name of the derived block
    :param dict system: parsed system, as returned by parse_equations
    :return: list of problems
    :rtype: list
    """
    n_equations = len(system["equations"])
    unknowns = system["unknowns"]

    if len(unknowns) == n_equations:
        return []

    spelled = ", ".join(system["names"][unknown] for unknown in unknowns)

    if len(unknowns) > n_equations:
        missing = len(unknowns) - n_equations
        return [
            f"{name}: {n_equations} equations for {len(unknowns)} unknowns, so "
            f"{missing} more {'is' if missing == 1 else 'are'} needed. The unknowns "
            f"are {spelled}. An unknown left free is usually an assumption of the "
            "study that has not been written down"
        ]

    return [
        f"{name}: {n_equations} equations for {len(unknowns)} unknowns, so the "
        f"system is over-determined and may have no solution. The unknowns are "
        f"{spelled}"
    ]


def _suggestion(identifier: str, known: dict) -> str:
    """
    Suggests parameters of the catalogue that look like the one that was meant

    :param str identifier: parameter that was not found
    :param dict known: parameters of the catalogue
    :return: a sentence naming the closest matches, or an empty string
    :rtype: str
    """
    lowered = identifier.lower()
    close = [
        candidate
        for candidate in known
        if lowered in candidate.lower() or candidate.lower() in lowered
    ]

    if not close:
        return ""

    return (
        ". Did you mean " + ", ".join(f"'{name}'" for name in sorted(close)[:5]) + "?"
    )
