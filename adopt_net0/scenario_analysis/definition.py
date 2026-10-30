"""
Reads and checks the definition of a scenario analysis.

A scenario analysis is defined in a json file next to the case study, and it
says three things: which parameters change, which values they take, and how
those values are combined into scenarios. Parameters come in three kinds, and
the kind decides how much of the definition can be checked before anything runs.

**Direct** parameters name something that exists in the input data folder, so
the catalogue knows about them and a name that does not exist is caught here
rather than after a sweep has started.

**Derived** parameters do not exist in the input data. They are quantities of
the study rather than of the model, such as the total demand of a system whose
demand is spread over several nodes, and they are defined by equations that
relate them to parameters that do exist. Writing them as equations rather than
as code has one property worth the trouble: the equations have to determine the
parameters they write, so an assumption that would otherwise sit unwritten
inside a function has to be stated. A demand split over four nodes needs four
equations, and if the study only names three, this module says so.

**Functions** are the escape hatch, for the cases no equation reaches: a profile
that is picked from a library and rescaled, or anything that reads a file. They
are marked as such in the definition so that a reader of the study can see where
the part that can be checked ends.

An equation over a parameter that is a time series constrains its yearly mean or
sum, never its shape. The shape comes from the base case and is scaled, which is
what keeps the operation well defined.

Examples::

    from adopt_net0.scenario_analysis import read_definition, validate_definition

    definition = read_definition("path to ScenarioAnalysis.json")
    problems = validate_definition(definition, "path to input data")
"""

import json
import logging
import re
from pathlib import Path

import sympy

from .catalogue import scan_input_data

log = logging.getLogger(__name__)

# Sets a parameter can be indexed over. An index of an equation is resolved by
# looking it up in these, so that the order the indices are written in does not
# matter and an index that belongs to no set is caught
INDEX_SETS = ["nodes", "carriers", "investment_periods"]

# Targets a parameter of the definition can have. The first five are applied by
# writing to the input data, the last one by calling a registered function
TARGETS = ["config", "carrier_data", "technology", "network", "structural", "function"]

OPERATIONS = ["set", "multiply"]
AGGREGATES = ["mean", "sum"]
COMBINATIONS = ["grid", "list", "lhs"]


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

    with open(folder_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)

    problems = []
    problems += _validate_analysis(definition.get("analysis", {}))
    problems += _validate_parameters(definition.get("parameters", {}), catalogue)
    problems += _validate_derived(definition.get("derived", {}), catalogue, topology)

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

        identifier = parameter.get("parameter", name)
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


def parse_equations(equations: list, topology: dict, free_symbols: set) -> dict:
    """
    Parses the equations of a derived parameter.

    A reference to a parameter of the case study is written as it is indexed,
    ``Demand[hydrogen, Large_cluster1]``, and the indices are resolved against
    the topology rather than by position, so they can be written in any order
    and an index that belongs to no set is caught. A sum over a set is written
    as a comprehension and is expanded before parsing.

    The symbols that are left once the free symbols are taken out are the
    unknowns: the parameters of the case study the equations write, plus any
    intermediate quantity the study names along the way.

    :param list equations: the equations, as strings
    :param dict topology: contents of the topology file
    :param set free_symbols: names of the symbols the analysis gives values to
    :return: dict with the parsed equations, the unknowns and the references
    :rtype: dict
    """
    references = {}
    names = {}
    parsed = []

    for equation in equations:
        expanded = _expand_comprehensions(equation, topology)
        encoded = _encode_references(expanded, topology, references)

        if "==" not in encoded:
            raise ValueError(f"equation '{equation}' has no '=='")

        left, right = encoded.split("==", 1)
        try:
            parsed.append(sympy.sympify(left) - sympy.sympify(right))
        except (sympy.SympifyError, SyntaxError, TypeError) as error:
            raise ValueError(f"equation '{equation}' does not parse: {error}")

    unknowns = []
    for expression in parsed:
        for symbol in expression.free_symbols:
            if str(symbol) in free_symbols or symbol in unknowns:
                continue
            unknowns.append(symbol)

    unknowns.sort(key=str)
    for symbol in unknowns:
        reference = references.get(str(symbol))
        names[symbol] = reference["written"] if reference else str(symbol)

    return {
        "equations": parsed,
        "unknowns": unknowns,
        "references": references,
        "names": names,
    }


def solve_derived(block: dict, sample: dict, topology: dict) -> dict:
    """
    Solves the equations of a derived parameter for one scenario.

    The free symbols are given their values of this scenario, which makes the
    system linear in the unknowns, and what comes out is the value every
    parameter of the case study has to be set to.

    :param dict block: one block of the derived section of the definition
    :param dict sample: value of every free symbol of the block in this scenario
    :param dict topology: contents of the topology file
    :return: dict mapping a written reference to the value it takes
    :rtype: dict
    """
    system = parse_equations(
        block.get("defines", []), topology, set(block.get("symbols", {}))
    )

    substituted = [
        expression.subs({sympy.Symbol(name): value for name, value in sample.items()})
        for expression in system["equations"]
    ]

    solution = sympy.solve(substituted, system["unknowns"], dict=True)
    if not solution:
        raise ValueError(
            "The equations have no solution for "
            + ", ".join(f"{name} = {value}" for name, value in sample.items())
        )

    resolved = {}
    for symbol, value in solution[0].items():
        reference = system["references"].get(str(symbol))
        if reference is None:
            # An intermediate quantity of the study, not something to write
            continue
        resolved[reference["written"]] = float(value)

    return resolved


def _expand_comprehensions(equation: str, topology: dict) -> str:
    """
    Writes a sum over a set out as an explicit sum.

    ``sum(Demand[hydrogen, n] for n in nodes)`` over three nodes becomes the sum
    of the three terms, so that what is parsed afterwards is plain arithmetic.

    :param str equation: the equation as written
    :param dict topology: contents of the topology file
    :return: the equation with its comprehensions expanded
    :rtype: str
    """
    pattern = re.compile(r"\bsum\s*\(")

    while True:
        match = pattern.search(equation)
        if match is None:
            return equation

        start = match.end() - 1
        end = _matching_parenthesis(equation, start)
        inside = equation[start + 1 : end]

        comprehension = re.match(
            r"^(.*?)\s+for\s+(\w+)\s+in\s+(\w+)\s*$", inside, re.DOTALL
        )
        if comprehension is None:
            raise ValueError(
                f"'sum({inside})' is not a sum over a set. Write it as "
                "'sum(<expression> for <index> in <set>)'"
            )

        expression, index, set_name = comprehension.groups()
        if set_name not in topology:
            raise ValueError(
                f"'{set_name}' is not a set of the topology, which has "
                f"{[key for key in INDEX_SETS if key in topology]}"
            )

        terms = [
            re.sub(rf"\b{re.escape(index)}\b", str(member), expression)
            for member in topology[set_name]
        ]
        equation = (
            equation[: match.start()]
            + "("
            + " + ".join(terms)
            + ")"
            + equation[end + 1 :]
        )


def _matching_parenthesis(text: str, start: int) -> int:
    """
    Finds the parenthesis that closes the one at a position

    :param str text: text to scan
    :param int start: position of the opening parenthesis
    :return: position of the closing parenthesis
    :rtype: int
    """
    depth = 0
    for position in range(start, len(text)):
        if text[position] == "(":
            depth += 1
        elif text[position] == ")":
            depth -= 1
            if depth == 0:
                return position

    raise ValueError(f"'{text}' has an unclosed parenthesis")


def _encode_references(equation: str, topology: dict, references: dict) -> str:
    """
    Replaces the indexed references of an equation by plain symbols.

    ``Demand[hydrogen, Large_cluster1]`` is not something sympy can read, so it
    becomes a single symbol and what it stood for is recorded, which is also how
    a solved value finds its way back to the parameter it belongs to.

    :param str equation: the equation, with its comprehensions already expanded
    :param dict topology: contents of the topology file
    :param dict references: record of the references seen so far, added to here
    :return: the equation with its references encoded
    :rtype: str
    """

    def encode(match):
        parameter = match.group(1)
        indices = [index.strip() for index in match.group(2).split(",")]
        scope = _resolve_indices(indices, topology)

        symbol = "ref_" + re.sub(r"\W", "_", parameter + "_" + "_".join(indices))
        written = f"{parameter}[{', '.join(indices)}]"

        if symbol in references and references[symbol]["written"] != written:
            raise ValueError(
                f"'{written}' and '{references[symbol]['written']}' encode to the "
                "same symbol, rename one of them"
            )

        references[symbol] = {
            "parameter": parameter,
            "scope": scope,
            "written": written,
        }
        return symbol

    return re.sub(r"(\w+)\s*\[([^\]]*)\]", encode, equation)


def _resolve_indices(indices: list, topology: dict) -> dict:
    """
    Says which set each index of a reference belongs to.

    Resolving by membership rather than by position means the indices can be
    written in any order, and that an index that is not a member of anything is
    caught where it is written rather than when the value is applied.

    :param list indices: the indices, as written
    :param dict topology: contents of the topology file
    :return: dict mapping a set name to the member that was named
    :rtype: dict
    """
    scope = {}
    for index in indices:
        for set_name in INDEX_SETS:
            if index in topology.get(set_name, []):
                scope[set_name] = [index]
                break
        else:
            raise ValueError(
                f"'{index}' is not a node, a carrier or an investment period of "
                "this case study"
            )

    return scope


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
