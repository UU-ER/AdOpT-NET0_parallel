"""
Equations of derived parameters: parsed with sympy and solved for each scenario.
An equation over a time series fixes its mean or sum, the shape comes from the
base case.
"""

import re

import sympy

from .utilities import INDEX_SETS


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
