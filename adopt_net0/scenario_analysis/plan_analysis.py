import csv
import hashlib
import itertools
import json
import logging
import os
import re
from pathlib import Path

from scipy.stats import qmc

from .define_analysis import NAMINGS

log = logging.getLogger(__name__)

# Longest scenario name. The name is a folder, and Windows refuses paths over
# 260 characters; elsewhere only the 255 characters of one folder name count
MAX_NAME_LENGTH = 30 if os.name == "nt" else 200

# Significant figures of a sampled value, so the table and the name agree
SAMPLE_DIGITS = 6


def plan_scenarios(definition: dict) -> list:
    """
    Expands a definition into its scenarios, combined as analysis/combination
    says (grid, list or lhs) and named as analysis/naming says (short, readable
    or hashed). A name depends on the values only, so that a scenario already
    solved is recognised when the analysis is run again.

    :param dict definition: the definition, as read by read_definition
    :return: list of scenarios, each a dict with its name, a readable label and
        its values
    :rtype: list
    """
    axes = _axes(definition)
    analysis = definition["analysis"]
    combination = analysis["combination"]["value"]

    if combination == "grid":
        rows = [dict(zip(axes, values)) for values in itertools.product(*axes.values())]
    elif combination == "list":
        lengths = {name: len(values) for name, values in axes.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"combination is 'list' but the parameters have different numbers "
                f"of values: {lengths}"
            )
        rows = [dict(zip(axes, values)) for values in zip(*axes.values())]
    elif combination == "lhs":
        rows = _latin_hypercube(
            axes, analysis["n_samples"]["value"], analysis.get("seed", {}).get("value")
        )
    else:
        raise ValueError(f"combination '{combination}' is not known")

    naming = analysis.get("naming", {}).get("value", "short")
    if naming not in NAMINGS:
        raise ValueError(f"naming '{naming}' is not one of {NAMINGS}")

    short = short_names(definition)
    scenarios = [
        {"name": scenario_name(row, naming, short), "label": _label(row), "values": row}
        for row in rows
    ]

    longest = max(scenarios, key=lambda scenario: len(scenario["name"]))["name"]
    if len(longest) > MAX_NAME_LENGTH:
        raise ValueError(
            f"naming is '{naming}' but names reach {len(longest)} characters, more "
            f"than the {MAX_NAME_LENGTH} a folder can take here: '{longest}'. Give "
            f"the parameters shorter 'short' aliases, or set naming to 'hashed'"
        )

    names = [scenario["name"] for scenario in scenarios]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} scenarios share a name, so their values repeat: "
            f"{duplicates[:5]}"
        )

    log.info(f"{len(scenarios)} scenarios planned by {combination}")
    return scenarios


def scenario_name(values: dict, naming: str = "short", short: dict = None) -> str:
    """
    Names a scenario by its values: short (h2p-0.75__gam-0.5), readable (full
    parameter names) or hashed (s + ten characters of a hash)

    :param dict values: value of every parameter of the scenario
    :param str naming: 'short', 'readable' or 'hashed'
    :param dict short: alias of every parameter that has one, for 'short'
    :return: the name
    :rtype: str
    """
    if naming == "readable":
        return _label(values)
    if naming == "short":
        return _label(values, short or {})

    content = json.dumps(values, sort_keys=True)
    return "s" + hashlib.sha1(content.encode("utf-8")).hexdigest()[:10]


def short_names(definition: dict) -> dict:
    """
    Collects the 'short' alias of every parameter and every derived symbol that
    has one

    :param dict definition: the definition
    :return: dict of parameter name to its alias
    :rtype: dict
    """
    short = {
        name: parameter["short"]
        for name, parameter in definition.get("parameters", {}).items()
        if parameter.get("short")
    }
    for block in definition.get("derived", {}).values():
        for symbol, symbol_definition in block["symbols"].items():
            if symbol_definition.get("short"):
                short[symbol] = symbol_definition["short"]
    return short


def write_plan(scenarios: list, save_path: str | Path) -> Path:
    """
    Writes the scenarios to a csv file, one row per scenario

    :param list scenarios: scenarios, as returned by plan_scenarios
    :param str/Path save_path: path of the csv file
    :return: path of the csv file
    :rtype: Path
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(scenarios[0]["values"])
    with open(save_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["name", "label"] + columns)
        for scenario in scenarios:
            writer.writerow(
                [scenario["name"], scenario["label"]]
                + [scenario["values"][column] for column in columns]
            )

    return save_path


def read_plan(file_path: str | Path) -> list:
    """
    Reads the scenarios back from the csv file written by write_plan

    :param str/Path file_path: path of the csv file
    :return: list of scenarios, each a dict with its name and its values
    :rtype: list
    """
    with open(file_path, "r", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    return [
        {
            "name": row.pop("name"),
            "label": row.pop("label"),
            "values": {column: _number(value) for column, value in row.items()},
        }
        for row in rows
    ]


def _axes(definition: dict) -> dict:
    """
    Collects the values of every parameter of the analysis, the direct ones
    first and then the symbols of every derived parameter

    :param dict definition: the definition
    :return: dict of parameter name to its list of values
    :rtype: dict
    """
    axes = {
        name: parameter["values"]
        for name, parameter in definition.get("parameters", {}).items()
    }
    for block in definition.get("derived", {}).values():
        for symbol, symbol_definition in block["symbols"].items():
            if symbol in axes:
                raise ValueError(f"'{symbol}' is defined twice in the analysis")
            axes[symbol] = symbol_definition["values"]

    if not axes:
        raise ValueError("The analysis changes nothing: no parameters and no derived")

    return axes


def _latin_hypercube(axes: dict, n_samples: int, seed: int = None) -> list:
    """
    Draws a Latin hypercube over the range the values of every parameter span

    :param dict axes: dict of parameter name to its list of values
    :param int n_samples: number of scenarios to draw
    :param int seed: seed of the sampler
    :return: list of dicts of parameter name to value
    :rtype: list
    """
    lower = [min(values) for values in axes.values()]
    upper = [max(values) for values in axes.values()]
    sampler = qmc.LatinHypercube(d=len(axes), seed=seed)
    samples = qmc.scale(sampler.random(n_samples), lower, upper)

    return [
        {name: float(f"{value:.{SAMPLE_DIGITS}g}") for name, value in zip(axes, sample)}
        for sample in samples
    ]


def _label(values: dict, short: dict = None) -> str:
    """
    Spells every parameter of a scenario and its value, in the order of the
    definition

    :param dict values: value of every parameter of the scenario
    :param dict short: alias to spell a parameter by instead of its name, if any
    :return: the label
    :rtype: str
    """
    short = short or {}
    return "__".join(
        f"{_safe(short.get(parameter, parameter))}-{_safe(_format(value))}"
        for parameter, value in values.items()
    )


def _format(value) -> str:
    """
    Spells a value the way it appears in a scenario name

    :param value: the value
    :return: the value as text
    :rtype: str
    """
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _safe(text: str) -> str:
    """
    Replaces what a folder name cannot hold

    :param str text: the text
    :return: the text with only letters, digits, dots, dashes and underscores
    :rtype: str
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def _number(text: str):
    """
    Reads a value of the csv back as the number it was written as, or as text

    :param str text: the value as read from the csv
    :return: int, float or str
    """
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text
