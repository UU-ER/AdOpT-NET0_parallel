import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from adopt_net0.data_preprocessing.data_loading import CARRIER_COLUMNS

from .derived_parameters import parse_equations, solve_derived
from .utilities import carrier_data_file, component_folders, read_topology

log = logging.getLogger(__name__)

# Files of the base case that are not copied into a scenario
ANALYSIS_FILES = [
    "ScenarioAnalysis.json",
    "ParameterCatalogue.json",
    "ScenarioBuilder.html",
]

# File the values of a scenario are recorded in, inside its input data folder
SCENARIO_RECORD = "Scenario.json"


def write_scenario(
    base_path: str | Path,
    definition: dict,
    scenario: dict,
    input_path: str | Path,
    results_path: str | Path,
) -> Path:
    """
    Writes the input data folder of one scenario: a copy of the base case with
    the values of the scenario applied. set writes a value, multiply scales the
    base value. On a time series, set fixes its mean or sum and keeps its shape.
    Derived parameters are solved from their equations first.

    :param str/Path base_path: input data folder of the base case
    :param dict definition: the definition of the analysis
    :param dict scenario: the scenario, with its name and its values
    :param str/Path input_path: folder the input data of the scenario is written to
    :param str/Path results_path: folder adopt writes the results of the scenario to
    :return: the input data folder of the scenario
    :rtype: Path
    """
    base_path = Path(base_path)
    input_path = Path(input_path)

    # A folder left by a failed attempt is replaced
    if input_path.exists():
        shutil.rmtree(input_path)
    shutil.copytree(
        base_path, input_path, ignore=shutil.ignore_patterns(*ANALYSIS_FILES)
    )

    topology = read_topology(input_path)

    values = scenario["values"]

    for name, parameter in definition.get("parameters", {}).items():
        _apply(input_path, topology, parameter, values[name])

    for block in definition.get("derived", {}).values():
        _apply_derived(input_path, topology, block, values)

    _point_results(input_path, Path(results_path), scenario["name"])

    with open(input_path / SCENARIO_RECORD, "w") as json_file:
        json.dump(scenario, json_file, indent=4)

    log.info(f"Scenario {scenario['name']} written to {input_path}")
    return input_path


def _apply(input_path: Path, topology: dict, parameter: dict, value):
    """
    Applies one value of a direct parameter

    :param Path input_path: input data folder of the scenario
    :param dict topology: contents of the topology file
    :param dict parameter: the parameter, as in the definition
    :param value: its value in this scenario
    """
    target = parameter["target"]
    operation = parameter.get("operation", "set")
    scope = parameter.get("scope", {})

    identifiers = parameter["parameter"]
    if isinstance(identifiers, str):
        identifiers = [identifiers]

    for identifier in identifiers:
        if target == "config":
            _apply_config(input_path, identifier, operation, value)
        elif target == "technology":
            files = _component_files(input_path, topology, identifier, scope, True)
            _apply_component(files, identifier, operation, value)
        elif target == "network":
            files = _component_files(input_path, topology, identifier, scope, False)
            _apply_component(files, identifier, operation, value)
        elif target == "carrier_data":
            _apply_carrier_data(
                input_path,
                topology,
                identifier,
                scope,
                operation,
                value,
                parameter.get("aggregate", "mean"),
            )
        else:
            raise NotImplementedError(
                f"Parameters of target '{target}' cannot be written yet"
            )


def _apply_derived(input_path: Path, topology: dict, block: dict, values: dict):
    """
    Solves a derived parameter for this scenario and sets what it gives

    :param Path input_path: input data folder of the scenario
    :param dict topology: contents of the topology file
    :param dict block: the derived parameter, as in the definition
    :param dict values: values of the scenario, the symbols of the block among them
    """
    sample = {symbol: values[symbol] for symbol in block["symbols"]}
    resolved = solve_derived(block, sample, topology)

    system = parse_equations(block.get("defines", []), topology, set(block["symbols"]))
    references = {
        reference["written"]: reference for reference in system["references"].values()
    }

    for written, value in resolved.items():
        reference = references[written]
        if reference["parameter"] not in CARRIER_COLUMNS:
            raise NotImplementedError(
                f"'{written}': derived parameters can only write carrier data yet"
            )
        _apply_carrier_data(
            input_path,
            topology,
            reference["parameter"],
            reference["scope"],
            "set",
            value,
            block.get("aggregate", "mean"),
        )


def _apply_config(input_path: Path, identifier: str, operation: str, value):
    """
    Applies a value to a leaf of the configuration

    :param Path input_path: input data folder of the scenario
    :param str identifier: path of the leaf, e.g. optimization/typicaldays/N
    :param str operation: set or multiply
    :param value: the value
    """
    config_file = input_path / "ConfigModel.json"
    with open(config_file, "r") as json_file:
        config = json.load(json_file)

    leaf = config
    for key in identifier.split("/"):
        leaf = leaf[key]
    leaf["value"] = _operate(leaf["value"], operation, value)

    with open(config_file, "w") as json_file:
        json.dump(config, json_file, indent=4)


def _component_files(
    input_path: Path, topology: dict, identifier: str, scope: dict, per_node: bool
) -> list:
    """
    Lists the files of a technology or a network that a parameter writes to

    :param Path input_path: input data folder of the scenario
    :param dict topology: contents of the topology file
    :param str identifier: component and path, e.g. Electrolyzer/Economics/unit_capex
    :param dict scope: nodes or periods the parameter is narrowed to
    :param bool per_node: True for a technology, False for a network
    :return: list of paths of the files
    """
    component = identifier.split("/")[0]
    periods = scope.get("investment_periods", topology["investment_periods"])

    # The catalogue offers the nodes of a technology as period/node
    nodes = [node.split("/")[-1] for node in scope.get("nodes", topology["nodes"])]

    files = [
        place / f"{component}.json"
        for _, place in component_folders(input_path, periods, nodes, per_node)
        if (place / f"{component}.json").exists()
    ]
    if not files:
        raise FileNotFoundError(f"'{component}' has no file in the scope given")
    return files


def _apply_component(files: list, identifier: str, operation: str, value):
    """
    Applies a value inside the json files of a technology or a network

    :param list files: the files
    :param str identifier: component and path, e.g. Electrolyzer/Economics/unit_capex
    :param str operation: set or multiply
    :param value: the value
    """
    keys = identifier.split("/")[1:]

    for file_path in files:
        with open(file_path, "r") as json_file:
            contents = json.load(json_file)

        node = contents
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = _operate(node[keys[-1]], operation, value)

        with open(file_path, "w") as json_file:
            json.dump(contents, json_file, indent=4)


def _apply_carrier_data(
    input_path: Path,
    topology: dict,
    column: str,
    scope: dict,
    operation: str,
    value: float,
    aggregate: str,
):
    """
    Applies a value to a column of the carrier data

    :param Path input_path: input data folder of the scenario
    :param dict topology: contents of the topology file
    :param str column: the column, e.g. Import price
    :param dict scope: carriers, nodes and periods the parameter is narrowed to
    :param str operation: set or multiply
    :param float value: the value, or the factor
    :param str aggregate: what set fixes, the yearly mean or the sum
    """
    carriers = scope.get("carriers", topology["carriers"])
    nodes = scope.get("nodes", topology["nodes"])
    periods = scope.get("investment_periods", topology["investment_periods"])

    for period in periods:
        for node in nodes:
            for carrier in carriers:
                file_path = carrier_data_file(input_path, period, node, carrier)
                if not file_path.exists():
                    continue

                frame = pd.read_csv(file_path, sep=";", index_col=0)
                frame[column] = _scale_series(
                    frame[column].to_numpy(dtype=float), operation, value, aggregate
                )
                frame.to_csv(file_path, sep=";")


def _scale_series(series: np.ndarray, operation: str, value: float, aggregate: str):
    """
    Applies a value to a time series, keeping its shape

    :param np.ndarray series: the series of the base case
    :param str operation: set or multiply
    :param float value: the value, or the factor
    :param str aggregate: what set fixes, the mean or the sum
    :return: the new series
    """
    if operation == "multiply":
        return series * value

    current = series.mean() if aggregate == "mean" else series.sum()
    if current == 0:
        flat = value if aggregate == "mean" else value / len(series)
        return np.full(len(series), flat)
    return series * (value / current)


def _operate(current, operation: str, value):
    """
    Applies set or multiply to a single value

    :param current: what the base case holds
    :param str operation: set or multiply
    :param value: the value, or the factor
    :return: the new value
    """
    if operation == "multiply":
        return current * value
    return value


def _point_results(input_path: Path, results_path: Path, name: str):
    """
    Points the results and the summary of the scenario to its own folder, as a
    summary shared by scenarios solved at the same time gets corrupted

    :param Path input_path: input data folder of the scenario
    :param Path results_path: folder the results of the scenario go to
    :param str name: name of the scenario
    """
    config_file = input_path / "ConfigModel.json"
    with open(config_file, "r") as json_file:
        config = json.load(json_file)

    results_path.mkdir(parents=True, exist_ok=True)
    config["reporting"]["save_path"]["value"] = str(results_path.resolve())
    config["reporting"]["save_summary_path"]["value"] = str(results_path.resolve())
    config["reporting"]["case_name"]["value"] = name

    with open(config_file, "w") as json_file:
        json.dump(config, json_file, indent=4)
