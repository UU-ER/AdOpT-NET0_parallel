"""
Reads an input data folder and lists the parameters a scenario analysis can
change.

A scenario analysis starts from one base case that has already been built, and
every scenario is that base case with a few values changed. So the question the
user has to answer first is what can be changed at all, and that question is
answered by the input data folder itself: the configuration file carries its own
descriptions and options, the carrier data files have a fixed set of columns,
and the technology and network files are json trees whose leaves are values.

This module walks all of that and returns one entry per parameter. The entries
are what a user interface offers to choose from, and they are also what the
definition of a scenario analysis is checked against, so that a parameter that
does not exist is caught before anything is run rather than after.

A parameter is listed at the widest scope it has, not once per node and
carrier. A carrier data column exists for every node and every investment
period, and a scenario that does not say otherwise changes all of them, which
is the same convention ``fill_carrier_data`` follows with its empty selectors.
The ``scope`` of an entry lists what it can be narrowed to.

Examples::

    from adopt_net0.scenario_analysis import scan_input_data

    catalogue = scan_input_data("path to input data")
    prices = [entry for entry in catalogue if "price" in entry["parameter"]]
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_preprocessing.data_loading import CARRIER_COLUMNS

log = logging.getLogger(__name__)

# Keys of a technology or network file that are prose rather than values. They
# are skipped so that the catalogue only offers things that can be varied
SKIPPED_KEYS = ["comment"]


def scan_input_data(folder_path: str | Path) -> list:
    """
    Lists every parameter of an input data folder that a scenario can change.

    The folder has to exist and carry at least a topology file. Parts that have
    not been created yet, such as technology data before it has been copied in,
    are skipped rather than raising, so that the catalogue can also be read
    while a case study is still being built.

    :param str/Path folder_path: path to the folder containing the case study data
    :return: list of parameter entries, see the module docstring
    :rtype: list
    """
    folder_path = Path(folder_path)

    topology_file = folder_path / "Topology.json"
    if not topology_file.exists():
        raise FileNotFoundError(
            f"No Topology.json in '{folder_path}', so the folder is not an input "
            "data folder of a case study"
        )

    with open(topology_file, "r") as json_file:
        topology = json.load(json_file)

    catalogue = []
    catalogue += _scan_configuration(folder_path)
    catalogue += _scan_topology(topology)
    catalogue += _scan_carrier_data(folder_path, topology)
    catalogue += _scan_technologies(folder_path, topology)
    catalogue += _scan_networks(folder_path, topology)

    log.info(f"Catalogue of '{folder_path}': {len(catalogue)} parameters")
    return catalogue


def write_catalogue(folder_path: str | Path, save_path: str | Path = None) -> list:
    """
    Writes the catalogue of an input data folder to a json file

    :param str/Path folder_path: path to the folder containing the case study data
    :param str/Path save_path: file the catalogue is written to, by default
        ParameterCatalogue.json next to the input data folder
    :return: list of parameter entries
    :rtype: list
    """
    folder_path = Path(folder_path)
    catalogue = scan_input_data(folder_path)

    if save_path is None:
        save_path = folder_path / "ParameterCatalogue.json"
    save_path = Path(save_path)

    with open(save_path, "w") as json_file:
        json.dump(catalogue, json_file, indent=4, default=str)

    log.info(f"Catalogue written to {save_path}")
    return catalogue


def _scan_configuration(folder_path: Path) -> list:
    """
    Lists the parameters of the model configuration file.

    Every leaf of the configuration carries its own description and, where the
    value is restricted, its options, so the entries come out documented without
    anything having to be written here.

    :param Path folder_path: path to the folder containing the case study data
    :return: list of parameter entries
    :rtype: list
    """
    config_file = folder_path / "ConfigModel.json"
    if not config_file.exists():
        return []

    with open(config_file, "r") as json_file:
        config = json.load(json_file)

    entries = []
    for path, leaf in _walk_configuration(config):
        entries.append(
            {
                "parameter": path,
                "target": "config",
                "description": leaf.get("description", ""),
                "value_type": _value_type(leaf.get("value")),
                "options": leaf.get("options"),
                "current": leaf.get("value"),
                "scope": {},
            }
        )

    return entries


def _walk_configuration(node: dict, prefix: str = ""):
    """
    Walks the configuration tree and yields its leaves.

    A leaf is a dict carrying a "value" key, which is the convention the
    configuration template follows. Everything above it is a group.

    :param dict node: subtree of the configuration
    :param str prefix: path of the subtree, empty at the root
    :return: yields (path, leaf) pairs
    """
    for key, value in node.items():
        if not isinstance(value, dict):
            continue

        path = f"{prefix}/{key}" if prefix else key

        if "value" in value:
            yield path, value
        else:
            yield from _walk_configuration(value, path)


def _scan_topology(topology: dict) -> list:
    """
    Lists the parts of the topology a scenario can change.

    These are the sets of the model rather than values in it, so a scenario that
    changes one of them changes which files have to exist and has to be applied
    before the data is copied in. They are listed as structural for that reason.

    :param dict topology: contents of the topology file
    :return: list of parameter entries
    :rtype: list
    """
    entries = []
    for key in ["nodes", "carriers", "investment_periods"]:
        entries.append(
            {
                "parameter": f"topology/{key}",
                "target": "structural",
                "description": f"The {key} of the case study. Changing this changes "
                "which data files the case study needs, so it is applied before "
                "the data is copied in.",
                "value_type": "set",
                "options": None,
                "current": topology.get(key),
                "scope": {},
            }
        )

    return entries


def _scan_carrier_data(folder_path: Path, topology: dict) -> list:
    """
    Lists the carrier data columns, one entry per carrier and column.

    A column exists for every node and every investment period, and a scenario
    that does not narrow the scope changes all of them, so the entry is made at
    that level and the nodes and periods are offered as a scope. What the base
    case holds is summarised rather than listed, both because a column is a
    year of hourly values and because the summary says whether the value differs
    between nodes, which is what decides whether the scope matters.

    :param Path folder_path: path to the folder containing the case study data
    :param dict topology: contents of the topology file
    :return: list of parameter entries
    :rtype: list
    """
    nodes = topology.get("nodes", [])
    carriers = topology.get("carriers", [])
    periods = topology.get("investment_periods", [])

    entries = []
    for carrier in carriers:
        # Read every file of this carrier once, rather than once per column
        frames = {}
        for period in periods:
            for node in nodes:
                file_path = (
                    folder_path
                    / period
                    / "node_data"
                    / node
                    / "carrier_data"
                    / f"{carrier}.csv"
                )
                if file_path.exists():
                    frames[(period, node)] = pd.read_csv(
                        file_path, sep=";", index_col=0
                    )

        if not frames:
            continue

        for column in CARRIER_COLUMNS:
            entries.append(
                {
                    "parameter": column,
                    "target": "carrier_data",
                    "description": f"{column} of {carrier}, as an hourly series.",
                    "value_type": "timeseries",
                    "options": None,
                    "current": _summarise_column(frames, column),
                    "scope": {
                        "carriers": [carrier],
                        "nodes": nodes,
                        "investment_periods": periods,
                    },
                }
            )

    return entries


def _summarise_column(frames: dict, column: str) -> dict:
    """
    Summarises one carrier data column over the nodes and periods it exists in.

    A column that is empty everywhere is reported as such, which matters: it
    means the base case does not use it, and a scenario that sets it is adding
    something rather than changing it.

    :param dict frames: carrier data frames, keyed by (period, node)
    :param str column: name of the column to summarise
    :return: dict describing what the base case holds
    :rtype: dict
    """
    per_node = {}
    for (period, node), frame in frames.items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            per_node[f"{period}/{node}"] = float(np.nanmean(values))

    if not per_node:
        return {"defined": False}

    means = np.array(list(per_node.values()))
    return {
        "defined": True,
        "mean": float(means.mean()),
        "min_over_nodes": float(means.min()),
        "max_over_nodes": float(means.max()),
        "differs_between_nodes": bool(means.max() - means.min() > 1e-9),
        "defined_in": len(per_node),
    }


def _scan_technologies(folder_path: Path, topology: dict) -> list:
    """
    Lists the values inside the technology files.

    The same technology is copied into every node that uses it, so it is listed
    once with those nodes as its scope. A technology that appears in no node has
    no file to read and is not listed.

    :param Path folder_path: path to the folder containing the case study data
    :param dict topology: contents of the topology file
    :return: list of parameter entries
    :rtype: list
    """
    return _scan_component_files(
        folder_path,
        topology,
        target="technology",
        sub_folder="technology_data",
        per_node=True,
    )


def _scan_networks(folder_path: Path, topology: dict) -> list:
    """
    Lists the values inside the network files.

    Networks are defined per investment period rather than per node, so their
    scope is the periods they appear in.

    :param Path folder_path: path to the folder containing the case study data
    :param dict topology: contents of the topology file
    :return: list of parameter entries
    :rtype: list
    """
    return _scan_component_files(
        folder_path,
        topology,
        target="network",
        sub_folder="network_data",
        per_node=False,
    )


def _scan_component_files(
    folder_path: Path,
    topology: dict,
    target: str,
    sub_folder: str,
    per_node: bool,
) -> list:
    """
    Lists the values of the json files of one kind of component.

    Technologies and networks differ only in where their files sit and in what
    their scope is, so both go through here.

    :param Path folder_path: path to the folder containing the case study data
    :param dict topology: contents of the topology file
    :param str target: which writer applies these parameters
    :param str sub_folder: folder the files sit in, below the period or the node
    :param bool per_node: True if the files sit below a node, False if below the
        investment period
    :return: list of parameter entries
    :rtype: list
    """
    nodes = topology.get("nodes", [])
    periods = topology.get("investment_periods", [])

    # Component name to the places it was found in, and to the contents of the
    # first file found, which is what the values are read from
    found = {}
    contents = {}

    for period in periods:
        places = (
            [folder_path / period / "node_data" / node / sub_folder for node in nodes]
            if per_node
            else [folder_path / period / sub_folder]
        )
        labels = [f"{period}/{node}" for node in nodes] if per_node else [period]

        for place, label in zip(places, labels):
            if not place.is_dir():
                continue
            for file_path in sorted(place.glob("*.json")):
                name = file_path.stem
                found.setdefault(name, []).append(label)
                if name not in contents:
                    with open(file_path, "r") as json_file:
                        contents[name] = json.load(json_file)

    entries = []
    for name in sorted(found):
        for path, value in _walk_component(contents[name]):
            entries.append(
                {
                    "parameter": f"{name}/{path}",
                    "target": target,
                    "description": f"{path} of {name}.",
                    "value_type": _value_type(value),
                    "options": None,
                    "current": value,
                    "scope": (
                        {"nodes": sorted(set(found[name]))}
                        if per_node
                        else {"investment_periods": sorted(set(found[name]))}
                    ),
                }
            )

    return entries


def _walk_component(node: dict, prefix: str = ""):
    """
    Walks a technology or network file and yields its leaves.

    Unlike the configuration, these files are plain nested values with no
    description around them, so a leaf is anything that is not a dict. Keys that
    hold prose are skipped, as they are not something a scenario varies.

    :param dict node: subtree of the file
    :param str prefix: path of the subtree, empty at the root
    :return: yields (path, value) pairs
    """
    for key, value in node.items():
        if key in SKIPPED_KEYS:
            continue

        path = f"{prefix}/{key}" if prefix else key

        if isinstance(value, dict):
            yield from _walk_component(value, path)
        else:
            yield path, value


def _value_type(value) -> str:
    """
    Names the kind of a value, so that a user interface knows what to offer

    :param value: value to classify
    :return: one of scalar, list, text or unknown
    :rtype: str
    """
    if isinstance(value, bool):
        return "scalar"
    if isinstance(value, (int, float)):
        return "scalar"
    if isinstance(value, list):
        return "list"
    if isinstance(value, str):
        return "text"
    return "unknown"
