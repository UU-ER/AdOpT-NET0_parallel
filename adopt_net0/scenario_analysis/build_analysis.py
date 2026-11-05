"""
Lists every parameter of an input data folder that a scenario analysis can
change (the catalogue), and writes ScenarioBuilder.html, a page to search the
catalogue and put an analysis together. A parameter is listed once at its widest
scope, and its scope says what it can be narrowed to.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data_preprocessing.data_loading import CARRIER_COLUMNS
from .utilities import INDEX_SETS, carrier_data_file, component_folders, read_topology

log = logging.getLogger(__name__)

# Keys of a technology or network file that are prose rather than values. They
# are skipped so that the catalogue only offers things that can be varied
SKIPPED_KEYS = ["comment"]

TEMPLATE_FILE = Path(__file__).parent / "templates" / "scenario_builder.html"

# The template carries an empty set of data between these two markers, which is
# what keeps it valid javascript on its own. Everything between them, the markers
# included, is replaced by the data of one case study
PLACEHOLDER = "/*__ADOPT_DATA__*/"
END_PLACEHOLDER = "/*__END_DATA__*/"


def scan_input_data(folder_path: str | Path) -> list:
    """
    Lists every parameter of an input data folder that a scenario can change.

    The folder has to exist and carry at least a topology file. Parts that have
    not been created yet, such as technology data before it has been copied in,
    are skipped rather than raising, so that the catalogue can also be read
    while a case study is still being built.

    :param str/Path folder_path: path to the folder containing the case study data
    :return: list of parameter entries
    :rtype: list
    """
    folder_path = Path(folder_path)

    topology_file = folder_path / "Topology.json"
    if not topology_file.exists():
        raise FileNotFoundError(
            f"No Topology.json in '{folder_path}', so the folder is not an input "
            "data folder of a case study"
        )

    topology = read_topology(folder_path)

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


def write_builder(folder_path: str | Path, save_path: str | Path = None) -> Path:
    """
    Writes the scenario builder of a case study to an html file.

    The catalogue is read from the case study and put inside the page, so the
    file that comes out is self contained and belongs to that case study only.
    Scanning the folder again and writing it again is how the page is kept up to
    date after the case study changes.

    :param str/Path folder_path: path to the folder containing the case study data
    :param str/Path save_path: file the page is written to, by default
        ScenarioBuilder.html inside the input data folder
    :return: path of the page that was written
    :rtype: Path
    """
    folder_path = Path(folder_path)

    topology = read_topology(folder_path)

    data = {
        "catalogue": scan_input_data(folder_path),
        "topology": {key: topology.get(key, []) for key in INDEX_SETS},
        "case_name": folder_path.name,
    }

    with open(TEMPLATE_FILE, "r", encoding="utf-8") as html_file:
        page = html_file.read()

    start = page.find(PLACEHOLDER)
    end = page.find(END_PLACEHOLDER)
    if start < 0 or end < start:
        raise ValueError(
            f"{TEMPLATE_FILE.name} has no {PLACEHOLDER} ... {END_PLACEHOLDER} for "
            "the data of the case study"
        )

    # A parameter whose name contains the end of a script tag would otherwise
    # close the script the data sits in
    encoded = json.dumps(data, default=str).replace("</", "<\\/")
    page = page[:start] + encoded + page[end + len(END_PLACEHOLDER) :]

    if save_path is None:
        save_path = folder_path / "ScenarioBuilder.html"
    save_path = Path(save_path)

    with open(save_path, "w", encoding="utf-8") as html_file:
        html_file.write(page)

    log.info(
        f"Scenario builder of '{folder_path.name}' written to {save_path}, "
        f"{len(data['catalogue'])} parameters"
    )
    return save_path


def create_scenario_builder(path: str | Path) -> Path:
    """
    Writes the parameter catalogue and the builder page of a case study into its
    input data folder. The page defines the analysis: its json is saved as
    ScenarioAnalysis.json in the same folder, where ScenarioHub reads it.

    :param str/Path path: input data folder of the case study
    :return: path of the page that was written
    :rtype: Path
    """
    write_catalogue(path)
    page = write_builder(path)

    log_msg = f"--- Builder page written to {page} ---"
    print(log_msg)
    log.info(log_msg)
    return page


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
    for key in INDEX_SETS:
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
                file_path = carrier_data_file(folder_path, period, node, carrier)
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
        per_node=False,
    )


def _scan_component_files(
    folder_path: Path,
    topology: dict,
    target: str,
    per_node: bool,
) -> list:
    """
    Lists the values of the json files of one kind of component.

    Technologies and networks differ only in where their files sit and in what
    their scope is, so both go through here.

    :param Path folder_path: path to the folder containing the case study data
    :param dict topology: contents of the topology file
    :param str target: technology or network
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

    for label, place in component_folders(folder_path, periods, nodes, per_node):
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
