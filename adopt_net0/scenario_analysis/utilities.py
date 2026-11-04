import json
from pathlib import Path

# Sets of the topology a parameter can be indexed and narrowed over
INDEX_SETS = ["nodes", "carriers", "investment_periods"]


def read_topology(folder_path: Path) -> dict:
    """
    Reads the topology of an input data folder

    :param Path folder_path: input data folder of the case study
    :return: contents of the topology file
    :rtype: dict
    """
    with open(Path(folder_path) / "Topology.json", "r") as json_file:
        return json.load(json_file)


def carrier_data_file(folder_path: Path, period: str, node: str, carrier: str) -> Path:
    """
    Path of the carrier data file of one carrier at one node and period

    :param Path folder_path: input data folder of the case study
    :param str period: investment period
    :param str node: node
    :param str carrier: carrier
    :return: path of the csv file, which may not exist
    :rtype: Path
    """
    return folder_path / period / "node_data" / node / "carrier_data" / f"{carrier}.csv"


def component_folders(
    folder_path: Path, periods: list, nodes: list, per_node: bool
) -> list:
    """
    Lists the folders the json files of technologies or networks sit in:
    technology_data below every node, network_data below every period

    :param Path folder_path: input data folder of the case study
    :param list periods: investment periods
    :param list nodes: nodes, used if per_node
    :param bool per_node: True for technologies, False for networks
    :return: list of (label, folder) pairs, label being period/node or period
    :rtype: list
    """
    if per_node:
        return [
            (
                f"{period}/{node}",
                folder_path / period / "node_data" / node / "technology_data",
            )
            for period in periods
            for node in nodes
        ]
    return [(period, folder_path / period / "network_data") for period in periods]
