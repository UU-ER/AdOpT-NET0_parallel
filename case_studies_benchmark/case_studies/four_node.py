"""
Four-nodal hydrogen system with a two-pressure pipeline network.

Ported from the four_node_configuration case study of the AdOpT-NET0-RegToNation
repository, with the Latin hypercube parameters frozen at the middle of their
sampling ranges, so that the case is static and reproducible.

Two large industrial clusters and two small clusters have a hydrogen demand.
Hydrogen can be imported at the large clusters, produced by electrolysis at all
nodes, stored, and transported over a low pressure distribution network and a
high pressure transmission network, both fully meshed.

The case exposes the knobs that drive the size and the difficulty of the
resulting problem, so that their effect on resource usage can be measured:

- ``typicaldays``: number of typical days, 0 for full resolution
- ``electricity_price``: constant or fluctuating import price
- ``hydrogen_demand``: constant or fluctuating demand
- ``pipeline_capex``: with or without a fixed CAPEX term, which decides whether
  the arc CAPEX needs a big-M disjunction
- ``bidirectional_precise``: precise or relaxed formulation of bidirectionality
- ``pipeline_size_min``: a minimum size on the transmission pipeline, which adds
  a disjunction per arc
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import adopt_net0 as adopt
import adopt_net0.data_preprocessing as dp
from adopt_net0.utilities import GUROBI_PARAMETERS
from adopt_net0.data_preprocessing.data_loading import import_jrc_climate_data

NAME = "four_node"

DATA_PATH = Path(__file__).parent / "data" / "four_node"

LARGE_NODES = ["Large_cluster1", "Large_cluster2"]
SMALL_NODES = ["Small_cluster1", "Small_cluster2"]
NODES = LARGE_NODES + SMALL_NODES

TIMESTEPS = 8760

# Latin hypercube parameters of the original study, frozen at the middle of
# their ranges. The ranges they were sampled from are given in the comments.
STATIC_PARAMETERS = {
    "total_demand_TWh": 10,  # [5, 10, 15]
    "demand_level_ratio": 15,  # [8, 15, 20]
    "unbalance_ratio": 3,  # [2, 3, 5]
    "import_availability_ratio": 0.3,  # [0, 0.2, 0.3, 0.4, 0.6]
    "electricity_price_avg": 100,  # [20, 50, 100, 150, 250]
    "electricity_standard_dev": 50,  # [10, 50, 100]
    "electricity_availability_small": 50,  # [30, 50, 100]
    "electricity_availability_large": 1000,  # [500, 1000, 1500]
    "hydrogen_import_price": 225,  # [150, 200, 250, 300]
}


def setup(
    input_data_path: Path,
    results_path: Path,
    typicaldays: int = 20,
    typicaldays_method: int = 1,
    electricity_price: str = "fluctuating",
    hydrogen_demand: str = "fluctuating",
    pipeline_capex: str = "fixed_plus_linear",
    bidirectional_precise: int = 0,
    pipeline_size_min: int = 250,
    storage: str = "on",
    storage_precise: int = 0,
    pv: str = "on",
    mipgap: float = 0.01,
    time_limit: float = 2,
    threads: int = 0,
    solver: str = "gurobi",
    method: int = -1,
    crossover: int = -1,
    barhomogeneous: int = -1,
    scaleflag: int = -1,
    concurrentmethod: int = -1,
    nodemethod: int = -1,
    presolve: int = -1,
    cuts: int = -1,
    cutpasses: int = -1,
    mipfocus: int = 0,
    heuristics: float = 0.05,
    norelheurtime: float = 0,
    numericfocus: int = 0,
    lpwarmstart: int = 0,
    branchdir: int = 0,
    sampling_interval: float = 0.5,
    case_name: str = None,
):
    """
    Creates the input data of the case study and switches profiling on.

    The folder structure, the technology and network data and the climate data
    are only created the first time. The carrier data and the component
    specifications are rewritten on every call, as they carry the knobs.

    :param Path input_data_path: folder the input data is written to
    :param Path results_path: folder the results are written to
    :param int typicaldays: number of typical days, 0 for full resolution
    :param int typicaldays_method: 1 clusters both the model and the data,
        2 clusters the model but keeps the data at full resolution
    :param str electricity_price: "constant" or "fluctuating"
    :param str hydrogen_demand: "constant" or "fluctuating"
    :param str pipeline_capex: "linear" or "fixed_plus_linear"
    :param int bidirectional_precise: 1 for the precise formulation
    :param int pipeline_size_min: minimum size of the transmission pipeline
    :param str storage: "on" or "off", "off" removes every storage from the
        model instead of only making it unattractive
    :param int storage_precise: 1 forbids simultaneous charging and
        discharging with a binary per storage and timestep
    :param str pv: "on" or "off". Photovoltaics carry the solar profile, which
        is the last source of time variability once the price and the demand
        are constant. Switching them off as well makes every hour identical,
        so that storage has no reason to exist
    :param float mipgap: MILP gap of the solver
    :param float time_limit: solver time limit in hours
    :param int threads: number of threads the solver may use, 0 for every core
        of the machine. This is what a job would ask a cluster for
    :param str solver: solver used
    :param int method: gurobi Method, the algorithm of the root relaxation.
        The root of a MIP runs single threaded and holds a large share of the
        time, so it is the option most likely to interact with the thread count
    :param int crossover: gurobi Crossover, 0 skips it after the barrier
    :param int barhomogeneous: gurobi BarHomogeneous
    :param int scaleflag: gurobi ScaleFlag
    :param int concurrentmethod: gurobi ConcurrentMethod
    :param int nodemethod: gurobi NodeMethod, the algorithm at the nodes
    :param int presolve: gurobi Presolve
    :param int cuts: gurobi Cuts
    :param int cutpasses: gurobi CutPasses, the maximum number of cut
        rounds at the root. The first round carries nearly all of the
        bound on this model family and the rest cost a third of the solve
    :param int mipfocus: gurobi MIPFocus
    :param float heuristics: gurobi Heuristics
    :param float norelheurtime: gurobi NoRelHeurTime in seconds
    :param int numericfocus: gurobi NumericFocus
    :param int lpwarmstart: gurobi LPWarmStart
    :param int branchdir: gurobi BranchDir
    :param float sampling_interval: sampling interval of the resource monitor
    :param str case_name: name added to the results folder
    """
    input_data_path = Path(input_data_path)
    results_path = Path(results_path)
    input_data_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    is_new = not (input_data_path / "Topology.json").exists()

    if is_new:
        dp.create_optimization_templates(input_data_path)
        _write_topology(input_data_path)

    _write_configuration(
        input_data_path,
        results_path,
        typicaldays=typicaldays,
        typicaldays_method=typicaldays_method,
        mipgap=mipgap,
        time_limit=time_limit,
        threads=threads,
        solver=solver,
        gurobi_options={
            "method": method,
            "crossover": crossover,
            "barhomogeneous": barhomogeneous,
            "scaleflag": scaleflag,
            "concurrentmethod": concurrentmethod,
            "nodemethod": nodemethod,
            "presolve": presolve,
            "cuts": cuts,
            "cutpasses": cutpasses,
            "mipfocus": mipfocus,
            "heuristics": heuristics,
            "norelheurtime": norelheurtime,
            "numericfocus": numericfocus,
            "lpwarmstart": lpwarmstart,
            "branchdir": branchdir,
        },
        sampling_interval=sampling_interval,
        case_name=case_name,
    )

    if is_new:
        _write_node_locations(input_data_path)

    # The technologies are rewritten on every call, as the storage knob
    # changes which of them exist at all
    _write_technologies(input_data_path, storage=storage, pv=pv)

    if is_new:
        _write_networks(input_data_path)

    _write_component_specifications(
        input_data_path,
        pipeline_capex=pipeline_capex,
        bidirectional_precise=bidirectional_precise,
        pipeline_size_min=pipeline_size_min,
        storage=storage,
        storage_precise=storage_precise,
    )

    _write_carrier_data(
        input_data_path,
        electricity_price=electricity_price,
        hydrogen_demand=hydrogen_demand,
    )

    if is_new:
        _write_climate_data(input_data_path)

    return input_data_path


def _write_climate_data(input_data_path: Path):
    """
    Loads climate data for every node from the JRC api.

    The synthetic coordinates of the original study place Small_cluster1 in the
    North Sea, where the api has no data and answers with a 400. For any node
    the api rejects, the climate data of the nearest node that did succeed is
    reused, so that the photovoltaics can still be fitted.

    :param Path input_data_path: folder the input data is written to
    """
    node_data_path = input_data_path / "period1" / "node_data"
    coordinates = pd.read_csv(input_data_path / "NodeLocations.csv", sep=";")
    coordinates = coordinates.set_index(coordinates.columns[0])

    with open(input_data_path / "Topology.json") as json_file:
        year = int(json.load(json_file)["start_date"].split("-")[0])

    # The api is called per node rather than through
    # load_climate_data_from_api, which stops at the first node it cannot fetch
    loaded = []
    failed = []
    for node in NODES:
        output_file = node_data_path / node / "ClimateData.csv"
        try:
            data = import_jrc_climate_data(
                coordinates.at[node, "lon"],
                coordinates.at[node, "lat"],
                year,
                coordinates.at[node, "alt"],
            )
            climate_data = pd.read_csv(output_file, sep=";")
            for column, value in data["dataframe"].items():
                climate_data[column] = value.values[: len(climate_data)]
            climate_data.to_csv(output_file, index=False, sep=";")
            loaded.append(node)
        except Exception as error:
            print(f"[CLIMATE] {node} rejected by the api ({error}), using a fallback")
            failed.append(node)

    if not loaded:
        raise RuntimeError("The climate api did not answer for any node")

    for node in failed:
        nearest = min(
            loaded,
            key=lambda candidate: _haversine(
                coordinates.at[node, "lon"],
                coordinates.at[node, "lat"],
                coordinates.at[candidate, "lon"],
                coordinates.at[candidate, "lat"],
            ),
        )
        climate_data = pd.read_csv(
            node_data_path / nearest / "ClimateData.csv", sep=";", index_col=0
        )
        climate_data.to_csv(node_data_path / node / "ClimateData.csv", sep=";")
        print(f"[CLIMATE] {node} uses the climate data of {nearest}")


def _write_topology(input_data_path: Path):
    """
    Writes the topology of the case study

    :param Path input_data_path: folder the input data is written to
    """
    with open(input_data_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)

    topology["nodes"] = NODES
    topology["carriers"] = ["electricity", "hydrogen"]
    topology["investment_periods"] = ["period1"]

    with open(input_data_path / "Topology.json", "w") as json_file:
        json.dump(topology, json_file, indent=4)


def _write_configuration(
    input_data_path: Path,
    results_path: Path,
    typicaldays: int,
    typicaldays_method: int,
    mipgap: float,
    time_limit: float,
    threads: int,
    solver: str,
    gurobi_options: dict,
    sampling_interval: float,
    case_name: str,
):
    """
    Writes the model configuration, including the profiling settings

    :param Path input_data_path: folder the input data is written to
    :param Path results_path: folder the results are written to
    :param int typicaldays: number of typical days, 0 for full resolution
    :param int typicaldays_method: clustering method, 1 or 2
    :param float mipgap: MILP gap of the solver
    :param float time_limit: solver time limit in hours
    :param str solver: solver used
    :param dict gurobi_options: solver options passed to the configuration
        untouched. The keys are those of ConfigModel.json, which
        GUROBI_PARAMETERS in adopt_net0/utilities.py maps to the gurobi names
    :param float sampling_interval: sampling interval of the resource monitor
    :param str case_name: name added to the results folder
    """
    with open(input_data_path / "ConfigModel.json", "r") as json_file:
        configuration = json.load(json_file)

    configuration["optimization"]["typicaldays"]["N"]["value"] = typicaldays
    if typicaldays != 0:
        if typicaldays_method not in [1, 2]:
            raise ValueError("typicaldays_method must be 1 or 2")
        configuration["optimization"]["typicaldays"]["method"][
            "value"
        ] = typicaldays_method
    configuration["optimization"]["objective"]["value"] = "costs"

    # Checked against the options adopt knows how to hand to gurobi, rather
    # than against the configuration on disk: the input data folder is written
    # once and reused, so a configuration created before an option existed is
    # still there and its key has to be created rather than only set
    for option, value in gurobi_options.items():
        if option not in GUROBI_PARAMETERS:
            raise KeyError(
                f"'{option}' is not a gurobi option adopt passes on, "
                f"available: {sorted(GUROBI_PARAMETERS)}"
            )
        configuration["solveroptions"].setdefault(option, {})["value"] = value

    configuration["solveroptions"]["solver"]["value"] = solver
    configuration["solveroptions"]["mipgap"]["value"] = mipgap
    # In hours. Without it a pathological run can hold the queue for the
    # template default of 100 hours
    configuration["solveroptions"]["timelim"]["value"] = time_limit
    # 0 lets gurobi use every core of the machine, which is the template
    # default. Setting it is how a run is made to look like a cluster job that
    # asked for a given number of cores
    configuration["solveroptions"]["threads"]["value"] = threads

    # Pressure levels are what makes the two pipeline networks distinct
    configuration["performance"]["pressure"]["pressure_on"]["value"] = 1
    configuration["performance"]["pressure"]["pressure_carriers"]["value"] = [
        "hydrogen"
    ]

    configuration["profiling"]["profiling_on"]["value"] = 1
    configuration["profiling"]["sampling_interval"]["value"] = sampling_interval
    configuration["profiling"]["monitor_network"]["value"] = 1

    configuration["reporting"]["save_path"]["value"] = str(results_path)
    configuration["reporting"]["save_summary_path"]["value"] = str(results_path)
    if case_name is not None:
        configuration["reporting"]["case_name"]["value"] = case_name

    with open(input_data_path / "ConfigModel.json", "w") as json_file:
        json.dump(configuration, json_file, indent=4)


def _write_node_locations(input_data_path: Path):
    """
    Creates the folder structure and writes the node locations

    :param Path input_data_path: folder the input data is written to
    """
    dp.create_input_data_folder_template(input_data_path)

    scenario_nodes = pd.read_csv(DATA_PATH / "NodeLocations_0001.csv", sep=";")
    node_location = pd.read_csv(
        input_data_path / "NodeLocations.csv", sep=";", index_col=0, header=0
    )
    for _, row in scenario_nodes.iterrows():
        if row["Node"] in NODES:
            node_location.at[row["Node"], "lon"] = row["lon"]
            node_location.at[row["Node"], "lat"] = row["lat"]
            node_location.at[row["Node"], "alt"] = row["alt"]
    node_location = node_location.reset_index()
    node_location.to_csv(input_data_path / "NodeLocations.csv", sep=";", index=False)


def storages_at_node(storage: str):
    """
    Returns the storages of every node.

    With storage switched off, the storages are removed from the model
    altogether, rather than only being made unattractive. Their variables and
    constraints then do not exist at all.

    :param str storage: "on" or "off"
    :return: dict of node to (new storages, existing storages)
    """
    if storage not in ["on", "off"]:
        raise ValueError("storage must be 'on' or 'off'")

    if storage == "off":
        return {node: ([], {}) for node in NODES}

    return {
        "Small_cluster1": (["Storage_H2_lowP"], {}),
        "Small_cluster2": (["Storage_H2_lowP"], {}),
        "Large_cluster1": (["Storage_H2_highP"], {"Storage_H2_Cavern": 10000}),
        "Large_cluster2": (["Storage_H2_highP"], {}),
    }


def _write_technologies(input_data_path: Path, storage: str, pv: str):
    """
    Writes the technologies available at each node and copies their data

    :param Path input_data_path: folder the input data is written to
    :param str storage: "on" or "off"
    :param str pv: "on" or "off"
    """
    if pv not in ["on", "off"]:
        raise ValueError("pv must be 'on' or 'off'")

    storages = storages_at_node(storage)
    photovoltaics = ["Photovoltaic"] if pv == "on" else []

    other_technologies = {
        "Small_cluster1": ["Electrolyzer_small"] + photovoltaics,
        "Small_cluster2": ["Electrolyzer_small"] + photovoltaics,
        "Large_cluster1": ["Electrolyzer_big"],
        "Large_cluster2": ["Electrolyzer_big"],
    }

    for node in NODES:
        new_storages, existing_storages = storages[node]
        path = input_data_path / "period1" / "node_data" / node / "Technologies.json"
        with open(path, "w") as json_file:
            json.dump(
                {
                    "new": other_technologies[node] + new_storages,
                    "existing": existing_storages,
                },
                json_file,
                indent=4,
            )

    dp.copy_technology_data(input_data_path)
    dp.copy_compressor_data(input_data_path)


def _write_networks(input_data_path: Path):
    """
    Writes the two fully meshed hydrogen networks

    :param Path input_data_path: folder the input data is written to
    """
    with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
        json.dump(
            {
                "new": [
                    "hydrogenPipelineOnshore_lowP",
                    "hydrogenPipelineOnshore_highP",
                ],
                "existing": [],
            },
            json_file,
            indent=4,
        )

    topology_path = input_data_path / "period1" / "network_topology" / "new"
    distance = _calculate_distances(input_data_path)

    for network in ["hydrogenPipelineOnshore_lowP", "hydrogenPipelineOnshore_highP"]:
        os.makedirs(topology_path / network, exist_ok=True)

        for file, value in [("connection", 1), ("size_max_arcs", 1000)]:
            matrix = pd.read_csv(topology_path / f"{file}.csv", sep=";", index_col=0)
            for node_from in NODES:
                for node_to in NODES:
                    if node_from != node_to:
                        matrix.loc[node_from, node_to] = value
            matrix.to_csv(topology_path / network / f"{file}.csv", sep=";")

        distance.to_csv(topology_path / network / "distance.csv", sep=";")

    dp.copy_network_data(input_data_path)


def _calculate_distances(input_data_path: Path):
    """
    Calculates the distance between all nodes from their coordinates

    :param Path input_data_path: folder the input data is written to
    :return: pandas DataFrame with the distances in km
    """
    coordinates = pd.read_csv(input_data_path / "NodeLocations.csv", sep=";")
    coordinates = coordinates.set_index(coordinates.columns[0])

    distance = pd.DataFrame(0.0, index=NODES, columns=NODES)
    for node_from in NODES:
        for node_to in NODES:
            if node_from == node_to:
                continue
            distance.loc[node_from, node_to] = round(
                _haversine(
                    coordinates.at[node_from, "lon"],
                    coordinates.at[node_from, "lat"],
                    coordinates.at[node_to, "lon"],
                    coordinates.at[node_to, "lat"],
                ),
                1,
            )

    return distance


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float):
    """
    Great circle distance between two points in km

    :param float lon1: longitude of the first point
    :param float lat1: latitude of the first point
    :param float lon2: longitude of the second point
    :param float lat2: latitude of the second point
    :return: float distance in km
    """
    radius = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * radius * np.arcsin(np.sqrt(a))


def _write_component_specifications(
    input_data_path: Path,
    pipeline_capex: str,
    bidirectional_precise: int,
    pipeline_size_min: int,
    storage: str,
    storage_precise: int,
):
    """
    Writes the specifications of pipelines, storages and electrolyzers.

    The pipeline settings carry three of the complexity knobs. A fixed CAPEX
    term (gamma1) makes the arc CAPEX need a big-M disjunction, and a minimum
    size adds a second condition to the same disjunction.

    :param Path input_data_path: folder the input data is written to
    :param str pipeline_capex: "linear" or "fixed_plus_linear"
    :param int bidirectional_precise: 1 for the precise formulation
    :param int pipeline_size_min: minimum size of the transmission pipeline
    """
    if pipeline_capex not in ["linear", "fixed_plus_linear"]:
        raise ValueError("pipeline_capex must be 'linear' or 'fixed_plus_linear'")

    network_path = input_data_path / "period1" / "network_data"

    # Distribution network, low pressure
    pipelines = {
        "hydrogenPipelineOnshore_lowP": {
            "size_min": 0,
            "size_max": 250,
            "gamma1": 5000000 if pipeline_capex == "fixed_plus_linear" else 0,
            "gamma4": 1700,
            "pressure": 15,
        },
        # Transmission network, high pressure
        "hydrogenPipelineOnshore_highP": {
            "size_min": pipeline_size_min,
            "size_max": 1000,
            "gamma1": 25000000 if pipeline_capex == "fixed_plus_linear" else 0,
            "gamma4": 1000,
            "pressure": None,
        },
    }

    for network, settings in pipelines.items():
        with open(network_path / f"{network}.json", "r") as json_file:
            network_data = json.load(json_file)

        network_data["size_min"] = settings["size_min"]
        network_data["size_max"] = settings["size_max"]

        network_data["Economics"]["gamma1"] = settings["gamma1"]
        network_data["Economics"]["gamma2"] = 0
        network_data["Economics"]["gamma3"] = 0
        network_data["Economics"]["gamma4"] = settings["gamma4"]

        network_data["Performance"]["bidirectional_network"] = 1
        network_data["Performance"]["bidirectional_network_precise"] = int(
            bidirectional_precise
        )
        network_data["Performance"]["min_transport"] = 0
        network_data["Performance"]["loss"] = 0

        if settings["pressure"] is not None:
            network_data["Performance"]["pressure"]["hydrogen"]["inlet"] = settings[
                "pressure"
            ]
            network_data["Performance"]["pressure"]["hydrogen"]["outlet"] = settings[
                "pressure"
            ]

        with open(network_path / f"{network}.json", "w") as json_file:
            json.dump(network_data, json_file, indent=4)

    # Storages, only if they are part of the model at all
    if storage == "on":
        settings = {
            "Storage_H2_lowP": (50, 15000, 0.8),
            "Storage_H2_highP": (500, 35000, 0.9),
            # The cavern is an existing technology and keeps its size
            "Storage_H2_Cavern": (None, None, 0.9),
        }
        for node, (new_storages, existing_storages) in storages_at_node(
            storage
        ).items():
            for name in list(new_storages) + list(existing_storages):
                size_max, unit_capex, rate = settings[name]
                _write_storage(
                    input_data_path,
                    node,
                    name,
                    size_max,
                    unit_capex,
                    rate,
                    storage_precise,
                )

    # Electrolyzers and photovoltaics
    sizes = {
        "Large_cluster1": [("Electrolyzer_big", 3500)],
        "Large_cluster2": [("Electrolyzer_big", 3500)],
        "Small_cluster1": [("Electrolyzer_small", 150), ("Photovoltaic", 25)],
        "Small_cluster2": [("Electrolyzer_small", 150), ("Photovoltaic", 25)],
    }
    for node, technologies in sizes.items():
        path = input_data_path / "period1" / "node_data" / node / "technology_data"
        for technology, size_max in technologies:
            if not (path / f"{technology}.json").exists():
                # The technology is switched off and was never copied over
                continue
            with open(path / f"{technology}.json", "r") as json_file:
                tec_data = json.load(json_file)
            tec_data["size_min"] = 0
            tec_data["size_max"] = size_max
            with open(path / f"{technology}.json", "w") as json_file:
                json.dump(tec_data, json_file, indent=4)


def _write_storage(
    input_data_path: Path,
    node: str,
    storage: str,
    size_max: float,
    unit_capex: float,
    rate: float,
    storage_precise: int,
):
    """
    Writes the specification of one storage.

    Note that this repository holds the charge and discharge rates under
    "Flow_capacity_relation", where the original case study used "Flexibility".

    :param Path input_data_path: folder the input data is written to
    :param str node: node the storage is at
    :param str storage: name of the storage technology
    :param float size_max: maximum size, None to keep the one of the template
    :param float unit_capex: unit capex, None to keep the one of the template
    :param float rate: charge and discharge rate
    :param int storage_precise: 1 forbids simultaneous charging and
        discharging with a binary per timestep
    """
    path = (
        input_data_path
        / "period1"
        / "node_data"
        / node
        / "technology_data"
        / f"{storage}.json"
    )
    with open(path, "r") as json_file:
        storage_data = json.load(json_file)

    if size_max is not None:
        storage_data["size_min"] = 0
        storage_data["size_max"] = size_max
    if unit_capex is not None:
        storage_data["Economics"]["unit_capex"] = unit_capex

    storage_data["Performance"]["allow_only_one_direction"] = 1
    storage_data["Performance"]["allow_only_one_direction_precise"] = int(
        storage_precise
    )
    storage_data["Performance"]["performance"]["eta_in"] = 0.95
    storage_data["Performance"]["performance"]["eta_out"] = 0.95
    storage_data["Flow_capacity_relation"]["charge_rate"] = rate
    storage_data["Flow_capacity_relation"]["discharge_rate"] = rate

    with open(path, "w") as json_file:
        json.dump(storage_data, json_file, indent=4)


def _demand_per_node():
    """
    Splits the total hydrogen demand over the four nodes

    :return: dict with the demand per node in TWh
    """
    total = STATIC_PARAMETERS["total_demand_TWh"]
    ratio = STATIC_PARAMETERS["demand_level_ratio"]
    unbalance = STATIC_PARAMETERS["unbalance_ratio"]

    small = total / (1 + ratio)
    large = total - small
    large2 = large / (unbalance + 1)
    large1 = large - large2

    return {
        "Large_cluster1": large1,
        "Large_cluster2": large2,
        "Small_cluster1": small / 2,
        "Small_cluster2": small / 2,
    }


def _fluctuation_factors(series: np.ndarray):
    """
    Normalizes a series to fluctuation factors with a mean of one

    :param np.ndarray series: hourly series
    :return: np.ndarray fluctuation factors
    """
    mean = float(np.mean(series))
    if mean == 0:
        raise ValueError("Mean of the example series is zero, cannot normalize")
    return series / mean


def _hydrogen_profiles(hydrogen_demand: str):
    """
    Builds the hydrogen demand profile of every node

    :param str hydrogen_demand: "constant" or "fluctuating"
    :return: dict with an hourly profile in MW per node
    """
    if hydrogen_demand not in ["constant", "fluctuating"]:
        raise ValueError("hydrogen_demand must be 'constant' or 'fluctuating'")

    demand = _demand_per_node()
    profiles = {}

    if hydrogen_demand == "constant":
        for node, demand_twh in demand.items():
            average = demand_twh * 1e6 / TIMESTEPS
            profiles[node] = np.full(TIMESTEPS, average)
        return profiles

    example = pd.read_excel(
        DATA_PATH / "Example_hydrogen_demand.xlsx", header=0, thousands=","
    ).iloc[:TIMESTEPS]

    # The large clusters follow the chemical industry, the small clusters the
    # remaining industry, as in the original case study
    factors = {
        "large": _fluctuation_factors(
            example["Industry_chemicals"].astype(float).to_numpy()
        ),
        "small": _fluctuation_factors(
            example["Industry_other"].astype(float).to_numpy()
        ),
    }

    for node, demand_twh in demand.items():
        average = demand_twh * 1e6 / TIMESTEPS
        group = "large" if node in LARGE_NODES else "small"
        profiles[node] = average * factors[group]

    return profiles


def _electricity_price_profile(electricity_price: str):
    """
    Builds the electricity import price profile.

    A fluctuating profile is taken from the example series and rescaled to the
    average and the standard deviation of the static parameters.

    :param str electricity_price: "constant" or "fluctuating"
    :return: np.ndarray hourly price in EUR/MWh
    """
    if electricity_price not in ["constant", "fluctuating"]:
        raise ValueError("electricity_price must be 'constant' or 'fluctuating'")

    average = STATIC_PARAMETERS["electricity_price_avg"]

    if electricity_price == "constant":
        return np.full(TIMESTEPS, float(average))

    example = pd.read_excel(DATA_PATH / "Example_electricity_prices.xlsx", header=0)
    series = example["E-MBT (ammonia)"].astype(float).to_numpy()[:TIMESTEPS]
    series = _fluctuation_factors(series) * average

    # Rescale to the target standard deviation, keeping the average
    target_std = STATIC_PARAMETERS["electricity_standard_dev"]
    current_std = float(np.std(series))
    if current_std > 0:
        series = (series - average) * (target_std / current_std) + average

    return np.clip(series, 0, None)


def _write_carrier_data(
    input_data_path: Path, electricity_price: str, hydrogen_demand: str
):
    """
    Writes demand, import limits, import prices and pressure levels

    :param Path input_data_path: folder the input data is written to
    :param str electricity_price: "constant" or "fluctuating"
    :param str hydrogen_demand: "constant" or "fluctuating"
    """
    hydrogen_profiles = _hydrogen_profiles(hydrogen_demand)
    electricity_prices = _electricity_price_profile(electricity_price)

    # Hydrogen import is only available at the large clusters and is split
    # between them with the same unbalance as the demand
    average_demand = STATIC_PARAMETERS["total_demand_TWh"] * 1e6 / TIMESTEPS
    import_limit = average_demand * STATIC_PARAMETERS["import_availability_ratio"]
    import_limit_2 = import_limit / (STATIC_PARAMETERS["unbalance_ratio"] + 1)
    import_limit_1 = import_limit - import_limit_2
    hydrogen_import_limit = {
        "Large_cluster1": import_limit_1,
        "Large_cluster2": import_limit_2,
    }

    pressures = {"large": 25, "small": 15}

    for node in NODES:
        group = "large" if node in LARGE_NODES else "small"

        dp.fill_carrier_data(
            input_data_path,
            value_or_data=pd.Series(hydrogen_profiles[node]),
            columns=["Demand"],
            carriers=["hydrogen"],
            nodes=[node],
        )
        dp.fill_carrier_data(
            input_data_path,
            value_or_data=pd.Series(electricity_prices),
            columns=["Import price"],
            carriers=["electricity"],
            nodes=[node],
        )
        dp.fill_carrier_data(
            input_data_path,
            value_or_data=STATIC_PARAMETERS[f"electricity_availability_{group}"],
            columns=["Import limit"],
            carriers=["electricity"],
            nodes=[node],
        )

        for connection, pressure in [
            ("Demand", pressures[group]),
            ("Export", 40),
            ("Import", 40),
            ("Generic production", 0),
        ]:
            dp.fill_carrier_pressure_data(
                input_data_path,
                pressure_value_bar=pressure,
                connection=[connection],
                carriers=["hydrogen"],
                nodes=[node],
            )

    for node, limit in hydrogen_import_limit.items():
        dp.fill_carrier_data(
            input_data_path,
            value_or_data=limit,
            columns=["Import limit"],
            carriers=["hydrogen"],
            nodes=[node],
        )
        dp.fill_carrier_data(
            input_data_path,
            value_or_data=STATIC_PARAMETERS["hydrogen_import_price"],
            columns=["Import price"],
            carriers=["hydrogen"],
            nodes=[node],
        )
