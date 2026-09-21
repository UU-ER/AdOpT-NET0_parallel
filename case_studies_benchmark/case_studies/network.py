"""
Two-nodal energy system with networks.

Follows docs/source/case_studies/CaseStudy_Networks.ipynb: a city node and a
rural node, each with a heat and an electricity demand, connected by an
existing and a new electricity network.

The size of the resulting optimization problem is mainly steered by the number
of typical days.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import adopt_net0 as adopt
import adopt_net0.data_preprocessing as dp
from adopt_net0.utilities import GUROBI_PARAMETERS

NAME = "network"


def setup(
    input_data_path: Path,
    results_path: Path,
    typicaldays: int = 30,
    typicaldays_method: int = 1,
    mipgap: float = 0.02,
    time_limit: float = 2,
    threads: int = 0,
    carbon_price: float = 0,
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

    Climate data is downloaded from an external api the first time a case is
    set up. If the input data already exists, only the configuration is
    updated, which keeps repeated runs of the same case fast.

    :param Path input_data_path: folder the input data is written to
    :param Path results_path: folder the results are written to
    :param int typicaldays: number of typical days, 0 for full resolution
    :param int typicaldays_method: 1 clusters both the model and the data,
        2 clusters the model but keeps the data at full resolution
    :param float mipgap: MILP gap of the solver
    :param float time_limit: solver time limit in hours
    :param int threads: number of threads the solver may use, 0 for every core
        of the machine. This is what a job would ask a cluster for
    :param float carbon_price: carbon price in EUR/t
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
        _write_nodes_and_technologies(input_data_path)
        _write_networks(input_data_path)
        _write_carrier_data(input_data_path)
        dp.load_climate_data_from_api(input_data_path)

    _write_carbon_price(input_data_path, carbon_price)

    return input_data_path


def _write_topology(input_data_path: Path):
    """
    Writes the topology of the case study

    :param Path input_data_path: folder the input data is written to
    """
    with open(input_data_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)

    topology["nodes"] = ["city", "rural"]
    topology["carriers"] = ["electricity", "heat", "gas", "hydrogen"]
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
    :param int threads: number of threads the solver may use, 0 for all of them
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

    configuration["solveroptions"]["mipgap"]["value"] = mipgap
    # In hours. Without it a pathological run can hold the queue for the
    # template default of 100 hours
    configuration["solveroptions"]["timelim"]["value"] = time_limit
    configuration["solveroptions"]["solver"]["value"] = solver
    # 0 lets gurobi use every core of the machine, which is the template
    # default. Setting it is how a run is made to look like a cluster job that
    # asked for a given number of cores
    configuration["solveroptions"]["threads"]["value"] = threads

    configuration["profiling"]["profiling_on"]["value"] = 1
    configuration["profiling"]["sampling_interval"]["value"] = sampling_interval
    configuration["profiling"]["monitor_network"]["value"] = 1

    configuration["reporting"]["save_path"]["value"] = str(results_path)
    configuration["reporting"]["save_summary_path"]["value"] = str(results_path)
    if case_name is not None:
        configuration["reporting"]["case_name"]["value"] = case_name

    with open(input_data_path / "ConfigModel.json", "w") as json_file:
        json.dump(configuration, json_file, indent=4)


def _write_nodes_and_technologies(input_data_path: Path):
    """
    Writes node locations, technologies and their maximum sizes

    :param Path input_data_path: folder the input data is written to
    """
    dp.create_input_data_folder_template(input_data_path)

    node_location = pd.read_csv(
        input_data_path / "NodeLocations.csv", sep=";", index_col=0, header=0
    )
    locations = {
        "city": {"lon": 5.1214, "lat": 52.0907, "alt": 5},
        "rural": {"lon": 5.24, "lat": 51.9561, "alt": 10},
    }
    for node, location in locations.items():
        for column, value in location.items():
            node_location.at[node, column] = value
    node_location = node_location.reset_index()
    node_location.to_csv(input_data_path / "NodeLocations.csv", sep=";", index=False)

    technologies_at_node = {
        "city": {
            "new": ["HeatPump_AirSourced", "Storage_Battery", "Photovoltaic"],
            "existing": {"Boiler_Small_NG": 1000},
        },
        "rural": {
            "new": [
                "HeatPump_AirSourced",
                "Storage_Battery",
                "Photovoltaic",
                "WindTurbine_Onshore_4000",
            ],
            "existing": {"Boiler_Small_NG": 350, "GasTurbine_simple": 1000},
        },
    }
    for node, technologies in technologies_at_node.items():
        path = input_data_path / "period1" / "node_data" / node / "Technologies.json"
        with open(path, "w") as json_file:
            json.dump(technologies, json_file, indent=4)

    dp.copy_technology_data(input_data_path)

    # The default sizes are on a household level and need to be increased
    size_max = {"Boiler_Small_NG": 2000, "HeatPump_AirSourced": 3000}
    for node in ["city", "rural"]:
        path = input_data_path / "period1" / "node_data" / node / "technology_data"
        for technology, size in size_max.items():
            with open(path / f"{technology}.json", "r") as json_file:
                tec_data = json.load(json_file)
            tec_data["size_max"] = size
            with open(path / f"{technology}.json", "w") as json_file:
                json.dump(tec_data, json_file, indent=4)


def _write_networks(input_data_path: Path):
    """
    Writes the existing and the new electricity network between the two nodes

    :param Path input_data_path: folder the input data is written to
    """
    with open(input_data_path / "period1" / "Networks.json", "w") as json_file:
        json.dump(
            {"new": ["electricityOnshore"], "existing": ["electricityOnshore"]},
            json_file,
            indent=4,
        )

    topology_path = input_data_path / "period1" / "network_topology"
    values = {"connection": 1, "distance": 50, "size": 1000}

    for status, files in [
        ("existing", ["connection", "distance", "size"]),
        ("new", ["connection", "distance"]),
    ]:
        os.makedirs(topology_path / status / "electricityOnshore", exist_ok=True)
        for file in files:
            matrix = pd.read_csv(
                topology_path / status / f"{file}.csv", sep=";", index_col=0
            )
            matrix.loc["city", "rural"] = values[file]
            matrix.loc["rural", "city"] = values[file]
            matrix.to_csv(
                topology_path / status / "electricityOnshore" / f"{file}.csv", sep=";"
            )

    dp.copy_network_data(input_data_path)

    path = input_data_path / "period1" / "network_data" / "electricityOnshore.json"
    with open(path, "r") as json_file:
        network_data = json.load(json_file)
    network_data["Economics"]["gamma2"] = 40000
    network_data["Economics"]["gamma4"] = 300
    with open(path, "w") as json_file:
        json.dump(network_data, json_file, indent=4)


def _write_carrier_data(input_data_path: Path):
    """
    Writes demand, import limits and import prices

    :param Path input_data_path: folder the input data is written to
    """
    hourly_data = {
        "city": adopt.load_network_city_data(),
        "rural": adopt.load_network_rural_data(),
    }

    for node, data in hourly_data.items():
        dp.fill_carrier_data(
            input_data_path,
            value_or_data=data.iloc[:, 1],
            columns=["Demand"],
            carriers=["electricity"],
            nodes=[node],
        )
        dp.fill_carrier_data(
            input_data_path,
            value_or_data=data.iloc[:, 0],
            columns=["Demand"],
            carriers=["heat"],
            nodes=[node],
        )

        for value, column, carrier in [
            (4000, "Import limit", "gas"),
            (2000, "Import limit", "electricity"),
            (0.25, "Import emission factor", "electricity"),
            (40, "Import price", "gas"),
            (120, "Import price", "electricity"),
        ]:
            dp.fill_carrier_data(
                input_data_path,
                value_or_data=value,
                columns=[column],
                carriers=[carrier],
                nodes=[node],
            )


def _write_carbon_price(input_data_path: Path, carbon_price: float):
    """
    Writes a constant carbon price to all nodes

    :param Path input_data_path: folder the input data is written to
    :param float carbon_price: carbon price in EUR/t
    """
    for node in ["city", "rural"]:
        path = input_data_path / "period1" / "node_data" / node / "CarbonCost.csv"
        carbon_cost = pd.read_csv(path, sep=";", index_col=0, header=0)
        carbon_cost["price"] = np.ones(len(carbon_cost)) * carbon_price
        carbon_cost = carbon_cost.reset_index()
        carbon_cost.to_csv(path, sep=";", index=False)
