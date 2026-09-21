"""
Nine-nodal Dutch hydrogen system with a two-pressure pipeline network.

Ported from the linepack_nl example, with the pipelines replaced by the two
fluid networks of the four_node case study. The example was written against a
version of adopt that carries a ``fluidynamic_pipeline`` network type and three
fixed pipeline size classes; neither exists here, so the corridors of the
example are kept and the physics of four_node is put on them. The two cases
then differ in topology alone, which is the point of this one.

Nine nodes of demand and generation, 42 of the 72 ordered pairs connected, so
the candidate graph is a real Dutch corridor map rather than a full mesh. Each
corridor can be built at low or at high pressure, which is 84 candidate arcs
against the 24 of four_node.

Why the topology is the interesting axis. Across every run of the profiling
study the median branch and bound tree is one node, and the deepest ever seen
is 248. The binaries of ``bidirectional_precise`` are one per arc and timestep,
tens of thousands of them, and the relaxation picks a flow direction almost for
free. The binaries of ``pipeline_size_min`` are one per arc, a few dozen, and
each one decides whether a corridor exists at all. Those are the hard ones, and
they scale with the topology rather than with the resolution, which is why more
nodes is expected to do what more typical days never did.

The supply of the example had to be changed for that to be true. It gives five
of the nine nodes enough existing wind, PV and electrolysis to cover their own
demand, so the only nodes that ever needed a pipeline were the three with
neither generation nor import: the first solve built 5 of the 42 corridors and
left 79 of the 84 size binaries at zero. Chemelot and Zeeland are therefore
stripped of their existing generation, and what is left at the other nodes is
halved, so that no node covers itself and the corridors carry the difference.

The hydrogen import is tightened in the same place, to 0.6 of the node's own
average demand at 300 EUR/MWh, but it is a ceiling and not a lever: at the
sizes of the example the import is zero at every node of the solution, and
tightening it alone changes the objective in no digit. The four nodes the
example leaves without import still have none.

The case exposes the knobs that drive the size and the difficulty of the
problem, so that their effect on resource usage can be measured:

- ``typicaldays``: number of typical days, 0 for full resolution
- ``bidirectional_precise``: precise or relaxed formulation of bidirectionality
- ``pipeline_size_min``: a minimum size on the transmission pipeline, which
  makes each arc either absent or at least that large, through a disjunction
- ``pipeline_capex``: with or without a fixed CAPEX term, which decides whether
  the arc CAPEX needs a big-M disjunction as well
- ``backbone``: whether the existing pipeline backbone is part of the system.
  Without it every corridor has to be built, which is the harder problem and
  the less realistic one. It defaults to off, and not only for difficulty:
  with pressure on, a compressor whose input or output is an existing network
  reads ``para_size_initial`` off the network block in
  ``adopt_net0/utilities.py``, while networks define it on the arc block, so
  the model cannot be built. four_node never meets this because it has no
  existing network
- ``storage``: removes every storage from the model rather than making it
  unattractive
"""

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

import adopt_net0.data_preprocessing as dp
from adopt_net0.utilities import GUROBI_PARAMETERS

NAME = "nl_node"

DATA_PATH = Path(__file__).parent / "data" / "nl_node"

# The pipeline JSONs of four_node, which are the formulation this case is meant
# to carry. They are read from the generated input data of that case study,
# which is where the two pressure levels live
FOUR_NODE_NETWORK_DATA = (
    Path(__file__).parent.parent / "inputData" / "four_node" / "period1" / "network_data"
)

LOW_PRESSURE = "hydrogenPipelineOnshore_lowP"
HIGH_PRESSURE = "hydrogenPipelineOnshore_highP"
BACKBONE = "hydrogenPipelineOnshore_backbone"

# The corridor map of the example, which every new network is written onto. The
# three size classes of the example share one map, so any of them will do
SOURCE_CORRIDORS = "H2Pipeline_large"
SOURCE_BACKBONE = "H2Pipeline_backbone"

# Storages of the example, removed together when the storage knob is off
STORAGES = ["Storage_H2", "Storage_H2_Cavern"]

# Hydrogen import is the outside option the corridors compete against. The
# example allows twice the demand at 210 EUR/MWh, and at the sizes it ships
# nobody imports at all: the import is zero at every node of the solution and
# tightening it changes the objective in no digit. It is kept as a ceiling
# rather than as a lever, for the runs in which the existing generation is cut
# back far enough for it to start binding. four_node caps its import at 0.3 of
# the demand, so this is the milder version of the same number.
IMPORT_AVAILABILITY_RATIO = 0.6
IMPORT_PRICE = 300

# The existing generation is what really decides whether a corridor is needed.
# At the sizes of the example each of the five supply nodes covers its own
# demand, so the only nodes that needed a pipeline were the three with neither
# generation nor import, and the first solve built 5 of the 42 corridors with
# 79 of the 84 size binaries at zero. Two things are done about it: two more
# nodes are made demand only, and what is left elsewhere is scaled back so
# that no node covers itself.
DEMAND_ONLY_NODES = ["Chemelot", "Zeeland"]
EXISTING_GENERATION_SCALE = 0.5

# The technologies that produce, as opposed to the storages, which the storage
# knob owns
GENERATION = ["WindTurbine_Onshore_4000", "Photovoltaic", "Electrolyzer"]


def setup(
    input_data_path: Path,
    results_path: Path,
    typicaldays: int = 16,
    typicaldays_method: int = 1,
    bidirectional_precise: int = 0,
    pipeline_size_min: int = 250,
    pipeline_capex: str = "fixed_plus_linear",
    backbone: str = "off",
    compression: str = "off",
    storage: str = "on",
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

    The node data, the profiles and the corridor map are copied from the
    vendored example the first time and left alone afterwards. What is
    rewritten on every call is the configuration, the network specifications
    and the technology lists, since those carry the knobs.

    :param Path input_data_path: folder the input data is written to
    :param Path results_path: folder the results are written to
    :param int typicaldays: number of typical days, 0 for full resolution
    :param int typicaldays_method: 1 clusters both the model and the data,
        2 clusters the model but keeps the data at full resolution
    :param int bidirectional_precise: 1 for the precise formulation, which is
        a binary per arc and timestep
    :param int pipeline_size_min: minimum size of the transmission pipeline. A
        non zero value makes each arc either absent or at least this large,
        which is a binary per arc and the hardest structure in the model
    :param str pipeline_capex: "linear" or "fixed_plus_linear"
    :param str backbone: "on" or "off". "off" removes the existing pipeline
        network, so that every corridor has to be built
    :param str compression: "on" or "off". Off switches the pressure levels
        and with them the compressors out of the model, which is what the
        four_node case carries them for. This case is about the topology, and
        a compressor per node is a technology per node, not a corridor
    :param str storage: "on" or "off", "off" removes every storage
    :param float mipgap: MILP gap of the solver
    :param float time_limit: solver time limit in hours
    :param int threads: number of threads the solver may use, 0 for every core
    :param str solver: solver used
    :param int method: gurobi Method, the algorithm of the root relaxation
    :param int crossover: gurobi Crossover
    :param int barhomogeneous: gurobi BarHomogeneous
    :param int scaleflag: gurobi ScaleFlag
    :param int concurrentmethod: gurobi ConcurrentMethod
    :param int nodemethod: gurobi NodeMethod, the algorithm at the nodes
    :param int presolve: gurobi Presolve
    :param int cuts: gurobi Cuts
    :param int cutpasses: gurobi CutPasses, the maximum number of cut rounds
        at the root
    :param int mipfocus: gurobi MIPFocus
    :param float heuristics: gurobi Heuristics
    :param float norelheurtime: gurobi NoRelHeurTime
    :param int numericfocus: gurobi NumericFocus
    :param int lpwarmstart: gurobi LPWarmStart
    :param int branchdir: gurobi BranchDir
    :param float sampling_interval: sampling interval of the resource monitor
    :param str case_name: name added to the results folder
    :return: Path of the input data folder
    """
    if pipeline_capex not in ["linear", "fixed_plus_linear"]:
        raise ValueError("pipeline_capex must be 'linear' or 'fixed_plus_linear'")
    if backbone not in ["on", "off"]:
        raise ValueError("backbone must be 'on' or 'off'")
    if compression not in ["on", "off"]:
        raise ValueError("compression must be 'on' or 'off'")
    if storage not in ["on", "off"]:
        raise ValueError("storage must be 'on' or 'off'")

    input_data_path = Path(input_data_path)
    results_path = Path(results_path)
    input_data_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    is_new = not (input_data_path / "Topology.json").exists()

    if is_new:
        _copy_source_data(input_data_path)

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
        compression=compression,
    )

    # The networks are rewritten on every call, as the backbone knob changes
    # which of them exist at all
    _write_networks(input_data_path, backbone=backbone)

    _write_technologies(input_data_path, storage=storage)

    _write_carrier_data(input_data_path)

    _write_component_specifications(
        input_data_path,
        pipeline_capex=pipeline_capex,
        bidirectional_precise=bidirectional_precise,
        pipeline_size_min=pipeline_size_min,
        backbone=backbone,
    )

    return input_data_path


def _copy_source_data(input_data_path: Path):
    """
    Copies the vendored example into the input data folder.

    Everything that is not a knob comes across untouched: the topology, the
    node data, the demand and climate profiles and the corridor map.

    :param Path input_data_path: folder the input data is written to
    """
    if not DATA_PATH.is_dir():
        raise FileNotFoundError(
            f"The source data of the case study is missing at {DATA_PATH}"
        )

    shutil.copytree(DATA_PATH, input_data_path, dirs_exist_ok=True)


def _template_configuration():
    """
    A configuration template of this adopt version.

    The example was written against another version, so its ConfigModel.json
    is missing the sections this one adds, the profiling settings among them.
    Rather than write those out by hand, a template is generated and the
    missing sections are taken from it.

    :return: dict of the template configuration
    """
    with tempfile.TemporaryDirectory() as scratch:
        dp.create_optimization_templates(Path(scratch))
        with open(Path(scratch) / "ConfigModel.json", "r") as json_file:
            return json.load(json_file)


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
    compression: str,
):
    """
    Writes the model configuration, including the profiling settings

    :param Path input_data_path: folder the input data is written to
    :param Path results_path: folder the results are written to
    :param int typicaldays: number of typical days, 0 for full resolution
    :param int typicaldays_method: clustering method, 1 or 2
    :param float mipgap: MILP gap of the solver
    :param float time_limit: solver time limit in hours
    :param int threads: threads the solver may use
    :param str solver: solver used
    :param dict gurobi_options: solver options passed to the configuration
        untouched. The keys are those of ConfigModel.json, which
        GUROBI_PARAMETERS in adopt_net0/utilities.py maps to the gurobi names
    :param float sampling_interval: sampling interval of the resource monitor
    :param str case_name: name added to the results folder
    :param str compression: "on" or "off", whether pressure levels and the
        compressors that go with them are part of the model
    """
    with open(input_data_path / "ConfigModel.json", "r") as json_file:
        configuration = json.load(json_file)

    # Sections this adopt knows about and the example did not carry
    for section, block in _template_configuration().items():
        configuration.setdefault(section, block)

    configuration["optimization"]["typicaldays"]["N"]["value"] = typicaldays
    if typicaldays != 0:
        if typicaldays_method not in [1, 2]:
            raise ValueError("typicaldays_method must be 1 or 2")
        configuration["optimization"]["typicaldays"]["method"][
            "value"
        ] = typicaldays_method
    configuration["optimization"]["objective"]["value"] = "costs"

    for option, value in gurobi_options.items():
        if option not in GUROBI_PARAMETERS:
            raise KeyError(
                f"'{option}' is not a gurobi option adopt passes on, "
                f"available: {sorted(GUROBI_PARAMETERS)}"
            )
        configuration["solveroptions"].setdefault(option, {})["value"] = value

    configuration["solveroptions"]["solver"]["value"] = solver
    configuration["solveroptions"]["mipgap"]["value"] = mipgap
    configuration["solveroptions"]["timelim"]["value"] = time_limit
    configuration["solveroptions"]["threads"]["value"] = threads

    # With pressure on, every node gets a compressor, which is a technology
    # per node rather than a corridor. This case is about the topology, so the
    # default is off and the two pipelines are then told apart by their size
    # and their cost rather than by their pressure
    configuration["performance"]["pressure"]["pressure_on"]["value"] = int(
        compression == "on"
    )
    configuration["performance"]["pressure"]["pressure_carriers"]["value"] = (
        ["hydrogen"] if compression == "on" else []
    )

    configuration["profiling"]["profiling_on"]["value"] = 1
    configuration["profiling"]["sampling_interval"]["value"] = sampling_interval
    configuration["profiling"]["monitor_network"]["value"] = 1

    configuration["reporting"]["save_path"]["value"] = str(results_path)
    configuration["reporting"]["save_summary_path"]["value"] = str(results_path)
    if case_name is not None:
        configuration["reporting"]["case_name"]["value"] = case_name

    with open(input_data_path / "ConfigModel.json", "w") as json_file:
        json.dump(configuration, json_file, indent=4)


def _write_networks(input_data_path: Path, backbone: str):
    """
    Puts the two four_node pipelines onto the corridor map of the example.

    The example carries three fixed size classes and a fluidynamic backbone,
    none of which this adopt can build. They are replaced by the low and the
    high pressure fluid pipeline, written onto the same corridors, so that the
    two case studies differ in topology and not in physics.

    :param Path input_data_path: folder the input data is written to
    :param str backbone: "on" or "off"
    """
    period_path = input_data_path / "period1"
    topology_path = period_path / "network_topology"
    network_data_path = period_path / "network_data"
    network_data_path.mkdir(parents=True, exist_ok=True)

    existing = [BACKBONE] if backbone == "on" else []

    with open(period_path / "Networks.json", "w") as json_file:
        json.dump(
            {"new": [LOW_PRESSURE, HIGH_PRESSURE], "existing": existing},
            json_file,
            indent=4,
        )

    # The corridor map, one folder per network, copied from the example rather
    # than invented, so the graph stays the Dutch one
    source = topology_path / "new" / SOURCE_CORRIDORS
    for network in [LOW_PRESSURE, HIGH_PRESSURE]:
        target = topology_path / "new" / network
        if source.is_dir() and not target.is_dir():
            shutil.copytree(source, target)

    for unused in ["H2Pipeline_large", "H2Pipeline_medium", "H2Pipeline_small"]:
        shutil.rmtree(topology_path / "new" / unused, ignore_errors=True)

    backbone_source = topology_path / "existing" / SOURCE_BACKBONE
    backbone_target = topology_path / "existing" / BACKBONE
    if backbone_source.is_dir() and not backbone_target.is_dir():
        shutil.copytree(backbone_source, backbone_target)
    shutil.rmtree(topology_path / "existing" / SOURCE_BACKBONE, ignore_errors=True)

    # The pipeline data itself, taken from four_node
    if not FOUR_NODE_NETWORK_DATA.is_dir():
        raise FileNotFoundError(
            "The four_node pipeline data is missing at "
            f"{FOUR_NODE_NETWORK_DATA}. Run the four_node case study once, "
            "which generates it, before running this one"
        )

    for network in [LOW_PRESSURE, HIGH_PRESSURE]:
        shutil.copyfile(
            FOUR_NODE_NETWORK_DATA / f"{network}.json",
            network_data_path / f"{network}.json",
        )

    if backbone == "on":
        # The backbone is an existing network, so it is the same pipeline with
        # a size already in the ground. It is the transmission one, since that
        # is what a backbone is
        shutil.copyfile(
            FOUR_NODE_NETWORK_DATA / f"{HIGH_PRESSURE}.json",
            network_data_path / f"{BACKBONE}.json",
        )

    for unused in ["H2Pipeline_large", "H2Pipeline_medium", "H2Pipeline_small"]:
        (network_data_path / f"{unused}.json").unlink(missing_ok=True)


def _write_technologies(input_data_path: Path, storage: str):
    """
    Writes the technology list of every node.

    Three things happen here. The storage knob removes the storages from the
    model rather than making them unattractive, which is how the four_node
    case study spells the same knob. The nodes in DEMAND_ONLY_NODES lose their
    existing generation altogether, so they have to be supplied over the
    network. What generation is left anywhere else is scaled by
    EXISTING_GENERATION_SCALE, so that no node covers its own demand and the
    corridors have to carry the difference.

    What a node may still build itself is left alone: local construction
    competing against transport is the decision the case study is about.

    :param Path input_data_path: folder the input data is written to
    :param str storage: "on" or "off"
    """
    source_nodes = DATA_PATH / "period1" / "node_data"
    node_path = input_data_path / "period1" / "node_data"

    for node in sorted(p.name for p in node_path.iterdir() if p.is_dir()):
        # Read from the vendored copy, so that switching the knob back on
        # restores what was removed
        with open(source_nodes / node / "Technologies.json", "r") as json_file:
            technologies = json.load(json_file)

        existing = dict(technologies.get("existing", {}))

        if storage == "off":
            existing = {
                name: size for name, size in existing.items() if name not in STORAGES
            }
            technologies["new"] = [
                name for name in technologies.get("new", []) if name not in STORAGES
            ]

        if node in DEMAND_ONLY_NODES:
            existing = {
                name: size for name, size in existing.items() if name not in GENERATION
            }
        else:
            existing = {
                name: (
                    size * EXISTING_GENERATION_SCALE if name in GENERATION else size
                )
                for name, size in existing.items()
            }

        technologies["existing"] = existing

        with open(node_path / node / "Technologies.json", "w") as json_file:
            json.dump(technologies, json_file, indent=4)


def _write_carrier_data(input_data_path: Path):
    """
    Rewrites the hydrogen import of every node that the example gives one.

    The limit becomes a fraction of the node's own average demand, so that a
    node covers part of its demand from outside and has to be connected for
    the rest. The nodes the example leaves without import keep none: those are
    the ones a corridor has to reach, and they are what makes the topology the
    lever of this case study.

    The profiles themselves are left alone, import limit and import price are
    the only columns touched.

    :param Path input_data_path: folder the input data is written to
    """
    source_nodes = DATA_PATH / "period1" / "node_data"
    node_path = input_data_path / "period1" / "node_data"

    for node in sorted(p.name for p in node_path.iterdir() if p.is_dir()):
        # Read from the vendored copy, so that the limit is a fraction of the
        # original demand however often this runs
        source_file = source_nodes / node / "carrier_data" / "hydrogen.csv"
        carrier = pd.read_csv(source_file, sep=";", index_col=0)

        if carrier["Import limit"].max() > 0:
            carrier["Import limit"] = (
                carrier["Demand"].mean() * IMPORT_AVAILABILITY_RATIO
            )
            carrier["Import price"] = IMPORT_PRICE

        carrier.to_csv(node_path / node / "carrier_data" / "hydrogen.csv", sep=";")


def _write_component_specifications(
    input_data_path: Path,
    pipeline_capex: str,
    bidirectional_precise: int,
    pipeline_size_min: int,
    backbone: str,
):
    """
    Writes the specifications of the pipelines.

    The settings carry three of the complexity knobs. A fixed CAPEX term
    (gamma1) makes the arc CAPEX need a big-M disjunction, a minimum size adds
    a second condition to the same disjunction, and the precise
    bidirectionality adds a binary per arc and timestep.

    The numbers are those of four_node, so that a corridor here costs and
    carries what a corridor there does.

    :param Path input_data_path: folder the input data is written to
    :param str pipeline_capex: "linear" or "fixed_plus_linear"
    :param int bidirectional_precise: 1 for the precise formulation
    :param int pipeline_size_min: minimum size of the transmission pipeline
    :param str backbone: "on" or "off"
    """
    network_path = input_data_path / "period1" / "network_data"

    pipelines = {
        # Distribution network, low pressure
        LOW_PRESSURE: {
            "size_min": 0,
            "size_max": 250,
            "gamma1": 5000000 if pipeline_capex == "fixed_plus_linear" else 0,
            "gamma4": 1700,
            "pressure": 15,
        },
        # Transmission network, high pressure
        HIGH_PRESSURE: {
            "size_min": pipeline_size_min,
            "size_max": 1000,
            "gamma1": 25000000 if pipeline_capex == "fixed_plus_linear" else 0,
            "gamma4": 1000,
            "pressure": None,
        },
    }

    if backbone == "on":
        # An existing network is not sized by the model, so a minimum size
        # would only forbid what is already there
        pipelines[BACKBONE] = {
            "size_min": 0,
            "size_max": 1000,
            "gamma1": 0,
            "gamma4": 1000,
            "pressure": None,
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


def corridors():
    """
    The candidate corridors of the case study, for reporting and for tests

    :return: tuple of the node count and the number of connected ordered pairs
    """
    matrix = pd.read_csv(
        DATA_PATH
        / "period1"
        / "network_topology"
        / "new"
        / SOURCE_CORRIDORS
        / "connection.csv",
        sep=";",
        index_col=0,
    )
    return matrix.shape[0], int((matrix.values != 0).sum())
