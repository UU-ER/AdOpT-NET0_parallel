from warnings import warn

from pyomo.environ import SolverFactory


# Key of a solver option in the configuration, to the name gurobi knows it by.
#
# The spelling on the right is gurobi's own, taken from its parameter list. A
# solver option that is not in this table never reaches the solver, and it is
# dropped without a word, so a configuration can look carefully tuned and still
# run at the defaults. Anything added to the configuration has to be added here
# as well.
GUROBI_PARAMETERS = {
    "mipgap": "MIPGap",
    "mipfocus": "MIPFocus",
    "threads": "Threads",
    "nodefilestart": "NodefileStart",
    "heuristics": "Heuristics",
    "presolve": "Presolve",
    "branchdir": "BranchDir",
    "lpwarmstart": "LPWarmStart",
    "intfeastol": "IntFeasTol",
    "feastol": "FeasibilityTol",
    "cuts": "Cuts",
    "numericfocus": "NumericFocus",
    # The root relaxation. On a MIP the root goes to a single thread and is
    # where most of the time is spent, so the algorithm that solves it, and the
    # way it is finished off, weigh more than anything in the tree
    "method": "Method",
    "crossover": "Crossover",
    "barhomogeneous": "BarHomogeneous",
    "scaleflag": "ScaleFlag",
    "concurrentmethod": "ConcurrentMethod",
    # The tree
    "nodemethod": "NodeMethod",
    "norelheurtime": "NoRelHeurTime",
}

# Solver options that are not gurobi parameters. "solver" names the solver, and
# the time limit is in hours in the configuration and in seconds for gurobi, so
# it is converted rather than passed on
NON_PARAMETER_OPTIONS = {"solver", "timelim"}


def get_gurobi_parameters(solveroptions: dict):
    """
    Initiates the gurobi solver and defines solver parameters

    Options the configuration does not carry are left at the gurobi default,
    which is what an older configuration, written before a parameter was added,
    relies on.

    :param dict solveroptions: dict with solver parameters
    :return: Gurobi Solver
    """
    solver = SolverFactory(solveroptions["solver"]["value"], solver_io="python")

    # In hours in the configuration, in seconds for gurobi
    solver.options["TimeLimit"] = solveroptions["timelim"]["value"] * 3600

    for option, parameter in GUROBI_PARAMETERS.items():
        if option in solveroptions:
            solver.options[parameter] = solveroptions[option]["value"]

    # A solver option nobody reads is worse than no option at all, as it looks
    # like it is doing something
    unknown = set(solveroptions) - set(GUROBI_PARAMETERS) - NON_PARAMETER_OPTIONS
    if unknown:
        warn(
            f"Solver options {sorted(unknown)} are not gurobi parameters and "
            "were ignored. Check the spelling against GUROBI_PARAMETERS in "
            "adopt_net0/utilities.py"
        )

    return solver


def get_glpk_parameters(solveroptions: dict):
    """
    Initiates the glpk solver and defines solver parameters

    :param dict solveroptions: dict with solver parameters
    :return: Gurobi Solver
    """
    solver = SolverFactory("glpk")

    return solver


def get_set_t(config: dict, model_block):
    """
    Returns the correct set_t for different clustering options

    :param dict config: config dict
    :param model_block: pyomo block holding set_t_full and set_t_clustered
    :return: set_t
    """
    if config["optimization"]["typicaldays"]["N"]["value"] == 0:
        return model_block.set_t_full
    elif config["optimization"]["typicaldays"]["method"]["value"] == 1:
        return model_block.set_t_clustered
    elif config["optimization"]["typicaldays"]["method"]["value"] == 2:
        return model_block.set_t_full


def get_data_for_investment_period(
    data, investment_period: str, aggregation_model: str
) -> dict:
    """
    Gets data from DataHandle for specific investement_period. Writes it to a dict.

    :param data: data to use
    :param str investment_period: investment period
    :param str aggregation_model: aggregation type
    :return: data of respective investment period
    :rtype: dict
    """
    data_period = {}
    data_period["period_name"] = investment_period
    data_period["topology"] = data.topology
    data_period["technology_data"] = data.technology_data[investment_period]
    data_period["time_series"] = data.time_series[aggregation_model].loc[
        :, investment_period
    ]
    data_period["network_data"] = data.network_data[investment_period]
    data_period["energybalance_options"] = data.energybalance_options[investment_period]
    data_period["config"] = data.model_config
    if data.model_config["optimization"]["typicaldays"]["N"]["value"] != 0:
        data_period["k_means_specs"] = data.k_means_specs[investment_period]
        # data_period["averaged_specs"] = data.averaged_specs[investment_period]
    if data.model_config["performance"]["pressure"]["pressure_on"]["value"] == 1:
        data_period["compressor_data"] = data.compressor_data[investment_period]

    # Hour multiplication factors
    if data.model_config["optimization"]["typicaldays"]["N"]["value"] == 0:
        data_period["hour_factors"] = [1] * len(
            data_period["topology"]["time_index"]["full"]
        )
    elif data.model_config["optimization"]["typicaldays"]["method"]["value"] == 1:
        data_period["hour_factors"] = data_period["k_means_specs"]["factors"]
    elif data.model_config["optimization"]["typicaldays"]["method"]["value"] == 2:
        data_period["hour_factors"] = [1] * len(
            data_period["topology"]["time_index"]["full"]
        )

    # Nr timesteps averaged
    if data.model_config["optimization"]["timestaging"]["value"] != 0:
        data_period["nr_timesteps_averaged"] = data.model_config["optimization"][
            "timestaging"
        ]["value"]
    else:
        data_period["nr_timesteps_averaged"] = 1

    return data_period


def determine_flow_existing_compressors(self, compressor, b_period, node):
    """
    Determines the flow capacity of an existing compressor connection by returning
    the minimum available capacity between the output and input components

    :param compressor: tuple with carrier, component1, component 2
    :param b_period: pyomo block data for period
    :param node: pyomo block data for node
    :return float: minimum capacity between input and output component
    """
    component_output_bound = float("inf")
    component_input_bound = float("inf")
    period_name = b_period.name.split("[")[-1].rstrip("]")
    type_component = [compressor.output_type, compressor.input_type]

    if type_component[0] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.output_component]
            .var_output
        )
        component_output_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[0] == "Network":
        component_output_bound = next(
            iter(
                b_period.network_block[
                    compressor.output_component
                ].para_size_initial.values()
            )
        )
    elif type_component[0] == "Import":
        component_output_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Import limit"]
        )

    elif type_component[0] == "Generic production":
        component_output_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Generic production"]
        )

    if type_component[1] == "Technology":
        var_output = (
            b_period.node_blocks[node]
            .tech_blocks_active[compressor.input_component]
            .var_output
        )
        component_input_bound = max(var_output[idx].ub for idx in var_output)
    elif type_component[1] == "Network":
        component_input_bound = next(
            iter(
                b_period.network_block[
                    compressor.input_component
                ].para_size_initial.values()
            )
        )
    elif type_component[1] == "Demand":
        component_input_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Demand"]
        )
    elif type_component[1] == "Export":
        component_input_bound = max(
            self.data.time_series["full"][period_name][node]["CarrierData"][
                compressor.carrier
            ]["Export limit"]
        )

    size = min(component_output_bound, component_input_bound)

    return size
