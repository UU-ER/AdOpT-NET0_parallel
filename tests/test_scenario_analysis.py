import json

import numpy as np
import pandas as pd
import pytest

import adopt_net0.data_preprocessing as dp
from adopt_net0.data_preprocessing.data_loading import CARRIER_COLUMNS
from adopt_net0.scenario_analysis import (
    ScenarioHub,
    collect_results,
    parse_equations,
    plan_scenarios,
    read_definition,
    read_plan,
    scenario_name,
    scan_input_data,
    solve_derived,
    validate_definition,
    write_catalogue,
    write_plan,
    write_scenario,
)

# The topology template has two nodes, which is the smallest case in which a
# total that is spread over the nodes needs an assumption about the split
TEMPLATE_TOPOLOGY = {"nodes": ["node1", "node2"], "carriers": ["electricity"]}

DEMAND_SYMBOLS = {
    "total_demand_TWh": {"description": "Total demand.", "values": [5, 10]}
}


def _built_case_study(request):
    """
    Builds the template case study the tests run against

    :param request: pytest request, carrying the temporary folders
    :return: path of the input data folder
    """
    data_folder_path = request.config.data_folder_path
    dp.create_optimization_templates(data_folder_path)
    dp.create_input_data_folder_template(data_folder_path)
    return data_folder_path


def test_catalogue_needs_a_topology(request):
    """
    Tests that scanning a folder that is not a case study says so.

    A missing topology file is the one thing the catalogue cannot work around,
    since it is where the nodes, carriers and investment periods come from.
    """
    with pytest.raises(FileNotFoundError):
        scan_input_data(request.config.result_folder_path)


def test_catalogue_of_a_template(request):
    """
    Tests standard behavior of scan_input_data on a freshly created case study.

    Checks that
    - the configuration is listed, with the description and the options that the
      configuration file carries itself
    - the topology is listed as structural
    - every carrier data column is listed once per carrier
    - the entries are unique, so that a scenario naming one of them is
      unambiguous
    """
    data_folder_path = request.config.data_folder_path
    dp.create_optimization_templates(data_folder_path)
    dp.create_input_data_folder_template(data_folder_path)

    catalogue = scan_input_data(data_folder_path)
    by_target = {}
    for entry in catalogue:
        by_target.setdefault(entry["target"], []).append(entry)

    # The configuration documents itself, so at least one entry has to come out
    # with a description and one with options
    config_entries = by_target["config"]
    assert any(entry["description"] for entry in config_entries)
    assert any(entry["options"] for entry in config_entries)

    objective = next(
        entry
        for entry in config_entries
        if entry["parameter"] == "optimization/objective"
    )
    assert objective["current"] == "costs"
    assert "costs" in objective["options"]

    # The sets of the model are structural, as they decide which files exist
    structural = {entry["parameter"] for entry in by_target["structural"]}
    assert structural == {
        "topology/nodes",
        "topology/carriers",
        "topology/investment_periods",
    }

    # A carrier data column is listed once per carrier, at the widest scope, so
    # the nodes and periods it can be narrowed to are on the entry instead
    with open(data_folder_path / "Topology.json", "r") as json_file:
        topology = json.load(json_file)

    carrier_entries = by_target["carrier_data"]
    assert len(carrier_entries) == len(CARRIER_COLUMNS) * len(topology["carriers"])
    for entry in carrier_entries:
        assert entry["scope"]["nodes"] == topology["nodes"]
        assert entry["scope"]["investment_periods"] == topology["investment_periods"]

    # An entry has to be identified by its parameter and its scope together: the
    # same column name exists for every carrier
    identifiers = [
        (entry["parameter"], json.dumps(entry["scope"], sort_keys=True))
        for entry in catalogue
    ]
    assert len(identifiers) == len(set(identifiers))


def test_catalogue_reports_what_the_base_case_holds(request):
    """
    Tests that the summary of a carrier data column says whether it is defined.

    A column that is empty in the base case is not the same as one that is set
    to zero: a scenario that fills it is adding something to the case study
    rather than changing it, and the catalogue has to make that visible.
    """
    data_folder_path = request.config.data_folder_path
    dp.create_optimization_templates(data_folder_path)
    dp.create_input_data_folder_template(data_folder_path)

    demand = 42.0
    dp.fill_carrier_data(data_folder_path, demand, columns=["Demand"])

    catalogue = scan_input_data(data_folder_path)
    filled = [
        entry
        for entry in catalogue
        if entry["target"] == "carrier_data" and entry["parameter"] == "Demand"
    ]
    assert filled
    for entry in filled:
        assert entry["current"]["defined"]
        assert entry["current"]["mean"] == pytest.approx(demand)
        assert not entry["current"]["differs_between_nodes"]

    # A column nothing was written to stays empty
    untouched = next(
        entry
        for entry in catalogue
        if entry["target"] == "carrier_data" and entry["parameter"] == "Export price"
    )
    assert not untouched["current"]["defined"]


def test_write_catalogue(request):
    """
    Tests standard behavior of write_catalogue
    - Tests that the file is written and can be read back
    """
    data_folder_path = request.config.data_folder_path
    dp.create_optimization_templates(data_folder_path)
    dp.create_input_data_folder_template(data_folder_path)

    save_path = data_folder_path / "ParameterCatalogue.json"
    written = write_catalogue(data_folder_path, save_path)

    assert save_path.exists()
    with open(save_path, "r") as json_file:
        assert len(json.load(json_file)) == len(written)


def test_scenario_analysis_template(request):
    """
    Tests standard behavior of
    - initialize_scenario_analysis_template
    - create_scenario_analysis_template
    - read_definition
    """
    data_folder_path = _built_case_study(request)
    dp.create_scenario_analysis_template(data_folder_path)

    scenario_file = data_folder_path / "ScenarioAnalysis.json"
    assert scenario_file.exists()

    definition = read_definition(scenario_file)
    assert definition["analysis"]["combination"]["value"] in ["grid", "list", "lhs"]

    # A fresh template changes nothing yet, and says so rather than passing
    problems = validate_definition(definition, data_folder_path)
    assert any("changes nothing" in problem for problem in problems)


def test_scenario_hub_plans_from_the_definition(request):
    """
    Tests that ScenarioHub reads, checks and plans an analysis into the output
    folder of its resources, and refuses a definition with problems
    """
    data_folder_path = _built_case_study(request)
    output_folder = request.config.result_folder_path / "hub"

    definition = dp.initialize_scenario_analysis_template()
    definition["analysis"]["name"]["value"] = "days"
    definition["resources"]["output_folder"]["value"] = str(output_folder)
    definition["parameters"]["days"] = {
        "short": "td",
        "target": "config",
        "parameter": "optimization/typicaldays/N",
        "operation": "set",
        "values": [4, 8],
    }
    with open(data_folder_path / "ScenarioAnalysis.json", "w") as json_file:
        json.dump(definition, json_file)

    analysis = ScenarioHub()
    analysis.read_definition(data_folder_path)
    assert analysis.analysis_path == output_folder / "days"

    analysis.plan()
    assert [scenario["name"] for scenario in analysis.scenarios] == ["td-4", "td-8"]
    assert (output_folder / "days" / "scenarios.csv").exists()

    analysis.definition["resources"]["cores"]["value"] = 0
    with pytest.raises(Exception, match="cores"):
        analysis.check()


def test_definition_of_a_direct_parameter(request):
    """
    Tests that a parameter of the catalogue validates and a made up one does not
    """
    data_folder_path = _built_case_study(request)

    sound = {
        "analysis": {
            "name": {"value": "test"},
            "combination": {"value": "grid"},
        },
        "parameters": {
            "gas_price": {
                "description": "Import price of electricity.",
                "target": "carrier_data",
                "parameter": "Import price",
                "scope": {"carriers": ["electricity"]},
                "operation": "set",
                "values": [20, 40, 60],
            }
        },
    }
    assert validate_definition(sound, data_folder_path) == []

    # A parameter that does not exist is caught before anything runs, and the
    # message points at what the case study does have
    made_up = json.loads(json.dumps(sound))
    made_up["parameters"]["gas_price"]["parameter"] = "Import prices"
    problems = validate_definition(made_up, data_folder_path)
    assert len(problems) == 1
    assert "is not a parameter of this case study" in problems[0]
    assert "Import price" in problems[0]

    # So is a scope that names a carrier the case study does not have
    wrong_scope = json.loads(json.dumps(sound))
    wrong_scope["parameters"]["gas_price"]["scope"] = {"carriers": ["naturalgas"]}
    problems = validate_definition(wrong_scope, data_folder_path)
    assert len(problems) == 1
    assert "naturalgas" in problems[0]


def test_derived_parameter_must_be_determined(request):
    """
    Tests the check that gives the equations their point.

    A total spread over two nodes is one equation for two unknowns, so the study
    is relying on an assumption about the split that it has not written down.
    Adding it makes the system determined.
    """
    data_folder_path = _built_case_study(request)

    definition = {
        "analysis": {"name": {"value": "test"}, "combination": {"value": "grid"}},
        "derived": {
            "demand": {
                "description": "Total demand spread over the nodes.",
                "symbols": DEMAND_SYMBOLS,
                "defines": [
                    "sum(Demand[electricity, n] for n in nodes) == "
                    "total_demand_TWh * 1e6 / 8760"
                ],
            }
        },
    }

    problems = validate_definition(definition, data_folder_path)
    assert len(problems) == 1
    assert "1 equations for 2 unknowns" in problems[0]
    assert "1 more is needed" in problems[0]

    # Stating the assumption closes the system
    definition["derived"]["demand"]["defines"].append(
        "Demand[electricity, node1] == Demand[electricity, node2]"
    )
    assert validate_definition(definition, data_folder_path) == []


def test_derived_parameter_is_checked_against_the_case_study(request):
    """
    Tests that an equation naming something the case study does not have is
    caught, both for the parameter and for the index
    """
    data_folder_path = _built_case_study(request)

    definition = {
        "analysis": {"name": {"value": "test"}, "combination": {"value": "grid"}},
        "derived": {
            "demand": {
                "symbols": DEMAND_SYMBOLS,
                "defines": ["Demand[electricity, node3] == total_demand_TWh"],
            }
        },
    }
    problems = validate_definition(definition, data_folder_path)
    assert any("'node3' is not a node" in problem for problem in problems)

    definition["derived"]["demand"]["defines"] = [
        "Consumption[electricity, node1] == total_demand_TWh"
    ]
    problems = validate_definition(definition, data_folder_path)
    assert any("'Consumption' is not a parameter" in problem for problem in problems)


def test_solve_derived():
    """
    Tests standard behavior of solve_derived
    - Tests that the solution satisfies the equations it was built from
    - Tests that an intermediate quantity does not end up as something to write
    """
    topology = TEMPLATE_TOPOLOGY
    block = {
        "symbols": {
            "total_demand_TWh": {"values": [10]},
            "unbalance_ratio": {"values": [3]},
        },
        "defines": [
            "mean_demand_MW == total_demand_TWh * 1e6 / 8760",
            "sum(Demand[electricity, n] for n in nodes) == mean_demand_MW",
            "Demand[electricity, node1] == unbalance_ratio * Demand[electricity, node2]",
        ],
    }

    system = parse_equations(block["defines"], topology, set(block["symbols"]))
    assert len(system["equations"]) == len(system["unknowns"]) == 3

    solution = solve_derived(
        block, {"total_demand_TWh": 10, "unbalance_ratio": 3}, topology
    )

    # The intermediate is used to state the equations, not written anywhere
    assert set(solution) == {"Demand[electricity, node1]", "Demand[electricity, node2]"}

    total = 10 * 1e6 / 8760
    assert sum(solution.values()) == pytest.approx(total)
    assert solution["Demand[electricity, node1]"] == pytest.approx(
        3 * solution["Demand[electricity, node2]"]
    )


def test_write_builder(request):
    """
    Tests standard behavior of write_builder
    - Tests that the page is written and carries the data of the case study
    - Tests that both markers of the template are gone, so nothing of the empty
      data the template carries is left next to the data of the case study
    - Tests that the data still sits inside a statement, which is what a page
      that renders nothing at all looks like when it does not
    """
    import json as json_module

    from adopt_net0.scenario_analysis import write_builder
    from adopt_net0.scenario_analysis.build_analysis import END_PLACEHOLDER, PLACEHOLDER

    data_folder_path = _built_case_study(request)
    page_path = write_builder(data_folder_path)

    assert page_path.exists()
    page = page_path.read_text(encoding="utf-8")
    assert PLACEHOLDER not in page
    assert END_PLACEHOLDER not in page

    start = page.index("const DATA = ") + len("const DATA = ")
    data, end = json_module.JSONDecoder().raw_decode(page, start)

    assert data["case_name"] == data_folder_path.name
    assert len(data["catalogue"]) == len(scan_input_data(data_folder_path))

    with open(data_folder_path / "Topology.json", "r") as json_file:
        topology = json_file.read()
    assert data["topology"]["nodes"] == json_module.loads(topology)["nodes"]

    # Anything other than the end of the statement here is a second expression
    # sitting next to the first one, which is a syntax error, and a page whose
    # script does not parse draws nothing at all
    assert page[end:].lstrip().startswith(";")

    # The data sits inside a script tag, so nothing in it may close one
    assert "</script" not in page[start:end]


def _plan_definition(combination: str, naming: str = "readable") -> dict:
    """
    A definition with two direct parameters and one derived symbol, enough to
    tell the three combinations apart

    :param str combination: grid, list or lhs
    :param str naming: short, readable or hashed
    :return: the definition
    """
    return {
        "analysis": {
            "name": {"value": "test"},
            "combination": {"value": combination},
            "n_samples": {"value": 5},
            "seed": {"value": 42},
            "naming": {"value": naming},
        },
        "parameters": {
            "price": {"values": [100, 200]},
            "capex": {"values": [0.5, 1.5]},
        },
        "derived": {"demand": {"symbols": {"total": {"values": [5, 10]}}}},
    }


def test_plan_grid_and_list():
    """
    Tests that grid crosses every value and list pairs them element by element
    """
    grid = plan_scenarios(_plan_definition("grid"))
    assert len(grid) == 8
    assert grid[0]["values"] == {"price": 100, "capex": 0.5, "total": 5}
    assert grid[0]["label"] == "price-100__capex-0.5__total-5"
    assert grid[0]["name"] == grid[0]["label"]

    paired = plan_scenarios(_plan_definition("list"))
    assert [scenario["values"]["price"] for scenario in paired] == [100, 200]
    assert [scenario["values"]["total"] for scenario in paired] == [5, 10]

    uneven = _plan_definition("list")
    uneven["parameters"]["price"]["values"] = [100, 200, 300]
    with pytest.raises(ValueError):
        plan_scenarios(uneven)


def test_plan_lhs_is_reproducible():
    """
    Tests that the same seed draws the same scenarios, inside the range given
    """
    first = plan_scenarios(_plan_definition("lhs", "hashed"))
    second = plan_scenarios(_plan_definition("lhs", "hashed"))
    assert len(first) == 5
    assert first == second
    for scenario in first:
        assert 100 <= scenario["values"]["price"] <= 200
        assert 0.5 <= scenario["values"]["capex"] <= 1.5


def test_scenario_name_depends_on_values_only():
    """
    Tests that a name is a function of the values, and that a hashed name is
    the same whatever order the values come in
    """
    values = {"price": 100, "capex": 0.5}
    assert scenario_name(values) == "price-100__capex-0.5"
    assert scenario_name(values, "hashed") == scenario_name(
        {"capex": 0.5, "price": 100}, "hashed"
    )
    assert scenario_name(values, "hashed") != scenario_name(
        {"price": 100, "capex": 0.6}, "hashed"
    )

    assert len(scenario_name(values, "hashed")) == 11

    short = {"price": "p"}
    assert scenario_name(values, "short", short) == "p-100__capex-0.5"
    assert scenario_name(values, "readable", short) == "price-100__capex-0.5"


def test_plan_names_by_the_choice_of_the_analysis():
    """
    Tests that the names follow analysis/naming and the aliases, that the label
    stays the full spelling, and that a name too long is refused rather than
    hashed behind the user's back
    """
    definition = _plan_definition("grid", "short")
    definition["parameters"]["price"]["short"] = "p"
    definition["derived"]["demand"]["symbols"]["total"]["short"] = "d"
    scenarios = plan_scenarios(definition)
    assert scenarios[0]["name"] == "p-100__capex-0.5__d-5"
    assert scenarios[0]["label"] == "price-100__capex-0.5__total-5"

    hashed = plan_scenarios(_plan_definition("grid", "hashed"))
    assert all(len(scenario["name"]) == 11 for scenario in hashed)
    assert hashed[0]["label"] == "price-100__capex-0.5__total-5"

    too_long = _plan_definition("grid", "readable")
    too_long["parameters"]["a_parameter_with_a_very_long_name" * 7] = {"values": [1]}
    with pytest.raises(ValueError, match="hashed"):
        plan_scenarios(too_long)


def test_short_aliases_are_checked():
    """
    Tests that an alias a folder cannot hold, and two parameters spelled the
    same, are reported
    """
    from adopt_net0.scenario_analysis.define_analysis import _validate_short

    definition = _plan_definition("grid", "short")
    assert _validate_short(definition) == []

    definition["parameters"]["price"]["short"] = "capex"
    assert any("'capex'" in problem for problem in _validate_short(definition))

    definition["parameters"]["price"]["short"] = "p/q"
    assert any("letters, digits" in problem for problem in _validate_short(definition))


def test_write_and_read_plan(request):
    """
    Tests that the plan reads back as it was written
    """
    scenarios = plan_scenarios(_plan_definition("grid"))
    plan_file = write_plan(
        scenarios, request.config.result_folder_path / "plan" / "scenarios.csv"
    )
    assert read_plan(plan_file) == scenarios


def test_write_scenario(request):
    """
    Tests that a scenario is written to a copy, with its values applied and the
    base case left as it was
    """
    base = _built_case_study(request)
    carrier_file = base / "period1" / "node_data" / "node1" / "carrier_data"
    carrier_file = carrier_file / "electricity.csv"
    frame = pd.read_csv(carrier_file, sep=";", index_col=0)
    frame["Demand"] = np.linspace(1, 3, len(frame))
    frame.to_csv(carrier_file, sep=";")

    definition = {
        "parameters": {
            "days": {
                "target": "config",
                "parameter": "optimization/typicaldays/N",
                "operation": "set",
                "values": [4],
            },
            "demand": {
                "target": "carrier_data",
                "parameter": "Demand",
                "scope": {"nodes": ["node1"]},
                "operation": "set",
                "aggregate": "mean",
                "values": [10],
            },
        }
    }
    scenario = {"name": "a", "label": "a", "values": {"days": 4, "demand": 10}}
    folder = request.config.result_folder_path / "scenario"
    input_path = write_scenario(
        base, definition, scenario, folder / "input", folder / "results"
    )

    with open(input_path / "ConfigModel.json", "r") as json_file:
        config = json.load(json_file)
    assert config["optimization"]["typicaldays"]["N"]["value"] == 4
    assert config["reporting"]["case_name"]["value"] == "a"

    written = pd.read_csv(
        input_path / carrier_file.relative_to(base), sep=";", index_col=0
    )
    assert written["Demand"].mean() == pytest.approx(10)
    # The shape is kept, only the level moves
    assert written["Demand"].iloc[-1] / written["Demand"].iloc[0] == pytest.approx(3)

    base_frame = pd.read_csv(carrier_file, sep=";", index_col=0)
    assert base_frame["Demand"].mean() == pytest.approx(2)


def test_collect_results_leaves_out_what_is_not_solved(request):
    """
    Tests that the table holds the solved scenarios only, with their values and
    how they ended
    """
    folder = request.config.result_folder_path / "collect"
    scenarios = plan_scenarios(_plan_definition("list"))
    write_plan(scenarios, folder / "scenarios.csv")

    done = folder / "done" / f"{scenarios[0]['name']}.json"
    done.parent.mkdir(parents=True)
    with open(done, "w") as json_file:
        json.dump({"termination": "optimal", "wall_s": 1.0, "threads": 1}, json_file)

    table = collect_results(folder, folder / "summary.csv")
    assert list(table["name"]) == [scenarios[0]["name"]]
    assert table["price"].iloc[0] == 100
    assert table["termination"].iloc[0] == "optimal"
    assert (folder / "summary.csv").exists()
