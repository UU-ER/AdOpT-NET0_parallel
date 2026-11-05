from adopt_net0.scenario_analysis import ScenarioHub, create_scenario_builder

# Specify the path to the input data of a case study that is already built
path = "case_studies_benchmark/inputData/nl_node_demo/nl_node"

# Write the builder page to define the analysis with (comment this line if already
# defined). Open ScenarioBuilder.html in the input data folder and save its json
# as ScenarioAnalysis.json in the same folder
create_scenario_builder(path)

# Read the scenario analysis, check it and plan the scenarios
analysis = ScenarioHub()
analysis.read_definition(path)
analysis.plan()

# Solve the scenarios in parallel (run again to continue after an interruption)
analysis.run()

# Collect the results of all scenarios into summary.csv
analysis.collect()
