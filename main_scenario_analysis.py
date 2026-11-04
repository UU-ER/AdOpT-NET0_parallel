import adopt_net0.data_preprocessing as dp
from adopt_net0.scenario_analysis import ScenarioHub

# Specify the path to the input data of a case study that is already built
path = "case_studies_benchmark/inputData/nl_node_demo/nl_node"

# Create the scenario analysis template (comment this line if already defined)
#dp.create_scenario_analysis_template(path)

# Read the scenario analysis and write the builder page to define it with
# (open ScenarioBuilder.html in the input data folder)
analysis = ScenarioHub()
analysis.read_definition(path)
analysis.write_builder()

# Check the definition and plan the scenarios
analysis.plan()

# Solve the scenarios in parallel (run again to continue after an interruption)
analysis.run()

# Collect the results of all scenarios into summary.csv
analysis.collect()
