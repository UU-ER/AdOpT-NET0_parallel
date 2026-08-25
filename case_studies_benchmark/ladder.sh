set -e
R="python run_benchmark.py run --case four_node --typicaldays 4 --mipgap 0.02"
$R --set pv=off --set storage=off --set electricity_price=constant --set hydrogen_demand=constant --set pipeline_capex=linear --set bidirectional_precise=0 --set pipeline_size_min=0
$R --set pv=on  --set storage=off --set electricity_price=constant --set hydrogen_demand=constant --set pipeline_capex=linear --set bidirectional_precise=0 --set pipeline_size_min=0
$R --set pv=on  --set storage=on  --set electricity_price=constant --set hydrogen_demand=constant --set pipeline_capex=linear --set bidirectional_precise=0 --set pipeline_size_min=0
$R --set pv=on  --set storage=on  --set electricity_price=fluctuating --set hydrogen_demand=fluctuating --set pipeline_capex=linear --set bidirectional_precise=0 --set pipeline_size_min=0
$R --set pv=on  --set storage=on  --set electricity_price=fluctuating --set hydrogen_demand=fluctuating --set pipeline_capex=fixed_plus_linear --set bidirectional_precise=0 --set pipeline_size_min=0
$R --set pv=on  --set storage=on  --set electricity_price=fluctuating --set hydrogen_demand=fluctuating --set pipeline_capex=fixed_plus_linear --set bidirectional_precise=0 --set pipeline_size_min=250
$R --set pv=on  --set storage=on  --set electricity_price=fluctuating --set hydrogen_demand=fluctuating --set pipeline_capex=fixed_plus_linear --set bidirectional_precise=1 --set pipeline_size_min=250
$R --set pv=on  --set storage=on  --set electricity_price=fluctuating --set hydrogen_demand=fluctuating --set pipeline_capex=fixed_plus_linear --set bidirectional_precise=1 --set pipeline_size_min=250 --set storage_precise=1
