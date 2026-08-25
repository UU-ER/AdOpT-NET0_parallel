set -e
# PV off, storage on: the only source of time variability is the price and
# the demand, so switching them isolates difficulty at identical size
R="python run_benchmark.py run --case four_node --typicaldays 4 --mipgap 0.02 --set pv=off --set storage=on --set pipeline_capex=linear --set bidirectional_precise=0 --set pipeline_size_min=0"
$R --set electricity_price=constant    --set hydrogen_demand=constant
$R --set electricity_price=fluctuating --set hydrogen_demand=constant
$R --set electricity_price=constant    --set hydrogen_demand=fluctuating
$R --set electricity_price=fluctuating --set hydrogen_demand=fluctuating
