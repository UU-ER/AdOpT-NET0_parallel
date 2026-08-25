set -e
for TD in 2 4 8; do
  B="python run_benchmark.py run --case four_node --typicaldays $TD --mipgap 0.02 --set pipeline_capex=linear --set bidirectional_precise=0 --set pipeline_size_min=0"
  $B --set pv=off --set storage=off --set electricity_price=constant --set hydrogen_demand=constant
  $B --set pv=off --set storage=on  --set electricity_price=constant --set hydrogen_demand=constant
  $B --set pv=off --set storage=on  --set electricity_price=fluctuating --set hydrogen_demand=fluctuating
done
