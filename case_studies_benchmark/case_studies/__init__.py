from . import network
from . import four_node

# Every case study module needs a NAME attribute and a setup function
CASE_STUDIES = {
    network.NAME: network,
    four_node.NAME: four_node,
}
