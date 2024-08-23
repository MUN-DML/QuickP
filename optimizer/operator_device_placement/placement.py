from enum import Enum

from optimizer.model.graph import CompGraph, DeviceGraph
from optimizer.operator_device_placement.hetero_aware_load_balanced.hetero_balanced_placement import \
    get_hetero_balanced_placement
from optimizer.operator_device_placement.metis.metis_partition import metis_partition
from optimizer.operator_device_placement.metis.subgraph_util import map_subgraph_to_device, construct_sub_graph, \
    identify_edges_cut
from optimizer.operator_device_placement.optimal.optimal_placement import get_optimize_placement


class PlacementGenerator(Enum):
    METIS = "METIS"
    OPTIMIZED = "OPTIMIZED",
    HETERO_BALANCED = "HETERO_BALANCED",


def get_placement_info(placement_type: str, comp_graph: CompGraph, device_topo: DeviceGraph):
    if placement_type == PlacementGenerator.METIS.value:
        partition_dict, edge_cut_list, edge_cut_weight_sum = metis_partition(comp_graph,
                        num_partitions=len(device_topo.getDeviceIDs()),
            )

        # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
        operator_device_mapping = map_subgraph_to_device(partition_dict, device_topo.getDeviceIDs())

    elif placement_type == PlacementGenerator.OPTIMIZED.value:
        operator_device_mapping = get_optimize_placement(comp_graph, device_topo)
        edge_cut_list, edge_cut_weight_sum = identify_edges_cut(comp_graph, operator_device_mapping)

    elif placement_type == PlacementGenerator.HETERO_BALANCED.value:
        operator_device_mapping = get_hetero_balanced_placement(comp_graph, device_topo)
        edge_cut_list, edge_cut_weight_sum = identify_edges_cut(comp_graph, operator_device_mapping)

    return operator_device_mapping, edge_cut_list, edge_cut_weight_sum
