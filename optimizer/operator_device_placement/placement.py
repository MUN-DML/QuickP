from enum import Enum

from optimizer.model.graph import CompGraph, DeviceGraph
from optimizer.operator_device_placement.non_contiguous_metis.near_optimal_placement import get_near_optimal_placement
from optimizer.operator_device_placement.metis.metis_partition import metis_partition
from optimizer.operator_device_placement.metis.subgraph_util import map_subgraph_to_device, construct_sub_graph, \
    identify_edges_cut
from optimizer.operator_device_placement.optimal.optimal_placement import get_optimize_placement
from optimizer.operator_device_placement.optimal_homo.optimal_placement_homo import get_optimize_placement_homo
from optimizer.optimization_problems.gurobi_util import get_subgraph_op_num_weight_sum_dict


class PlacementGenerator(Enum):
    METIS = "METIS"
    OPTIMIZED = "OPTIMIZED"
    NEAR_OPTIMAL = "NEAR_OPTIMAL"
    OPTIMIZED_HOMO = "OPTIMIZED_HOMO"


def get_placement_info(placement_type: str, comp_graph: CompGraph, device_topo: DeviceGraph):
    if placement_type == PlacementGenerator.METIS.value:
        partition_dict, edge_cut_list, edge_cut_weight_sum = metis_partition(comp_graph,
                        num_partitions=len(device_topo.getDeviceIDs()),
            )
        device_computing_cost_dict = comp_graph.get_comp_cost_sum_ratio(len(device_topo.getDeviceIDs()))
        _, partition_weights = get_subgraph_op_num_weight_sum_dict(comp_graph, partition_dict)
        # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
        operator_device_mapping = map_subgraph_to_device(partition_dict, device_topo.getDeviceIDs(), device_computing_cost_dict, partition_weights)

    elif placement_type == PlacementGenerator.OPTIMIZED.value:
        operator_device_mapping = get_optimize_placement(comp_graph, device_topo)
        edge_cut_list, edge_cut_weight_sum = identify_edges_cut(comp_graph, operator_device_mapping)

    elif placement_type == PlacementGenerator.OPTIMIZED_HOMO.value:
        operator_device_mapping = get_optimize_placement_homo(comp_graph, device_topo)
        edge_cut_list, edge_cut_weight_sum = identify_edges_cut(comp_graph, operator_device_mapping)

    elif placement_type == "NEAR_OPTIMAL":
        operator_device_mapping = get_near_optimal_placement(comp_graph, device_topo)
        print("fuck", operator_device_mapping)
        edge_cut_list, edge_cut_weight_sum = identify_edges_cut(comp_graph, operator_device_mapping)

    else:
        print(placement_type, "does not support this placement type")

    return operator_device_mapping, edge_cut_list, edge_cut_weight_sum
