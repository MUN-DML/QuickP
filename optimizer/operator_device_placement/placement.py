from enum import Enum

from optimizer.model.graph import CompGraph, DeviceGraph
from optimizer.operator_device_placement.metis.metis_partition import metis_partition
from optimizer.operator_device_placement.metis.subgraph_util import map_subgraph_to_device, identify_edges_cut
from optimizer.operator_device_placement.optimal.optimal_placement_with_grouper import \
    get_optimize_placement_with_grouper
from optimizer.main_simulator.gurobi_util import get_subgraph_op_num_weight_sum_dict
from optimizer.operator_device_placement.single_device_placement import get_mcmc_init_device_placement
from optimizer.scheduling.random_placement import get_random_device_placement


class PlacementGenerator(Enum):
    METIS = "METIS"
    INIT_MCMC = "INIT_MCMC"
    OPTIMIZED_GROUPER = "OPTIMIZED_GROUPER"
    RANDOM = "RANDOM"


def get_placement_info(placement_type: str, comp_graph: CompGraph, device_topo: DeviceGraph, M, model_type):
    if placement_type == PlacementGenerator.METIS.value:
        partition_dict, edge_cut_list, edge_cut_weight_sum = metis_partition(comp_graph,
                        num_partitions=len(device_topo.getDeviceIDs()),
            )
        device_computing_cost_dict = comp_graph.get_comp_cost_sum_ratio(len(device_topo.getDeviceIDs()))
        _, partition_weights = get_subgraph_op_num_weight_sum_dict(comp_graph, partition_dict)
        # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
        operator_device_mapping = map_subgraph_to_device(partition_dict, device_topo.getDeviceIDs(), device_computing_cost_dict, partition_weights)

    elif placement_type == PlacementGenerator.INIT_MCMC.value:
        operator_device_mapping = get_mcmc_init_device_placement(comp_graph, device_topo, M)
        edge_cut_list, edge_cut_weight_sum = identify_edges_cut(comp_graph, operator_device_mapping)

    elif placement_type == PlacementGenerator.RANDOM.value:
        operator_device_mapping = get_random_device_placement(comp_graph, device_topo, M)
        edge_cut_list, edge_cut_weight_sum = identify_edges_cut(comp_graph, operator_device_mapping)

    elif placement_type == PlacementGenerator.OPTIMIZED_GROUPER.value:
        operator_device_mapping = get_optimize_placement_with_grouper(comp_graph, device_topo, M, model_type)
        edge_cut_list, edge_cut_weight_sum = identify_edges_cut(comp_graph, operator_device_mapping)

    else:
        print(placement_type, "does not support this placement type")

    return operator_device_mapping, edge_cut_list, edge_cut_weight_sum
