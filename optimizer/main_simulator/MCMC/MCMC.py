import random

from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.main_simulator.MCMC.cost_model import evaluate_mcmc
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph
from optimizer.model.graph import CompGraph, find_non_connected_pairs
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction, init_graph_weight, \
    construct_sub_graph
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.operator_device_placement.placement import get_placement_info
from optimizer.scheduling.scheduling_order_only import FIFO_scheduling_order


def mcmc_search(comp_graph: CompGraph, deviceTopo):
    all_non_connected_pairs = []
    # Partition the computation graph
    operator_device_mapping, edge_cut_list, edge_cut_weight_sum = (
        get_placement_info("METIS", comp_graph, deviceTopo))
    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
    device_subgraph_mapping = construct_sub_graph(comp_graph, operator_device_mapping)
    # device order mapping
    init_order_dict, _ = FIFO_scheduling_order(comp_graph, device_subgraph_mapping, edge_cut_list,
                                               operator_device_mapping)
    # Execute the simulation
    init_latency = evaluate_mcmc(comp_graph, deviceTopo, operator_device_mapping, edge_cut_list, init_order_dict)

    current_strategy = {"scheduling": init_order_dict, "latency": init_latency}

    for device, subgraph in device_subgraph_mapping.items():
        # there will be no pairs with the same element
        non_connected_pairs = find_non_connected_pairs(subgraph)
        all_non_connected_pairs.extend(non_connected_pairs)

    for i in range(0, 100):
        random_tuple = random.choice(all_non_connected_pairs)
        assigned_device = operator_device_mapping[random_tuple[0]]
        new_strategy = current_strategy["scheduling"].copy()
        assigned_sequence = new_strategy[assigned_device]
        # Find the indices of the two nodes
        index1 = assigned_sequence.index(random_tuple[0])
        index2 = assigned_sequence.index(random_tuple[1])

        # Swap the two nodes using multiple assignment
        assigned_sequence[index1], assigned_sequence[index2] = assigned_sequence[index2], assigned_sequence[index1]
        new_latency = evaluate_mcmc(comp_graph, deviceTopo, operator_device_mapping, edge_cut_list, init_order_dict)
        if new_latency < current_strategy["latency"]:
            current_strategy["scheduling"] = new_strategy
            current_strategy["latency"] = new_latency


if __name__ == '__main__':
    graph_init = {
        "number_of_devices": 6,
        "model_type": TFModelEnum.ALEXNET,
        "node_weight_function": NodeWeightFunction.AVE_COMP_COST,
        "edge_weight_function": EdgeWeightFunction.SOURCE_OUTPUT_TENSOR,
        "weight_norm_function": WeightNormalizationFunction.MIN_MAX,
    }

    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(graph_init["number_of_devices"], None,
                                                             model_type=graph_init["model_type"])
    # init graph node/edge weight
    init_graph_weight(comp_graph, graph_init["node_weight_function"], graph_init["edge_weight_function"],
                      graph_init["weight_norm_function"])

    mcmc_search(comp_graph, deviceTopo)
