import copy
import random

from DNN_model_tf.tf_model_enum import TFModelEnum
from baseline.MCMC.cost_model import evaluate_mcmc
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph, get_proper_M
from optimizer.model.graph import CompGraph, find_non_connected_pairs
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction, init_graph_weight, \
    construct_sub_graph
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.operator_device_placement.placement import get_placement_info


def mcmc_search(comp_graph: CompGraph, deviceTopo):
    all_non_connected_pairs = []
    M = get_proper_M(graph_init["model_type"])
    # Partition the computation graph
    operator_device_mapping, edge_cut_list, edge_cut_weight_sum = (
        get_placement_info("RANDOM", comp_graph, deviceTopo, M))
    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
    device_subgraph_mapping = construct_sub_graph(comp_graph, operator_device_mapping)

    # Execute the simulation
    init_latency = evaluate_mcmc(comp_graph, deviceTopo, operator_device_mapping, edge_cut_list)

    current_strategy = {"placement": operator_device_mapping, "latency": init_latency}


    for i in range(0, 500):
        random_node = random.choice(comp_graph.getOperatorIDs())
        random_device = random.choice(deviceTopo.getDeviceIDs())
        new_placement = copy.deepcopy(current_strategy["placement"])
        new_placement[random_node] = random_device

        # Swap the two nodes using multiple assignment
        new_latency = evaluate_mcmc(comp_graph, deviceTopo, new_placement, edge_cut_list)
        if new_latency < current_strategy["latency"]:
            current_strategy["placement"] = new_placement
            current_strategy["latency"] = new_latency

    print(current_strategy["latency"])


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
