# Author: Hong Wang
# Description: The is the deprecated file for MCMC execution, I code a new one with a much higher efficiency in MCMC_new/MCMC.py

import copy
import datetime
import os
import random
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(project_root)
from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.operator_device_placement.MCMC_deprecated.cost_model import evaluate_mcmc
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph, get_proper_M
from optimizer.model.graph import CompGraph
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction, init_graph_weight, \
    construct_sub_graph
from optimizer.operator_device_placement.placement import get_placement_info


def mcmc_search(comp_graph: CompGraph, deviceTopo):
    all_non_connected_pairs = []
    M = get_proper_M(graph_init["model_type"])
    # Partition the computation graph
    operator_device_mapping, edge_cut_list, edge_cut_weight_sum = (
        get_placement_info("RANDOM", comp_graph, deviceTopo, M, graph_init["model_type"]))
    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
    device_subgraph_mapping = construct_sub_graph(comp_graph, operator_device_mapping)

    # Execute the simulation
    init_latency = evaluate_mcmc(comp_graph, deviceTopo, operator_device_mapping, edge_cut_list)

    current_strategy = {"placement": operator_device_mapping, "latency": init_latency}

    beginning_time = datetime.datetime.now()

    for i in range(0, 1000000):
        random_node = random.choice(comp_graph.getOperatorIDs())
        random_device = random.choice(deviceTopo.getDeviceIDs())
        new_placement = copy.deepcopy(current_strategy["placement"])
        new_placement[random_node] = random_device

        # Swap the two nodes using multiple assignment
        new_latency = evaluate_mcmc(comp_graph, deviceTopo, new_placement, edge_cut_list)
        if new_latency < current_strategy["latency"]:
            current_strategy["placement"] = new_placement
            current_strategy["latency"] = new_latency
        if i % 100 == 0:
            ending_time = datetime.datetime.now()
            print("device number ", graph_init["number_of_devices"], "model",  graph_init["model_type"]
                  ,"The latency is ", current_strategy["latency"], "Step number is ", i, "time lapsed in seconds",
                  datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))


if __name__ == '__main__':
    graph_init = {
        "number_of_devices": 2,
        "model_type": TFModelEnum.ALEXNET,
    }

    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(graph_init["number_of_devices"],
                                                             model_type=graph_init["model_type"])

    mcmc_search(comp_graph, deviceTopo)
