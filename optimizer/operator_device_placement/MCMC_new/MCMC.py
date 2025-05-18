import copy
import datetime
import os
import random
import sys
from collections import defaultdict

import networkx as nx

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.append(project_root)
from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.main_simulator.simulator_util import get_comp_cost_dict, get_comm_cost_dict
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph, get_proper_M
from optimizer.model.graph import CompGraph, DeviceGraph
from optimizer.operator_device_placement.MCMC_new.cost_model_priority import evaluate_mcmc_priority
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction, init_graph_weight, \
    construct_sub_graph, identify_edges_cut
from optimizer.operator_device_placement.placement import get_placement_info


def invert_mapping(op_device_mapping):
    device_ops_mapping = defaultdict(list)
    for op, device in op_device_mapping.items():
        device_ops_mapping[device].append(op)
    return dict(device_ops_mapping)


def mcmc_search(comp_graph: CompGraph, deviceTopo: DeviceGraph):
    devices = deviceTopo.getDeviceIDs()
    # build the global rank
    op_priority_map = {}
    topo_sorted = list(nx.topological_sort(comp_graph))

    for current_node in reversed(topo_sorted):
        # Check if the current node has any predecessors
        successors = list(comp_graph.successors(current_node))

        if successors:  # If there are predecessors, compute the max computing cost
            max_suc_computing_cost = max(
                op_priority_map[succ_node] for succ_node in successors
            )
        else:  # If there are no predecessors, set the max computing cost to 0
            max_suc_computing_cost = 0

        # Calculate the global rank for the current node
        op_priority_map[current_node] = (
                max_suc_computing_cost + comp_graph.getOperatorCompCostByDevice(current_node, devices[0])
        )

    M = get_proper_M(graph_init["model_type"])
    # Partition the computation graph
    operator_device_mapping, edge_cut_list, edge_cut_weight_sum = (
        get_placement_info("INIT_MCMC", comp_graph, deviceTopo, M, graph_init["model_type"]))
    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict

    op_computing_cost_mapping = get_comp_cost_dict(comp_graph, operator_device_mapping)
    edge_cut_communication_cost_mapping = get_comm_cost_dict(comp_graph, deviceTopo, edge_cut_list, operator_device_mapping)
    device_ops_mapping = invert_mapping(operator_device_mapping)

    # Execute the simulation
    init_latency = evaluate_mcmc_priority(comp_graph, device_ops_mapping, op_computing_cost_mapping, edge_cut_communication_cost_mapping, op_priority_map)

    current_strategy = {"placement": operator_device_mapping, "latency": init_latency}

    beginning_time = datetime.datetime.now()

    for i in range(0, 25000):
        random_node = random.choice(comp_graph.getOperatorIDs())
        random_device = random.choice(deviceTopo.getDeviceIDs())
        new_placement = copy.deepcopy(current_strategy["placement"])
        new_placement[random_node] = random_device
        new_edge_cut_list, _ = identify_edges_cut(comp_graph, new_placement)
        edge_cut_communication_cost_mapping = get_comm_cost_dict(comp_graph, deviceTopo, new_edge_cut_list, new_placement)
        new_device_ops_mapping = invert_mapping(new_placement)

        # Swap the two nodes using multiple assignment
        new_latency = evaluate_mcmc_priority(comp_graph, new_device_ops_mapping, op_computing_cost_mapping,
                                    edge_cut_communication_cost_mapping, op_priority_map)
        if new_latency < current_strategy["latency"]:
            current_strategy["placement"] = new_placement
            current_strategy["latency"] = new_latency
        if i % 500 == 0:
            ending_time = datetime.datetime.now()
            print("device number ", graph_init["number_of_devices"], "intel-band", graph_init["intel_band"],"model", graph_init["model_type"]
                  , "The latency is ", current_strategy["latency"], "Step number is ", i, "time lapsed in seconds",
                  datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))


if __name__ == '__main__':
    graph_init = {
        "number_of_devices": 10,
        "model_type": TFModelEnum.ALEXNET,
        'intel_band': 25,
    }

    deviceTopo, comp_graph = init_computing_and_device_graph(graph_init["number_of_devices"],
                                                             model_type=graph_init["model_type"], gpu_per_server=1,
                                                             intra_server_bandwidth=None,
                                                             intel_server_bandwidth=graph_init["intel_band"])

    mcmc_search(comp_graph, deviceTopo)
