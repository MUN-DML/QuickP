from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.main_simulator.MCMC.cost_model import evaluate_mcmc
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph
from optimizer.model.graph import CompGraph
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction, init_graph_weight, \
    construct_sub_graph
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.operator_device_placement.placement import get_placement_info
from optimizer.scheduling.scheduling_order_only import FIFO_scheduling_order


def mcmc_search(comp_graph: CompGraph, deviceTopo,
                ):
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

    # for device, subgraph in device_subgraph_mapping.items():




    time_limit = 120


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
