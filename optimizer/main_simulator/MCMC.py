from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph
from optimizer.main_simulator.simulator import simulate
from optimizer.model.graph import CompGraph
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction, init_graph_weight
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.scheduling.near_optimal_scheduling_with_sampling import SamplingFunction
from optimizer.scheduling.scheduling_order_only import FIFO_scheduling_order


def mcmc_search(comp_graph: CompGraph, deviceTopo,
                    device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict, ):
    init_order_dict, _ = FIFO_scheduling_order(comp_graph, device_subgraph_mapping, edge_cut_list,
                                               operator_device_mapping)
    time_limit = 120

    # Execute the simulation
    expected_training = simulate(comp_graph, deviceTopo, placement='METIS', scheduling_function='mcmc')


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
