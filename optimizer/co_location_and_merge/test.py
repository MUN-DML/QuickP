import argparse

from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.co_location_and_merge.grouper_util import create_colocation_group_to_ops_map, label_all_node_with_group, \
    merge_group, analyze_group, edge_based_label
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction, init_graph_weight
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for optimization problem after graph partitioning')
    parser.add_argument('--number_of_device', type=int, default=6)
    # TEST SMALL
    parser.add_argument('--model', type=str, default='VGG')
    parser.add_argument('--normalization_function', default='MIN_MAX', type=str, help='')
    # NEAR_OPTIMAL OPTIMIZED METIS TEST OPTIMIZED_HOMO INCONTIGUOUS_METIS


    args = parser.parse_args()

    # Dynamically access attributes using getattr
    model_type = getattr(TFModelEnum, args.model, None)
    weight_norm_function = getattr(WeightNormalizationFunction, args.normalization_function.upper(), None)

    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(args.number_of_device, None, model_type=TFModelEnum.TEST)

    computing_cost_dict = {"a": 0, "b": 0, "c": 0, "d": 100, "e": 0, "f": 0, "g":50}
    edge_based_label(comp_graph, deviceTopo, computing_cost_dict)
    for op_id, op_data in comp_graph.nodes(data=True):
        if 'colocation_group' in op_data:
            print(op_id, op_data['colocation_group'])
    print('Ready to merge')
    merge_group(comp_graph)
    print('After merge')
    for op_id, op_data in comp_graph.nodes(data=True):
        if 'colocation_group' in op_data:
            print(op_id, op_data['colocation_group'])
    map = create_colocation_group_to_ops_map(comp_graph)
    analyze_group(map, computing_cost_dict)