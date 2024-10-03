import networkx as nx
from gurobipy import Model, GRB
from networkx import Graph

from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph
from optimizer.main_simulator.quicks.cost_model_quicks import evaluate_quicks
from optimizer.model.graph import CompGraph
from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction, init_graph_weight, \
    construct_sub_graph
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.operator_device_placement.placement import get_placement_info
from optimizer.scheduling.near_optimal_scheduling_with_sampling import get_device_unreachable_pairs_mapping, split_nodes
from optimizer.main_simulator.quicks.quicks_list_schedule import quicks_list_schedule
from optimizer.scheduling.scheduling_util import split_three_stage_subgraph


def quickS(comp_graph: CompGraph, deviceTopo):
    # Partition the computation graph
    operator_device_mapping, edge_cut_list, edge_cut_weight_sum = (
        get_placement_info("METIS", comp_graph, deviceTopo))
    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
    device_subgraph_mapping = construct_sub_graph(comp_graph, operator_device_mapping)

    rank_map = evaluate_quicks(comp_graph, deviceTopo, operator_device_mapping, edge_cut_list)
    quicks_list_schedule(model, start, finish, comm_start, comm_end, comp_graph,
                         device_subgraph_mapping, operator_device_mapping, rank_map)


def calculate_rank_map(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                       device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict,
                       rho, sampling_function):
    rank_map = {}
    M = 1000000
    topological_order = list(nx.topological_sort(comp_graph))
    topological_order_mapping = {node: index for index, node in enumerate(topological_order)}
    order = {}
    stage_to_be_optimize_mapping: dict[any, Graph] = {}
    stage_two_finish = model.addVars(device_subgraph_mapping.keys(), vtype=GRB.CONTINUOUS, lb=0.0,
                                     name="non_isolated_part_finish")

    # form new device non-isolated part mapping
    # split into isolated and non-isolated part
    for device, subgraph in device_subgraph_mapping.items():
        # Simply the search space by
        stage_one, stage_two, stage_three = split_three_stage_subgraph(
            subgraph, operator_device_mapping, edge_cut_list)
        # give stage one the highest rank and the three the lowest rank
        for stage_one_node in stage_one:
            rank_map[stage_one_node] = 10000000
        for stage_three_node in stage_three:
            rank_map[stage_three_node] = 0

        # Map stage two to device
        stage_to_be_optimize_mapping[device] = stage_two

        # we only care about
        # independent relied part => random topo sort, this stage should always be first, we don't care know
        stage_one_sorted = sorted(list(stage_one.nodes), key=lambda node: topological_order_mapping[node])
        for a, b in zip(stage_one_sorted, stage_one_sorted[1:]):
            model.addConstr(finish[a] <= start[b])
        # stage_three => non-exporting part, FIFO is okay, we don't care
        stage_three_sorted = sorted(list(stage_one.nodes), key=lambda node: topological_order_mapping[node])
        for a, b in zip(stage_three_sorted, stage_three_sorted[1:]):
            model.addConstr(finish[a] <= start[b])

        # nonoverlapping between stage 1 and 2
        for stage_two_node in stage_two:
            model.addConstr(stage_two_finish[device] >= finish[stage_two_node])
            model.addConstr(start[stage_two_node] >= finish[stage_one_sorted[-1]])
        # # nonoverlapping between stage 2 and 3
        model.addConstr(start[stage_three_sorted[0]] >= stage_two_finish[device])

    device_unreachable_pairs_mapping, global_set_with_nr = get_device_unreachable_pairs_mapping(
        stage_to_be_optimize_mapping)
    global_node_split_by_device = split_nodes(comp_graph, global_set_with_nr, list(device_subgraph_mapping.keys()),
                                              operator_device_mapping, r=rho,
                                              sampling_function=sampling_function)

    for device, non_iso_part in stage_to_be_optimize_mapping.items():

        # flatten the pairs to get all the non-repeated nodes, convert to list
        selected_nodes, other_nodes = global_node_split_by_device.get(device)["selected_list"], \
            global_node_split_by_device.get(device)["unselected_list"]

        # use the FIFO order to sort other_nodes;
        local_fifo_order = fifo_operator_order[device]
        node_order_dict = {op: idx for idx, op in enumerate(local_fifo_order)}
        # sort other_nodes based on the FIFO order
        other_nodes = sorted(other_nodes, key=lambda op: node_order_dict[op])
        # Apply sequential constraint to non-selected nodes
        for op_a, op_b in zip(other_nodes, other_nodes[1:]):
            model.addConstr(finish[op_a] <= start[op_b])

        # apply optimization to all pairs involving these high-cost nodes, these pair also include the unselected nodes,
        # To optimize one node, all of its non-reachable node should be included
        important_pairs = [pair for pair in device_unreachable_pairs_mapping[device] if
                           pair[0] in selected_nodes or pair[1] in selected_nodes]
        for op_a, op_b in important_pairs:
            current_subgraph_dc = stage_to_be_optimize_mapping[device]
            assert op_a in current_subgraph_dc.nodes and op_b in current_subgraph_dc.nodes
            assert not nx.has_path(current_subgraph_dc, op_a, op_b)
            order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"order_{op_a}_{op_b}")
            model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]), name=f"NoOverlap1_{op_a}_{op_b}")
            model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")

    return rank_map


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

    quickS(comp_graph, deviceTopo)
