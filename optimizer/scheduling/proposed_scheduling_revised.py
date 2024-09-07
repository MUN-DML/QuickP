import itertools
import random
from enum import Enum
from itertools import combinations

import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import CompGraph, find_non_connected_pairs, is_not_connected
from optimizer.scheduling.scheduling_order_only import FIFO_scheduling_order


def near_optimal_scheduling_revised(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                                    device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict, rho):
    # The global data dependency is already applied
    M = 1000000
    order = {}
    fifo_operator_order, _ = FIFO_scheduling_order(comp_graph, device_subgraph_mapping, edge_cut_list,
                                                   operator_device_mapping)

    global_set_with_nr = list(get_global_node_set_with_nr(device_subgraph_mapping))
    global_node_split_by_device = split_list_based_on_score(comp_graph, global_set_with_nr, device_subgraph_mapping,
                                                                 edge_cut_list, operator_device_mapping, r=rho)

    for device, subgraph in device_subgraph_mapping.items():
        # there will be no pairs with the same element
        non_connected_pairs = find_non_connected_pairs(subgraph)
        # flatten the pairs to get all the non-repeated nodes, convert to list
        selected_nodes, other_nodes = global_node_split_by_device.get(device)["selected_list"], global_node_split_by_device.get(device)["unselected_list"]
        # print('fuck', comp_graph.getOperatorCompCostByDevice(high_cost_nodes[-1], operator_device_mapping[high_cost_nodes[-1]]),
        #       comp_graph.getOperatorCompCostByDevice(other_nodes[0], operator_device_mapping[other_nodes[0]]))

        # use the FIFO order to sort other_nodes;
        local_fifo_order = fifo_operator_order[device]
        node_order_dict = {op: idx for idx, op in enumerate(local_fifo_order)}
        # sort other_nodes based on the FIFO order
        other_nodes = sorted(other_nodes, key=lambda op: node_order_dict[op])
        print('other', len(other_nodes), other_nodes)
        # Apply sequential constraint to non-selected nodes
        for op_a, op_b in zip(other_nodes, other_nodes[1:]):
            model.addConstr(finish[op_a] <= start[op_b])

        # use the topo order to sort selected nodes;
        topological_order = list(nx.topological_sort(subgraph))
        topological_order_mapping = {node: index for index, node in enumerate(topological_order)}
        selected_nodes = sorted(selected_nodes, key=lambda node: topological_order_mapping[node])
        # apply optimization to these high-cost node pairs
        for high_cost_node in selected_nodes:
            non_connected_pair = [pair for pair in non_connected_pairs if high_cost_node in pair]
            for op_a, op_b in non_connected_pair:
                order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"order_{op_a}_{op_b}")
                model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]),
                                name=f"NoOverlap1_{op_a}_{op_b}")
                model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")
        print('selected', len(selected_nodes), selected_nodes)

    for device, subgraph in device_subgraph_mapping.items():
        outgoings = [edge for edge in edge_cut_list if edge[0] in subgraph]
        topo_order = list(nx.topological_sort(subgraph))
        # Create a mapping of nodes to their topological order position
        topo_order_map = {node: index for index, node in enumerate(topo_order)}
        # Sort the edges based on the topological order of the source nodes
        sorted_outgoings = sorted(outgoings, key=lambda edge: topo_order_map[edge[0]])
        for comm1, comm2 in combinations(sorted_outgoings, 2):
            source_node_1 = comm1[0]
            source_node_2 = comm2[0]
            # in this case, these two nodes does not have dependency, implement FCFS policy
            if is_not_connected(subgraph, source_node_1, source_node_2):
                order_1_first = model.addVar(vtype=GRB.BINARY, name=f"order_{source_node_1}_first_{source_node_2}")
                # Enforce the order based on the finish times using Big M

                # If order_1_first == 1, communication 1 finishes before communication 2 starts
                model.addConstr(comm_start[comm2] >= comm_end[comm1] - M * (1 - order_1_first),
                                name=f"FCFS_comm1_first_{source_node_1}_{source_node_2}")

                # If order_1_first == 0, communication 2 finishes before communication 1 starts
                model.addConstr(comm_start[comm1] >= comm_end[comm2] - M * order_1_first,
                                name=f"FCFS_comm2_first_{source_node_1}_{source_node_2}")
            # in this case, a must be b's preceding node
            else:
                assert nx.has_path(subgraph, source_node_1, source_node_2)
                model.addConstr(comm_end[comm1] <= comm_start[comm2])


class SamplingFunction(Enum):
    PROBABILISTIC_SAMPLING = "probabilistic"
    RANDOM = "random"
    HEAVY_HITTER = "heavy_hitter"


def get_global_node_set_with_nr(device_subgraph_mapping: dict):
    global_all_nodes = set()
    for device, subgraph in device_subgraph_mapping.items():
        # there will be no pairs with the same element
        non_connected_pairs = find_non_connected_pairs(subgraph)
        # flatten the pairs to get all the non-repeated nodes, convert to list
        all_nodes = list({node for pair in non_connected_pairs for node in pair})
        global_all_nodes.update(all_nodes)
    return global_all_nodes


def split_list_based_on_score(graph: CompGraph, node_list, device_subgraph_mapping: dict[any, CompGraph], edge_cut_list,
                              operator_device_mapping, r, sampling_function=SamplingFunction.HEAVY_HITTER) -> dict:

    result_dict = {device_id: {"selected_list": [], "unselected_list": []} for device_id in device_subgraph_mapping.keys()}

    def get_related_subgraph_num(node):
        device = operator_device_mapping[node]
        current_subgraph = device_subgraph_mapping.get(device)

        outgoing_edges = [(u, v) for u, v in edge_cut_list if operator_device_mapping.get(u) == device]
        related_devices = set()
        outgoing_edges_depended = [(u, v) for u, v in outgoing_edges
                                    if nx.has_path(graph, node, u)]
        for u, v in outgoing_edges_depended:
            assert device_subgraph_mapping.get(operator_device_mapping.get(u)) == current_subgraph
            related_devices.add(operator_device_mapping.get(v))

    def evaluate_node(node):
        assigned_device = operator_device_mapping[node]
        computing_cost = graph.getOperatorCompCostByDevice(node, assigned_device)
        # Set to track unique sub_graphs that depend on this operator
        return computing_cost

    node_score_mapping = {node: evaluate_node(node) for node in node_list}
    if sampling_function == SamplingFunction.HEAVY_HITTER:
        # List to store pairs where both nodes have a computing cost > 5
        # First sort the node list based on the computing cost condition
        node_list_sorted = sorted(node_list, key=lambda node: (node_score_mapping[node], get_related_subgraph_num(node)), reverse=True)
        threshold_index = int(len(node_list_sorted) * r)
        print('number of selected in total', threshold_index)
        selected_nodes, unselected_nodes = node_list_sorted[:threshold_index], node_list_sorted[threshold_index:]
    elif sampling_function == SamplingFunction.RANDOM:
        num_to_select = int(len(node_list) * r)
        selected_nodes = random.sample(node_list, num_to_select)
        unselected_nodes = [node for node in node_list if node not in selected_nodes]
    else:
        assert sampling_function == SamplingFunction.PROBABILISTIC_SAMPLING
        # Calculate the total score for all nodes
        total_score = sum(node_score_mapping.values())
        # Calculate the probability for each node
        node_probabilities = [node_score_mapping[node] / total_score for node in node_list]
        # Sample nodes probabilistically based on the computed probabilities
        selected_nodes = random.choices(node_list, weights=node_probabilities, k=int(len(node_list) * r))
        # Get the unselected nodes (those not in selected_nodes)
        unselected_nodes = [node for node in node_list if node not in selected_nodes]

    # Map selected_nodes and unselected_nodes to the corresponding device in result_dict
    for selected_node in selected_nodes:
        result_dict[operator_device_mapping[selected_node]]["selected_list"].append(selected_node)
    for unselected_node in unselected_nodes:
        result_dict[operator_device_mapping[unselected_node]]["unselected_list"].append(unselected_node)

    return result_dict
