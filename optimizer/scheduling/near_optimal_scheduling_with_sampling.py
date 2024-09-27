import itertools
import random
from enum import Enum
from itertools import combinations

import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import CompGraph, find_non_connected_pairs, is_not_connected
from optimizer.scheduling.scheduling_util import split_subgraph, handle_terminal_components_with_comm_end_point
from optimizer.scheduling.scheduling_order_only import FIFO_scheduling_order, heteroG_scheduling_order


def near_optimal_scheduling_with_sampling(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                                          device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict,
                                          rho, sampling_function):
    # The global data dependency is already applied
    M = 1000000
    order = {}
    device_DC_mapping = {}
    fifo_operator_order, _ = FIFO_scheduling_order(comp_graph, device_subgraph_mapping, edge_cut_list,
                                                   operator_device_mapping)
    heteroG_operator_order, _ = heteroG_scheduling_order(comp_graph, device_subgraph_mapping, edge_cut_list,
                                                   operator_device_mapping)

    device_DC_finish = model.addVars(device_subgraph_mapping.keys(), vtype=GRB.CONTINUOUS, lb=0.0,
                                                    name="non_isolated_part_finish")
    topological_order = list(nx.topological_sort(comp_graph))
    topological_order_mapping = {node: index for index, node in enumerate(topological_order)}

    # form new device non-isolated part mapping
    # split into isolated and non-isolated part
    for device, subgraph in device_subgraph_mapping.items():
        # Simply the search space by
        depended_component, wccs, isolated_node_list, terminal_nodes_without_comm_np= split_subgraph(subgraph, operator_device_mapping, edge_cut_list)
        # Map non_iso_part to device
        device_DC_mapping[device] = depended_component

        # Merge isolated_node_list and sink_with_source_node_dependency
        stage_two = isolated_node_list | terminal_nodes_without_comm_np
        # stage_2_3 = stage_two | set(wccs.nodes())

        # Sort the isolated node list according to topo order and apply a sequential constraint, from set to sorted list
        # stage_two = sorted(list(stage_two), key=lambda node: topological_order_mapping[node])
        # local_fifo_order = fifo_operator_order[device]
        local_fifo_order = heteroG_operator_order[device]
        node_order_dict = {op: idx for idx, op in enumerate(local_fifo_order)}
        stage_two = sorted(list(stage_two), key=lambda node: node_order_dict[node])

        for a, b in zip(stage_two, stage_two[1:]):
            model.addConstr(finish[a] <= start[b])
        # the isolated part will start after the non-isolated part finished
        for depended_node in depended_component.getOperatorIDs():
            model.addConstr(device_DC_finish[device] >= finish[depended_node])
        if len(stage_two) > 0:
            model.addConstr(start[stage_two[0]] >= device_DC_finish[device])

        # Sort the sink_components
        handle_terminal_components_with_comm_end_point(subgraph, wccs, device, operator_device_mapping, edge_cut_list, model, start, finish, stage_two[-1], heteroG_operator_order)

    device_unreachable_pairs_mapping, global_set_with_nr = get_device_unreachable_pairs_mapping(device_DC_mapping)
    global_node_split_by_device = split_nodes(comp_graph, global_set_with_nr, list(device_subgraph_mapping.keys()), operator_device_mapping, r=rho,
                                              sampling_function=sampling_function)

    for device, non_iso_part in device_DC_mapping.items():

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
            current_subgraph_dc = device_DC_mapping[device]
            assert op_a in current_subgraph_dc.nodes and op_b in current_subgraph_dc.nodes
            assert not nx.has_path(current_subgraph_dc, op_a, op_b)
            order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"order_{op_a}_{op_b}")
            model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]), name=f"NoOverlap1_{op_a}_{op_b}")
            model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")

    '''
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
        '''


class SamplingFunction(Enum):
    PROBABILISTIC_SAMPLING = "PROBABILISTIC_SAMPLING"
    RANDOM = "RANDOM"
    HEAVY_HITTER = "HEAVY_HITTER"


def get_device_unreachable_pairs_mapping(device_DC_mapping: dict):
    mapping = {}
    global_all_nodes = set()
    for device, non_iso_part in device_DC_mapping.items():
        # there will be no pairs with the same element
        non_connected_pairs = find_non_connected_pairs(non_iso_part)
        mapping[device] = non_connected_pairs
        # flatten the pairs to get all the non-repeated nodes, convert to list
        global_all_nodes.update({node for pair in non_connected_pairs for node in pair})
    return mapping, list(global_all_nodes)


def split_nodes(graph: CompGraph, node_list, device_list: list,
                operator_device_mapping, r, sampling_function) -> dict:
    result_dict = {device_id: {"selected_list": [], "unselected_list": [], "isolated_part": []} for device_id in
                   device_list}

    def evaluate_node(node):
        assigned_device = operator_device_mapping[node]
        computing_cost = graph.getOperatorCompCostByDevice(node, assigned_device)
        # Set to track unique sub_graphs that depend on this operator
        return computing_cost

    node_score_mapping = {node: evaluate_node(node) for node in node_list}
    # First sort the node list based on the computing cost condition
    node_list = sorted(node_list, key=lambda node: node_score_mapping[node], reverse=True)
    num_to_select = int(len(node_list) * r)

    if sampling_function == SamplingFunction.HEAVY_HITTER:
        selected_nodes, unselected_nodes = node_list[:num_to_select], node_list[num_to_select:]

    elif sampling_function == SamplingFunction.RANDOM:
        selected_nodes = random.sample(node_list, num_to_select)
        unselected_nodes = [node for node in node_list if node not in selected_nodes]
    else:
        # Calculate the total score for all nodes
        total_score = sum(node_score_mapping.values())

        # Calculate the probability for each node
        node_probabilities = [node_score_mapping[node] / total_score for node in node_list]

        # Weighted sampling without replacement based on computed probabilities
        def weighted_sample_without_replacement(node_list, node_probabilities, k):
            selected_nodes = []
            nodes = node_list[:]
            probs = node_probabilities[:]

            for _ in range(k):
                # Calculate the sum of the remaining probabilities
                total_prob = sum(probs)

                # Check if the total probability is zero, which would cause division by zero
                if total_prob == 0:
                    # If the total probability is zero, select randomly from the remaining nodes
                    selected_nodes += random.sample(nodes, k - len(selected_nodes))
                    break

                # Normalize probabilities to sum to 1
                norm_probs = [p / total_prob for p in probs]

                # Select a node based on the normalized probabilities
                selected_node = random.choices(nodes, weights=norm_probs, k=1)[0]
                selected_nodes.append(selected_node)

                # Remove the selected node and its probability
                index = nodes.index(selected_node)
                nodes.pop(index)
                probs.pop(index)

            return selected_nodes

        # Sample nodes probabilistically based on the computed probabilities without replacement
        selected_nodes = weighted_sample_without_replacement(node_list, node_probabilities, k=int(len(node_list) * r))

        # Get the unselected nodes (those not in selected_nodes)
        unselected_nodes = [node for node in node_list if node not in selected_nodes]

    # Map selected_nodes and unselected_nodes to the corresponding device in result_dict
    for selected_node in selected_nodes:
        result_dict[operator_device_mapping[selected_node]]["selected_list"].append(selected_node)
    for unselected_node in unselected_nodes:
        result_dict[operator_device_mapping[unselected_node]]["unselected_list"].append(unselected_node)

    return result_dict
