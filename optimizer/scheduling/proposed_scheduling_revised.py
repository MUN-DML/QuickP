import itertools

import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import CompGraph, find_non_connected_pairs, split_non_connected_pairs, is_not_connected, \
    split_list_based_on_computing_cost
from optimizer.scheduling.scheduling_order_only import FIFO_scheduling_order


def near_optimal_scheduling_revised(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                                    device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict):
    # The global data dependency is already applied
    M = 1000000
    order = {}
    fifo_operator_order, _ = FIFO_scheduling_order(comp_graph, device_subgraph_mapping, edge_cut_list,
                                                   operator_device_mapping)
    for device, subgraph in device_subgraph_mapping.items():
        # there will be no pairs with the same element
        non_connected_pairs = find_non_connected_pairs(subgraph)
        # flatten the pairs to get all the non-repeated nodes, convert to list
        all_nodes = list({node for pair in non_connected_pairs for node in pair})
        high_cost_nodes, other_nodes = split_list_based_on_computing_cost(comp_graph, device, all_nodes)

        # get the FIFO order
        local_fifo_order = fifo_operator_order[device]
        node_order_dict = {op: idx for idx, op in enumerate(local_fifo_order)}
        # sort other_nodes based on the FIFO order
        other_nodes = sorted(other_nodes, key=lambda op: node_order_dict[op])

        # apply optimization to these high-cost node pairs
        for high_cost_node in high_cost_nodes:
            non_connected_pair = [pair for pair in non_connected_pairs if high_cost_node in pair]
            for op_a, op_b in non_connected_pair:
                order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"order_{op_a}_{op_b}")
                model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]), name=f"NoOverlap1_{op_a}_{op_b}")
                model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")
        for op_a, op_b in zip(other_nodes, other_nodes[1:]):
            model.addConstr(finish[op_a] <= start[op_b])

    for device, subgraph in device_subgraph_mapping.items():
        outgoings = [edge for edge in edge_cut_list if edge[0] in subgraph]
        topo_order = list(nx.topological_sort(subgraph))
        # Create a mapping of nodes to their topological order position
        topo_order_map = {node: index for index, node in enumerate(topo_order)}
        # Sort the edges based on the topological order of the source nodes
        sorted_outgoings = sorted(outgoings, key=lambda edge: topo_order_map[edge[0]])
        for comm1, comm2 in zip(sorted_outgoings, sorted_outgoings[1:]):
            source_node_1 = comm1[0]
            source_node_2 = comm2[0]
            # in this case, these two nodes does not have dependency, implement FCFS policy
            if is_not_connected(subgraph, source_node_1, source_node_2):
                order_1_first = model.addVar(vtype=GRB.BINARY, name=f"order_{source_node_1}_first_{source_node_2}")
                # Enforce the order based on the finish times using Big M
                # If finish[source_node_1] <= finish[source_node_2], set order_1_first = 1
                model.addConstr(finish[source_node_1] - finish[source_node_2] <= M * (1 - order_1_first),
                                name=f"order_decision_1_{source_node_1}_{source_node_2}")

                # If finish[source_node_2] < finish[source_node_1], set order_1_first = 0
                model.addConstr(finish[source_node_2] - finish[source_node_1] <= M * order_1_first,
                                name=f"order_decision_2_{source_node_1}_{source_node_2}")

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