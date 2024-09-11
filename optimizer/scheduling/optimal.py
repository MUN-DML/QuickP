import itertools

import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import CompGraph


def optimal_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph, device_subgraph_mapping: dict[any, CompGraph], edge_cut_list):
    # The global data dependency is already applied
    M = 1000000
    order = {}
    for subgraph in device_subgraph_mapping.values():
        for a, b in itertools.combinations(subgraph.getOperatorIDs(), 2):
            # Initialize order variables
            order[a, b] = model.addVar(vtype=GRB.BINARY, name=f"order_{a}_{b}")
            if nx.has_path(subgraph, b, a):
                model.addConstr(start[a] >= finish[b])
            elif nx.has_path(subgraph, a, b):
                model.addConstr(start[b] >= finish[a])
            else:
                model.addConstr(start[b] >= finish[a] - M * (1 - order[a, b]),name=f"NoOverlap1_{a}_{b}")
                model.addConstr(start[a] >= finish[b] - M * order[a, b], name=f"NoOverlap2_{a}_{b}")
    '''
    # Add constraint to ensure each device can only send one link at a time, communication scheduling
    # Only edges in the edge_cut_list will bring communication cost
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
    '''
