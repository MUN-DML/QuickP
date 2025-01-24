import itertools

import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import CompGraph, find_non_connected_pairs


def optimal_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph, device_subgraph_mapping: dict[any, CompGraph], edge_cut_list, M):
    # The global data dependency is already applied
    order = {}
    for subgraph in device_subgraph_mapping.values():
        non_connected_pairs, _ = find_non_connected_pairs(subgraph)
        for a, b in non_connected_pairs:
            # Initialize order variables
            order[a, b] = model.addVar(vtype=GRB.BINARY)
            model.addConstr(start[b] >= finish[a] - M * (1 - order[a, b]))
            model.addConstr(start[a] >= finish[b] - M * order[a, b])
