import hashlib
from collections import defaultdict, deque
from typing import Dict, List

import networkx as nx
from networkx.classes import DiGraph

from optimizer.model.graph import CompGraph, DeviceGraph


def get_op_group_map(groups_op_mapping: Dict[any, List[any]]) -> Dict[str, any]:
    op_group_map = {}

    for group, ops in groups_op_mapping.items():
        for op in ops:
            op_group_map[op] = group

    return op_group_map


def sort_by_critical_score(computing_graph: CompGraph, computing_cost_dict):
    global_score = {}
    topo_sorted = list(nx.topological_sort(computing_graph))

    for current_node in reversed(topo_sorted):
        # Check if the current node has any predecessors
        successors = list(computing_graph.successors(current_node))

        if successors:  # If there are predecessors, compute the max computing cost
            max_suc_computing_cost = sum(
                global_score[succ_node] for succ_node in successors
            )
        else:  # If there are no predecessors, set the max computing cost to 0
            max_suc_computing_cost = 0

        # Calculate the global rank for the current node
        global_score[current_node] = (max_suc_computing_cost + computing_cost_dict[current_node])

    all_nodes = list(computing_graph.nodes)
    all_nodes = sorted(all_nodes, key=global_score.get, reverse=True)

    return all_nodes


# _run_colocation_step
def create_eligible_edge_subgraph(graph: CompGraph, device_topo: DeviceGraph, computing_cost_dict):
    fast_link = device_topo.get_fastest_link()
    eligible_edges = []
    for edge in graph.edges:
        source, destination = edge
        destination_computing_cost = computing_cost_dict[destination]
        communication_cost = graph.getEdgeTensorSize(source, destination) * device_topo.calUnitCommCostInUS(
            fast_link[0], fast_link[1])
        # the source only has one outgoing edge and communication cost if on different device is higher than
        # and graph.in_degree(destination) == 1 will minimize the performance loss
        if (communication_cost >= destination_computing_cost or computing_cost_dict[source] == 0) and graph.out_degree(source) == 1:
            # label both end the group of source node. One node will probably have more than one group. Waiting to merge groups
            eligible_edges.append((source, destination))
    return graph.edge_subgraph(eligible_edges)


def analyze_group(graph: CompGraph, node_computing_cost_dict):
    group_ops_mapping = graph.create_colocation_group_to_ops_map()
    print({gid: len(group) for gid, group in group_ops_mapping.items()})
    print('group number', len(group_ops_mapping.keys()))
    print('number of labelled ', sum(len(group) for group in group_ops_mapping.values()))
    print('labelled computing cost', sum(sum( node_computing_cost_dict[node] for node in group) for group in group_ops_mapping.values())
          , 'all computing cost', sum(node_computing_cost_dict.values()))
    group_computing_cost_sum = {
        gid: sum(node_computing_cost_dict[op] for op in group)
        for gid, group in group_ops_mapping.items()
    }
    print("group_computing_cost_sum", group_computing_cost_sum)


def label_group(sub_graph: CompGraph):
    weakly_connected_components = list(nx.weakly_connected_components(sub_graph))

    for wcc in weakly_connected_components:
        merged_string = ''.join(wcc)
        hashed_string = hashlib.md5(merged_string.encode()).hexdigest()
        for node in wcc:
            sub_graph.set_colocation_group(node, hashed_string)
