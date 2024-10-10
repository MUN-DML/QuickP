from collections import defaultdict, deque
from typing import Dict, List

import networkx as nx
from networkx.classes import DiGraph

from optimizer.model.graph import CompGraph


def create_colocation_group_to_ops_map(op_graph: DiGraph) -> Dict[any, List[str]]:
    """Generate a dict that maps a colocation group to its op id list."""
    colocation_group_map = defaultdict(list)

    for op_id, op_data in op_graph.nodes(data=True):
        # Check if the node has a 'colocation_group' attribute
        group = op_data.get('colocation_group')
        if group is not None:
            colocation_group_map[group].append(op_id)
    # {'':list(op_graph.nodes)[0:40], '1': list(op_graph.nodes)[41:80], '2': list(op_graph.nodes)[80:121]}
    # {'':list(op_graph.nodes)[0:600], '1': list(op_graph.nodes)[601:1200], '2': list(op_graph.nodes)[1201:1600]}
    return dict(colocation_group_map)


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


def bfs_with_colocation(graph: CompGraph, device_topo, start_node, computing_cost_dict):
    """
    Perform BFS on a NetworkX DiGraph starting from any node,
    skipping nodes that already have a 'colocation_group' attribute.

    Args:
        :param graph:
        :param device_topo:
        :param start_node:
        :param computing_cost_dict:
    """
    # Queue to keep track of nodes to explore (starting with start_node)
    queue = deque([start_node])

    # label the beginning node
    graph.set_colocation_group(start_node, start_node)

    # Perform BFS
    while queue:
        # Pop the leftmost (oldest) node in the queue
        node = queue.popleft()

        # Check if the node has a 'colocation_group' attribute
        if 'colocation_group' not in graph.nodes[node]:
            raise ValueError("Dequeued node should be labelled already")

        # Explore neighbors (outgoing edges in DiGraph)
        for neighbor in graph.successors(node):  # Only follow outgoing edges
            if 'colocation_group' not in graph.nodes[neighbor]:  # If not already visited
                # potential communication cost and comp_cost
                computing_cost = computing_cost_dict[node]
                communication_cost = graph.getEdgeTensorSize(node, neighbor
                                      * device_topo.calUnitCommCostInUS())
                # Mark the node as visited by adding the 'colocation_group' attribute
                if communication_cost >= computing_cost:
                    graph.nodes[node]['colocation_group'] = start_node
                    # only expand node labelled in the same group
                    queue.append(neighbor)