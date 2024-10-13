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
def edge_based_label(graph: CompGraph, device_topo: DeviceGraph, computing_cost_dict):
    fast_link = device_topo.get_fastest_link()
    for edge in graph.edges:
        source, destination = edge
        destination_computing_cost = computing_cost_dict[destination]
        communication_cost = graph.getEdgeTensorSize(source, destination) * device_topo.calUnitCommCostInUS(
            fast_link[0], fast_link[1])
        # the source only has one outgoing edge and communication cost if on different device is higher than
        if (communication_cost >= destination_computing_cost or computing_cost_dict[source] == 0) and graph.out_degree(
                source) == 1:
            # label both end the group of source node. One node will probably have more than one group. Waiting to merge groups
            graph.update_colocation_group(source, source)
            graph.update_colocation_group(destination, source)


def label_all_node_with_group(graph: CompGraph, device_topo: DeviceGraph, computing_cost_dict):
    """
    Perform BFS on a NetworkX DiGraph starting from any node,
    skipping nodes that already have a 'colocation_group' attribute.

    Args:
        :param graph:
        :param device_topo:
        :param start_node:
        :param computing_cost_dict:
    """

    def BFS_label(from_node):
        # Queue to keep track of nodes to explore (starting with start_node)
        queue = deque([from_node])

        # label the beginning node
        graph.set_colocation_group(from_node, from_node)

        # Perform BFS
        while queue:
            # Pop the leftmost (oldest) node in the queue
            node = queue.popleft()

            # Check if the node has a 'colocation_group' attribute
            if 'colocation_group' not in graph.nodes[node]:
                raise ValueError("Dequeued node should be labelled already")

            # Explore neighbors (outgoing edges in DiGraph)
            for neighbor in graph.successors(node):  # Only follow outgoing edges
                # potential communication cost and comp_cost
                computing_cost = computing_cost_dict[neighbor]
                communication_cost = graph.getEdgeTensorSize(node, neighbor) * device_topo.calUnitCommCostInUS(
                    fast_link[0], fast_link[1])
                # Mark the node as visited by adding the 'colocation_group' attribute
                if communication_cost >= computing_cost or (
                        computing_cost_dict[node] == 0 and graph.out_degree(node) == 1):
                    # If not already visited, will expand this successor
                    if 'colocation_group' not in graph.nodes[neighbor]:
                        graph.set_colocation_group(neighbor, from_node)
                        queue.append(neighbor)
                    # If already visited, add this new group to its existing groups but will not expand further
                    else:
                        graph.update_colocation_group(neighbor, from_node)

    fast_link = device_topo.get_fastest_link()
    node_order = sort_by_critical_score(graph, computing_cost_dict)
    for node in node_order:
        # skip already labelled node
        if 'colocation_group' in graph.nodes[node]:
            continue
        BFS_label(node)


def analyze_group(graph: CompGraph, node_computing_cost_dict):
    group_ops_mapping = graph.create_colocation_group_to_ops_map()
    print({gid: len(group) for gid, group in group_ops_mapping.items()})
    print('group number', len(group_ops_mapping.keys()))
    print('number of labelled ', sum(len(group) for group in group_ops_mapping.values()))
    group_computing_cost_sum = {
        gid: sum(node_computing_cost_dict[op] for op in group)
        for gid, group in group_ops_mapping.items()
    }
    print("group_computing_cost_sum", group_computing_cost_sum)


# if there is any node which has two groups labelled, this two groups get merged
def merge_group(computing_graph: CompGraph):
    merged_group_op_set_map = {}

    # a list of set [(a, b), (c), (d, e, g), (b, c), (e, p)]
    # I want to get [(a, b, c), (d, e, g, p)]
    def merge_sets(sets):
        merged = []

        for current_set in sets:
            # Find all existing sets that share elements with current_set
            overlapping = []
            for s in merged:
                if s & current_set:  # if there's an intersection
                    overlapping.append(s)

            # If no overlap, add the current_set as a new group
            if not overlapping:
                merged.append(current_set)
            else:
                # Merge all overlapping sets with the current_set
                merged = [s for s in merged if s not in overlapping]
                merged.append(set.union(*overlapping, current_set))

        return merged

    def get_group_op_set_map():
        colocation_group_map = defaultdict(set)

        for op_id, op_data in computing_graph.nodes(data=True):
            # Check if the node has a 'colocation_group' attribute
            group_list = op_data.get('colocation_group')
            # every node should have colocation group
            if group_list is None or not group_list:
                # raise ValueError(f'colocation group {op_id} has no colocation_group')
                continue
            for group_id in group_list:
                colocation_group_map[group_id].add(op_id)
        return dict(colocation_group_map)

    # A 2D array, each list in groups_to_join indicate multiple groups which should be merged
    groups_to_join = []
    for op_id, op_data in computing_graph.nodes(data=True):
        # Check if the node has a 'colocation_group' attribute
        group_list = op_data.get('colocation_group')
        # every node should have colocation group
        if group_list is None or not group_list:
            # raise ValueError(f'colocation group {op_id} has no colocation_group')
            continue
        if len(group_list) < 1:
            raise ValueError(f'colocation group {op_id} has empty colocation_group')
        if len(group_list) > 1:
            groups_to_join.append(set(group_list))

    merged_sets = merge_sets(groups_to_join)
    print(groups_to_join, 'merged_sets', merged_sets)
    group_op_set = get_group_op_set_map()
    print('current group_op_set', group_op_set)
    for multiple_groups_to_be_join in merged_sets:
        merged_nodes = set()
        merged_string = ''.join(multiple_groups_to_be_join)
        hashed_string = hashlib.md5(merged_string.encode()).hexdigest()
        for group_id in multiple_groups_to_be_join:
            merged_nodes = merged_nodes | group_op_set[group_id]

        merged_group_op_set_map[hashed_string] = merged_nodes

    for new_group_id, op_set in merged_group_op_set_map.items():
        for op_id in op_set:
            computing_graph.set_colocation_group(op_id, new_group_id)
