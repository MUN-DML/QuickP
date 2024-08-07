import heapq
from collections import deque
from enum import Enum

from networkx import topological_sort

from optimizer.model.graph import CompGraph
from py_util import convert_data_size, convert_time


def topo_sort_Kahn(graph: CompGraph):
    # Compute the in-degree of each node
    in_degrees = {node: graph.in_degree(node) for node in graph.nodes()}

    # Queue of nodes with no incoming edges
    zero_in_degree_queue = deque([node for node in graph.nodes() if in_degrees[node] == 0])

    topo_order = []

    while zero_in_degree_queue:
        current = zero_in_degree_queue.popleft()
        topo_order.append(current)

        for neighbor in graph.successors(current):
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)
    assert is_topological_sort(graph, topo_order)
    return topo_order


def topo_sort_default(graph: CompGraph):
    return topological_sort(graph)


def compute_priority(graph, edge_cuts):
    priority = {node: graph.getOperatorCompCostAve(node) for node in graph.nodes()}
    speed = convert_data_size(20, 'GB', 'bit')
    # Adjust priority based on edge cuts
    for (u, v) in edge_cuts:
        priority[v] += int(convert_time(graph.getOperatorOutputInBit(v) / speed, 's', 'us'))

    return priority


def topo_sort_Kahn_priority(graph: CompGraph, edge_cuts: list):
    priority = compute_priority(graph, edge_cuts)

    # Compute the in-degree of each node
    in_degrees = {node: graph.in_degree(node) for node in graph.nodes()}

    # Priority queue of nodes with no incoming edges
    zero_in_degree_queue = [(priority[node], node) for node in graph.nodes() if in_degrees[node] == 0]
    heapq.heapify(zero_in_degree_queue)

    topo_order = []

    while zero_in_degree_queue:
        _, current = heapq.heappop(zero_in_degree_queue)
        topo_order.append(current)

        for neighbor in graph.successors(current):
            in_degrees[neighbor] -= 1
            if in_degrees[neighbor] == 0:
                heapq.heappush(zero_in_degree_queue, (priority[neighbor], neighbor))

    assert is_topological_sort(graph, topo_order)
    return topo_order


def is_topological_sort(graph: CompGraph, node_list):
    # Create a dictionary to record the position of each node in the node_list
    position = {node: i for i, node in enumerate(node_list)}

    # Iterate through all edges in the graph
    for u, v in graph.edges():
        # For a valid topological sort, u must come before v
        if position[u] > position[v]:
            return False

    return True


class TopoSortFunction(Enum):
    KAHN = topo_sort_Kahn
    DEFAULT = topo_sort_default
    KAHN_PRIORITY = topo_sort_Kahn_priority


def from_topo_list_to_dict(sorted_nodes):
    return {node: pos for pos, node in enumerate(sorted_nodes)}


def create_topological_position_dict(graph: CompGraph, sort_function: TopoSortFunction, edge_cuts: list):
    """
    Creates a dictionary mapping each node to its position in the topologically sorted order.

    Parameters:
    graph (nx.DiGraph): The directed graph.

    Returns:
    dict: A dictionary where keys are nodes and values are their positions in the topologically sorted order.
    """
    if sort_function == TopoSortFunction.KAHN_PRIORITY:
        sorted_nodes = sort_function(graph, edge_cuts)
    else:
        sorted_nodes = sort_function(graph)
    return from_topo_list_to_dict(sorted_nodes)
