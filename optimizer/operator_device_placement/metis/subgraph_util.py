from enum import Enum

import networkx as nx
import numpy as np
import pandas as pd
from networkx import DiGraph

from optimizer.model.graph import CompGraph
from optimizer.weight_adjustment_before_partition.weight_adjustment_function import recalculate_weights_critical_score


# expand the subgraph by one node and make it still a subgraph of original_graph
def expand_subgraph(sub_graph, node, original_graph):
    if node not in original_graph.nodes:
        raise ValueError("node {} not in the original graph".format(node))
    # Add the node to the subgraph
    sub_graph.add_node(node, **original_graph.nodes[node])

    # Add edges from the original graph to the subgraph
    for predecessor in original_graph.predecessors(node):
        # predecessor must be already in the subgraph to prevent bring extra nodes and edges
        if predecessor in sub_graph.nodes and not sub_graph.has_edge(predecessor, node):
            sub_graph.add_edge(predecessor, node, **original_graph.get_edge_data(predecessor, node) or {})

    for successor in original_graph.successors(node):
        # successor must be already in the subgraph to prevent bring extra nodes and edges
        if successor in sub_graph.nodes and not sub_graph.has_edge(node, successor):
            sub_graph.add_edge(node, successor, **original_graph.get_edge_data(node, successor) or {})


def shrink_subgraph(sub_graph, node):
    if node not in sub_graph.nodes:
        raise ValueError(f"node {node} not in the subgraph")

    # Remove the node along with all its edges
    sub_graph.remove_node(node)


def creates_cycle(subgraph, node, G):
    # Make the update on the copied graph
    subgraph_copy = subgraph.copy()

    expand_subgraph(subgraph_copy, node, G)

    # Check for cycles
    has_cycle = not nx.is_directed_acyclic_graph(subgraph_copy)

    return has_cycle


def identify_edges_cut(weighted_digraph: DiGraph, partition_dict: dict[str, int]) -> tuple[list[tuple], int]:
    cut_edges = [
        (u, v)
        for u, v in weighted_digraph.edges()
        if partition_dict[u] != partition_dict[v]
    ]
    sum_of_weights = sum(weighted_digraph[u][v].get('edge_weight', 0) for u, v in cut_edges)

    return cut_edges, sum_of_weights


def map_subgraph_to_device(partition_dict, device_id_list, computing_cost_dict: dict[str, float] = None, subgraph_weight_dict: dict = None):
    # Extract unique subgraph IDs
    subgraph_id_list = list(set(partition_dict.values()))
    assert len(subgraph_id_list) == len(device_id_list)
    # Sort to ensure consistency
    if subgraph_weight_dict:
        # Sort subgraph_id_list based on the weight sum of each graph, the subgraph with a higher weight sum will be placed first
        subgraph_id_list.sort(key=lambda subgraph_id: subgraph_weight_dict.get(subgraph_id, float('inf')), reverse=True)
    if computing_cost_dict:
        # Sort device_id_list based on the values in computing_cost_dict, the faster device will be placed first
        device_id_list.sort(key=lambda device_id: computing_cost_dict.get(device_id, float('inf')))
    # Ensure they have the same length
    if len(subgraph_id_list) != len(device_id_list):
        raise ValueError("Subgraph ID list and device ID list must have the same length")
    # Create a mapping from subgraph IDs to device IDs
    subgraph_to_device_map = {subgraph_id: device_id for subgraph_id, device_id in
                              zip(subgraph_id_list, device_id_list)}

    # Update partition_dict with device IDs
    updated_partition_dict = {node: subgraph_to_device_map[subgraph_id] for node, subgraph_id in partition_dict.items()}

    return updated_partition_dict


def construct_sub_graph(digraph: CompGraph, placement: dict[str, int]) -> dict[int, CompGraph]:
    subgraph_dict = {}
    for operator, placement_index in placement.items():
        # init a Digraph obj if it does not exist in the subgraph_dict
        if placement_index not in subgraph_dict:
            subgraph_dict[placement_index] = CompGraph()
        expand_subgraph(subgraph_dict[placement_index], operator, digraph)
    return subgraph_dict


def normalize_weights_min_max(dag: CompGraph, to_range: tuple = (1, 1000)):
    from sklearn.preprocessing import MinMaxScaler

    # extract
    node_weights = {node: attributes['node_weight'] for node, attributes in dag.nodes(data=True)}
    edge_weights = {(source, target): attributes['edge_weight'] for source, target, attributes in dag.edges(data=True)}

    # Convert to pandas DataFrame
    df_nodes = pd.DataFrame(node_weights.items(), columns=['node', 'node_weight'])
    df_edges = pd.DataFrame(edge_weights.items(), columns=['edge', 'edge_weight'])

    node_range = (max(to_range[0], df_nodes.node_weight.min()), min(to_range[1], df_nodes.node_weight.max()))
    edge_range = (max(to_range[0], df_edges.edge_weight.min()), min(to_range[1], df_edges.edge_weight.max()))

    scaler_node = MinMaxScaler(feature_range=node_range)
    scaler_edge = MinMaxScaler(feature_range=edge_range)

    # Fit and transform nodes/edges weights
    df_nodes['node_weight'] = scaler_node.fit_transform(df_nodes[['node_weight']])
    df_edges['edge_weight'] = scaler_edge.fit_transform(df_edges[['edge_weight']])

    # Convert back to dictionary
    scaled_node_weights = dict(zip(df_nodes['node'], df_nodes['node_weight'].astype(int)))
    scaled_edge_weights = dict(zip(df_edges['edge'], df_edges['edge_weight'].astype(int)))

    # Update the node and edge weight
    for node, weight in scaled_node_weights.items():
        dag.nodes[node]['node_weight'] = weight
    for (source, dest), weight in scaled_edge_weights.items():
        dag[source][dest]['edge_weight'] = weight


class WeightNormalizationFunction(Enum):
    MIN_MAX = normalize_weights_min_max


def normalize_list(weight_sum_list: list) -> list[float]:
    array = np.array(weight_sum_list)
    normalized_array = array / array.sum()
    return normalized_array.tolist()
