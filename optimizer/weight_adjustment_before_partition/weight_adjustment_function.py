import math
from enum import Enum
from typing import TypedDict

import networkx as nx

from optimizer.model.graph import CompGraph


def recursive_increase(dag: CompGraph, adjust_matrix: dict = None):
    # By using a weighted sum of predecessor nodes' weights, the function accounts for the interconnected nature of
    # nodes within subgraphs. This ensures that the influence of a node on its successors is proportional to its weight
    # and the weight of the connecting edge. This dependency chain leads to a smoother distribution of weights since
    # each node’s weight is a blend of its own cost and the cumulative influence of its predecessors.

    # Process nodes in topological order to ensure dependencies are respected
    topo_sorted_nodes = list(nx.topological_sort(dag))

    if adjust_matrix["node_enable"]:
        for node in topo_sorted_nodes:
            # Calculate the weighted sum of predecessors' weights
            for pred in dag.predecessors(node):
                source_node_weight = dag.nodes[pred]['node_weight']
                if source_node_weight == 0:
                    continue
                dag.nodes[node]['node_weight'] += int(source_node_weight * adjust_matrix["adjustment_ratio"])

    if adjust_matrix["edge_enable"]:
        for node in topo_sorted_nodes:
            for successor in dag.successors(node):
                # Calculate the new weight for the edge (node, succ)
                for pred in dag.predecessors(node):
                    incoming_edge_weight = dag[pred][node]["edge_weight"]
                    dag[node][successor]['edge_weight'] += int(incoming_edge_weight * adjust_matrix["adjustment_ratio"])


def recalculate_weights_critical_score(dag: CompGraph, adjust_matrix: dict = None):
    # By using a weighted sum of predecessor nodes' weights, the function accounts for the interconnected nature of
    # nodes within subgraphs. This ensures that the influence of a node on its successors is proportional to its weight
    # and the weight of the connecting edge. This dependency chain leads to a smoother distribution of weights since
    # each node’s weight is a blend of its own cost and the cumulative influence of its predecessors.

    # Process nodes in topological order to ensure dependencies are respected
    topo_sorted_nodes = list(nx.topological_sort(dag))

    if adjust_matrix.get("node_enable", False):
        max_degree = max(dag.in_degree(node) + dag.out_degree(node) for node in topo_sorted_nodes)

        for node in topo_sorted_nodes:
            # Calculate a criticality score based on normalized degree
            in_degree = dag.in_degree(node)
            out_degree = dag.out_degree(node)
            criticality_score = (in_degree + out_degree) / max_degree

            # Adjust node weight using a logarithmic scale to handle large weights
            current_weight = dag.nodes[node].get('node_weight', 1)
            adjustment = int(math.log(1 + current_weight) * criticality_score * adjust_matrix["adjustment_ratio"])
            dag.nodes[node]['node_weight'] += adjustment

    if adjust_matrix.get("edge_enable", False):
        for node in topo_sorted_nodes:
            for successor in dag.successors(node):
                # Calculate the new weight for the edge (node, successor) based on node weights
                source_node_weight = dag.nodes[node].get('node_weight', 1)
                target_node_weight = dag.nodes[successor].get('node_weight', 1)
                average_node_weight = (source_node_weight + target_node_weight) / 2

                current_edge_weight = dag[node][successor].get('edge_weight', 1)
                adjustment = int(current_edge_weight + average_node_weight * adjust_matrix["adjustment_ratio"])
                dag[node][successor]['edge_weight'] = adjustment


def longest_dependency_chain(dag: CompGraph, adjust_matrix: dict = None):
    if adjust_matrix.get("edge_enable", False):
        # 1. Identify the longest path
        longest_path = nx.dag_longest_path(dag, weight='edge_weight')

        # 2. Increase weights for the longest path
        for u, v in zip(longest_path[:-1], longest_path[1:]):
            dag[u][v]['edge_weight'] *= 5  # Increase weight for the critical longest path

        # 3. Optionally, identify and weight significant sub-paths
        for node in longest_path:
            for target in dag.nodes():
                if node != target:
                    for sub_path in nx.all_simple_paths(dag, source=node, target=target):
                        if len(sub_path) > 1 and sub_path != longest_path:
                            for u, v in zip(sub_path[:-1], sub_path[1:]):
                                dag[u][v]['edge_weight'] *= 2  # Increase weight for significant sub-paths


class WeightAdjustmentFunction(Enum):
    Recursive_Increase = recursive_increase
    CRITICAL_SCORE = recalculate_weights_critical_score


class WeightAdjustMatrix(TypedDict):
    function_type: WeightAdjustmentFunction
    adjustment_ratio: int
    node_enable: bool
    edge_enable: bool


def recalculate_node_weights(dag: CompGraph, adjust_matrix: WeightAdjustMatrix = None):
    if not adjust_matrix:
        return

    if "adjustment_ratio" not in adjust_matrix.keys():
        raise ValueError("adjustment_ratio must be specified")

    if adjust_matrix["adjustment_ratio"] <= 0:
        return

    adjust_matrix["function_type"](dag, adjust_matrix)
