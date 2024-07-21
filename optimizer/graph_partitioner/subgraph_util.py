import networkx as nx
from networkx import DiGraph


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
    sum_of_weights = sum(weighted_digraph[u][v]['edge_weight'] for u, v in cut_edges)

    return cut_edges, sum_of_weights


def construct_sub_graph(digraph: DiGraph, placement: dict[str, int]) -> dict[int, DiGraph]:
    subgraph_dict = {}
    for operator, placement_index in placement.items():
        # init a Digraph obj if it does not exist in the subgraph_dict
        if placement_index not in subgraph_dict:
            subgraph_dict[placement_index] = nx.DiGraph()
        expand_subgraph(subgraph_dict[placement_index], operator, digraph)
    return subgraph_dict
