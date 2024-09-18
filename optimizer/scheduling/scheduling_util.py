import networkx as nx

from optimizer.model.graph import CompGraph


def split_subgraph(graph: CompGraph, operator_device_mapping, edge_cut_list):
    def get_depended_node_set(node):
        device = operator_device_mapping[node]
        outgoing_edges = [(u, v) for u, v in edge_cut_list if operator_device_mapping.get(u) == device]
        destination_node_depended = set(v for (u, v) in outgoing_edges if nx.has_path(graph, node, u))
        for node in destination_node_depended:
            assert operator_device_mapping.get(node) != device
        return destination_node_depended

    def get_depending_node_set(node):
        device = operator_device_mapping[node]
        incoming_edges = [(u, v) for u, v in edge_cut_list if operator_device_mapping.get(v) == device]
        source_node_depended = set(u for (u, v) in incoming_edges if nx.has_path(graph, v, node))
        for node in source_node_depended:
            assert operator_device_mapping.get(node) != device
        return source_node_depended

    # Create a copy of the original graph
    new_graph = graph.copy()

    # Iterate over the nodes and remove those with 0 related subgraphs
    nodes_to_remove = [node for node in graph.nodes if len(get_depended_node_set(node)) == 0]

    totally_isolated_nodes = [node for node in nodes_to_remove if len(get_depending_node_set(node)) == 0]

    # Remove the nodes from the copied graph
    new_graph.remove_nodes_from(nodes_to_remove)

    return new_graph, nodes_to_remove
