import networkx as nx

from optimizer.model.graph import CompGraph

'''
Split a graph into three parts
Source Nodes: Nodes that serve as dependencies for other subgraphs.
Sink Nodes: Nodes that do not serve as dependencies for other subgraphs but have dependencies from other graphs. These can be seen as "end points."
Isolated Nodes: Nodes that neither serve as dependencies for other subgraphs nor have dependencies from other graphs. These are independent nodes within the graph structure.
'''


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
    non_source_node = set(node for node in graph.nodes if len(get_depended_node_set(node)) == 0)

    isolate_nodes = set(node for node in non_source_node if len(get_depending_node_set(node)) == 0)

    sink_nodes = non_source_node - isolate_nodes

    # Remove the nodes from the copied graph
    new_graph.remove_nodes_from(non_source_node)

    return new_graph, non_source_node
