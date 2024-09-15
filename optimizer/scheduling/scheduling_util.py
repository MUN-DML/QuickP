import networkx as nx

from optimizer.model.graph import CompGraph


def get_op_related_node_set_mapping(graph: CompGraph, operator_device_mapping, edge_cut_list) -> dict[any, set]:
    def get_related_node_set(node):
        device = operator_device_mapping[node]

        outgoing_edges = [(u, v) for u, v in edge_cut_list if operator_device_mapping.get(u) == device]
        destination_node_depended = set(v for (u, v) in outgoing_edges if nx.has_path(graph, node, u))
        '''
        current_subgraph = device_subgraph_mapping.get(device)
        related_devices = set()
        outgoing_edges_depended = [(u, v) for u, v in outgoing_edges if nx.has_path(graph, node, u)]
        for u, v in outgoing_edges_depended:
            assert device_subgraph_mapping.get(operator_device_mapping.get(u)) == current_subgraph
            assert operator_device_mapping.get(v) != device
            related_devices.add(operator_device_mapping.get(v))
        return len(related_devices)
        '''
        for node in destination_node_depended:
            assert operator_device_mapping.get(node) != device
        return destination_node_depended

    return {op: get_related_node_set(op) for op in graph.nodes}


def split_subgraph(graph: CompGraph, operator_device_mapping, device_subgraph_mapping, edge_cut_list):
    """
    Return a new subgraph with nodes that have 0 related subgraphs removed.

    Parameters:
    - graph: networkx.DiGraph, the original graph.
    - op_related_subgraph_mapping: dict, a mapping of nodes to the number of related subgraphs.

    Returns:
    - A new subgraph with nodes having 0 related subgraphs removed.
    """
    # Create a copy of the original graph
    new_graph = graph.copy()
    op_related_node_set_mapping = get_op_related_node_set_mapping(new_graph, operator_device_mapping, edge_cut_list)

    # Iterate over the nodes and remove those with 0 related subgraphs
    nodes_to_remove = [node for node, node_set in op_related_node_set_mapping.items() if
                       len(node_set) == 0]

    # Remove the nodes from the copied graph
    new_graph.remove_nodes_from(nodes_to_remove)

    return new_graph, nodes_to_remove
