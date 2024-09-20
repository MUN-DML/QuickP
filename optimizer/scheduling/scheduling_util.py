from typing import Tuple, Any, Set, Iterator

import networkx as nx
from matplotlib import pyplot as plt

from optimizer.model.graph import CompGraph

'''
Split a graph into three parts
Source Nodes: Nodes that serve as dependencies for other subgraphs.
Sink Nodes: Nodes that do not serve as dependencies for other subgraphs but have dependencies from other graphs. These can be seen as "end points."
Isolated Nodes: Nodes that neither serve as dependencies for other subgraphs nor have dependencies from other graphs. These are independent nodes within the graph structure.
'''


def split_subgraph(graph: CompGraph, operator_device_mapping, edge_cut_list) -> tuple[
    CompGraph, Any, set[Any], set[Any]]:
    device = operator_device_mapping[list(graph.nodes)[0]]
    outgoing_edges = [(u, v) for u, v in edge_cut_list if
                      operator_device_mapping.get(u) == device and operator_device_mapping.get(v) != device]
    incoming_edges = [(u, v) for u, v in edge_cut_list if
                      operator_device_mapping.get(v) == device and operator_device_mapping.get(u) != device]
    incoming_nodes = set(v for u, v in edge_cut_list if operator_device_mapping.get(v) == device)

    def get_depended_node_set(node):
        destination_node_depended = set(v for (u, v) in outgoing_edges if nx.has_path(graph, node, u))
        for node in destination_node_depended:
            assert operator_device_mapping.get(node) != device
        return destination_node_depended

    def get_depending_node_set(node):
        source_node_depended = set(u for (u, v) in incoming_edges if nx.has_path(graph, v, node))
        for node in source_node_depended:
            assert operator_device_mapping.get(node) != device
        return source_node_depended

    # Create a copy of the original graph
    new_graph = graph.copy()

    # Iterate over the nodes and remove those with 0 related subgraphs
    terminal_node = set(node for node in graph.nodes if len(get_depended_node_set(node)) == 0)
    depended_node = set(graph.nodes) - terminal_node

    isolate_terminal_nodes = set(node for node in terminal_node if len(get_depending_node_set(node)) == 0)

    non_isolated_terminal_nodes = terminal_node - isolate_terminal_nodes

    # Remove the nodes from the copied graph
    new_graph.remove_nodes_from(terminal_node)

    print('ff', len(terminal_node), len(isolate_terminal_nodes), len(non_isolated_terminal_nodes))

    non_isolated_terminal_components = graph.subgraph(non_isolated_terminal_nodes).copy()

    # Identify weakly connected components whose entire predecessors are from source nodes
    terminal_nodes_without_incoming_edge = set()
    weakly_connected_components: list[set] = list(nx.weakly_connected_components(non_isolated_terminal_components))



    for wcc in weakly_connected_components:
        wcc_predecessors = set()
        for node in wcc:
            # Get all predecessors of the current node
            for predecessor in graph.predecessors(node):
                # Only add the predecessor if it's not part of the weakly_connected_component
                if predecessor not in wcc:
                    wcc_predecessors.add(predecessor)
        if wcc_predecessors.issubset(depended_node | isolate_terminal_nodes) and wcc.isdisjoint(incoming_nodes):
            terminal_nodes_without_incoming_edge.update(wcc)
            # remove this part from sink_components
            non_isolated_terminal_components.remove_nodes_from(wcc)

    print('ff2', len(terminal_node), len(isolate_terminal_nodes), len(non_isolated_terminal_components.nodes), len(terminal_nodes_without_incoming_edge))

    def visualize():
        # Draw the nodes with different colors based on their group
        color_map = []
        for node, data in graph.nodes(data=True):  # Unpack node and attributes
            if node in depended_node:
                color_map.append('red')
            elif node in isolate_terminal_nodes:
                color_map.append('blue')
            elif node in non_isolated_terminal_components:
                color_map.append('green')
            elif node in terminal_nodes_without_incoming_edge:
                color_map.append('purple')
            else:
                color_map.append('gray')  # Optional: to handle nodes not in any of the sets

        pos = nx.spring_layout(graph)
        # Plot the graph
        plt.figure(figsize=(10, 8))
        nx.draw(graph, pos, node_color=color_map, with_labels=False, node_size=200)
        plt.title("Visualization of Node Groups")
        plt.show()

    return new_graph, non_isolated_terminal_components, isolate_terminal_nodes, terminal_nodes_without_incoming_edge


def handle_sink_components_with_no_source_predecessors(subgraph, sink_components: nx.DiGraph, device, operator_device_mapping, cut_off, topological_order_mapping, model, start, finish):
    weakly_connected_components: list[set] = list(nx.weakly_connected_components(sink_components))
    sink_nodes = set(sink_components.nodes)
    incoming_nodes = set(v for u, v in cut_off if operator_device_mapping.get(v) == device)
    # check all node has dependency from outside nodes
    for i in sink_nodes:
        assert i in subgraph.nodes
        indicator = False
        for incoming in incoming_nodes:
            if nx.has_path(subgraph, incoming, i):
                indicator = True
                continue
        if indicator == False:
            raise ValueError(i, "does not have depedency from other subgraph")

    # node that directly connected with a cross device dependency
    joint_nodes = sink_nodes.intersection(incoming_nodes)
    other_nodes = sink_nodes - joint_nodes
    print('ppp', joint_nodes)
    print('kkk', weakly_connected_components)

