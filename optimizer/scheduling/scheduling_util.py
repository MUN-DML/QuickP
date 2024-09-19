from typing import Tuple

import networkx as nx

from optimizer.model.graph import CompGraph

'''
Split a graph into three parts
Source Nodes: Nodes that serve as dependencies for other subgraphs.
Sink Nodes: Nodes that do not serve as dependencies for other subgraphs but have dependencies from other graphs. These can be seen as "end points."
Isolated Nodes: Nodes that neither serve as dependencies for other subgraphs nor have dependencies from other graphs. These are independent nodes within the graph structure.
'''


def split_subgraph(graph: CompGraph, operator_device_mapping, edge_cut_list) -> Tuple[CompGraph, CompGraph, set]:
    device = operator_device_mapping[list(graph.nodes)[0]]
    outgoing_edges = [(u, v) for u, v in edge_cut_list if
                      operator_device_mapping.get(u) == device and operator_device_mapping.get(v) != device]
    incoming_edges = [(u, v) for u, v in edge_cut_list if
                      operator_device_mapping.get(v) == device and operator_device_mapping.get(u) != device]

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
    non_source_node = set(node for node in graph.nodes if len(get_depended_node_set(node)) == 0)
    source_node = set(graph.nodes) - non_source_node

    isolate_nodes = set(node for node in non_source_node if len(get_depending_node_set(node)) == 0)

    sink_nodes = non_source_node - isolate_nodes

    # Remove the nodes from the copied graph
    new_graph.remove_nodes_from(non_source_node)

    print('ff', len(non_source_node), len(isolate_nodes), len(sink_nodes))

    sink_components = graph.subgraph(sink_nodes)
    weakly_connected_components: list[set] = list(nx.weakly_connected_components(sink_components))
    for weakly_connected_component in weakly_connected_components:
        wcc_predecessors = set()
        for node in weakly_connected_component:
            wcc_predecessors.update(graph.predecessors(node))
        if not wcc_predecessors.issubset(source_node):
            print("jjjbd")

    return new_graph, sink_components, isolate_nodes


def handle_sink_components(subgraph, sink_components: nx.DiGraph, device, operator_device_mapping, cut_off):
    topological_order = list(nx.topological_sort(subgraph))
    topological_order_mapping = {node: index for index, node in enumerate(topological_order)}
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
    for weakly_connected_component in weakly_connected_components:
        # weak_connected_subgraph = sink_components.subgraph(weakly_connected_component)
        weakly_connected_component = sorted(weakly_connected_component, key=lambda node: topological_order_mapping[node])
