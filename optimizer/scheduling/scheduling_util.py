from itertools import combinations
from typing import Tuple, Any, Set, Iterator

import networkx as nx
from gurobipy import Model, GRB
from matplotlib import pyplot as plt
from networkx.classes import Graph

from optimizer.model.graph import CompGraph, combine_graphs

'''
Split a graph into three parts
Source Nodes: Nodes that serve as dependencies for other subgraphs.
Sink Nodes: Nodes that do not serve as dependencies for other subgraphs but have dependencies from other graphs. These can be seen as "end points."
Isolated Nodes: Nodes that neither serve as dependencies for other subgraphs nor have dependencies from other graphs. These are independent nodes within the graph structure.
'''


def split_subgraph(subgraph: CompGraph, operator_device_mapping, edge_cut_list) -> tuple[
    Graph, set, set[Any], set[Any], Any]:
    device = operator_device_mapping[list(subgraph.nodes)[0]]
    outgoing_edges = [(u, v) for u, v in edge_cut_list if
                      operator_device_mapping.get(u) == device and operator_device_mapping.get(v) != device]
    incoming_edges = [(u, v) for u, v in edge_cut_list if
                      operator_device_mapping.get(v) == device and operator_device_mapping.get(u) != device]
    comm_end_nodes = set(v for u, v in edge_cut_list if operator_device_mapping.get(v) == device)

    def get_depended_node_set(node):
        destination_node_depended = set(v for (u, v) in outgoing_edges if nx.has_path(subgraph, node, u))
        for node in destination_node_depended:
            assert operator_device_mapping.get(node) != device
        return destination_node_depended

    def get_depending_node_set(node):
        source_node_depended = set(u for (u, v) in incoming_edges if nx.has_path(subgraph, v, node))
        for node in source_node_depended:
            assert operator_device_mapping.get(node) != device
        return source_node_depended

    # Iterate over the nodes and remove those with 0 related subgraphs
    terminal_node = set(node for node in subgraph.nodes if len(get_depended_node_set(node)) == 0)
    isolate_terminal_nodes = set(node for node in terminal_node if len(get_depending_node_set(node)) == 0)
    non_isolated_terminal_nodes = terminal_node - isolate_terminal_nodes

    depended_node = set(subgraph.nodes) - terminal_node
    independent_depended = set(node for node in depended_node if len(get_depending_node_set(node)) == 0)
    dependent_depended = depended_node - independent_depended

    # Remove the nodes from the copied graph
    stage_one = subgraph.subgraph(independent_depended)
    stage_three = subgraph.subgraph(non_isolated_terminal_nodes).copy()

    # Identify weakly connected components whose entire predecessors are from depdended and isolated nodes
    terminal_nodes_without_comm_np = set()
    weakly_connected_components: list[set] = list(nx.weakly_connected_components(stage_three))

    for wcc in weakly_connected_components:
        wcc_predecessors = set()
        for node in wcc:
            # Get all predecessors of the current node
            for predecessor in subgraph.predecessors(node):
                # Only add the predecessor if it's not part of the weakly_connected_component
                if predecessor not in wcc:
                    wcc_predecessors.add(predecessor)
        if wcc_predecessors.issubset(depended_node | isolate_terminal_nodes) and wcc.isdisjoint(comm_end_nodes):
            terminal_nodes_without_comm_np.update(wcc)
            # remove this part from sink_components
            stage_three.remove_nodes_from(wcc)

    def visualize():
        # Draw the nodes with different colors based on their group
        color_map = []
        for node, data in subgraph.nodes(data=True):  # Unpack node and attributes
            if node in depended_node:
                color_map.append('red')
            elif node in isolate_terminal_nodes:
                color_map.append('blue')
            elif node in stage_three:
                color_map.append('green')
            elif node in terminal_nodes_without_comm_np:
                color_map.append('purple')
            else:
                color_map.append('gray')  # Optional: to handle nodes not in any of the sets

        pos = nx.spring_layout(subgraph)
        # Plot the graph
        plt.figure(figsize=(10, 8))
        nx.draw(subgraph, pos, node_color=color_map, with_labels=False, node_size=200)
        plt.title("Visualization of Node Groups")
        plt.show()
    assert len(subgraph.nodes) == len(stage_one.nodes) + len(dependent_depended)  + len(isolate_terminal_nodes) + len(terminal_nodes_without_comm_np) + len(stage_three.nodes)
    return stage_one, dependent_depended, isolate_terminal_nodes, terminal_nodes_without_comm_np, stage_three


def handle_terminal_components_with_comm_end_point(subgraph, components_to_be_op: nx.DiGraph, device, operator_device_mapping, cut_off, model: Model, start, finish, last_stage_finish, topo_dict):
    weakly_connected_components: list[set] = list(nx.weakly_connected_components(components_to_be_op))
    # Convert each wcc (which is a set) to a tuple and store it in a list
    wcc_tuples = [tuple(wcc) for wcc in weakly_connected_components]

    # Use addVars to create wcc_start and wcc_finish variables for each wcc
    wcc_start = model.addVars(wcc_tuples, vtype=GRB.CONTINUOUS)
    wcc_finish = model.addVars(wcc_tuples, vtype=GRB.CONTINUOUS)
    all_nodes = set(components_to_be_op.nodes)
    comm_end_nodes = set(v for u, v in cut_off if operator_device_mapping.get(v) == device)
    # check all node has dependency from outside nodes
    for i in all_nodes:
        assert i in subgraph.nodes
        indicator = False
        for incoming in comm_end_nodes:
            if nx.has_path(subgraph, incoming, i):
                indicator = True
                continue
        if indicator == False:
            raise ValueError(i, "does not have depedency from other subgraph")

    # node that directly connected with a cross device dependency
    for wcc in wcc_tuples:
        sorted_nodes = sorted(list(wcc), key=lambda node: topo_dict[node])
        # check the first node in each wcc is a comm end node
        assert sorted_nodes[0] in comm_end_nodes
        # Apply sequential constraint
        model.addConstr(wcc_start[wcc] == start[sorted_nodes[0]])
        model.addConstr(wcc_start[wcc] >= finish[last_stage_finish])
        model.addConstr(wcc_finish[wcc] == finish[sorted_nodes[-1]])
        for a, b in zip(sorted_nodes, sorted_nodes[1:]):
            model.addConstr(finish[a] <= start[b])

    order_wcc = {}
    M = 1000000
    # add non-overalpping constarints to wcc
    for wcc, wcc2 in combinations(wcc_tuples, 2):
        order_wcc[wcc, wcc2] = model.addVar(vtype=GRB.BINARY)
        model.addConstr(wcc_start[wcc2] >= wcc_finish[wcc] - M * (1 - order_wcc[wcc, wcc2]))
        model.addConstr(wcc_start[wcc] >= wcc_finish[wcc2] - M * order_wcc[wcc, wcc2])


def handle_stage_two(subgraph: Graph, dependent_depended, isolated_node_list, terminal_nodes_without_comm_np, model: Model, start, finish, last_stage_finish):
    weakly_connected_components: list[set] = list(nx.weakly_connected_components(subgraph.subcomponents_to_be_op))
    # Convert each wcc (which is a set) to a tuple and store it in a list
    wcc_tuples = [tuple(wcc) for wcc in weakly_connected_components]

    # Use addVars to create wcc_start and wcc_finish variables for each wcc
    wcc_start = model.addVars(wcc_tuples, vtype=GRB.CONTINUOUS)
    wcc_finish = model.addVars(wcc_tuples, vtype=GRB.CONTINUOUS)
    topological_order = list(nx.topological_sort(subgraph))
    topological_order_mapping = {node: index for index, node in enumerate(topological_order)}

    # node that directly connected with a cross device dependency
    for wcc in wcc_tuples:
        sorted_nodes = sorted(list(wcc), key=lambda node: topological_order_mapping[node])
        # Apply sequential constraint
        model.addConstr(wcc_start[wcc] == start[sorted_nodes[0]])
        model.addConstr(wcc_start[wcc] >= finish[last_stage_finish])
        model.addConstr(wcc_finish[wcc] == finish[sorted_nodes[-1]])
        for a, b in zip(sorted_nodes, sorted_nodes[1:]):
            model.addConstr(finish[a] <= start[b])


def handle_stage_three(subgraph, components_to_be_op: nx.DiGraph, device, operator_device_mapping, cut_off, model: Model, start, finish, last_stage_finish):
    weakly_connected_components: list[set] = list(nx.weakly_connected_components(components_to_be_op))
    # Convert each wcc (which is a set) to a tuple and store it in a list
    wcc_tuples = [tuple(wcc) for wcc in weakly_connected_components]

    # Use addVars to create wcc_start and wcc_finish variables for each wcc
    wcc_start = model.addVars(wcc_tuples, vtype=GRB.CONTINUOUS)
    wcc_finish = model.addVars(wcc_tuples, vtype=GRB.CONTINUOUS)
    topological_order = list(nx.topological_sort(subgraph))
    topological_order_mapping = {node: index for index, node in enumerate(topological_order)}
    all_nodes = set(components_to_be_op.nodes)
    comm_end_nodes = set(v for u, v in cut_off if operator_device_mapping.get(v) == device)
    # check all node has dependency from outside nodes
    for i in all_nodes:
        assert i in subgraph.nodes
        indicator = False
        for incoming in comm_end_nodes:
            if nx.has_path(subgraph, incoming, i):
                indicator = True
                continue
        if indicator == False:
            raise ValueError(i, "does not have depedency from other subgraph")

    # node that directly connected with a cross device dependency
    for wcc in wcc_tuples:
        sorted_nodes = sorted(list(wcc), key=lambda node: topological_order_mapping[node])
        # check the first node in each wcc is a comm end node
        assert sorted_nodes[0] in comm_end_nodes
        # Apply sequential constraint
        model.addConstr(wcc_start[wcc] == start[sorted_nodes[0]])
        model.addConstr(wcc_start[wcc] >= last_stage_finish[device])
        model.addConstr(wcc_finish[wcc] == finish[sorted_nodes[-1]])
        for a, b in zip(sorted_nodes, sorted_nodes[1:]):
            model.addConstr(finish[a] <= start[b])

    order_wcc = {}
    M = 1000000
    # add non-overalpping constarints to wcc
    for wcc, wcc2 in combinations(wcc_tuples, 2):
        order_wcc[wcc, wcc2] = model.addVar(vtype=GRB.BINARY)
        model.addConstr(wcc_start[wcc2] >= wcc_finish[wcc] - M * (1 - order_wcc[wcc, wcc2]))
        model.addConstr(wcc_start[wcc] >= wcc_finish[wcc2] - M * order_wcc[wcc, wcc2])


def three_stage_split_subgraph(subgraph: CompGraph, operator_device_mapping, edge_cut_list) -> tuple[
    Graph, Graph, Graph]:

    device = operator_device_mapping[list(subgraph.nodes)[0]]
    outgoing_edges = [(u, v) for u, v in edge_cut_list if
                      operator_device_mapping.get(u) == device and operator_device_mapping.get(v) != device]
    incoming_edges = [(u, v) for u, v in edge_cut_list if
                      operator_device_mapping.get(v) == device and operator_device_mapping.get(u) != device]
    comm_end_nodes = set(v for u, v in edge_cut_list if operator_device_mapping.get(v) == device)

    def get_depended_node_set(node):
        destination_node_depended = set(v for (u, v) in outgoing_edges if nx.has_path(subgraph, node, u))
        for node in destination_node_depended:
            assert operator_device_mapping.get(node) != device
        return destination_node_depended

    def get_depending_node_set(node):
        source_node_depended = set(u for (u, v) in incoming_edges if nx.has_path(subgraph, v, node))
        for node in source_node_depended:
            assert operator_device_mapping.get(node) != device
        return source_node_depended

    # Iterate over the nodes and remove those with 0 related subgraphs
    terminal_node = set(node for node in subgraph.nodes if len(get_depended_node_set(node)) == 0)
    isolate_terminal_nodes = set(node for node in terminal_node if len(get_depending_node_set(node)) == 0)
    non_isolated_terminal_nodes = terminal_node - isolate_terminal_nodes

    depended_node = set(subgraph.nodes) - terminal_node
    independent_depended = set(node for node in depended_node if len(get_depending_node_set(node)) == 0)
    dependent_depended = depended_node - independent_depended

    # Remove the nodes from the copied graph
    stage_one = subgraph.subgraph(independent_depended)
    stage_three = subgraph.subgraph(non_isolated_terminal_nodes).copy()

    # Identify weakly connected components whose entire predecessors are from depdended and isolated nodes
    terminal_nodes_without_comm_np = set()
    weakly_connected_components: list[set] = list(nx.weakly_connected_components(stage_three))

    for wcc in weakly_connected_components:
        wcc_predecessors = set()
        for node in wcc:
            # Get all predecessors of the current node
            for predecessor in subgraph.predecessors(node):
                # Only add the predecessor if it's not part of the weakly_connected_component
                if predecessor not in wcc:
                    wcc_predecessors.add(predecessor)
        if wcc_predecessors.issubset(depended_node | isolate_terminal_nodes) and wcc.isdisjoint(comm_end_nodes):
            terminal_nodes_without_comm_np.update(wcc)
            # remove this part from sink_components
            stage_three.remove_nodes_from(wcc)

    stage_two = subgraph.subgraph(isolate_terminal_nodes | dependent_depended | terminal_nodes_without_comm_np)


    assert len(subgraph.nodes) == len(stage_one.nodes) + len(stage_two.nodes) + len(stage_three.nodes)
    return stage_one, stage_two, stage_three