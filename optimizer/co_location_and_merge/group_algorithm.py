import hashlib
from collections import deque

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path

from optimizer.model.graph import CompGraph, DeviceGraph, visualize_graph


def traverse_merge_loop_no_performance_degradation(comp_graph: CompGraph, device_topo: DeviceGraph):

    def traverse_and_merge_no_performance_degradation(comp_graph: CompGraph, device_topo: DeviceGraph):
        any_data_update = False
        random_device = comp_graph.getDeviceList()[0]
        # set is implemented by hashtable, fast deletion and adding
        edges_to_process = set(comp_graph.edges())
        while edges_to_process:
            u, v = edges_to_process.pop()
            if not comp_graph.is_edge_mergable(u, v):
                continue
            # Check if the edge is marked with the attribute 'ismerge'
            # if (self.getOperatorCompCostByDevice(u, random_device) == 0 or self.getOperatorCompCostByDevice(v, random_device) == 0) and (self.out_degree(u) == 1 ):
            if comp_graph.out_degree(u) + comp_graph.in_degree(v) == 2:
                data = comp_graph.merge_edge(u, v)
            elif (comp_graph.getOperatorCompCostByDevice(u,random_device) == 0 and comp_graph.getOperatorCompCostByDevice(
                    v, random_device) == 0):
                data = comp_graph.merge_edge(u, v)
            elif comp_graph.getOperatorCompCostByDevice(u, random_device) == 0 and comp_graph.out_degree(u) == 1:
                data = comp_graph.merge_edge(u, v)
            elif comp_graph.getOperatorCompCostByDevice(v, random_device) == 0 and comp_graph.in_degree(v) == 1:
                data = comp_graph.merge_edge(u, v)
            else:
                data = None

            if data:
                new_edges, deleted_edges = data
                edges_to_process -= deleted_edges
                edges_to_process |= new_edges
                if not any_data_update:
                    any_data_update = True

        assert nx.is_directed_acyclic_graph(comp_graph)
        print("current op number", comp_graph.number_of_nodes())
        return any_data_update


    while True:
        any_update = traverse_and_merge_no_performance_degradation(comp_graph, device_topo)
        if not any_update:
            break


def traverse_merge_loop(comp_graph: CompGraph, device_topo: DeviceGraph):
    while True:
        any_update = traverse_and_merge(comp_graph, device_topo)
        if not any_update:
            break


def traverse_and_merge(comp_graph: CompGraph, device_topo: DeviceGraph):
    any_data_update = False
    random_device = comp_graph.getDeviceList()[0]
    fast_link = device_topo.get_fastest_link()
    # set is implemented by hashtable, fast deletion and adding
    edges_to_process = set(comp_graph.edges())
    while edges_to_process:
        u, v = edges_to_process.pop()
        if not comp_graph.is_edge_mergable(u, v):
            continue
        # Check if the edge is marked with the attribute 'ismerge'
        # if (self.getOperatorCompCostByDevice(u, random_device) == 0 or self.getOperatorCompCostByDevice(v, random_device) == 0) and (self.out_degree(u) == 1 ):
        if comp_graph.out_degree(u) + comp_graph.in_degree(v) == 2:
            data = comp_graph.merge_edge(u, v)
            update_shortest_path_cost(comp_graph, u, device_topo, fast_link)
        elif (comp_graph.getOperatorCompCostByDevice(u, random_device) == 0 or comp_graph.getOperatorCompCostByDevice(v,random_device) == 0):
            if comp_graph.getOperatorCompCostByDevice(v, random_device) == 0 and comp_graph.getOperatorCompCostByDevice(
                    u, random_device) > 0 and comp_graph.in_degree(v) > 1:
                continue
            if comp_graph.getOperatorCompCostByDevice(u, random_device) == 0 and comp_graph.getOperatorCompCostByDevice(
                    v, random_device) > 0 and comp_graph.out_degree(u) > 1:
                continue
            data = comp_graph.merge_edge(u, v)
            update_shortest_path_cost(comp_graph, u, device_topo, fast_link)
        elif (comp_graph.get_shortest_path_cost(u) - comp_graph.getOperatorCompCostByDevice(u,random_device) >=
              sum(comp_graph.get_shortest_path_cost(succ) for succ in comp_graph.successors(u))):
            data = comp_graph.merge_edge(u, v)
            update_shortest_path_cost(comp_graph, u, device_topo, fast_link)
        else:
            data = None

        if data:
            new_edges, deleted_edges = data
            edges_to_process -= deleted_edges
            edges_to_process |= new_edges
            if not any_data_update:
                any_data_update = True

    assert nx.is_directed_acyclic_graph(comp_graph)
    print("current op number", comp_graph.number_of_nodes())
    return any_data_update


def apply_co_location_constraint(comp_graph: CompGraph, device_topo: DeviceGraph):
    all_node_set = set()
    for edge in comp_graph.edges():
        if not comp_graph.is_edge_mergable(edge[0], edge[1]) and len(
                nx.minimum_edge_cut(comp_graph, edge[0], edge[1], flow_func=shortest_augmenting_path)) == 2:
            all_paths = list(nx.node_disjoint_paths(comp_graph, edge[0], edge[1]))
            flattened_set = set([element for sublist in all_paths for element in sublist])
            all_node_set.update(flattened_set)
    subgraph = comp_graph.subgraph(all_node_set)
    visualize_graph(subgraph, show_edge_labels=False, show_node_labels=False)
    wcc_node_sets = list(nx.weakly_connected_components(subgraph))
    for node_set in wcc_node_sets:
        new_id = hashlib.md5("&".join(node_set).encode()).hexdigest()
        for node in node_set:
            comp_graph.set_colocation_group(node, new_id)


def get_longest_path(comp_graph, device_topo: DeviceGraph):
    random_device = comp_graph.getDeviceList()[0]
    slow_link = device_topo.get_slowest_link()
    global_rank = {}
    best_successor = {}  # To store the best successor of each node for path reconstruction
    topo_sorted = list(nx.topological_sort(comp_graph))

    for current_node in reversed(topo_sorted):
        # Check if the current node has any predecessors
        successors = list(comp_graph.successors(current_node))

        if successors:  # If there are predecessors, compute the max computing cost
            best_successor[current_node], max_suc_total_cost = max(
                ((succ_node, global_rank[succ_node] + comp_graph.getEdgeTensorSize(current_node, succ_node)
                  * device_topo.calUnitCommCostInUS(slow_link[0], slow_link[1])) for succ_node in successors),
                key=lambda x: x[1]
            )
        else:  # If there are no predecessors, set the max computing cost to 0
            max_suc_total_cost = 0
            best_successor[current_node] = None  # No successor for sink nodes

        # Calculate the global rank for the current node
        global_rank[current_node] = max_suc_total_cost + comp_graph.getOperatorCompCostByDevice(current_node,
                                                                                                random_device)

    max_rank_node = max(global_rank, key=global_rank.get)

    # Reconstruct the longest path using the best_successor dictionary
    longest_path = []
    current_node = max_rank_node

    while current_node is not None:
        longest_path.append(current_node)
        current_node = best_successor[current_node]

    print('fuck', longest_path)

    new_id = hashlib.md5("&".join(longest_path).encode()).hexdigest()
    for node in longest_path:
        comp_graph.set_colocation_group(node, new_id)

    return longest_path


def apply_critical_path_based_co_location(comp_graph: CompGraph, device_topo: DeviceGraph):
    random_device = comp_graph.getDeviceList()[0]
    slow_link = device_topo.get_slowest_link()
    global_rank = {}
    best_successor = {}  # To store the best successor of each node for path reconstruction
    topo_sorted = list(nx.topological_sort(comp_graph))

    for current_node in reversed(topo_sorted):
        # Check if the current node has any predecessors
        successors = list(comp_graph.successors(current_node))

        if successors:  # If there are predecessors, compute the max computing cost
            best_successor[current_node], max_suc_total_cost = max(
                ((succ_node, global_rank[succ_node] + comp_graph.getEdgeTensorSize(current_node, succ_node)
                  * device_topo.calUnitCommCostInUS(slow_link[0], slow_link[1])) for succ_node in successors),
                key=lambda x: x[1]
            )
        else:  # If there are no predecessors, set the max computing cost to 0
            max_suc_total_cost = 0
            best_successor[current_node] = None  # No successor for sink nodes

        # Calculate the global rank for the current node
        global_rank[current_node] = max_suc_total_cost + comp_graph.getOperatorCompCostByDevice(current_node,
                                                                                                random_device)
    edge_set = set()
    for node, best_succ in best_successor.items():
        if comp_graph.out_degree(node) > 1:
            edge_set.add((node, best_succ))
    print("number of edges", len(edge_set))
    subgraph = comp_graph.edge_subgraph(edge_set)
    visualize_graph(subgraph, show_edge_labels=False, show_node_labels=False)
    wcc_node_sets = list(nx.weakly_connected_components(subgraph))
    for node_set in wcc_node_sets:
        new_id = hashlib.md5("&".join(node_set).encode()).hexdigest()
        for node in node_set:
            comp_graph.set_colocation_group(node, new_id)


def min_rank_calculation(comp_graph: CompGraph, device_topo: DeviceGraph):
    random_device = comp_graph.getDeviceList()[0]
    fast_link = device_topo.get_fastest_link()
    global_rank = {}
    topo_sorted = list(nx.topological_sort(comp_graph))

    for current_node in reversed(topo_sorted):
        # Check if the current node has any predecessors
        successors = list(comp_graph.successors(current_node))

        if successors:  # If there are predecessors, compute the max computing cost
            min_suc_total_cost = min(global_rank[succ_node] + comp_graph.getEdgeTensorSize(current_node, succ_node)*
                                     device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1])
                                     for succ_node in successors)

        else:  # If there are no predecessors, set the max computing cost to 0
            min_suc_total_cost = 0

        # Calculate the global rank for the current node
        global_rank[current_node] = min_suc_total_cost + comp_graph.getOperatorCompCostByDevice(current_node,
                                                                                                random_device)
        comp_graph.nodes[current_node]["shortest_path_cost"] = global_rank[current_node]

def update_shortest_path_cost(comp_graph: CompGraph, node, device_topo, fast_link):
    successors = list(comp_graph.successors(node))
    random_device = comp_graph.getDeviceList()[0]
    if successors:  # If there are predecessors, compute the max computing cost
        comp_graph.nodes[node]["shortest_path_cost"] = min(comp_graph.get_shortest_path_cost(succ_node) + comp_graph.getEdgeTensorSize(node, succ_node) *
                                device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1])
                                for succ_node in successors) + comp_graph.getOperatorCompCostByDevice(node,random_device)
    else:
        comp_graph.nodes[node]["shortest_path_cost"] = comp_graph.getOperatorCompCostByDevice(node,random_device)