import hashlib
from collections import deque

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path

from optimizer.model.graph import CompGraph, DeviceGraph, visualize_graph


def traverse_merge_loop(comp_graph: CompGraph, device_topo: DeviceGraph, alpha: int):
    while True:
        any_update = traverse_and_merge(comp_graph, device_topo, alpha)
        if not any_update:
            break


def traverse_and_merge(comp_graph: CompGraph, device_topo: DeviceGraph, alpha: int):
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
        elif (comp_graph.getOperatorCompCostByDevice(u, random_device) < alpha and comp_graph.getOperatorCompCostByDevice(v, random_device) < alpha):
            data = comp_graph.merge_edge(u, v)
        elif comp_graph.getOperatorCompCostByDevice(u, random_device) < alpha  and comp_graph.out_degree(u) == 1:
            data = comp_graph.merge_edge(u, v)
        elif comp_graph.getOperatorCompCostByDevice(v, random_device) < alpha and comp_graph.in_degree(v) == 1:
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


def apply_all_co_location_constraint(comp_graph: CompGraph, device_topo: DeviceGraph, number_of_device):
    random_device = comp_graph.getDeviceList()[0]
    slow_link = device_topo.get_slowest_link()
    fast_link = device_topo.get_fastest_link()
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
    '''
    if False:
        edge_set = set()
        for node, best_succ in best_successor.items():
            if comp_graph.out_degree(node) > 1:
                edge_set.add((node, best_succ))

        # find the correct way but need to update group computing cost
        for i, j in comp_graph.edges:
            if comp_graph.out_degree(i) <= 1 or (i, j) in edge_set:
                continue
            if min(comp_graph.get_group_cost_by_edge_set(succ, edge_set) + comp_graph.getEdgeTensorSize(i,succ) * device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1]) for succ in comp_graph.successors(i)) >= sum(comp_graph.get_group_cost_by_edge_set(succ, edge_set) for succ in comp_graph.successors(i)):
                print("added fucker1")
                edge_set.update(comp_graph.out_edges(i))

        for i, j in comp_graph.edges:
            if comp_graph.in_degree(j) <= 1 or (i, j) in edge_set:
                continue
            if min(comp_graph.get_group_cost_by_edge_set(pre, edge_set) + comp_graph.getEdgeTensorSize(pre,j) * device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1]) for pre in comp_graph.predecessors(j)) >= sum(comp_graph.get_group_cost_by_edge_set(pre, edge_set) for pre in comp_graph.predecessors(j)):
                print("added fucker2")
                edge_set.update(comp_graph.in_edges(j))

        edge_subgraph = comp_graph.edge_subgraph(edge_set)
        print("number of nodes in edge-subg", edge_subgraph.number_of_nodes())
        visualize_graph(edge_subgraph, show_edge_labels=False, show_node_labels=False)
        wcc_node_sets = list(nx.weakly_connected_components(edge_subgraph))
        print("number of wcc in edge_subg", len(wcc_node_sets))
    '''

    node_set = set()
    for node, best_succ in best_successor.items():
        if comp_graph.out_degree(node) > 1:
            node_set.update([node, best_succ])

    # find the correct way but need to update group computing cost
    for i, j in comp_graph.edges:
        if comp_graph.out_degree(i) <= 1 or {i, j}.issubset(node_set):
            continue
        if min(comp_graph.get_group_cost_by_node_set(succ, node_set) + comp_graph.getEdgeTensorSize(i,succ) * device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1]) for succ in comp_graph.successors(i)) >= sum(comp_graph.get_group_cost_by_node_set(succ, node_set) for succ in comp_graph.successors(i)):
            print("added successors")
            node_set.update(comp_graph.successors(i))

    for i, j in comp_graph.edges:
        if comp_graph.in_degree(j) <= 1 or {i, j}.issubset(node_set):
            continue
        if min(comp_graph.get_group_cost_by_node_set(pre, node_set) + comp_graph.getEdgeTensorSize(pre,j) * device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1]) for pre in comp_graph.predecessors(j)) >= sum(comp_graph.get_group_cost_by_node_set(pre, node_set) for pre in comp_graph.predecessors(j)):
            print("added predecessors")
            node_set.update(comp_graph.predecessors(j))

    subgraph = comp_graph.subgraph(node_set)
    print("number of nodes in subg", subgraph.number_of_nodes())

    subgraph.visualize_graphviz()
    wcc_node_sets = list(nx.weakly_connected_components(subgraph))
    print("number of wcc in node_subgraph", len(wcc_node_sets))

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