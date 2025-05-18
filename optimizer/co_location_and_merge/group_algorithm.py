import hashlib
import random
from collections import deque
from enum import Enum

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
        if comp_graph.out_degree(u) >= 2 and comp_graph.in_degree(v) >= 2:
            continue
        if comp_graph.out_degree(u) + comp_graph.in_degree(v) == 2:
            data = comp_graph.merge_edge(u, v)
        elif comp_graph.getOperatorCompCostByDevice(u, random_device) < alpha and comp_graph.out_degree(u) == 1:
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


def group_longest_path(comp_graph: CompGraph, device_topo: DeviceGraph):
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
        global_rank[current_node] = max_suc_total_cost + comp_graph.getOperatorCompCostByDevice(current_node, random_device)

    edge_set = set()
    for node, best_succ in best_successor.items():
        if comp_graph.out_degree(node) > 1:
            edge_set.add((node, best_succ))

    subgraph = comp_graph.edge_subgraph(edge_set)
    print("number of nodes in subg", subgraph.number_of_nodes())

    wcc_node_sets = list(nx.weakly_connected_components(subgraph))
    print("number of wcc in node_subgraph", len(wcc_node_sets))

    for node_set in wcc_node_sets:
        new_id = hashlib.md5("&".join(node_set).encode()).hexdigest()
        for node in node_set:
            comp_graph.set_colocation_group(node, new_id)


def iteratively_expand_wcc(comp_graph: CompGraph, deviceTopo: DeviceGraph, beta=10):
    q = set(op for op in comp_graph.nodes if 'colocation_group' in comp_graph.nodes[op])
    print("number of nodes in all wcc before expanding", len(q))
    any_d = deviceTopo.getDeviceIDs()[0]
    while True:
        any_update = False
        eligible_nodes = {
            node for node in comp_graph.nodes()
            if 'colocation_group' not in comp_graph.nodes[node] and comp_graph.getOperatorCompCostByDevice(node, any_d) < beta
        }
        while eligible_nodes:
            node = eligible_nodes.pop()
            grouped_neighbour = {
                n for n in set(comp_graph.predecessors(node)).union(comp_graph.successors(node))
                if 'colocation_group' in comp_graph.nodes[n] and comp_graph.nodes[n]['colocation_group'] is not None
            }
            # no grouped neighbour
            if not grouped_neighbour:
                continue
            random_neighbour = random.choice(list(grouped_neighbour))
            comp_graph.set_colocation_group(node, comp_graph.get_colocation_group(random_neighbour))
            any_update = True
        if not any_update:
            break
        q = set(op for op in comp_graph.nodes if 'colocation_group' in comp_graph.nodes[op])
        print("number of nodes in all wcc after expanding", len(q))

# deprecated
def fuse_weakly_connected_components(computation_graph: CompGraph):
    group_ops_mapping = computation_graph.create_colocation_group_to_ops_map()
    for node_set in group_ops_mapping.values():
        wcc_graph: CompGraph = computation_graph.subgraph(node_set)
        edges_to_process = set(wcc_graph.edges())
        while len(edges_to_process) > 1:
            u, v = edges_to_process.pop()
            # if computation_graph.out_degree(u) == 1 or computation_graph.in_degree(v) == 1:
            if computation_graph.is_edge_mergable(u, v):
                data = computation_graph.merge_edge(u, v)
                if data:
                    new_edges, deleted_edges = data
                    for a, b in new_edges:
                        # if two ends of a newly created edge also have a co-location relationship
                        if all("colocation_group" in computation_graph.nodes[node] for node in (a, b)):
                            if computation_graph.get_colocation_group(a) == computation_graph.get_colocation_group(b):
                                edges_to_process.add((a, b))
                    edges_to_process -= deleted_edges
    print("current op number", computation_graph.number_of_nodes())


def QuickP_op_fusion_and_colocation(comp_graph, deviceTopo: DeviceGraph, alpha, beta):
    traverse_merge_loop(comp_graph, deviceTopo, alpha)
    group_longest_path(comp_graph, deviceTopo)
    iteratively_expand_wcc(comp_graph, deviceTopo, beta)


class SearchSpaceReductionScheme(Enum):
    QuickP = QuickP_op_fusion_and_colocation


def apply_search_space_reduction_scheme(function: SearchSpaceReductionScheme, **kwargs):
    # Define the required arguments for each scheduling algorithm
    required_args = {
        SearchSpaceReductionScheme.QuickP: ['comp_graph', 'deviceTopo', 'alpha', 'beta'],
    }

    if function not in required_args:
        raise ValueError(f"Unknown scheduling algorithm: {function}")

    # Check if all required arguments are provided
    missing_args = [arg for arg in required_args[function] if arg not in kwargs]
    if missing_args:
        raise ValueError(f"Missing arguments for {function}: {', '.join(missing_args)}")

    # Select the appropriate arguments for the scheduling function
    selected_kwargs = {key: kwargs[key] for key in required_args[function]}

    # Dynamically dispatch to the appropriate scheduling function
    function(**selected_kwargs)
