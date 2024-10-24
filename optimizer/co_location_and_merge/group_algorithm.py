import hashlib
from collections import deque

import networkx as nx
from networkx.algorithms.flow import shortest_augmenting_path

from optimizer.co_location_and_merge.grouper_util import create_eligible_edge_subgraph, label_group, analyze_group
from optimizer.model.graph import CompGraph, DeviceGraph, visualize_graph


def group_and_fuse_op_incrementally(comp_graph, deviceTopo):
    # there should be many iterations until there is no eligible ops to merge
    while True:
        # After each merge, the comp_cost map will change
        comp_cost = comp_graph.getOpCompCostMapByDevice(deviceTopo.getDeviceIDs()[0])
        subgraph_of_wcc = create_eligible_edge_subgraph(comp_graph, deviceTopo, comp_cost)
        # if no node is labeled with the co-location attribute, there is no need to merge operators any more
        if len(subgraph_of_wcc.nodes) == 0:
            break
        # After all node get labelled, merge groups
        label_group(subgraph_of_wcc)
        analyze_group(comp_graph, comp_cost)
        # merge ops based on the merged groups
        graph_coarsen(comp_graph, subgraph_of_wcc, comp_cost)
    print("new computing graph node number", comp_graph.number_of_nodes())
    comp_graph.save_to_file('grouped_computing_graph.json')


# _generate_fused_op_graph
def graph_coarsen(computing_graph: CompGraph, sub_graph_of_wcc: CompGraph, computing_cost_dict):
    def merge_operators(ops_to_be_merged: set):

        # double check if those nodes are connected, forming one weakly connected component
        wcc = computing_graph.subgraph(ops_to_be_merged)
        if not nx.is_weakly_connected(wcc):
            raise ValueError(f"{ops_to_be_merged} are not connected")

        internal_edges = deque(wcc.edges)

        while len(internal_edges) > 0:
            op1, op2 = internal_edges.popleft()
            if computing_graph.out_degree(op1) > 1 and computing_graph.in_degree(op2) > 1:
                # CAVEATS: finding disjoint paths may take long time
                paths = list(nx.node_disjoint_paths(computing_graph, op1, op2))
                if len(paths) > 1:
                    raise ValueError(f"{op1} and {op2} has more than one disjoint path")

        # create attributes for the new node
        random_node_cost_dict = computing_graph.getCompCostMapByOp(list(ops_to_be_merged)[0])
        new_computing_cost = sum(computing_cost_dict[op] for op in ops_to_be_merged)
        new_comp_cost_dict = {op: new_computing_cost for op in random_node_cost_dict.keys()}
        new_memory = sum(computing_graph.getMemorySize(op) for op in ops_to_be_merged)

        # add the new node
        new_id = hashlib.md5("&".join(ops_to_be_merged).encode()).hexdigest()
        computing_graph.add_new_node(new_id, "merged",
                                     memory=new_memory, comp_cost_map=new_comp_cost_dict)

        # restore the dependency relationship
        # Redirect in-edges (predecessors of the nodes to merge)
        for node in ops_to_be_merged:
            for pred in computing_graph.predecessors(node):
                if pred not in ops_to_be_merged:  # Avoid self-loops
                    computing_graph.add_edge(pred, new_id, **computing_graph.get_edge_data(pred, node))

        # Redirect out-edges (successors of the nodes to merge)
        for node in ops_to_be_merged:
            for succ in computing_graph.successors(node):
                if succ not in ops_to_be_merged:  # Avoid self-loops
                    computing_graph.add_edge(new_id, succ, **computing_graph.get_edge_data(node, succ))

        # Remove the original nodes
        computing_graph.remove_nodes_from(ops_to_be_merged)

        # Double check if the graph after merge is still DAG
        assert nx.is_directed_acyclic_graph(computing_graph)

    weakly_connected_components = list(nx.weakly_connected_components(sub_graph_of_wcc))
    for wcc_set in weakly_connected_components:
        merge_operators(wcc_set)


'''
The below is the latest algorithm
'''


def get_subgraph_of_eligible_edges(graph: CompGraph, device_topo: DeviceGraph, computing_cost_dict):
    fast_link = device_topo.get_fastest_link()
    eligible_edges = set()
    for edge in graph.edges:
        source, destination = edge
        if computing_cost_dict[source] == 0:
            eligible_edges.add(source)
            eligible_edges.add(destination)
            continue
        communication_cost = graph.getEdgeTensorSize(source, destination) * device_topo.calUnitCommCostInUS(
            fast_link[0], fast_link[1])
        # the source only has one outgoing edge and communication cost if on different device is higher than
        # and graph.in_degree(destination) == 1 will minimize the performance loss
        if communication_cost >= computing_cost_dict[destination]:
            # label both end the group of source node. One node will probably have more than one group. Waiting to merge groups
            eligible_edges.add(source)
            eligible_edges.add(destination)
    return graph.subgraph(eligible_edges)


def coarsen_weakly_connected_component(wcc_set: set, computation_graph: CompGraph, computing_cost_dict):
    sub_graph = computation_graph.subgraph(list(wcc_set))
    if not nx.is_weakly_connected(sub_graph):
        raise ValueError(f"{sub_graph.nodes} are not connected")

    internal_edges = deque(sub_graph.edges)
    while len(internal_edges) > 0:
        source, target = internal_edges.popleft()
        if is_worth_merging(source, target, ):
            merge_node_pair(source, target, computation_graph, computing_cost_dict)


def merge_node_pair(u, v, computation_graph: CompGraph, computing_cost_dict):
    if not computation_graph.is_edge_mergable(u, v):
        return
    # create attributes for the new node
    random_node_cost_dict = computation_graph.getCompCostMapByOp(u)
    new_computing_cost = sum(computing_cost_dict[op] for op in [u, v])
    new_comp_cost_dict = {op: new_computing_cost for op in random_node_cost_dict.keys()}
    new_memory = sum(computation_graph.getMemorySize(op) for op in [u, v])
    # new tensorsize should also be created

    # add the new node
    new_id = hashlib.md5("&".join([u, v]).encode()).hexdigest()
    computation_graph.add_new_node(new_id, "merged",
                                   memory=new_memory, comp_cost_map=new_comp_cost_dict)

    # Redirect in-edges (predecessors of the nodes to merge)
    for node in [u, v]:
        for pred in computation_graph.predecessors(node):
            if pred not in [u, v]:  # Avoid self-loops
                computation_graph.add_edge(pred, new_id, **computation_graph.get_edge_data(pred, node))

    # Redirect out-edges (successors of the nodes to merge)
    for node in [u, v]:
        for succ in computation_graph.successors(node):
            if succ not in [u, v]:  # Avoid self-loops
                computation_graph.add_edge(new_id, succ, **computation_graph.get_edge_data(node, succ))

    # Remove the original nodes
    computation_graph.remove_nodes_from([u, v])

    # Double check if the graph after merge is still DAG
    assert nx.is_directed_acyclic_graph(computation_graph)


# Since new node and edge are formed after op-fusion
def is_worth_merging(source, target, computation_graph: CompGraph, device_topo, fast_link, computing_cost_dict):
    destination_computing_cost = computing_cost_dict[target]
    communication_cost = computation_graph.getEdgeTensorSize(source, target) * device_topo.calUnitCommCostInUS(
        fast_link[0], fast_link[1])
    if communication_cost >= destination_computing_cost or computing_cost_dict[source] == 0:
        return True
    else:
        return False


'traverse and merge function'


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
        elif (comp_graph.getOperatorCompCostByDevice(u, random_device) == 0 or comp_graph.getOperatorCompCostByDevice(v,random_device) == 0):
            if comp_graph.getOperatorCompCostByDevice(v, random_device) == 0 and comp_graph.getOperatorCompCostByDevice(
                    u, random_device) > 0 and comp_graph.in_degree(v) > 1:
                continue
            if comp_graph.getOperatorCompCostByDevice(u, random_device) == 0 and comp_graph.getOperatorCompCostByDevice(
                    v, random_device) > 0 and comp_graph.out_degree(u) > 1:
                continue
            # Merge nodes u and v, by default merge v into u
            # This function only merge mergable edge
            data = comp_graph.merge_edge(u, v)
        #elif (min(comp_graph.getOperatorCompCostByDevice(pre, random_device) + comp_graph.getEdgeTensorSize(pre, v) *
                  #device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1]) for pre in comp_graph.predecessors(v)) >=
              #sum(comp_graph.getOperatorCompCostByDevice(pre, random_device) for pre in comp_graph.predecessors(v))):
            #data = comp_graph.merge_edge(u, v)
        elif (min(comp_graph.getOperatorCompCostByDevice(succ, random_device) + comp_graph.getEdgeTensorSize(u, succ) *
                  device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1]) for succ in comp_graph.successors(u)) >=
              sum(comp_graph.getOperatorCompCostByDevice(succ, random_device) for succ in comp_graph.successors(u))):
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
            '''
            max_suc_total_cost = max(
                global_rank[succ_node] +
                comp_graph.getEdgeTensorSize(current_node,succ_node)
                * device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1]) for succ_node in successors
            )
            # Store the best successor for path reconstruction using max()
            best_successor[current_node] = max(
                successors, key=lambda succ_node: global_rank[succ_node] +
                comp_graph.getEdgeTensorSize(current_node,succ_node)
                * device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1])
            )
            '''
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
            '''
            max_suc_total_cost = max(
                global_rank[succ_node] +
                comp_graph.getEdgeTensorSize(current_node,succ_node)
                * device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1]) for succ_node in successors
            )
            # Store the best successor for path reconstruction using max()
            best_successor[current_node] = max(
                successors, key=lambda succ_node: global_rank[succ_node] +
                comp_graph.getEdgeTensorSize(current_node,succ_node)
                * device_topo.calUnitCommCostInUS(fast_link[0], fast_link[1])
            )
            '''
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
