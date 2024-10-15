import hashlib
from collections import deque

import networkx as nx

from optimizer.co_location_and_merge.grouper_util import create_eligible_edge_subgraph, label_group, analyze_group
from optimizer.model.graph import CompGraph, DeviceGraph


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

def coarsen_weakly_connected_component(wcc_set: set, computation_graph: CompGraph, computing_cost_dict):
    sub_graph = computation_graph.subgraph(list(wcc_set))
    if not nx.is_weakly_connected(sub_graph):
        raise ValueError(f"{sub_graph.nodes} are not connected")

    internal_edges = deque(sub_graph.edges)
    while len(internal_edges) > 0:
        source, target = internal_edges.popleft()
        # if
        #     merge_node_pair(source, target, computation_graph, computing_cost_dict)



def merge_node_pair(u ,v, computation_graph, computing_cost_dict):
    if not is_edge_mergable(u, v, computation_graph):
        return
    # create attributes for the new node
    random_node_cost_dict = computation_graph.getCompCostMapByOp(u)
    new_computing_cost = sum(computing_cost_dict[op] for op in [u,v])
    new_comp_cost_dict = {op: new_computing_cost for op in random_node_cost_dict.keys()}
    new_memory = sum(computation_graph.getMemorySize(op) for op in [u,v])
    # new tensorsize should also be created

    # add the new node
    new_id = hashlib.md5("&".join([u,v]).encode()).hexdigest()
    computation_graph.add_new_node(new_id, "merged",
                                     memory=new_memory, comp_cost_map=new_comp_cost_dict)

    # Redirect in-edges (predecessors of the nodes to merge)
    for node in [u ,v]:
        for pred in computation_graph.predecessors(node):
            if pred not in [u,v]:  # Avoid self-loops
                computation_graph.add_edge(pred, new_id, **computation_graph.get_edge_data(pred, node))

    # Redirect out-edges (successors of the nodes to merge)
    for node in [u ,v]:
        for succ in computation_graph.successors(node):
            if succ not in [u,v]:  # Avoid self-loops
                computation_graph.add_edge(new_id, succ, **computation_graph.get_edge_data(node, succ))

    # Remove the original nodes
    computation_graph.remove_nodes_from([u, v])

    # Double check if the graph after merge is still DAG
    assert nx.is_directed_acyclic_graph(computation_graph)


def is_edge_mergable(source, target, computation_graph: CompGraph):
    if not computation_graph.has_edge(source, target):
        raise ValueError(f"Edge {source}, {target} does not exist")
    # merging 𝑢, 𝑣 is acyclic if and only if (𝑢, 𝑣) is the only path from 𝑢 to 𝑣 on G.
    # If u's out degree or v's in degree is 1, there can only exist one path
    if computation_graph.out_degree(source) == 1 or computation_graph.in_degree(target) == 1:
        return True
    #  the size of a minimum cut set is equal to the maximum number of disjoint paths that can be found between any pair of vertices.
    # paths = list(nx.node_disjoint_paths(computation_graph, source, target))
    min_cut_size = len(nx.minimum_edge_cut(computation_graph, source, target))
    return min_cut_size < 2


def is_worth_merging(source, target, computation_graph: CompGraph, device_topo, fast_link, computing_cost_dict):
    destination_computing_cost = computing_cost_dict[target]
    communication_cost = computation_graph.getEdgeTensorSize(source, target) * device_topo.calUnitCommCostInUS(
        fast_link[0], fast_link[1])
    if communication_cost >= destination_computing_cost or computing_cost_dict[source] == 0:
        return True
    else:
        return False
