from collections import deque

import networkx as nx

from optimizer.co_location_and_merge.grouper_util import merge_group, label_all_node_with_group, \
    create_eligible_edge_subgraph, label_group, analyze_group
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
                if len(paths) > 0:
                    raise ValueError(f"{op1} and {op2} has more than one disjoint path")

        # create attributes for the new node
        random_node_cost_dict = computing_graph.getCompCostMapByOp(list(ops_to_be_merged)[0])
        new_computing_cost = sum(computing_cost_dict[op] for op in ops_to_be_merged)
        new_comp_cost_dict = {op: new_computing_cost for op in random_node_cost_dict.keys()}
        new_memory = sum(computing_graph.getMemorySize(op) for op in ops_to_be_merged)

        # add the new node
        new_id = "&".join(ops_to_be_merged)
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
