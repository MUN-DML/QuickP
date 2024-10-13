from collections import deque

import networkx as nx
from networkx.classes import DiGraph, subgraph

from optimizer.co_location_and_merge.grouper_util import merge_group, label_all_node_with_group, edge_based_label
from optimizer.model.graph import CompGraph, DeviceGraph


def group_and_fuse_op_incrementally(comp_graph, deviceTopo):
    # there should be many iterations
    while True:
        # After each merge, the comp_cost map will change
        comp_cost = comp_graph.getOpCompCostMapByDevice(deviceTopo.getDeviceIDs()[0])
        edge_based_label(comp_graph, deviceTopo, comp_cost)
        # After all node get labelled, merge groups
        merge_group(comp_graph)
        # if no node is labeled with the co-location attribute, there is no need to merge operators any more
        # for node in comp_graph.nodes():
        #     if
        merge_operators(comp_graph, comp_cost)
        break


# _generate_fused_op_graph
def merge_operators(computing_graph: CompGraph, computing_cost_dict):

    def generate_new_operator(ops_to_be_merged):

        # double check if those nodes are connected, forming one weakly connected component
        sub_graph = computing_graph.subgraph(ops_to_be_merged)
        if not nx.is_weakly_connected(sub_graph):
            raise ValueError(f"{ops_to_be_merged} are not connected")

        internal_edges = deque(sub_graph.edges)

        # get predecessors and successors of this component
        component_incoming_nodes = set()
        component_outgoing_nodes = set()
        # Loop through each node in the subgraph
        for node in ops_to_be_merged:
            # Find all predecessors of the node (incoming nodes)
            component_incoming_nodes.update(
                pred for pred in computing_graph.predecessors(node) if pred not in ops_to_be_merged)
            # Find all successors of the node (outgoing nodes)
            component_outgoing_nodes.update(
                succ for succ in computing_graph.successors(node) if succ not in ops_to_be_merged)

        while len(internal_edges) > 0:
            op1, op2 = internal_edges.popleft()
            if computing_graph.out_degree(op1) > 1 and computing_graph.in_degree(op2) > 1:
                # CAVEATS: finding disjoint paths may take long time
                paths = list(nx.node_disjoint_paths(computing_graph, op1, op2))
                if len(paths) > 0:
                    raise ValueError(f"{op1} and {op2} has more than one disjoint path")

        # create a new node
        new_computing_cost = sum(computing_cost_dict[op] for op in ops_to_be_merged)
        new_memory = sum(computing_graph.getMemorySize(op) for op in ops_to_be_merged)

        # Remove the original nodes
        computing_graph.remove_nodes_from(ops_to_be_merged)

        # Double check if the graph after merge is still DAG
        assert nx.is_directed_acyclic_graph(computing_graph)
