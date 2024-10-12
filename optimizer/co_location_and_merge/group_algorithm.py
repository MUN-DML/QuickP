from collections import deque

import networkx as nx
from networkx.classes import DiGraph, subgraph

from optimizer.co_location_and_merge.grouper_util import merge_group, label_all_node_with_group, edge_based_label
from optimizer.model.graph import CompGraph, DeviceGraph


# group with nodes with small computing cost but a large communication cost if on different devices
def quickcut_group(computing_graph: CompGraph, device_topo: DeviceGraph):
    computing_cost_dict = computing_graph.getOpCompCostMapByDevice(device_topo.getDeviceIDs()[0])
    label_all_node_with_group(computing_graph, device_topo, computing_cost_dict)
    # After all node get labelled, merge groups
    merge_group(computing_graph)


def group_and_merge_group(computing_graph: CompGraph, device_topo: DeviceGraph):
    computing_cost_dict = computing_graph.getOpCompCostMapByDevice(device_topo.getDeviceIDs()[0])
    edge_based_label(computing_graph, device_topo, computing_cost_dict)
    # After all node get labelled, merge groups
    merge_group(computing_graph)


def merge_operators(computing_graph: CompGraph, operator_2d_list, computing_cost_dict):

    # _generate_fused_op_graph
    def generate_new_operator(ops_to_be_merged):

        # double check if those nodes are connected, forming one weakly connected component
        sub_graph = computing_graph.subgraph(ops_to_be_merged)
        if not nx.is_weakly_connected(sub_graph):
            raise ValueError(f"{ops_to_be_merged} are not connected")

        internal_edges = deque(sub_graph.edges)

        component_incoming_nodes= [u for (u, v) in computing_graph.edges if u not in sub_graph.nodes and v in sub_graph.nodes]
        component_outgoing_nodes = [v for (u, v) in computing_graph.edges if u in sub_graph.nodes and v not in sub_graph.nodes]

        while len(internal_edges) > 0:
            op1, op2 = internal_edges.popleft()
            if computing_graph.out_degree(op1) > 1 and computing_graph.in_degree(op2) > 1:
                pass

        # create a new node
        new_computing_cost = sum(computing_cost_dict[op] for op in ops_to_be_merged)
        new_memory = sum(computing_graph.nodes[op] for op in ops_to_be_merged)

        # Remove the original nodes
        computing_graph.remove_nodes_from(ops_to_be_merged)

        # Double check if the graph after merge is still DAG
        assert nx.is_directed_acyclic_graph(computing_graph)



