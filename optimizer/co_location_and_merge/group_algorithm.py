from collections import deque

import networkx as nx

from optimizer.co_location_and_merge.grouper_util import merge_group, label_all_node_with_group, \
    create_eligible_edge_subgraph, label_group
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
        # merge ops based on the merged groups
        # graph_coarsen(comp_graph, comp_cost)
        break


# _generate_fused_op_graph
def graph_coarsen(computing_graph: CompGraph, computing_cost_dict):

    def merge_operators(ops_to_be_merged):

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

        # create attributes for the new node
        random_node_cost_dict = computing_graph.getCompCostMapByOp(ops_to_be_merged[0])
        new_computing_cost = sum(computing_cost_dict[op] for op in ops_to_be_merged)
        new_comp_cost_dict = {op: new_computing_cost for op in random_node_cost_dict.keys()}
        new_memory = sum(computing_graph.getMemorySize(op) for op in ops_to_be_merged)

        # add the new node
        new_id = "&".join(ops_to_be_merged)
        computing_graph.add_new_node(new_id, "merged",
                                     memory=new_memory, comp_cost_map=new_comp_cost_dict)

        # restore the dependency relationship
        for incoming_node in component_incoming_nodes:
            computing_graph.add_new_edge(incoming_node, new_id)

        for outgoing_node in component_outgoing_nodes:
            computing_graph.add_new_edge(new_id, outgoing_node)

        # Remove the original nodes
        computing_graph.remove_nodes_from(ops_to_be_merged)

        # Double check if the graph after merge is still DAG
        assert nx.is_directed_acyclic_graph(computing_graph)

    group_ops_map = computing_graph.create_colocation_group_to_ops_map()
    for group in group_ops_map.values():
        merge_operators(group)
