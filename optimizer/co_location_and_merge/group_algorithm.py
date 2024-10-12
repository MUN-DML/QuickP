from collections import deque

from networkx.classes import DiGraph

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
        new_computing_cost = sum(computing_cost_dict[op] for op in ops_to_be_merged)

        internal_edges = deque(
            [(u, v) for u, v in computing_graph.edges()
             if u in ops_to_be_merged and v in ops_to_be_merged])
        while len(internal_edges) > 0:
            op1, op2 = internal_edges.popleft()


        # Add the new node to the graph
        computing_graph.add_node(new_node)

        # Redirect all incoming edges to the new node, avoiding duplicates
        predecessors = set()
        for node in ops_to_be_merged:
            predecessors.update(computing_graph.predecessors(node))  # Collect unique predecessors

        for pred in predecessors:
            if not computing_graph.has_edge(pred, new_node):  # Avoid adding duplicate edges
                computing_graph.add_edge(pred, new_node)

        # Redirect all outgoing edges from the merged nodes to the new node, avoiding duplicates
        successors = set()
        for node in ops_to_be_merged:
            successors.update(computing_graph.successors(node))  # Collect unique successors

        for succ in successors:
            if not computing_graph.has_edge(new_node, succ):  # Avoid adding duplicate edges
                computing_graph.add_edge(new_node, succ)

        # Remove the original nodes
        computing_graph.remove_nodes_from(ops_to_be_merged)



