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

    def generate_new_operator(ops_to_be_merged):
        pass

