from optimizer.model.graph import CompGraph, DeviceGraph


def get_comp_cost_dict(computation_graph, operator_device_mapping):
    comp_cost_dict = {}
    for node_id in computation_graph.getOperatorIDs():
        # Add constraints that each op's ending time = starting time + its computing time
        assigned_device = operator_device_mapping[node_id]
        comp_cost_dict[node_id] = computation_graph.getOperatorCompCostByDevice(node_id, assigned_device)
    return comp_cost_dict


def get_comm_cost_dict(computation_graph: CompGraph, device_topo: DeviceGraph, edge_cut_list, operator_device_mapping):
    comm_cost_dict = {}
    for edge_id_tuple in edge_cut_list:
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Aggregate communication cost
        comm_cost_dict[edge_id_tuple] = computation_graph.getEdgeTensorSize(source_op_ID, dest_op_ID) * device_topo.calUnitCommCostInUS(
            operator_device_mapping[source_op_ID], operator_device_mapping[dest_op_ID])
    return comm_cost_dict
