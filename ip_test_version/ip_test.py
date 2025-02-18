import argparse
import datetime
from collections import defaultdict

from gurobipy import *

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.main_simulator.gurobi_util import gurobi_setup, init_computing_and_device_graph, get_proper_M
from DNN_model_tf.tf_model_enum import TFModelEnum
from ICC2025.util_quickp import show_quick_p_result, get_proper_alpha, visualize_placement
from optimizer.co_location_and_merge.group_algorithm import traverse_merge_loop, group_longest_path, \
    fuse_weakly_connected_components, iteratively_expand_wcc
from optimizer.model.graph import CompGraph, find_non_connected_pairs

'''
This version is for model test. 
It is slower than the IP in ICC2025 file.
'''

def show_icc_logging(model, x, start, finish, comp_cost_map, model_type: TFModelEnum, comp_graph, deviceTopo, tensor_sizes, comm_cost, show_placement=True, show_communication=False):

    def get_operator_device_mapping_through_x(x):
        mapping = {}
        for (operator_id, device_id), var in x.items():
            # Check if the variable has a value of 1 (operator is assigned to device)
            if var.X > 0.5:  # Since the variable is binary, this checks if it is assigned
                mapping[operator_id] = device_id
        return mapping

    operator_device_placement = get_operator_device_mapping_through_x(x)
    assignment_dict = defaultdict(list)
    comm_cost_dict = defaultdict(float)
    device_total_cost_map = defaultdict(int)
    device_util_rate_map = defaultdict(int)
    # populate assignment_dict
    for op, device in operator_device_placement.items():
        # Assignment: {device: [(op1, start[op1], finish[op1]), (...)]}
        assignment_dict[device].append((op, start[op].X, finish[op].X))
    # Sort operators by their start times for each device
    for device, ops in assignment_dict.items():
        assignment_dict[device] = sorted(ops, key=lambda x: x[1])

    for i, j in comp_graph.getEdgeIDs():
            comm_cost_dict[i,j] = comm_cost[i,j].X

    # Print operator placement
    for device, op_info_tuples in assignment_dict.items():
        sum_comp = 0
        print(f"Device: {device}")
        for op_tuple in op_info_tuples:
            op = op_tuple[0]
            comp_cost = comp_cost_map[op]
            sum_comp += comp_cost
            if comp_cost == 0:
                continue
            if show_placement:
                print(f"  Operator: {op_tuple[0]}, Start: {op_tuple[1]}, Finish: {op_tuple[2]}, Comp Cost: {comp_cost}")
        device_utility_rate = sum_comp / model.ObjVal
        device_total_cost_map[device] = sum_comp
        device_util_rate_map[device] = device_utility_rate

    # print comm cost
    if show_communication:
        for i, j in comp_graph.getEdgeIDs():
            print(
                f"  : Edge {i, j}, source_placement: {operator_device_placement[i]}, end_placement: {operator_device_placement[j]}, "
                f"bandwidth: {0 if operator_device_placement[i] == operator_device_placement[j] else deviceTopo.get_link_bandwidth(operator_device_placement[i], operator_device_placement[j])} "
                f" Tensor Size: {tensor_sizes[i, j]}, Comm Cost: {comm_cost_dict[i, j]}")

    print('Expected Training time = ', model.ObjVal, 'us', sep=' ')
    print("Device Utility Rate:", device_util_rate_map)
    print("total_computing_time_per_device:", device_total_cost_map)
    print('The Placement Searching Runtime = ', "%.2f" % model.Runtime, 's', sep=' ')
    print('ALL Cross Device Communication Cost Sum =', sum(comm_cost_dict.values()))
    print(f"This is the near-optimal solution of such configuration: \n"
          f"model type: {model_type} \n"
          f"number of operators: {comp_graph.number_of_nodes()} \n"
          f"number of devices: {deviceTopo.number_of_nodes()} \n"
          f"The environment is homogenous")
    return operator_device_placement


def QuickP(comp_graph: CompGraph, deviceTopo, M, model_type) -> dict:

    # Init solver
    model = gurobi_setup("minimize_maxload")

    # get co-location info
    group_ops_mapping = comp_graph.create_colocation_group_to_ops_map()
    op_group_map = comp_graph.create_op_group_id_mapping()

    # Define variables
    x = model.addVars(comp_graph.getOperatorIDs(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device
    group_device_mapping = model.addVars(group_ops_mapping.keys(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                                         name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else "y_group")
    start = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else "start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else "finish")  # finish[node_id] represent the finish time of this node
    comm_cost = model.addVars(comp_graph.getEdgeIDs(), vtype=GRB.CONTINUOUS, lb=0.0)

    # Co-location constraint
    # Ensure each group is assigned to exactly one device
    for group in group_ops_mapping.keys():
        model.addConstr(quicksum(group_device_mapping[group, device] for device in deviceTopo.getDeviceIDs()) == 1)

    for group_id, group in group_ops_mapping.items():
        for device in deviceTopo.getDeviceIDs():
            for node in group:
                model.addConstr(x[node, device] == group_device_mapping[group_id, device])

    # Add constraints that schedule every node on exactly one machine
    for op in comp_graph.getOperatorIDs():
        if op not in op_group_map:
            model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1)

    # Add constraints that each op's ending time = starting time + its computing time. Homogeneous device
    any_d = deviceTopo.getDeviceIDs()[0]
    homo_op_cost_dict = comp_graph.getOpCompCostMapByDevice(any_d)
    for node_id in comp_graph.getOperatorIDs():
        model.addConstr(finish[node_id] == start[node_id] + homo_op_cost_dict[node_id])

    # Add constraint that if op2 depends on op1, the starting time of op2 will be the ending time of op1 + communication delay if these two ops are not placed on the same device
    device_pairs = {(src, dest) for src in deviceTopo.getDeviceIDs() for dest in deviceTopo.getDeviceIDs() if
                    src != dest}
    # unit_comm_costs[device_id_src, device_id_dest] means the com cost per bit from device with source device to dest device
    unit_comm_costs = {
        (src_device, dest_device): deviceTopo.calUnitCommCostInUS(src_device, dest_device)
        for src_device, dest_device in device_pairs
    }
    tensor_sizes = {
        (source_op_ID, dest_op_ID): comp_graph.getEdgeTensorSize(source_op_ID, dest_op_ID)
        for source_op_ID, dest_op_ID in comp_graph.getEdgeIDs()
    }
    for edge_id_tuple in list(comp_graph.getEdgeIDs()):

        # no communication cost for ops on the same device
        source_op_ID, dest_op_ID = edge_id_tuple

        # if tensor size is 0, even if two ops are on different device, no communication cost will exist
        if tensor_sizes[source_op_ID, dest_op_ID] == 0:
            model.addConstr(finish[source_op_ID] <= start[dest_op_ID])
            continue

        # no comm cost if on the same co-lo group
        if source_op_ID in op_group_map and dest_op_ID in op_group_map and op_group_map[source_op_ID] == op_group_map[dest_op_ID]:
            model.addConstr(finish[source_op_ID] <= start[dest_op_ID])
            continue

        # Since it is a Binary, its range will be either 0 or 1
        place_indicator = model.addVars(device_pairs, vtype=GRB.BINARY)
        model.addConstrs(
            (place_indicator[device_id_src, device_id_dest] >= x[source_op_ID, device_id_src] + x[
                dest_op_ID, device_id_dest] - 1
             for device_id_src, device_id_dest in device_pairs)
        )

        model.addConstrs(
            (place_indicator[device_id_src, device_id_dest] <= x[source_op_ID, device_id_src]
             for device_id_src, device_id_dest in device_pairs)
        )

        model.addConstrs(
            (place_indicator[device_id_src, device_id_dest] <= x[dest_op_ID, device_id_dest]
             for device_id_src, device_id_dest in device_pairs)
        )

        # Calculate the comm cost of on different devices
        comm_cost_expr = quicksum(
            tensor_sizes[source_op_ID, dest_op_ID] *
            unit_comm_costs[device_id_src, device_id_dest] *
            place_indicator[device_id_src, device_id_dest]
            for device_id_src, device_id_dest in device_pairs
        )

        model.addConstr(comm_cost[source_op_ID, dest_op_ID] == comm_cost_expr)

        # Ensures the communication duration covers the communication cost.
        model.addConstr(finish[source_op_ID] + comm_cost[source_op_ID, dest_op_ID] <= start[dest_op_ID])


    '''
    Scheduling Part
    '''
    op_group_mapping = comp_graph.create_op_group_id_mapping()
    non_reachable_pairs, topological_order_mapping = find_non_connected_pairs(comp_graph)
    ungrouped_non_reachable_pairs = []

    for i, j in non_reachable_pairs:
        if i in op_group_mapping and j in op_group_mapping and op_group_mapping[i] == op_group_mapping[j]:
            continue
        ungrouped_non_reachable_pairs.append((i, j))

    #  scheduling inside each group follows topo sort since each node pair in non_reachable_pairs is calculated by this sort algorithm
    for ops in group_ops_mapping.values():
        ordered_list = sorted(ops, key=lambda node: topological_order_mapping[node])
        for op_a, op_b in zip(ordered_list, ordered_list[1:]):
            model.addConstr(finish[op_a] <= start[op_b])

    # Iterate over topologically sorted nodes
    for a, b in ungrouped_non_reachable_pairs:
        if homo_op_cost_dict[a] == 0 or homo_op_cost_dict[b] == 0:
            continue
        # For each consecutive pair of operators, add a constraint for each device
        for device_id in deviceTopo.getDeviceIDs():
            # Ensure the correct order for each potential device assignment
            # This constraint will only apply if both a and b are assigned to the same device
            model.addConstr(finish[a] <= start[b] + M * (2 - x[a, device_id] - x[b, device_id]))

    # TotalLatency that we are minimizing
    TotalLatency = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    for op_end in finish.values():
        model.addConstr(TotalLatency >= op_end, "satisfy each deice's latency")

    # Set the target of solver
    model.setObjective(TotalLatency, GRB.MINIMIZE)

    # Run the solver
    sys.stdout.flush()
    model.optimize()

    # Check optimization status
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        model.computeIIS()
        model.write("model.ilp")
        print("IIS written to model.ilp")

        # Print the constraints that are in the IIS
        print("\nThe following constraints are in the IIS:")
        for constr in model.getConstrs():
            if constr.IISConstr:
                print(f"{constr.ConstrName}")
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif model.status == GRB.OPTIMAL:
        '''
        show_icc_logging will show a lot of logging. Normally, the communication part logging is hidden
        '''
        place = show_icc_logging(model, x, start, finish, homo_op_cost_dict, model_type, comp_graph, deviceTopo,
                                tensor_sizes, comm_cost, show_placement=True, show_communication=True)
        del model
        disposeDefaultEnv()
        return place
    else:
        print(f"Optimization ended with status {model.status}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for optimization problem after graph partitioning')
    parser.add_argument('--number_of_device', type=int, default=6,
                        help="Number of devices (must be >= 2 and divisible by 2)")
    parser.add_argument('--model', type=str, default='ALEXNET', choices=['ALEXNET', 'VGG', 'FNET', 'BERT'],
                        help="Model name")

    args = parser.parse_args()

    # Dynamically access attributes using getattr
    model_type = getattr(TFModelEnum, args.model, None)

    # init deviceTopo and comp_graph
    deviceTopo, comp_graph = init_computing_and_device_graph(args.number_of_device, None, model_type=model_type)
    alpha = get_proper_alpha(comp_graph, deviceTopo, if_visualize=False)
    print("alpha", alpha)
    # op-fusion
    beginning_time = datetime.datetime.now()
    traverse_merge_loop(comp_graph, deviceTopo, alpha)
    # apply co-location grouper
    wcc_node_set = group_longest_path(comp_graph, deviceTopo, args.number_of_device)
    iteratively_expand_wcc(comp_graph, deviceTopo)
    '''
    # Uncomment the following section to further reduce the solver's search latency.
    # Note: This optimization may result in a minor additional performance trade-off and increase the fusion runtime.

    # - Fuse weakly connected components in the computation graph.
    # - Further reduce the graph size.
    fuse_weakly_connected_components(comp_graph, wcc_node_set)
    traverse_merge_loop(comp_graph, deviceTopo, alpha)
    '''
    ending_time = datetime.datetime.now()
    print("op fusion run time", datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))
    placement = QuickP(comp_graph, deviceTopo, M=get_proper_M(model_type), model_type=model_type)
    # visualize_placement(comp_graph, placement) # uncomment to show a rough placement visualization
