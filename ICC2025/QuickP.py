import argparse
import datetime

from gurobipy import *

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.main_simulator.gurobi_util import gurobi_setup, init_computing_and_device_graph, get_proper_M
from DNN_model_tf.tf_model_enum import TFModelEnum
from ICC2025.util_quickp import show_quick_p_result
from optimizer.co_location_and_merge.group_algorithm import traverse_merge_loop, apply_all_co_location_constraint, \
    fuse_weakly_connected_components
from optimizer.model.graph import CompGraph, find_non_connected_pairs


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

    # Co-location constraint
    # Ensure each group is assigned to exactly one device
    for group in group_ops_mapping.keys():
        model.addConstr(quicksum(group_device_mapping[group, device] for device in deviceTopo.getDeviceIDs()) == 1,
                        name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"assign_group_{group}_to_one_device")

    for group_id, group in group_ops_mapping.items():
        for device in deviceTopo.getDeviceIDs():
            for node in group:
                model.addConstr(x[node, device] == group_device_mapping[group_id, device])

    # Add constraints that schedule every node on exactly one machine
    for op in comp_graph.getOperatorIDs():
        if op not in op_group_map:
            model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1, name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"one_device_{op}")

    # Add constraints that each op's ending time = starting time + its computing time. Homogeneous device
    any_d = deviceTopo.getDeviceIDs()[0]
    homo_op_cost_dict = comp_graph.getOpCompCostMapByDevice(any_d)
    for node_id in comp_graph.getOperatorIDs():
        model.addConstr(finish[node_id] == start[node_id] + homo_op_cost_dict[node_id],
                            name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"finish_start_{node_id}")

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
        if source_op_ID in op_group_map and dest_op_ID in op_group_map and op_group_map[source_op_ID] == op_group_map[dest_op_ID]:
            model.addConstr(finish[source_op_ID] <= start[dest_op_ID], "" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"data_dependency_{source_op_ID}_{dest_op_ID}")
            continue

        # Aggregate communication cost
        comm_cost_expr = quicksum(
            unit_comm_costs[device_id_src, device_id_dest] * tensor_sizes[source_op_ID, dest_op_ID] *
            x[source_op_ID, device_id_src] * x[dest_op_ID, device_id_dest]
            for device_id_src, device_id_dest in device_pairs
        )

        # Ensures the communication duration covers the communication cost.
        model.addConstr(finish[source_op_ID] + comm_cost_expr <= start[dest_op_ID],
                        "" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"data_dependency_{source_op_ID}_{dest_op_ID}")


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
        # For each consecutive pair of operators, add a constraint for each device
        for device_id in deviceTopo.getDeviceIDs():
            # Ensure the correct order for each potential device assignment
            # This constraint will only apply if both a and b are assigned to the same device
            model.addConstr(finish[a] <= start[b] + M * (2 - x[a, device_id] - x[b, device_id]),
                            name="" if model_type in [TFModelEnum.BERT,
                                                      TFModelEnum.FNET] else f"bigM_topo_order_{a}_{b}_on_device_{device_id}")

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
        show_quick_p_result(model, x, start, finish, homo_op_cost_dict, model_type, comp_graph, deviceTopo)
        del model
        disposeDefaultEnv()
    else:
        print(f"Optimization ended with status {model.status}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for optimization problem after graph partitioning')
    parser.add_argument('--number_of_device', type=int, default=2)
    # TEST SMALL
    parser.add_argument('--model', type=str, default='FNET')
    parser.add_argument('--alpha', type=int, default=200)

    args = parser.parse_args()

    # Dynamically access attributes using getattr
    model_type = getattr(TFModelEnum, args.model, None)

    # init deviceTopo and comp_graph
    deviceTopo, comp_graph = init_computing_and_device_graph(args.number_of_device, None, model_type=model_type)

    # op-fusion
    beginning_time = datetime.datetime.now()
    traverse_merge_loop(comp_graph, deviceTopo, args.alpha)
    ending_time = datetime.datetime.now()
    print("op fusion run time", datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp()))
    # apply co-location grouper
    wcc_node_set = apply_all_co_location_constraint(comp_graph, deviceTopo, args.number_of_device)
    # fuse weakly connected component
    fuse_weakly_connected_components(comp_graph, wcc_node_set)
    # comp_graph.visualize_graphviz()

    QuickP(comp_graph, deviceTopo, M=get_proper_M(model_type), model_type=model_type)