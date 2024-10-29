from gurobipy import *
from networkx import topological_sort

from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.model.graph import CompGraph
from optimizer.scheduling.scheduling import add_topo_order_constraints_with_grouper

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.main_simulator.gurobi_util import gurobi_setup, show_optimization_solution


def get_optimize_placement_with_grouper_homo(comp_graph: CompGraph, deviceTopo, M, model_type) -> dict:

    def get_operator_device_mapping_through_x(x):
        mapping = {}
        for (operator_id, device_id), var in x.items():
            # Check if the variable has a value of 1 (operator is assigned to device)
            if var.X > 0.5:  # Since the variable is binary, this checks if it is assigned
                mapping[operator_id] = device_id
        return mapping

    # Init solver
    model = gurobi_setup("minimize_maxload")

    # get co-location info
    group_ops_mapping = comp_graph.create_colocation_group_to_ops_map()
    op_group_map = comp_graph.create_op_group_id_mapping()

    any_d = deviceTopo.getDeviceIDs()[0]
    any_d_2 = deviceTopo.getDeviceIDs()[1]
    homo_op_cost_dict = comp_graph.getOpCompCostMapByDevice(any_d)
    # Get homo comm sot
    comm_cost_dict = {}
    for edge_id_tuple in comp_graph.getEdgeIDs():
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Aggregate communication cost
        comm_cost_dict[edge_id_tuple] = comp_graph.getEdgeTensorSize(source_op_ID,dest_op_ID) * deviceTopo.calUnitCommCostInUS(any_d, any_d_2)

    # Define variables
    x = model.addVars(comp_graph.getOperatorIDs(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device
    group_device_mapping = model.addVars(group_ops_mapping.keys(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                                         name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else "y_group")
    start = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else "start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else "finish")  # finish[node_id] represent the finish time of this node
    cross_device_indicator = model.addVars(comp_graph.getEdgeIDs(), vtype=GRB.BINARY, name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else "cross_device_indicator")

    # Co-location constraint
    # Ensure each group is assigned to exactly one device
    for group in group_ops_mapping.keys():
        model.addConstr(quicksum(group_device_mapping[group, device] for device in deviceTopo.getDeviceIDs()) == 1,
                        name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"assign_group_{group}_to_one_device")

    for group_id, group in group_ops_mapping.items():
        for device in deviceTopo.getDeviceIDs():
            for node in group:
                model.addConstr(x[node, device] == group_device_mapping[group_id, device])

    # Add constraints that schedule every ungrouped node on exactly one machine
    for op in comp_graph.getOperatorIDs():
        if op not in op_group_map:
            model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1, name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"one_device_{op}")

    # Add constraints that each op's ending time = starting time + its computing time. Homogeneous device
    for node_id in comp_graph.getOperatorIDs():
        model.addConstr(finish[node_id] == start[node_id] + homo_op_cost_dict[node_id],
                            name="" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"finish_start_{node_id}")

    for a, b in comp_graph.getEdgeIDs():
        # Use one quicksum to check if a and b are placed on different devices
        # 1 means different devices
        # ops in the same group have no comm cost
        if a in op_group_map and b in op_group_map and op_group_map[a] == op_group_map[b]:
            model.addConstr(cross_device_indicator[a, b] == 0)
        else:
            model.addConstr(
                cross_device_indicator[a, b] == 1 - quicksum(x[a, device] * x[b, device] for device in deviceTopo.getDeviceIDs()),
                name=f"split_indicator_{a}_{b}"
            )

    # unit_comm_costs[device_id_src, device_id_dest] means the com cost per bit from device with source device to dest device
    for edge_id_tuple in list(comp_graph.getEdgeIDs()):

        # no communication cost for ops on the same device
        source_op_ID, dest_op_ID = edge_id_tuple
        if source_op_ID in op_group_map and dest_op_ID in op_group_map and op_group_map[source_op_ID] == op_group_map[dest_op_ID]:
            model.addConstr(finish[source_op_ID] <= start[dest_op_ID], "" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"data_dependency_{source_op_ID}_{dest_op_ID}")
            continue

        # Aggregate communication cost
        comm_cost_expr = comm_cost_dict[edge_id_tuple] * cross_device_indicator[edge_id_tuple]

        # Ensures the communication duration covers the communication cost.
        model.addConstr(finish[source_op_ID] + comm_cost_expr <= start[dest_op_ID], "" if model_type in [TFModelEnum.BERT, TFModelEnum.FNET] else f"data_dependency_{source_op_ID}_{dest_op_ID}")

    add_topo_order_constraints_with_grouper(model, comp_graph, x, deviceTopo.getDeviceIDs(), finish, start, group_ops_mapping, M, model_type)

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
        # show_optimization_solution(model, x, comp_graph, deviceTopo, start, finish, comm_cost)
        print('Runtime = ', "%.2f" % model.Runtime, 's', sep='')

        operator_device_mapping = get_operator_device_mapping_through_x(x)
        del model
        disposeDefaultEnv()
        return operator_device_mapping
    else:
        print(f"Optimization ended with status {model.status}")