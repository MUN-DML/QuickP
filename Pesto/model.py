from gurobipy import *

from optimizer.model.graph import CompGraph, find_non_connected_pairs
from optimizer.scheduling.scheduling import add_topo_order_constraint

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.main_simulator.gurobi_util import gurobi_setup, show_optimization_solution


def get_optimize_placement(comp_graph: CompGraph, deviceTopo, M):

    # Init solver
    model = gurobi_setup("minimize_maxload")

    # Define variables
    x = model.addVars(comp_graph.getOperatorIDs(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device
    start = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # finish[node_id] represent the finish time of this node

    # Add constraints that schedule every node on exactly one machine
    for op in comp_graph.getOperatorIDs():
        model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1, name=f"one_device_{op}")

    # Add constraints that each op's ending time = starting time + its computing time. Homogeneous device
    any_d = deviceTopo.getDeviceIDs()[0]
    homo_op_cost_dict = comp_graph.getOpCompCostMapByDevice(any_d)
    for node_id in comp_graph.getOperatorIDs():
        model.addConstr(finish[node_id] == start[node_id] + homo_op_cost_dict[node_id],
                            name=f"finish_start_{node_id}")

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
        source_op_ID, dest_op_ID = edge_id_tuple
        # Aggregate communication cost
        comm_cost_expr = quicksum(
            unit_comm_costs[device_id_src, device_id_dest] * tensor_sizes[source_op_ID, dest_op_ID] *
            x[source_op_ID, device_id_src] * x[dest_op_ID, device_id_dest]
            for device_id_src, device_id_dest in device_pairs
        )

        # Ensures the communication duration covers the communication cost.
        model.addConstr(finish[source_op_ID] + comm_cost_expr <= start[dest_op_ID],
                        f"data_dependency_{source_op_ID}_{dest_op_ID}")

    # Iterate over topologically sorted nodes
    for a, b in find_non_connected_pairs(comp_graph):
        # For each consecutive pair of operators, add a constraint for each device
        for device_id in deviceTopo.getDeviceIDs():
            # Ensure the correct order for each potential device assignment
            # This constraint will only apply if both a and b are assigned to the same device
            model.addConstr(finish[a] <= start[b] + M * (2 - x[a, device_id] - x[b, device_id]),
                            name=f"bigM_topo_order_{a}_{b}_on_device_{device_id}")

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
        del model
        disposeDefaultEnv()
    else:
        print(f"Optimization ended with status {model.status}")