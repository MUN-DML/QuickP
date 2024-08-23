from gurobipy import *
from networkx import topological_sort

from optimizer.model.graph import find_non_connected_pairs, DeviceGraph, CompGraph
from optimizer.scheduling.scheduling import add_topo_order_constraints

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.optimization_problems.gurobi_util import gurobi_setup


def get_hetero_balanced_placement(comp_graph: CompGraph, deviceTopo: DeviceGraph) -> dict:

    def get_operator_device_mapping_through_x(x):
        mapping = {}
        for (operator_id, device_id), var in x.items():
            # Check if the variable has a value of 1 (operator is assigned to device)
            if var.X > 0.5:  # Since the variable is binary, this checks if it is assigned
                mapping[operator_id] = device_id
        return mapping

    # Init solver
    model = gurobi_setup("minimize_maxload")
    non_connected_pairs = find_non_connected_pairs(comp_graph)

    # Define variables
    x = model.addVars(comp_graph.getOperatorIDs(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device

    computing_cost = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="computing_cost")
    total_weight_device = model.addVars(deviceTopo.getDeviceIDs(), vtype=GRB.CONTINUOUS, name="total_weight_device")

    comm_cost = model.addVars(comp_graph.getEdgeIDs(), vtype=GRB.CONTINUOUS, lb=0.0, name="comm_cost")

    # Add constraints that schedule every node on exactly one machine
    for op in comp_graph.getOperatorIDs():
        model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1, name=f"one_device_{op}")


    # No connected pair cannot be allocated to the same device
    for device in deviceTopo.getDeviceIDs():
        for node_a, node_b in non_connected_pairs:
            model.addConstr(x[node_a, device] + x[node_b, device] <= 1)


    # Add constraints that each op's
    for node_id in comp_graph.getOperatorIDs():
        comp_cost = quicksum(x[node_id, device_id] * comp_graph.getOperatorCompCostByDevice(node_id, device_id)
                             for device_id in deviceTopo.getDeviceIDs())
        model.addConstr(computing_cost[node_id] == comp_cost, name=f"finish_start_{node_id}")


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
        model.addConstr(comm_cost[source_op_ID, dest_op_ID] == comm_cost_expr, f"comm_cost_{source_op_ID}_{dest_op_ID}")
    # Define a variable to represent the total sum of communication costs
    total_comm_cost = model.addVar(vtype=GRB.CONTINUOUS, name="total_comm_cost")
    # Set up the total communication cost as the sum of the communication costs between all pairs of operators
    model.addConstr(
        total_comm_cost == quicksum(comm_cost[source_op_ID, dest_op_ID] for source_op_ID, dest_op_ID in comm_cost),
        name="total_comm_cost_constraint"
    )

    # Constraints: Define total weight for each device based on assigned operators
    for device in deviceTopo.getDeviceIDs():
        model.addConstr(
            total_weight_device[device] == quicksum(computing_cost[op, device] * x[op, device] for op in comp_graph.getOperatorIDs()),
            name=f"weight_device_{device}")

    # Minimize the maximum weight assigned to any device
    max_weight = model.addVar(vtype=GRB.CONTINUOUS, name="max_weight")
    for device in deviceTopo.getDeviceIDs():
        model.addConstr(total_weight_device[device] <= max_weight, name=f"max_weight_constraint_{device}")

    # Set the target of solver
    model.setObjective(max_weight, GRB.MINIMIZE)

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
        print('Runtime = ', "%.2f" % model.Runtime, 's', sep='')

        operator_device_mapping = get_operator_device_mapping_through_x(x)
        del model
        disposeDefaultEnv()
        return operator_device_mapping
    else:
        print(f"Optimization ended with status {model.status}")