from gurobipy import *
from networkx import topological_sort

from optimizer.model.graph import find_non_connected_pairs, DeviceGraph, CompGraph
from optimizer.operator_device_placement.metis.metis_partition import metis_partition
from optimizer.operator_device_placement.metis.subgraph_util import construct_sub_graph
from optimizer.scheduling.scheduling import add_topo_order_constraints

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.optimization_problems.gurobi_util import gurobi_setup


def get_near_optimal_placement(comp_graph: CompGraph, deviceTopo: DeviceGraph, num_sub_graph_per_device=2) -> dict:
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

    # get metis partition
    total_number_of_sub_graph = num_sub_graph_per_device * len(deviceTopo.getDeviceIDs())
    partition_dict, cut_edge_list, _ = metis_partition(comp_graph, total_number_of_sub_graph)
    subgraph_dict = construct_sub_graph(comp_graph, partition_dict) # subgraph id - subgraph mapping

    # Get homo computing cost
    any_d = deviceTopo.getDeviceIDs()[0]
    homo_op_cost_dict = comp_graph.getOpCompCostMapByDevice(any_d)

    # Define variables
    # [operator_id, device_id] == 1 means this operator is assigned to this device
    x = model.addVars(comp_graph.getOperatorIDs(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")
    # subgraph device mapping
    y = model.addVars(subgraph_dict.keys(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY, name="y")
    split_indicator = model.addVars(non_connected_pairs, vtype=GRB.BINARY, name="split_indicator")

    # device-subgraph one-to-one mapping
    # Step 1: Ensure each subgraph is assigned to exactly one device
    # Constraint: Each subgraph is assigned to exactly one device
    model.addConstrs((quicksum(y[subgraph_id, device] for device in deviceTopo.getDeviceIDs()) == 1 for subgraph_id in
                      subgraph_dict.keys()), "SubgraphAssignment")
    # Constraint: Each device is assigned to exactly one subgraph
    model.addConstrs((quicksum(y[subgraph_id, device] for subgraph_id in subgraph_dict.keys()) == num_sub_graph_per_device for device in
                      deviceTopo.getDeviceIDs()), "DeviceAssignment")
    # Link the assignment of operators to devices with the assignment of subgraphs to devices
    for subgraph_id, subgraph in subgraph_dict.items():
        for device in deviceTopo.getDeviceIDs():
            for op in subgraph.getOperatorIDs():
                # Ensure that if the subgraph is assigned to the device, the operator is also assigned to the device
                model.addConstr(x[op, device] == y[subgraph_id, device])

    # Add constraints that schedule every node on exactly one machine
    for op in comp_graph.getOperatorIDs():
        model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1, name=f"one_device_{op}")

    # Constraint to enforce the split_indicator based on placement
    for a, b in non_connected_pairs:
        # Use one quicksum to check if a and b are placed on different devices
        # 1 means different devices
        model.addConstr(
            split_indicator[a, b] == 1 - quicksum(x[a, device] * x[b, device] for device in deviceTopo.getDeviceIDs()),
            name=f"split_indicator_{a}_{b}"
        )

    # Define a variable to represent the total score of splits (sum of split indicators)
    total_split_score = model.addVar(vtype=GRB.CONTINUOUS, name="total_splits")
    # Set the total_splits variable equal to the sum of split_indicator values. Splitting high-cost
    model.addConstr(
        total_split_score == quicksum(split_indicator[a, b] * (homo_op_cost_dict[a] + homo_op_cost_dict[b])
                                      for a, b in non_connected_pairs),
        name="total_splits_constraint"
    )

    # Set the target of solver
    model.setObjective(total_split_score, GRB.MAXIMIZE)

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
