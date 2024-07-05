# python3 estimator_test.py

from gurobipy import *
import torch
import tensorflow as tf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.computing_graph.computing_graph import get_computation_graph
from py_util import tensor_shape_to_bits
from optimizer.model.graph import DeviceGraph, CompGraph, has_more_than_one_component, keep_largest_component, \
    determine_node_order
from DNN_model_tf.small import small_tf

# init fake data
if not os.path.exists('comp_graph.json'):
    model = small_tf()
    comp_graph = get_computation_graph(model=model)
    comp_graph.generata_random_cost(2)
    comp_graph.save_to_file('comp_graph.json')

comp_graph = CompGraph.load_from_file('comp_graph.json')
if has_more_than_one_component(comp_graph):
    comp_graph = keep_largest_component(comp_graph)

deviceTopo = DeviceGraph()
deviceTopo.generata_fat_tree_topo(2, 30, 20, 1)

# Init solver
model = Model("minimize_maxload")
model.setParam("LogToConsole", 0)
model.setParam("LogFile", "gurobi.log")
model.setParam("MIPGap", 0.11)
model.setParam("TimeLimit", 2400)
model.setParam("MIPFocus", 1)

# if this is too large, then the reformulated
# ex-quadratic constraints can behave funky
model.setParam("IntFeasTol", 1e-6)
model.setParam("MemLimit", 4096)  # Example: Limit memory usage to 4 GB
model.setParam("Threads", 4)  # Example: Use 4 threads

# Define variables
x = {}  # key will be (operator_id, machine_id), value will be 1 or 0; x[3, 1] = 1 means operator 3 get allocated to device 1
start = {}  # start[node_id] represent the starting time of this node
finish = {}  # finish[node_id] represent the finish time of this node
comm_start = {}  # comm_start[source_op, dest_op] represent the communication
comm_end = {}
comm_cost = {}

# Initialize all variables with names
for node_id in comp_graph.getOperatorIDs():
    for machine_id in deviceTopo.getDeviceIDs():
        x[node_id, machine_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{node_id}_{machine_id}")
    start[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"start_{node_id}")
    finish[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"finish_{node_id}")

for edge_id_tuple in comp_graph.getEdgeIDs():
    source_op_ID, dest_op_ID = edge_id_tuple
    comm_start[source_op_ID, dest_op_ID] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                        name=f"comm_start_{source_op_ID}_{dest_op_ID}")
    comm_end[source_op_ID, dest_op_ID] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                      name=f"comm_end_{source_op_ID}_{dest_op_ID}")
    comm_cost[source_op_ID, dest_op_ID] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                       name=f"comm_cost_{source_op_ID}_{dest_op_ID}")

# Add constraints that schedule every node on exactly one machine
for op in comp_graph.getOperatorIDs():
    model.addConstr(quicksum(x[op, device] for device in deviceTopo.getDeviceIDs()) == 1, name=f"one_device_{op}")

# Add constraints that operators assigned cannot exceed the capacity
for machine_id in deviceTopo.getDeviceIDs():
    mem_sum = quicksum(x[node_id, machine_id] * comp_graph.getOperator(node_id)["mem"]
                       for node_id in comp_graph.getOperatorIDs())
    model.addConstr(mem_sum <= deviceTopo.getDeviceMaxMem(machine_id),
                    f"satisfy_memory_constraint_{machine_id}")

# Add constraints that each device should have at least one operator assigned
for machine_id in deviceTopo.getDeviceIDs():
    model.addConstr(quicksum(x[node_id, machine_id] for node_id in comp_graph.getOperatorIDs()) >= 1,
                    name=f"at_least_one_op_{machine_id}")

# Add constraints that each op's ending time = starting time + its computing time
for node_id in comp_graph.getOperatorIDs():
    comp_cost = quicksum(x[node_id, device_id] * comp_graph.getOperatorCompCostByDevice(node_id, device_id)
                         for device_id in deviceTopo.getDeviceIDs())
    model.addConstr(finish[node_id] == start[node_id] + comp_cost, name=f"finish_start_{node_id}")

# Add constraint that if op2 depends on op1, the starting time of op2 will be the ending time of op1 + communication delay if these two ops are not placed on the same device
# unit_comm_costs[device_id_src, device_id_dest] means the com cost per bit from device with source device to dest device
unit_comm_costs = {}
for device_id_src in deviceTopo.getDeviceIDs():
    for device_id_dest in deviceTopo.getDeviceIDs():
        unit_comm_costs[device_id_src, device_id_dest] = deviceTopo.calUnitCommCostInUS(device_id_src, device_id_dest)
for edge_id_tuple in list(comp_graph.getEdgeIDs()):
    # https://support.gurobi.com/hc/en-us/articles/360039628832-Constraint-has-no-bool-value-are-you-trying-lb-expr-ub
    # https://support.gurobi.com/hc/en-us/community/posts/360077951791-if-statement-in-constraint
    source_op_ID, dest_op_ID = edge_id_tuple
    shape, dtype = comp_graph.getOperatorOutputSizeAndType(source_op_ID)
    tensor_size = tensor_shape_to_bits(shape, dtype=dtype)

    # Aggregate communication cost
    comm_cost_expr = quicksum(
        unit_comm_costs[device_id_src, device_id_dest] * tensor_size * x[source_op_ID, device_id_src] * x[
            dest_op_ID, device_id_dest]
        for device_id_src in deviceTopo.getDeviceIDs()
        for device_id_dest in deviceTopo.getDeviceIDs()
        if device_id_src != device_id_dest
    )

    model.addConstr(comm_cost[source_op_ID, dest_op_ID] == comm_cost_expr, f"comm_cost_{source_op_ID}_{dest_op_ID}")

    # Ensures the communication starts only after the source operation finishes.
    model.addConstr(comm_start[source_op_ID, dest_op_ID] >= finish[source_op_ID],
                    f"bind_finish_to_comm_start_{source_op_ID}_{dest_op_ID}")

    # Ensures the communication ends before the destination operation starts.
    model.addConstr(comm_end[source_op_ID, dest_op_ID] <= start[dest_op_ID],
                    f"bind_comm_end_to_start_{source_op_ID}_{dest_op_ID}")

    # Ensures the communication duration covers the communication cost.
    model.addConstr(comm_end[source_op_ID, dest_op_ID] == comm_start[source_op_ID, dest_op_ID] + comm_cost[
        source_op_ID, dest_op_ID],
                    f"data_dependency_{source_op_ID}_{dest_op_ID}")

# Add constraint to ensure each device processes only one operator at a time. This is a SCHEDULING problem
for device in deviceTopo.getDeviceIDs():
    # ensures that each pair of operations is only considered once
    for op1, op2 in itertools.combinations(comp_graph.getOperatorIDs(), 2):
        if determine_node_order(comp_graph, op1, op2) == 1:
            y1 = model.addVar(vtype=GRB.BINARY, name=f"y1_{device}_{op1}_{op2}")
            model.addGenConstrIndicator(y1, True, finish[op1] <= start[op2])
            # If on the same device, ensure that the operators do not overlap
            model.addConstr(y1 >= x[op1, device] + x[op2, device] - 1, name=f"non_overlap_{op1}_{op2}_{device}")
        elif determine_node_order(comp_graph, op1, op2) == 2:
            y2 = model.addVar(vtype=GRB.BINARY, name=f"y2_{device}_{op1}_{op2}")
            model.addGenConstrIndicator(y2, True, finish[op2] <= start[op1])
            model.addConstr(y2 >= x[op1, device] + x[op2, device] - 1, name=f"non_overlap_{op1}_{op2}_{device}")
        else:
            raise ValueError("node not existing")

# Add constraint to ensure each device can only send or receive from one link
# Iterate over all pairs of communication edges
for (source_op_ID1, dest_op_ID1), (source_op_ID2, dest_op_ID2) in itertools.combinations(comp_graph.getEdgeIDs(), 2):
    for device_id_src, device_id_dest in itertools.combinations(deviceTopo.getDeviceIDs(), 2):
        # if two communications are using the same link. device a -> device b and device b -> device a are considered the same link,
        # these two communications cannot overlap
        if determine_node_order(comp_graph, source_op_ID1, source_op_ID2) == 1:
            no_overlap1 = model.addVar(vtype=GRB.BINARY)
            # Enforce non-overlapping constraints using indicator constraints
            model.addGenConstrIndicator(no_overlap1, True, comm_end[source_op_ID1, dest_op_ID1] <= comm_start[source_op_ID2, dest_op_ID2])
            model.addConstr(
                no_overlap1 >= x[source_op_ID1, device_id_src] + x[dest_op_ID1, device_id_dest] + x[source_op_ID2, device_id_src] + x[dest_op_ID2, device_id_dest] + x[source_op_ID1, device_id_dest] + x[dest_op_ID1, device_id_src] + x[source_op_ID2, device_id_dest] + x[dest_op_ID2, device_id_src] - 3)

        elif determine_node_order(comp_graph, source_op_ID1, source_op_ID2) == 2:
            no_overlap2 = model.addVar(vtype=GRB.BINARY)
            model.addGenConstrIndicator(no_overlap2, True, comm_end[source_op_ID2, dest_op_ID2] <= comm_start[source_op_ID1, dest_op_ID1])
            model.addConstr(
                no_overlap2 >= x[source_op_ID1, device_id_src] + x[dest_op_ID1, device_id_dest] + x[source_op_ID2, device_id_src] + x[dest_op_ID2, device_id_dest] + x[source_op_ID1, device_id_dest] + x[dest_op_ID1, device_id_src] + x[source_op_ID2, device_id_dest] + x[dest_op_ID2, device_id_src] - 3)

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
    print('Runtime = ', "%.2f" % model.Runtime, 's', sep='')
    print('Expected Traning time = ', TotalLatency.X, 's', sep='')
    # init result dict
    result = {'totalLatency': model.ObjVal, 'Assignment': {}, 'CommunicationCosts': [], "CommunicationTimeLine": {}}

    # populate result['Assignment']
    for key, value in x.items():
        # key[1] is the device id
        if key[1] not in result['Assignment']:
            result['Assignment'][key[1]] = []
        # key[0] is the operator id. Put id into the list assigned to the device
        if value.X > 0.5:
            # Assignment: {device: [(op1, start[op1], finish[op1]), (...)]}
            result['Assignment'][key[1]].append((key[0], start[key[0]].X, finish[key[0]].X))
    # Sort operators by their start times for each device
    for device, ops in result['Assignment'].items():
        result['Assignment'][device] = sorted(ops, key=lambda x: x[1])

    # populate result['CommunicationCosts'] and result['CommunicationTimeLine']
    for edge_id_tuple in list(comp_graph.getEdgeIDs()):
        source_op_ID, dest_op_ID = edge_id_tuple
        s_placement = None
        d_placement = None
        comm_cost_var = model.getVarByName(f"comm_cost_{source_op_ID}_{dest_op_ID}")
        comm_start_var = model.getVarByName(f"comm_start_{source_op_ID}_{dest_op_ID}")
        comm_end_var = model.getVarByName(f"comm_end_{source_op_ID}_{dest_op_ID}")
        if comm_cost_var and comm_start_var and comm_end_var:
            comm_cost = comm_cost_var.X
            comm_start_time = comm_start_var.X
            comm_end_time = comm_end_var.X
            if comm_cost == 0:
                continue
            shape, dtype = comp_graph.getOperatorOutputSizeAndType(source_op_ID)
            tensor_size = tensor_shape_to_bits(shape, dtype=dtype)
            for device, ops in result['Assignment'].items():
                if source_op_ID in [op[0] for op in ops]:
                    s_placement = device
                if dest_op_ID in [op[0] for op in ops]:
                    d_placement = device
                if s_placement and d_placement:
                    break
            # if both ops are placed on the same device, use 999 to represent that
            if s_placement == d_placement:
                bandwidth = 999
            else:
                bandwidth = deviceTopo.get_link_bandwidth(s_placement, d_placement)
            result['CommunicationCosts'].append(
                (source_op_ID, s_placement, dest_op_ID, d_placement, comm_cost, tensor_size, bandwidth))
            # Populate the communication timeline divided by device
            if s_placement not in result['CommunicationTimeLine']:
                result['CommunicationTimeLine'][s_placement] = []
            if d_placement not in result['CommunicationTimeLine']:
                result['CommunicationTimeLine'][d_placement] = []

            result['CommunicationTimeLine'][s_placement].append(
                (source_op_ID, dest_op_ID, comm_start_time, comm_end_time, comm_cost))
            result['CommunicationTimeLine'][d_placement].append(
                (source_op_ID, dest_op_ID, comm_start_time, comm_end_time, comm_cost))
    # Sort the communication timeline based on the starting time
    for device, timeline in result['CommunicationTimeLine'].items():
        result['CommunicationTimeLine'][device] = sorted(timeline, key=lambda x: x[2])

    # Print operator placement
    for device, ops in result['Assignment'].items():
        print(f"Device: {device}")
        for op in ops:
            comp_cost = 0  # Initialize computation cost for the current operator
            for device_id in deviceTopo.getDeviceIDs():
                comp_cost += x[op[0], device_id].X * comp_graph.getOperatorCompCostByDevice(op[0], device_id)
            if comp_cost == 0:
                continue
            print(f"  Operator: {op[0]}, Start: {op[1]}, Finish: {op[2]}, Comp Cost: {comp_cost}")

    # Print communication costs
    print("Communication Costs:")
    sum_of_communication = 0
    for source_op_ID, s_placement, dest_op_ID, d_placement, comm_cost, tensor_size, bandwidth in result[
        'CommunicationCosts']:
        sum_of_communication += comm_cost
        print(
            f"  From {source_op_ID} with placement {s_placement} to {dest_op_ID} with placement {d_placement}, Cost: {comm_cost}, Tensor size: {tensor_size}, Bandwidth: {bandwidth} GB/s")
    print(f"Total Communication Cost: {sum_of_communication}")

    # Print communication timeline divided by device
    print("Communication Timeline:")
    for device, timeline in result['CommunicationTimeLine'].items():
        print(f"Device: {device}")
        for source_op_ID, dest_op_ID, comm_start_time, comm_end_time, cost in timeline:
            print(
                f"  Communication from {source_op_ID} to {dest_op_ID} starts at {comm_start_time} and ends at {comm_end_time} with cost: {cost}")
    del model
    disposeDefaultEnv()
else:
    print(f"Optimization ended with status {model.status}")
