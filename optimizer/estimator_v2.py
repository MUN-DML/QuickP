import json
from gurobipy import *

from optimizer.data_structure.graph import DeviceGraph, CompGraph

# Load input
# graph = json.load(sys.stdin)  # operator graph in JSON format
comp_graph = CompGraph()
comp_graph.random_rebuild(8)
print(comp_graph.getAllOperators())

deviceTopo = DeviceGraph()
deviceTopo.random_rebuild(4)
print(deviceTopo.getAllDevices())
standard_tensor_size = 1000

# Init solver
model = Model("minimize_maxload")
model.setParam("LogToConsole", 0)
model.setParam("LogFile", "gurobi.log")
model.setParam("MIPGap", 0.01)
model.setParam("TimeLimit", 1200)
model.setParam("MIPFocus", 1)

# if this is too large, then the reformulated
# ex-quadratic constraints can behave funky
model.setParam("IntFeasTol", 1e-6)

# Define placement variable
# {node_id: device_id}
x = {}
for op_id in comp_graph.getOperatorIDs():
    x[op_id] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=max(deviceTopo.nodes))

# Add constraints that schedule every node on exactly one machine
node_schedule_count = {}
# map the node_id to the times it is assigned
for key, value in x.items():
    # (node_id, machine_id) is the key and key[0] is the node_id. Time complexity is only n
    node_id = key[0]
    if node_id not in node_schedule_count:
        node_schedule_count[node_id] = model.addVar(vtype=GRB.INTEGER, lb=0)
    node_schedule_count[node_id] += value
for times in list(node_schedule_count.values()):
    model.addConstr(times == 1, "one op can only be scheduled once")

# Add constraints that operators assigned cannot exceed the capacity
device_mem_count = {}
# map the node_id to the times it is assigned
for key, value in x.items():
    # (node_id, machine_id) is the key and key[1] is the machine_id.
    device_id = key[1]
    nodeId = key[0]
    if device_id not in device_mem_count:
        device_mem_count[device_id] = model.addVar(vtype=GRB.INTEGER, lb=0)
    # value is either 1 or 0
    device_mem_count[device_id] += value * comp_graph.getOperator(nodeId)["size"]
for key, value in device_mem_count.items():
    device_capacity = deviceTopo.getDevice(key)["memory_capacity"]
    model.addConstr(value <= device_capacity, "satisfy each deice's memory constraint")

# Add constraints that each device should have at least one operator assigned
device_op_count = {}
for key, value in x.items():
    # (node_id, machine_id) is the key and key[1] is the machine_id. Time complexity is only n
    device_id = key[1]
    if device_id not in device_op_count:
        device_op_count[device_id] = model.addVar(vtype=GRB.INTEGER, lb=0)
    device_op_count[device_id] += value
for number_op in list(device_op_count.values()):
    model.addConstr(number_op >= 1, "each device should have at least one op")

# Add constraints that later operator cannot begin before all previous ones finish computing and transmission
start = {}
finish = {}
for node_id in list(comp_graph.getOperatorIDs()):
    start[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    finish[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
for edge_id_tuple in list(comp_graph.getEdgeIDs()):
    print(edge_id_tuple)
    sourceID = edge_id_tuple[0]
    destID = edge_id_tuple[1]
    source_placement = 1
    dest_placement = 1
    model.addConstr(start[destID] >= finish[sourceID] + round(
        standard_tensor_size / deviceTopo.getConnection(source_placement, source_placement)["computing_speed"], 2),
                    "data dependency between source and destination nodes")

    for i in deviceTopo.getDeviceIDs():
        model.addConstr(start[destID] >= finish[sourceID] + round(
            standard_tensor_size / deviceTopo.getConnection(source_placement, source_placement)["computing_speed"], 2),
                        "data dependency between source and destination nodes")

# TotalLatency that we are minimizing
TotalLatency = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
model.addConstr(TotalLatency >= max(list(finish.values())), "satisfy each deice's latency")

# Set the target of solver
model.setObjective(TotalLatency, GRB.MINIMIZE)

# Run the solver
sys.stdout.flush()
model.optimize()

if model.Status == GRB.Status.INFEASIBLE:
    raise "infeasible"
elif model.Status == GRB.Status.OPTIMAL:
    print("Value is:", TotalLatency.X)
else:
    raise "Wrong status code"

print('Runtime = ', "%.2f" % model.Runtime, 's', sep='')

#populate the result dict
result = {'totalLatency': TotalLatency.X, 'Assignment': {}}
for key, value in x.items():
    # key[1] is the device id
    if key[1] not in result['Assignment']:
        result[key[1]] = []
    # key[0] is the operator id. Put id into the list assigned to the device
    if value.X > 0.99:
        result[key[1]].append(key[0])

del model
disposeDefaultEnv()
print(json.dumps(result))
