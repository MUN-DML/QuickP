import json
from gurobipy import *

from optimizer.model.graph import DeviceGraph, CompGraph

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
    x[op_id] = model.addVar(vtype=GRB.INTEGER, lb=min(deviceTopo.nodes), ub=max(deviceTopo.nodes))

# Add constraints that operators assigned cannot exceed the capacity
device_mem_count = {}
# map the node_id to the times it is assigned
for nodeId, deviceID in x.items():
    if deviceID not in device_mem_count:
        device_mem_count[deviceID] = model.addVar(vtype=GRB.INTEGER, lb=0)
    # value is either 1 or 0
    device_mem_count[deviceID] += comp_graph.getOperator(nodeId)["size"]
for deviceID, size_sum in device_mem_count.items():
    device_capacity = deviceTopo.getDevice(deviceID)["memory_capacity"]
    model.addConstr(size_sum <= device_capacity, "satisfy each deice's memory constraint")

# Add constraints that each device should have at least one operator assigned
device_op_count = {}
for nodeId, deviceID in x.items():
    # (node_id, machine_id) is the key and key[1] is the machine_id. Time complexity is only n
    if deviceID not in device_op_count:
        device_op_count[deviceID] = model.addVar(vtype=GRB.INTEGER, lb=0)
    device_op_count[deviceID] += 1
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
    source_placement = x[sourceID]
    dest_placement = x[destID]
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
for nodeId, deviceID in x.items():
    # key[1] is the device id
    if deviceID not in result['Assignment']:
        result['Assignment'][deviceID] = []
    # key[0] is the operator id. Put id into the list assigned to the device
    result['Assignment'][deviceID].append(nodeId)

del model
disposeDefaultEnv()
print(json.dumps(result))
