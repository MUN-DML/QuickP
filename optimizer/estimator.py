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

# Define variables
x = {}  # key will be (node_id, machine_id), value will be 1 or 0
d = {}  # key will be (node_id_1, node_id_2), value will be 1 or 0
for node_id in comp_graph.getOperatorIDs():
    for machine_id in deviceTopo.getDeviceIDs():
        x[node_id, machine_id] = model.addVar(vtype=GRB.BINARY)
for edge_id_tuple in comp_graph.getEdgeIDs():
    d[edge_id_tuple[0], edge_id_tuple[1]] = model.addVar(vtype=GRB.BINARY)
# for

# Add constraints that schedule every node on exactly one machine
for node_id in comp_graph.getOperatorIDs():
    times_scheduled = LinExpr()
    for machine_id in deviceTopo.getDeviceIDs():
        times_scheduled += x[node_id, machine_id]
    model.addConstr(times_scheduled == 1,"every node on exactly one machine")
'''
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
'''
# Add constraints that each device should have at least one operator assigned
for machine_id in deviceTopo.getDeviceIDs():
    op_count = LinExpr()
    for node_id in comp_graph.getOperatorIDs():
        op_count += x[node_id, machine_id]
    model.addConstr(op_count >= 1, "each device should have at least one op")

# Add constraints that later operator cannot begin before all previous ones finish computing and transmission
start = {}
finish = {}
for node_id in list(comp_graph.getOperatorIDs()):
    start[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    finish[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
for node_id in list(comp_graph.getOperatorIDs()):
    comp_cost = LinExpr()
    # since there is one placement, only one x[node_id, device_id] will be 1
    for device_id in deviceTopo.getDeviceIDs():
        comp_cost += x[node_id, device_id] * comp_graph.getOperator(node_id)["computing_cost"]
    model.addConstr(finish[node_id] == start[node_id] + comp_cost, "finish == start + process")

for edge_id_tuple in list(comp_graph.getEdgeIDs()):
    sourceID = edge_id_tuple[0]
    destID = edge_id_tuple[1]
    model.addConstr(start[destID] >= finish[sourceID])
    '''
    source_placement = 1
    dest_placement = 1
    model.addConstr(start[destID] >= finish[sourceID] + round(
        standard_tensor_size / deviceTopo.getConnection(source_placement, source_placement)["computing_speed"], 2),
                    "data dependency between source and destination nodes")

    for i in deviceTopo.getDeviceIDs():
        model.addConstr(start[destID] >= finish[sourceID] + round(
            standard_tensor_size / deviceTopo.getConnection(source_placement, source_placement)["computing_speed"], 2),
                        "data dependency between source and destination nodes")
    '''
# TotalLatency that we are minimizing
TotalLatency = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
for op_end in finish.values():
    model.addConstr(TotalLatency >= op_end, "satisfy each deice's latency")

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
        result['Assignment'][key[1]] = []
    # key[0] is the operator id. Put id into the list assigned to the device
    if value.X > 0.99:
        result['Assignment'][key[1]].append(key[0])

del model
disposeDefaultEnv()
print(json.dumps(result))
