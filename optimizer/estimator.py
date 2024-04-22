import json
from gurobipy import *

from optimizer.data_structure.graph import DAG, Tree, DeviceGraph

# Get the parameter values passed from the command
if len(sys.argv) < 2:
    raise 'no argument given'
elif sys.argv[1] == 'contig':
    FORCE_CONTIGUOUS_FORWARD = True
elif sys.argv[1] == 'noncontig':
    FORCE_CONTIGUOUS_FORWARD = False
else:
    raise 'argument should be contig/noncontig'

# Load input
# graph = json.load(sys.stdin)  # operator graph in JSON format
graph = DAG('')
graph.add_node(1, "relu", 1, 1.5)
graph.add_node(2, "relu", 4, 2.5)
graph.add_node(3, "matmul", 1, 1.5)
graph.add_node(4, "sigmoid", 1, 1.5)
graph.add_edge(1, 3, 0.3)
graph.add_edge(2, 3, 0.5)
graph.add_edge(3, 4, 0.3)

deviceTopo = DeviceGraph()

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
for node_id in list(graph.getNodes().keys()):
    for machine_id in list(devices.keys()):
        x[node_id, machine_id] = model.addVar(vtype=GRB.BINARY)
for edge in list(graph.getEdges().values()):
    d[edge.sourceID, edge.destID] = model.addVar(vtype=GRB.BINARY)
'''
for key, value in d.items():
    sourceId = key[0]
    destId = key[1]
    source_node_device
'''
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
    device_mem_count[device_id] += value * graph.getNodes()[nodeId].size
for key, value in device_mem_count.items():
    # devices[key] will return a device object
    device_capacity = devices[key].capacity
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
'''
for key, value in device_op_count.items():
    number_op = LinExpr()
    number_op += value
    model.addConstr(number_op >= 1, "each device should have at least one op")
'''

# Add constraints that later operator cannot begin before all previous ones finish computing and transmission
start = {}
finish = {}
for node_id in list(graph.getNodes().keys()):
    start[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    finish[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
for edge in list(graph.getEdges().values()):
    sourceID = edge.sourceID
    destID = edge.destID
    model.addConstr(start[destID] >= finish[sourceID] + graph.getEdges()[sourceID, destID].communicationCost,
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
