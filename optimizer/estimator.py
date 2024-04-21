import json
from gurobipy import *

from optimizer.data_structure.graph import DAG

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
devices = {}

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
for node in graph.getNodes():
    for machine_id in list(devices.keys()):
        x[node.id, machine_id] = model.addVar(vtype=GRB.BINARY)
for edge in graph.getEdges():
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
    if key[0] not in node_schedule_count:
        node_schedule_count[key[0]] = 0
    node_schedule_count[key[0]] += value
for key, value in node_schedule_count.items():
    times_scheduled = LinExpr()
    times_scheduled += value
    model.addConstr(times_scheduled == 1, "one op can only be scheduled once")

# Add constraints that operators assigned cannot exceed the capacity
device_mem_count = {}
# map the node_id to the times it is assigned
for key, value in x.items():
    # (node_id, machine_id) is the key and key[1] is the machine_id.
    if key[1] not in device_mem_count:
        device_mem_count[key[1]] = 0
    nodeId = key[0]
    # value is either 1 or 0
    device_mem_count[key[1]] += value * graph.getNodes()[key[0]].size
for key, value in device_mem_count.items():
    # devices[key] will return a device object
    device_capacity = devices[key].capacity
    memory_sum = LinExpr()
    memory_sum += value
    model.addConstr(memory_sum <= device_capacity, "satisfy each deice's memory constraint")

# Add constraints that each device should have at least one operator assigned
device_op_count = {}
for key, value in x.items():
    # (node_id, machine_id) is the key and key[1] is the machine_id. Time complexity is only n
    if key[1] not in device_op_count:
        device_op_count[key[1]] = 0
    device_op_count[key[1]] += value
for key, value in device_op_count.items():
    number_op = LinExpr()
    number_op += value
    model.addConstr(number_op >= 1, "each device should have at least one op")

# CommIn, CommOut
node_in = {}
comm_out = {}
for machine_id in list(devices.keys()):
    for node_id, node in nodes.items():
        comm_in[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
        comm_out[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
    for edge in graph['edges']:
        u = edge['sourceId']
        v = edge['destId']
        model.addConstr(comm_in[u, machine_id] >= x[v, machine_id] - x[u, machine_id])
        model.addConstr(comm_out[u, machine_id] >= x[u, machine_id] - x[v, machine_id])

# TotalLatency that we are minimizing
# Latency (only create variables)
latency = {}
for node in graph.getNodes():
    latency[node.id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
TotalLatency = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0)
for node in graph.getNodes():
    model.addConstr(TotalLatency >= latency[node.id])

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
