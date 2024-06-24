import json

from gurobipy import *

from DNN_model_tf.vgg_tf import VGG16_tf
from model.graph import visualize_graph
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.device_topo.device_graph import get_device_topo_ssh
from optimizer.cluster_info import servers

model = VGG16_tf()
comp_graph = get_computation_graph(model=model)
deviceTopo = get_device_topo_ssh(servers)

# init fake data
comp_graph.generata_random_cost(100)
deviceTopo.generata_random_nodes(100)

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
x1 = model.addVar(vtype=GRB.BINARY, name="w1")
x2 = model.addVar(vtype=GRB.BINARY, name="w2")
for node_id in comp_graph.getOperatorIDs():
    for machine_id in deviceTopo.getDeviceIDs():
        x[node_id, machine_id] = model.addVar(vtype=GRB.BINARY)
for source_id in comp_graph.getOperatorIDs():
    for dest_id in comp_graph.getOperatorIDs():
        d[source_id, dest_id] = model.addVar(vtype=GRB.BINARY)
        # If two nodes do not have dependency relationship
        if (source_id, dest_id) not in comp_graph.getEdgeIDs():
            model.addConstr(d[source_id, dest_id] == 0)
        else:
            for device_id in deviceTopo.getDeviceIDs():
                model.addConstr((x[source_id, device_id] == 1) >> (x1 == 1), "source node is placed on the device")
                model.addConstr((x[dest_id, device_id] == 1) >> (x2 == 1), "dest node is placed on the device")
                model.addGenConstrAnd(d[source_id, dest_id], [x1, x2], "andconstr")

# Add constraints that schedule every node on exactly one machine
for node_id in comp_graph.getOperatorIDs():
    times_scheduled = LinExpr()
    for machine_id in deviceTopo.getDeviceIDs():
        times_scheduled += x[node_id, machine_id]
    model.addConstr(times_scheduled == 1,"every node on exactly one machine")

# Add constraints that operators assigned cannot exceed the capacity
for machine_id in deviceTopo.getDeviceIDs():
    mem_sum = LinExpr()
    for node_id in comp_graph.getOperatorIDs():
        mem_sum += x[node_id, machine_id] * comp_graph.getOperator(node_id)["mem"]
    model.addConstr(mem_sum <= deviceTopo.getDevice(machine_id)["memory_capacity"], "satisfy each device's memory constraint")

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
        # consider the device heterogeneity
        comp_cost += x[node_id, device_id] * comp_graph.getOperator(node_id)["comp_cost"][device_id]
    model.addConstr(finish[node_id] == start[node_id] + comp_cost, "finish == start + process")

for edge_id_tuple in list(comp_graph.getEdgeIDs()):
    sourceID = edge_id_tuple[0]
    destID = edge_id_tuple[1]
    source_placement = model.addVar(vtype=GRB.INTEGER, name="w1")
    dest_placement = model.addVar(vtype=GRB.INTEGER, name="w1")
    # https://support.gurobi.com/hc/en-us/articles/360039628832-Constraint-has-no-bool-value-are-you-trying-lb-expr-ub
    # https://support.gurobi.com/hc/en-us/community/posts/360077951791-if-statement-in-constraint
    for device_id in deviceTopo.getDeviceIDs():
        model.addConstr((x[sourceID, device_id] == 1) >> (source_placement == device_id))
        model.addConstr((x[destID, device_id] == 1) >> (dest_placement == device_id))
    # model.update()
    # path = nx.shortest_path(deviceTopo, source=source_placement, target=dest_placement)
    # communication_cost = deviceTopo.calculateCommunicationCost(standard_tensor_size, path_list=path)
    model.addConstr(start[destID] >= finish[sourceID] + d[sourceID, destID] * 0, "data dependency between source and destination nodes")

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
