import json
from gurobipy import *

# Load input
graph = json.load(sys.stdin)  # operator graph in JSON format
nodes = {}

# Define variables
x = {}  # map from (node_id, machine_id) to variable

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

# TotalLatency that we are minimizing
TotalLatency = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
for node_id, node in nodes.items():
    model.addConstr(TotalLatency >= latency[node_id])

# Add constraints
# schedule every node on exactly one machine
for node_id, node in nodes.items():
    times_scheduled = LinExpr()
    for machine_id in range(1 + maxSubgraphs):
        times_scheduled += x[node_id, machine_id]
    model.addConstr(times_scheduled == 1)


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

result = {}

del model
disposeDefaultEnv()
print(json.dumps(result))