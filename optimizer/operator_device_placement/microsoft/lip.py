# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# python3 latency-ip/lip.py < latency-inputs/OperatorGraphs/bert_l-3_inference.json
import json
from gurobipy import *

from optimizer.model.graph import CompGraph, DeviceGraph

# graph = json.load(sys.stdin)

def get_microsoft_placement(graph: CompGraph, device_topo: DeviceGraph):

    outgoingConnectionCost = {}
    for edge in graph.getEdgeIDs():
        u, v = edge
        if u in outgoingConnectionCost:
            if abs(outgoingConnectionCost[u] - edge['cost']) > 1e-6:
                raise ("node " + str(u) + " has two different outgoing connection costs")
        else:
            outgoingConnectionCost[u] = edge['cost']
    # set 0 where a node had no outgoing edges
    for node_id, node in nodes.items():
        if node_id not in outgoingConnectionCost:
            outgoingConnectionCost[node_id] = 0.0

    # figure out a (loose) upper bound for optimum latency
    # we will just sum up all cpuLatencies
    latencyUpperBound = 0.0
    for node_id, node in nodes.items():
        latencyUpperBound += node['cpuLatency']
    print('latencyUpperBound = ', latencyUpperBound)

    # IP to minimize latency

    model = Model("minimize_latency")
    model.setParam("LogToConsole", 0)
    model.setParam("LogFile", "gurobi.log")
    model.setParam("MIPGap", 0.01)
    model.setParam("TimeLimit", 3600)
    model.setParam("MIPFocus", 1)

    # if this is too large, then the reformulated
    # ex-quadratic constraints can behave funky
    model.setParam("IntFeasTol", 1e-6)


    # create variables
    x = {} # map from (node_id, machine_id) to variable
    # all cpus together are 0
    # FPGA subgraphs start from 1 and are in blocks of MAX_SUBGRAPHS_PER_FPGA many
    # (like in the paper)
    x = model.addVars(graph.getOperatorIDs(), device_topo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device
    # contiguity constraints
    z = model.addVars(graph.getOperatorIDs(), device_topo.getDeviceIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                      name="x")
    comm_in = model.addVars(graph.getOperatorIDs(), device_topo.getDeviceIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                      name="x")
    comm_out = model.addVars(graph.getOperatorIDs(), device_topo.getDeviceIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                            name="x")
    latency = model.addVars(graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # finish[node_id] represent the finish time of this node

    # Add constraints that schedule every node on exactly one machine
    for op in graph.getOperatorIDs():
        model.addConstr(quicksum(x[op, device] for device in device_topo.getDeviceIDs()) == 1, name=f"one_device_{op}")

    # contiguity constraints
    for machine_id in device_topo.getDeviceIDs():
        for node_id in graph.getOperatorIDs():
            model.addConstr(z[node_id, machine_id] >= x[node_id, machine_id])
        for edge in graph.getEdgeIDs():
            u, v = edge
            model.addConstr(z[v, machine_id] <= z[u, machine_id])
            model.addConstr(z[v, machine_id] <= x[v, machine_id] - x[u, machine_id] + 1)


    # CommIn, CommOut
    for machine_id in device_topo.getDeviceIDs():
        for edge in graph.getEdgeIDs():
            u, v = edge
            model.addConstr(comm_in[u, machine_id] >= x[v, machine_id] - x[u, machine_id])
            model.addConstr(comm_out[u, machine_id] >= x[u, machine_id] - x[v, machine_id])


    # subgraph start, finish (only create variables)
    start = [None]
    finish = [None]
    for machine_id in device_topo.getDeviceIDs():
        start.append(model.addVar(vtype = GRB.CONTINUOUS, lb=0.0))
        finish.append(model.addVar(vtype = GRB.CONTINUOUS, lb=0.0))


    # subgraph can't start before the incoming results are ready
    for machine_id in device_topo.getDeviceIDs():
        for node_id, node in nodes.items():
            # quadratic constraint!
            # model.addConstr(start[machine_id] >= latency[node_id] * comm_in[node_id, machine_id])
            # rewrite it like so:
            model.addConstr(start[machine_id] >= latency[node_id]
                            - (1 - comm_in[node_id, machine_id]) * latencyUpperBound)


    # finishing time of a subgraph
    for machine_id in device_topo.getDeviceIDs():
        fpga_load = LinExpr()
        for node_id, node in nodes.items():
            fpga_load += node['fpgaLatency'] * x[node_id, machine_id]
            # model with "calls": communication NOT overlapped with compute
            # so we add communication here
            fpga_load += outgoingConnectionCost[node_id] * comm_in[node_id, machine_id]
            fpga_load += outgoingConnectionCost[node_id] * comm_out[node_id, machine_id]
        model.addConstr(finish[machine_id] == start[machine_id] + fpga_load)


    # latency constraints for nodes on CPU
    for node_id, node in nodes.items():
        model.addConstr(latency[node_id] >= node['cpuLatency'] * x[node_id, 0])
    for edge in graph['edges']:
        u = edge['sourceId']
        v = edge['destId']
        model.addConstr(latency[v] >= latency[u] + nodes[v]['cpuLatency'] * x[v, 0])


    # latency for nodes on a subgraph
    for machine_id in device_topo.getDeviceIDs():
        for node_id, node in nodes.items():
            # quadratic constraint!
            # model.addConstr(latency[node_id] >= x[node_id, machine_id] * finish[machine_id])
            # rewrite it like so:
            model.addConstr(latency[node_id] >= finish[machine_id]
                            - (1 - x[node_id, machine_id]) * latencyUpperBound)


    # TotalLatency that we are minimizing
    TotalLatency = model.addVar(vtype = GRB.CONTINUOUS, lb=0.0)
    for node_id, node in nodes.items():
        model.addConstr(TotalLatency >= latency[node_id])


    model.setObjective(TotalLatency, GRB.MINIMIZE)

    print('Running optimizer...')
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
    result['totalLatency'] = TotalLatency.X
    result['fpgas'] = []
    result['cpus'] = []
    for machine_id in device_topo.getDeviceIDs():
        resultMachine = {}

        resultMachine['nodes'] = []
        debugTotalSize = 0.0
        for node_id, node in nodes.items():
            if x[node_id, machine_id].X > 0.99:
                resultMachine['nodes'].append(node_id)
        if machine_id == 0:
            result['cpus'].append(resultMachine)
        else:
            result['fpgas'].append(resultMachine)

    del model
    disposeDefaultEnv()
    print(json.dumps(result))
