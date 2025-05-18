import os
import sys

from gurobipy import *

from optimizer.main_simulator.simulator_util import get_comp_cost_dict, get_comm_cost_dict
from optimizer.model.graph import CompGraph, DeviceGraph
from optimizer.scheduling.priority_heteroG import priority_queue_max_rank_heteroG
import gurobipy as gp
os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.operator_device_placement.metis.subgraph_util import construct_sub_graph


def evaluate_mcmc(computing_graph: CompGraph, device_topo: DeviceGraph, operator_device_mapping, edge_cut_list):

    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
    device_subgraph_mapping = construct_sub_graph(computing_graph, operator_device_mapping)

    # Get computation and communication cost
    op_computing_cost_mapping = get_comp_cost_dict(computing_graph, operator_device_mapping)
    edge_cut_communication_cost_mapping = get_comm_cost_dict(computing_graph, device_topo, edge_cut_list, operator_device_mapping)

    # two_dime_node_list is to test whether the
    two_dime_node_list: list[list] = [list(subgraph.nodes.keys()) for subgraph in device_subgraph_mapping.values()]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    # Init solver
    model = gp.Model("minimize_maxload", env=env)

    # Define variables

    start = model.addVars(computing_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(computing_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # finish[node_id] represent the finish time of this node

    '''
    Define Constraints
    '''

    for node_id in computing_graph.getOperatorIDs():
        # Add constraints that each op's ending time = starting time + its computing time
        model.addConstr(finish[node_id] == start[node_id] + op_computing_cost_mapping[node_id], name=f"finish_start_{node_id}")

    # Data dependency for same-device communication
    non_edge_cut_list = [edge for edge in computing_graph.getEdgeIDs() if edge not in edge_cut_list]
    for edge_id_tuple in non_edge_cut_list:
        source_op_ID, target_op_ID = edge_id_tuple
        model.addConstr(finish[source_op_ID] <= start[target_op_ID])

    # Data dependency for cross-device communication
    for edge_id_tuple in edge_cut_list:
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Ensures the communication starts only after the source operation finishes.
        model.addConstr(finish[source_op_ID] + edge_cut_communication_cost_mapping[edge_id_tuple] <= start[dest_op_ID],
                        f"data_dependency_{source_op_ID}_{dest_op_ID}")

    # It is an SCHEDULING problem within each device.
    priority_queue_max_rank_heteroG(model, start, finish, None, None, computing_graph, device_subgraph_mapping, edge_cut_list, operator_device_mapping)

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
        if model is not None:
            model.dispose()
        disposeDefaultEnv()
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    # this is the main process part after a solution is reached
    elif model.status == GRB.OPTIMAL:
        # show_optimization_solution(model, operator_device_mapping, computing_graph, device_topo, start, finish, edge_cut_communication_cost_mapping, True, two_dime_node_list)
        optimal_value = model.ObjVal
        if model is not None:
            model.dispose()
        disposeDefaultEnv()
        return optimal_value
    else:
        print(f"Optimization ended with status {model.status}")
        if model is not None:
            model.dispose()
        disposeDefaultEnv()