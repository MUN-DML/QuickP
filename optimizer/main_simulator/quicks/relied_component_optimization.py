from gurobipy import *

from optimizer.main_simulator.quicks.quicks import calculate_rank_map
from optimizer.main_simulator.simulator_util import get_comp_cost_dict, get_comm_cost_dict
from optimizer.model.graph import CompGraph, DeviceGraph
from optimizer.scheduling.mcmc_order import mcmc_schedule

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.operator_device_placement.metis.subgraph_util import construct_sub_graph, WeightNormalizationFunction, \
    init_graph_weight
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph, gurobi_setup, \
    show_optimization_solution, show_graph_partition_info


def get_relied_component_execution_order(relied_graph: CompGraph, edge_cut_list, op_computing_cost_mapping, edge_cut_communication_cost_mapping):

    # Init solver
    model = gurobi_setup("minimize_maxload")

    # Define variables

    start = model.addVars(relied_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(relied_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # finish[node_id] represent the finish time of this node
    comm_start = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0,
                               name="")  # comm_start[source_op, dest_op] represent the communication
    comm_end = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0, name="")

    '''
    Define Constraints
    '''

    for node_id in relied_graph.getOperatorIDs():
        # Add constraints that each op's ending time = starting time + its computing time
        model.addConstr(finish[node_id] == start[node_id] + op_computing_cost_mapping[node_id], name=f"finish_start_{node_id}")

    # Data dependency for same-device communication
    non_edge_cut_list = [edge for edge in relied_graph.getEdgeIDs() if edge not in edge_cut_list]
    for edge_id_tuple in non_edge_cut_list:
        source_op_ID, target_op_ID = edge_id_tuple
        model.addConstr(finish[source_op_ID] <= start[target_op_ID])

    # Data dependency for cross-device communication
    for edge_id_tuple in edge_cut_list:
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Ensures the communication starts only after the source operation finishes.
        model.addConstr(finish[source_op_ID] <= comm_start[source_op_ID, dest_op_ID],
                        name = "")
        # Ensures the communication duration covers the communication cost.
        model.addConstr(comm_start[source_op_ID, dest_op_ID] + edge_cut_communication_cost_mapping[edge_id_tuple] == comm_end[source_op_ID, dest_op_ID],
                        name = "")
        # Ensures the communication ends before the destination operation starts.
        model.addConstr(comm_end[source_op_ID, dest_op_ID] <= start[dest_op_ID],
                        name = "")

    # It is an SCHEDULING problem within each device.


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