# python3 after_graph_partition_hetero.py
import argparse

from gurobipy import *
from sklearn.cluster import KMeans
import numpy as np

from optimizer.model.graph import CompGraph, DeviceGraph
from optimizer.scheduling.proposed_scheduling_revised import SamplingFunction

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.operator_device_placement.metis.subgraph_util import construct_sub_graph, WeightNormalizationFunction, \
    map_subgraph_to_device, init_graph_weight
from optimizer.optimization_problems.gurobi_util import init_computing_and_device_graph, gurobi_setup, \
    show_optimization_solution, show_graph_partition_info
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum
from optimizer.operator_device_placement.placement import get_placement_info
from optimizer.scheduling.scheduling import execute_scheduling_function


def simulate(computing_graph: CompGraph, device_topo: DeviceGraph,
             scheduling_function: str = "FIFO",
             placement: str = 'METIS',
             rho=0.05,
             sampling_function=SamplingFunction.HEAVY_HITTER):

    # Partition the computation graph
    operator_device_mapping, edge_cut_list, edge_cut_weight_sum = (
        get_placement_info(placement, computing_graph, device_topo))

    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
    device_subgraph_mapping = construct_sub_graph(computing_graph, operator_device_mapping)

    # Get computation and communication cost
    op_computing_cost_mapping = get_comp_cost_dict(computing_graph, operator_device_mapping)
    edge_cut_communication_cost_mapping = get_comm_cost_dict(computing_graph, device_topo, edge_cut_list, operator_device_mapping)

    # Get all operator costs
    costs = np.array(list(op_computing_cost_mapping.values())).reshape(-1, 1)
    # Set the minimum cutoff to avoid near-zero values
    min_cutoff = 0.1
    # Calculate the 80th percentile of the costs
    percentile_threshold = np.percentile(costs, 80)
    # Set the final threshold as the maximum of the percentile threshold and the minimum cutoff
    threshold = max(percentile_threshold, min_cutoff)

    # two_dime_node_list is to test whether the
    two_dime_node_list: list[list] = [list(subgraph.nodes.keys()) for subgraph in device_subgraph_mapping.values()]

    # Init solver
    model = gurobi_setup("minimize_maxload")

    # Define variables
    x = model.addVars(computing_graph.getOperatorIDs(), device_topo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device
    start = model.addVars(computing_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(computing_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # finish[node_id] represent the finish time of this node
    comm_start = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0,
                               name="comm_start")  # comm_start[source_op, dest_op] represent the communication
    comm_end = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0, name="comm_end")

    '''
    Define Constraints
    '''

    # If we assume a homogeneous environment where each operator has the same time consumption on each device and the
    # bandwidth is also the same. Once we get the graph partition, the device-operation placement is already solved
    # because it does not matter where each sub-graph is placed.
    for op_id in computing_graph.getOperatorIDs():
        for device_id in device_topo.getDeviceIDs():
            if device_id == operator_device_mapping[op_id]:
                model.addConstr(x[op_id, device_id] == 1)
            else:
                model.addConstr(x[op_id, device_id] == 0)

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
        model.addConstr(finish[source_op_ID] <= comm_start[source_op_ID, dest_op_ID],
                        f"bind_finish_to_comm_start_{source_op_ID}_{dest_op_ID}")
        # Ensures the communication duration covers the communication cost.
        model.addConstr(comm_start[source_op_ID, dest_op_ID] + edge_cut_communication_cost_mapping[edge_id_tuple] == comm_end[source_op_ID, dest_op_ID],
                        f"data_dependency_{source_op_ID}_{dest_op_ID}")
        # Ensures the communication ends before the destination operation starts.
        model.addConstr(comm_end[source_op_ID, dest_op_ID] <= start[dest_op_ID],
                        f"bind_comm_end_to_start_{source_op_ID}_{dest_op_ID}")

    # It is an SCHEDULING problem within each device.
    execute_scheduling_function(scheduling_function, model, start=start, finish=finish, comm_start=comm_start,
                                comm_end=comm_end, comp_graph=computing_graph, device_subgraph_mapping=device_subgraph_mapping,
                                edge_cut_list=edge_cut_list, operator_device_mapping=operator_device_mapping, rho=rho,
                                sampling_function=sampling_function, threshold=threshold)

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
        show_optimization_solution(model, x, computing_graph, device_topo, start, finish, edge_cut_communication_cost_mapping, True, two_dime_node_list)
        print(f"This is the optimal solution of such configuration: \n"
              f"number of operators: {computing_graph.number_of_nodes()} \n"
              f"number of devices: {device_topo.number_of_nodes()} \n"
              f"placement way of this DNN graph: {placement} \n"
              f"scheduling method in each device: {scheduling_function} \n"
              f"The environment is homogenous")
        show_graph_partition_info(computing_graph, operator_device_mapping, edge_cut_list, edge_cut_weight_sum)
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


def get_comp_cost_dict(computation_graph, operator_device_mapping):
    comp_cost_dict = {}
    for node_id in computation_graph.getOperatorIDs():
        # Add constraints that each op's ending time = starting time + its computing time
        assigned_device = operator_device_mapping[node_id]
        comp_cost_dict[node_id] = computation_graph.getOperatorCompCostByDevice(node_id, assigned_device)
    return comp_cost_dict


def get_comm_cost_dict(computation_graph, device_topo, edge_cut_list, operator_device_mapping):
    comm_cost_dict = {}
    for edge_id_tuple in edge_cut_list:
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Aggregate communication cost
        comm_cost_dict[edge_id_tuple] = computation_graph.getEdgeTensorSize(source_op_ID, dest_op_ID) * device_topo.calUnitCommCostInUS(
            operator_device_mapping[source_op_ID], operator_device_mapping[dest_op_ID])
    return comm_cost_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for optimization problem after graph partitioning')
    parser.add_argument('--number_of_device', type=int, default=6)
    parser.add_argument('--model', type=str, default='ALEXNET')
    parser.add_argument('--normalization_function', default='MIN_MAX', type=str, help='')
    # NEAR_OPTIMAL OPTIMIZED METIS OPTIMIZED_HOMO
    parser.add_argument('--placement', default='METIS', type=str, help='')
    # PRIORITY_HETEROG  PRIORITY_MIN_COMP OPTIMIZED FIFO NEAR_OPTIMAL NEAR_OPTIMAL_REVISED
    parser.add_argument('--scheduling', default='NEAR_OPTIMAL_REVISED', type=str, help='')
    # parser.add_argument('--hetero_rate', default=None, type=int, help='')
    # rho == 0 is FIFO, rho == 1 is optimal; model.setParam("MIPGap", 0.01) will make it optimized
    parser.add_argument('--rho', default=0.05, type=float, help='')
    # PROBABILISTIC_SAMPLING RANDOM HEAVY_HITTER
    parser.add_argument('--sampling', default="HEAVY_HITTER", type=str, help='')
    parser.add_argument('--threshold', default=1, type=float, help='')

    args = parser.parse_args()

    # Dynamically access attributes using getattr
    model_type = getattr(TFModelEnum, args.model, None)
    weight_norm_function = getattr(WeightNormalizationFunction, args.normalization_function.upper(), None)
    sample_function = getattr(SamplingFunction, args.sampling.upper(), None)

    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(args.number_of_device, "comp_graph.json",
                                                             None, model_type=model_type)
    # init graph node/edge weight
    init_graph_weight(comp_graph, NodeWeightFunction.AVE_COMP_COST, EdgeWeightFunction.SOURCE_OUTPUT_TENSOR, weight_norm_function)

    simulate(comp_graph, deviceTopo,
             scheduling_function=args.scheduling,
             placement = args.placement,
             rho=args.rho,
             sampling_function=sample_function)
             # threshold = args.threshold
