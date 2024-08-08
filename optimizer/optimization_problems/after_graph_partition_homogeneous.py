# python3 after_graph_partition.py
import argparse
from itertools import combinations

from gurobipy import *
import torch
import tensorflow as tf

from optimizer.model.graph import find_non_connected_pairs, topological_sort_groups

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.optimization_problems.scheduling_algorithm import create_topological_position_dict, TopoSortFunction
from optimizer.graph_partitioner.metis_partition import metis_partition
from optimizer.graph_partitioner.subgraph_util import construct_sub_graph, WeightNormalizationFunction, \
    map_subgraph_to_device
from optimizer.optimization_problems.gurobi_util import init_computing_and_device_graph, gurobi_setup, \
    show_optimization_solution, show_graph_partition_info, get_subgraph_topo_dict, sort_edges_by_topo_order, \
    get_incoming_and_outing_cut_off_edges_in_subgraph
from optimizer.graph_partitioner.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum


def optimize_after_graph_partition(number_of_devices=2, model_type: TFModelEnum = TFModelEnum.SMALL,
                                   edge_weight_function=EdgeWeightFunction.MOCK_COMMUNICATION_COST_WITH_COMP,
                                   adjust_matrix=None, weight_norm_function: WeightNormalizationFunction = None,
                                   scheduling_algorithm=TopoSortFunction.KAHN_PRIORITY):
    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(number_of_devices, "comp_graph_after_partition.json",
                                                             model_type=model_type)
    # Init solver
    model = gurobi_setup("minimize_maxload")

    # Partition the computation graph
    partition_dict, edge_cut_list, weighted_graph, edge_cut_weight_sum = metis_partition(comp_graph,
                                                                                         num_partitions=number_of_devices,
                                                                                         edge_weight_function=edge_weight_function,
                                                                                         adjust_matrix=adjust_matrix,
                                                                                         weight_normalize=weight_norm_function)
    subgraph_dict = construct_sub_graph(comp_graph, partition_dict)

    operator_device_dict = map_subgraph_to_device(partition_dict, deviceTopo.getDeviceIDs())
    device_subgraph_dict = construct_sub_graph(comp_graph, operator_device_dict)

    # global_topo_dict will decide the
    global_topo_dict = create_topological_position_dict(comp_graph, scheduling_algorithm, edge_cut_list)
    # operator scheduling within each device; global_topo_dict.keys() maintains the self-defined topo sorting
    subgraph_topo_dict = get_subgraph_topo_dict(global_topo_dict.keys(), partition_dict)

    # two_dime_node_list is to test whether the
    two_dime_node_list: list[list] = [list(subgraph.nodes.keys()) for subgraph in subgraph_dict.values()]

    # Define variables
    x = {}  # [operator_id, device_id] == 1 means this operator is assigned to this device
    y = {}  # [subgraph_id, device_id] == 1 means this subgraph is assigned to this device
    start = {}  # start[node_id] represent the starting time of this node
    finish = {}  # finish[node_id] represent the finish time of this node
    comm_start = {}  # comm_start[source_op, dest_op] represent the communication
    comm_end = {}
    comm_cost = {}

    for node_id in comp_graph.getOperatorIDs():
        for machine_id in deviceTopo.getDeviceIDs():
            x[node_id, machine_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{node_id}_{machine_id}")
        start[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"start_{node_id}")
        finish[node_id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"finish_{node_id}")

    for subgraph_id in subgraph_dict.keys():
        for device in deviceTopo.getDeviceIDs():
            y[subgraph_id, device] = model.addVar(vtype=GRB.BINARY, name=f"y_{subgraph_id}_{device}")

    for edge_id_tuple in comp_graph.getEdgeIDs():
        source_op_ID, dest_op_ID = edge_id_tuple
        if edge_id_tuple in edge_cut_list:
            comm_start[source_op_ID, dest_op_ID] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                                name=f"comm_start_{source_op_ID}_{dest_op_ID}")
            comm_end[source_op_ID, dest_op_ID] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                                              name=f"comm_end_{source_op_ID}_{dest_op_ID}")

    '''
    Define Constraints
    '''

    # If we assume a homogeneous environment where each operator has the same time consumption on each device and the
    # bandwidth is also the same. Once we get the graph partition, the device-operation placement is already solved
    # because it does not matter where each sub-graph is placed.
    for device_id, subgraph in device_subgraph_dict.items():
        for node_id in subgraph.nodes:
            model.addConstr(x[node_id, device_id] == 1)

    for node_id in comp_graph.getOperatorIDs():
        # Add constraints that each op's ending time = starting time + its computing time
        assigned_device = operator_device_dict[node_id]
        comp_cost = comp_graph.getOperatorCompCostByDevice(node_id, assigned_device)
        model.addConstr(finish[node_id] == start[node_id] + comp_cost, name=f"finish_start_{node_id}")

        # Add constraints that schedule every node on exactly one machine
        model.addConstr(quicksum(x[node_id, device] for device in deviceTopo.getDeviceIDs()) == 1,
                        name=f"one_device_{node_id}")

    # Add constraint that if op2 depends on op1, the starting time of op2 will be the ending time of op1 + communication delay if these two ops are not placed on the same device
    # device_pairs is a Set obj with unique device pair

    for edge_id_tuple in edge_cut_list:
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Aggregate communication cost
        communication_cost = comp_graph.getOperatorOutputInBit(source_op_ID) * deviceTopo.calUnitCommCostInUS(
            operator_device_dict[source_op_ID], operator_device_dict[dest_op_ID])

        # Ensures the communication starts only after the source operation finishes.
        model.addConstr(comm_start[source_op_ID, dest_op_ID] >= finish[source_op_ID],
                        f"bind_finish_to_comm_start_{source_op_ID}_{dest_op_ID}")

        # Ensures the communication ends before the destination operation starts.
        model.addConstr(comm_end[source_op_ID, dest_op_ID] <= start[dest_op_ID],
                        f"bind_comm_end_to_start_{source_op_ID}_{dest_op_ID}")

        # Ensures the communication duration covers the communication cost.
        model.addConstr(comm_end[source_op_ID, dest_op_ID] == comm_start[source_op_ID, dest_op_ID] + communication_cost,
                        f"data_dependency_{source_op_ID}_{dest_op_ID}")

        # just for verification
        comm_cost[source_op_ID, dest_op_ID] = communication_cost

    # specify the data dependency
    for source_op_ID, dest_op_ID in comp_graph.getEdgeIDs():
        model.addConstr(finish[source_op_ID] <= start[dest_op_ID])
    # It is an SCHEDULING problem within each device. The scheduling must follow the topo sorting. Thus, a possible sort
    for subgraph in subgraph_dict.values():
        for level in topological_sort_groups(subgraph):
            for op_a, op_b in combinations(level, 2):
                z = model.addVar(vtype=GRB.BINARY, name=f"z_{op_a}_{op_b}")
                # Use logical constraints to enforce the or condition without big-M
                model.addGenConstrIndicator(z, True, finish[op_a] <= start[op_b])
                model.addGenConstrIndicator(z, False, finish[op_b] <= start[op_a])

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
        show_optimization_solution(model, x, comp_graph, deviceTopo, start, finish, True, two_dime_node_list)
        show_graph_partition_info(weighted_graph, partition_dict, edge_cut_list, edge_cut_weight_sum)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for optimization problem after graph partitioning')
    parser.add_argument('--number_of_device', type=int, default=2)
    parser.add_argument('--model', type=str, default='SMALL')
    parser.add_argument('--normalization_function', default='MinMax', type=str, help='')
    parser.add_argument('--node_weight_function', default='comp_cost', type=str, help='')
    parser.add_argument('--edge_weight_function', default='comm_cost', type=str, help='')
    parser.add_argument('--topo_sort_function', default='Kahn', type=str,
                        help='it is regarding operator and communication scheduling')

    args = parser.parse_args()

    model_mapping_dict = {'VGG': TFModelEnum.VGG, 'SMALL': TFModelEnum.SMALL, "ALEXNET": TFModelEnum.ALEXNET}
    weight_normalization_dict = {'MinMax': WeightNormalizationFunction.MIN_MAX}
    topo_sort_dict = {"Kahn": TopoSortFunction.KAHN, "Default": TopoSortFunction.DEFAULT,
                      "KahnPriority": TopoSortFunction.KAHN_PRIORITY}

    optimize_after_graph_partition(number_of_devices=args.number_of_device, model_type=model_mapping_dict[args.model],
                                   adjust_matrix={"node_enable": True, "edge_enable": False, 'adjustment_ratio': 0},
                                   weight_norm_function=weight_normalization_dict[args.normalization_function],
                                   scheduling_algorithm=topo_sort_dict[args.topo_sort_function])
