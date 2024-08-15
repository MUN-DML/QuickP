# python3 after_graph_partition_hetero.py
import argparse
from itertools import combinations

from gurobipy import *
import torch
import tensorflow as tf

os.environ['GRB_LICENSE_FILE'] = '/home/hola/solverLicense/gurobi.lic'

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.optimization_problems.scheduling_algorithm import TopoSortFunction
from optimizer.graph_partitioner.metis_partition import metis_partition
from optimizer.graph_partitioner.subgraph_util import construct_sub_graph, WeightNormalizationFunction, \
    map_subgraph_to_device
from optimizer.optimization_problems.gurobi_util import init_computing_and_device_graph, gurobi_setup, \
    show_optimization_solution, show_graph_partition_info, initialize_queues, update_queue
from optimizer.graph_partitioner.weight_functions import NodeWeightFunction, EdgeWeightFunction
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum
from optimizer.model.graph import find_non_connected_pairs, label_node_levels
from optimizer.weight_adjustment_before_partition.weight_adjustment_function import WeightAdjustMatrix, \
    WeightAdjustmentFunction


def optimize_after_graph_partition(number_of_devices=2, model_type: TFModelEnum = TFModelEnum.SMALL,
                                   node_weight_function=NodeWeightFunction.AVE_COMP_COST,
                                   edge_weight_function=EdgeWeightFunction.MOCK_COMMUNICATION_COST_WITH_COMP,
                                   adjust_matrix: WeightAdjustMatrix=None,
                                   weight_norm_function=WeightNormalizationFunction.MIN_MAX):
    # init fake data
    deviceTopo, comp_graph = init_computing_and_device_graph(number_of_devices, "comp_graph_after_partition.json",
                                                             None, model_type=model_type)
    # Init solver
    model = gurobi_setup("minimize_maxload")

    # Partition the computation graph
    partition_dict, edge_cut_list, edge_cut_weight_sum = metis_partition(comp_graph,
                                                                         num_partitions=number_of_devices,
                                                                         node_weight_function=node_weight_function,
                                                                         edge_weight_function=edge_weight_function,
                                                                         adjust_matrix=adjust_matrix,
                                                                         weight_normalize=weight_norm_function,
                                                                         sub_graph_weight_sum_ratio=None)
    subgraph_dict = construct_sub_graph(comp_graph, partition_dict)

    # Update the op_id-subgraph_id mapping dict to op_id-device_id mapping dict
    operator_device_dict = map_subgraph_to_device(partition_dict, deviceTopo.getDeviceIDs())

    # two_dime_node_list is to test whether the
    two_dime_node_list: list[list] = [list(subgraph.nodes.keys()) for subgraph in subgraph_dict.values()]

    # Define variables
    x = model.addVars(comp_graph.getOperatorIDs(), deviceTopo.getDeviceIDs(), vtype=GRB.BINARY,
                      name="x")  # [operator_id, device_id] == 1 means this operator is assigned to this device
    start = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="start")  # start[node_id] represent the starting time of this node
    finish = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # finish[node_id] represent the finish time of this node
    ready = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                           name="finish")  # ready[node_id] represent the ready time of this node, simulating Queue
    comm_start = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0,
                               name="comm_start")  # comm_start[source_op, dest_op] represent the communication
    comm_end = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0, name="comm_end")
    comm_cost = model.addVars(edge_cut_list, vtype=GRB.CONTINUOUS, lb=0.0, name="comm_cost")

    '''
    Define Constraints
    '''

    # If we assume a homogeneous environment where each operator has the same time consumption on each device and the
    # bandwidth is also the same. Once we get the graph partition, the device-operation placement is already solved
    # because it does not matter where each sub-graph is placed.
    for op_id in comp_graph.getOperatorIDs():
        for device_id in deviceTopo.getDeviceIDs():
            if device_id == operator_device_dict[op_id]:
                model.addConstr(x[op_id, device_id] == 1)
            else:
                model.addConstr(x[op_id, device_id] == 0)

    # Data dependency
    for source_op_ID, dest_op_ID in comp_graph.getEdgeIDs():
        model.addConstr(finish[source_op_ID] <= start[dest_op_ID])

    for node_id in comp_graph.getOperatorIDs():
        # Add constraints that each op's ending time = starting time + its computing time
        assigned_device = operator_device_dict[node_id]
        comp_cost = comp_graph.getOperatorCompCostByDevice(node_id, assigned_device)
        model.addConstr(finish[node_id] == start[node_id] + comp_cost, name=f"finish_start_{node_id}")

    for edge_id_tuple in edge_cut_list:
        # only the edge in the edge_cut_list will bring communication cost since the source_op and destination-op are
        # placed on different devices
        source_op_ID, dest_op_ID = edge_id_tuple
        # Aggregate communication cost
        communication_cost = comp_graph.getEdgeTensorSize(source_op_ID, dest_op_ID) * deviceTopo.calUnitCommCostInUS(
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
        model.addConstr(comm_cost[source_op_ID, dest_op_ID] == communication_cost,
                        f"comm_cost_{source_op_ID}_{dest_op_ID}")

    # It is an SCHEDULING problem within each device.
    device_queues = initialize_queues(subgraph_dict)

    # Initialize the set to track completed tasks
    completed_tasks = set()

    # This list will store all the constraints that we batch before optimization
    last_finish_time = {subgraph_id: None for subgraph_id in range(len(subgraph_dict))}
    # Process each subgraph independently
    while any(queue for queue in device_queues.values()):
        for subgraph_id, queue in device_queues.items():
            if queue:
                # Get the next task to execute for this subgraph
                task = queue.popleft()

                # Ensure the task starts after its ready time
                model.addConstr(start[task] >= ready[task],
                                                   name=f"start_after_ready_{task}_on_subgraph_{subgraph_id}")

                # Ensure that the task starts after the previous task finishes within the same subgraph
                if last_finish_time[subgraph_id] is not None:
                    model.addConstr(start[task] >= last_finish_time[subgraph_id],
                                                       name=f"start_after_prev_finish_{task}_on_subgraph_{subgraph_id}")

                # Track the finish time of the current task
                last_finish_time[subgraph_id] = finish[task]

                # Track task completion
                completed_tasks.add(task)

                # Update the queue based on the completion of the task
                update_queue(queue, task, comp_graph, subgraph_dict[subgraph_id], completed_tasks)

    # Add constraint to ensure each device can only send or receive from one link at a time, communication scheduling
    # Only edges in the edge_cut_list will bring communication cost
    M = 100000
    order_link = {}
    for communication_a, communication_b in combinations(edge_cut_list, 2):
        order_link[communication_a, communication_b] = model.addVar(vtype=GRB.BINARY)
        model.addConstr(comm_start[communication_b] >= comm_end[communication_a] - M * (1 - order_link[communication_a, communication_b]))
        model.addConstr(comm_start[communication_a] >= comm_end[communication_b] - M * order_link[communication_a, communication_b])

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
        show_graph_partition_info(comp_graph, partition_dict, edge_cut_list, edge_cut_weight_sum)
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
                                   adjust_matrix={"function_type": WeightAdjustmentFunction.Recursive_Increase, "node_enable": True, "edge_enable": False, 'adjustment_ratio': 0},
                                   weight_norm_function=weight_normalization_dict[args.normalization_function])
