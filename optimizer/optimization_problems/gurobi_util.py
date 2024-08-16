from collections import deque
from functools import cmp_to_key
from itertools import combinations
from typing import Tuple

import networkx as nx
from gurobipy import *
from networkx import DiGraph

from py_util import compare_2d_list

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.computing_graph.op_graph_util import get_proper_optimizer
from optimizer.model.graph import DeviceGraph, CompGraph, has_more_than_one_component, keep_largest_component, \
    visualize_graph, find_non_connected_pairs
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum


def gurobi_setup(name: str):
    # Init solver
    model = Model(name)
    model.setParam("LogToConsole", 0)
    model.setParam("LogFile", "gurobi.log")
    model.setParam("MIPGap", 0.21)
    model.setParam("TimeLimit", 2400)
    model.setParam("MIPFocus", 1)

    # if this is too large, then the reformulated
    # ex-quadratic constraints can behave funky
    model.setParam("IntFeasTol", 1e-6)
    model.setParam("MemLimit", 4096)  # Example: Limit memory usage to 4 GB
    model.setParam("Threads", 4)  # Example: Use 4 threads

    return model


def init_computing_and_device_graph(num_device, filename: str, hetero_adjust_rate, model_type=TFModelEnum.SMALL) \
        -> Tuple[DeviceGraph, CompGraph]:
    # init device topo
    deviceTopo = DeviceGraph()
    deviceTopo.generata_fat_tree_topo(num_device, 30, 20, 1)

    if not os.path.exists(filename):
        model = model_type()
        optimizer = get_proper_optimizer(model)
        comp_graph = get_computation_graph(model=model, optimizer=optimizer)
        comp_graph.generata_random_cost(num_device, hetero_adjust_rate)
        comp_graph.save_to_file(filename)

    comp_graph = CompGraph.load_from_file(filename)
    comp_graph = keep_largest_component(comp_graph)
    # comp_graph.clean_marginal_operators()
    # visualize_graph(comp_graph, show_node_labels=False, show_edge_labels=False)
    return deviceTopo, comp_graph


def show_optimization_solution(model, x: dict, comp_graph: CompGraph, deviceTopo: DeviceGraph, start: dict,
                               finish: dict, graph_partition=False, two_dime_node_list=None):
    if graph_partition and not two_dime_node_list:
        raise ValueError("should has a 2d list to represent the original graph partition")
    # init result dict
    result = {'totalLatency': model.ObjVal, 'Assignment': {}, 'CommunicationCosts': [], "CommunicationTimeLine": {},
              "device_utility_rate": {}, "total_communication_time": None, "total_computing_time_per_device": {}}
    # populate result['Assignment']
    for key, value in x.items():
        # key[1] is the device id
        if key[1] not in result['Assignment']:
            result['Assignment'][key[1]] = []
        # key[0] is the operator id. Put id into the list assigned to the device
        if value.X > 0.5:
            # Assignment: {device: [(op1, start[op1], finish[op1]), (...)]}
            result['Assignment'][key[1]].append((key[0], start[key[0]].X, finish[key[0]].X))
    # Sort operators by their start times for each device
    for device, ops in result['Assignment'].items():
        result['Assignment'][device] = sorted(ops, key=lambda x: x[1])

    # populate result['CommunicationCosts'] and result['CommunicationTimeLine']
    for edge_id_tuple in list(comp_graph.getEdgeIDs()):
        source_op_ID, dest_op_ID = edge_id_tuple
        s_placement = None
        d_placement = None
        comm_cost_var = model.getVarByName(f"comm_cost[{source_op_ID},{dest_op_ID}]")
        comm_start_var = model.getVarByName(f"comm_start[{source_op_ID},{dest_op_ID}]")
        comm_end_var = model.getVarByName(f"comm_end[{source_op_ID},{dest_op_ID}]")
        if comm_cost_var and comm_start_var and comm_end_var:
            comm_cost = comm_cost_var.X
            comm_start_time = comm_start_var.X
            comm_end_time = comm_end_var.X
            if comm_cost == 0:
                continue
            tensor_size = comp_graph.getEdgeTensorSize(source_op_ID, dest_op_ID)
            for device, ops in result['Assignment'].items():
                if source_op_ID in [op[0] for op in ops]:
                    s_placement = device
                if dest_op_ID in [op[0] for op in ops]:
                    d_placement = device
                if s_placement and d_placement:
                    break
            # if both ops are placed on the same device, use 999 to represent that
            if s_placement == d_placement:
                bandwidth = 999
            else:
                bandwidth = deviceTopo.get_link_bandwidth(s_placement, d_placement)
            result['CommunicationCosts'].append(
                (source_op_ID, s_placement, dest_op_ID, d_placement, comm_cost, tensor_size, bandwidth))
            # Populate the communication timeline divided by device
            if s_placement not in result['CommunicationTimeLine']:
                result['CommunicationTimeLine'][s_placement] = []
            if d_placement not in result['CommunicationTimeLine']:
                result['CommunicationTimeLine'][d_placement] = []

            result['CommunicationTimeLine'][s_placement].append(
                (source_op_ID, dest_op_ID, comm_start_time, comm_end_time, comm_cost))
            result['CommunicationTimeLine'][d_placement].append(
                (source_op_ID, dest_op_ID, comm_start_time, comm_end_time, comm_cost))
    # Sort the communication timeline based on the starting time
    for device, timeline in result['CommunicationTimeLine'].items():
        result['CommunicationTimeLine'][device] = sorted(timeline, key=lambda x: x[2])

    # Print operator placement
    for device, op_info_tuples in result['Assignment'].items():
        sum_comp = 0
        print(f"Device: {device}")
        for op_tuple in op_info_tuples:
            comp_cost = 0  # Initialize computation cost for the current operator
            for device_id in deviceTopo.getDeviceIDs():
                comp_cost += x[op_tuple[0], device_id].X * comp_graph.getOperatorCompCostByDevice(op_tuple[0],
                                                                                                  device_id)
            sum_comp += comp_cost
            if comp_cost == 0:
                continue
            print(f"  Operator: {op_tuple[0]}, Start: {op_tuple[1]}, Finish: {op_tuple[2]}, Comp Cost: {comp_cost}")
        device_utility_rate = sum_comp / model.ObjVal
        result['total_computing_time_per_device'][device] = sum_comp
        result['device_utility_rate'][device] = device_utility_rate

    # Print communication costs
    print("Communication Costs:")
    sum_of_communication = 0
    for source_op_ID, s_placement, dest_op_ID, d_placement, comm_cost, tensor_size, bandwidth in result[
        'CommunicationCosts']:
        sum_of_communication += comm_cost
        print(
            f"  From {source_op_ID} with placement {s_placement} to {dest_op_ID} with placement {d_placement}, Cost: {comm_cost}, Tensor size: {tensor_size}, Bandwidth: {bandwidth} GB/s")
    result['total_communication_time'] = sum_of_communication

    # Print communication timeline divided by device
    print("Communication Timeline:")
    for device, timeline in result['CommunicationTimeLine'].items():
        print(f"Device: {device}")
        for source_op_ID, dest_op_ID, comm_start_time, comm_end_time, cost in timeline:
            print(
                f"  Communication from {source_op_ID} to {dest_op_ID} starts at {comm_start_time} and ends at {comm_end_time} with cost: {cost}")

    # Print Summary
    print('Runtime = ', "%.2f" % model.Runtime, 's', sep='')
    print('Expected Training time = ', model.ObjVal, 's', sep='')
    print("Device Utility Rate:", result['device_utility_rate'])
    print("Total Communication Time:", result['total_communication_time'])
    print("total_computing_time_per_device:", result['total_computing_time_per_device'])


def get_subgraph_op_num_weight_sum_dict(weighted_graph: CompGraph, partition_dict):
    # Count the number of nodes in each partition
    # Count the number of nodes and sum of weights in each partition
    partition_counts = {}
    partition_weights = {}
    for node, part in partition_dict.items():
        if part not in partition_counts:
            partition_counts[part] = 0
        if part not in partition_weights:
            partition_weights[part] = 0
        partition_counts[part] += 1
        partition_weights[part] += weighted_graph.nodes[node]['node_weight']
    return partition_counts, partition_weights


def show_graph_partition_info(weighted_graph: CompGraph, partition_dict, edge_cut_list, weight_cut_sum):
    partition_counts, partition_weights = get_subgraph_op_num_weight_sum_dict(weighted_graph, partition_dict)
    print("how many operators for each subgraph", partition_counts, "the sum of weights for each subgraph",
          partition_weights)
    print("edges cut during partition", edge_cut_list)
    print("sum of weight of edge cut", weight_cut_sum)


def get_subgraph_topo_dict(original_topo_list, partition_dict) -> dict[int, list]:
    subgraph_topo_dict = {}
    for node in original_topo_list:
        subgraph_id = partition_dict[node]
        if subgraph_id not in subgraph_topo_dict.keys():
            subgraph_topo_dict[subgraph_id] = []
        subgraph_topo_dict[subgraph_id].append(node)

    return subgraph_topo_dict


# Function to sort edges based on the topological order of the source node and then the destination node
def sort_edges_by_topo_order(edges, topo_order, sort_dest=False):
    def key_func(edge):
        if sort_dest:
            return topo_order[edge[0]], topo_order[edge[1]]
        else:
            return topo_order[edge[0]]

    return sorted(edges, key=key_func)


def sort_edges_by_subgraph_and_dependency(edges, topo_order):
    # Function to get the relevant node within the subgraph for sorting
    def relevant_node(edge):
        src, dest = edge
        if src in topo_order.keys():
            return src
        if dest in topo_order.keys():
            return dest
        ValueError("both src and dest are not in topo_order")

    # Function to compare two edges
    def compare_edges(edge1, edge2):
        node1 = relevant_node(edge1)
        node2 = relevant_node(edge2)

        # Compare based on topological order of the relevant nodes
        if topo_order[node1] != topo_order[node2]:
            return topo_order[node1] - topo_order[node2]

        # If the relevant nodes are the same, compare based on dependency
        src1, dest1 = edge1
        src2, dest2 = edge2

        # Check if the first edge should come before the second based on dependency
        if dest1 == node1 and src2 == node2:
            return -1
        if src1 == node1 and dest2 == node2:
            return 1

        # If none of the above conditions apply, consider them equal
        return 0

    # Sort the edges using the custom comparison function
    sorted_edges = sorted(edges, key=cmp_to_key(compare_edges))

    return sorted_edges


def FIFO_scheduling(model: Model, start, ready, finish, comp_graph, subgraph_dict, partition_dict):
    def initialize_queues(subgraph_dict, dependency_graph):
        # Initialize a queue for each subgraph (device)
        device_queues = {subgraph_id: deque() for subgraph_id, subgraph in subgraph_dict.items()}

        # Initialize with tasks that have no predecessors in the global graph
        for subgraph_id, subgraph in subgraph_dict.items():
            for operator_id in subgraph.nodes():
                # Check if the node has no predecessors in the global dependency graph
                global_predecessors = list(dependency_graph.predecessors(operator_id))

                # If the node has no predecessors in the global graph, it can be added to the queue
                if not global_predecessors:
                    # Add to the appropriate subgraph's queue
                    device_queues[subgraph_id].append(operator_id)

        return device_queues

    def update_queue(device_queues, finished_task, dependency_graph, completed_tasks, partition_dict):
        # Check all successors of the finished task in the global dependency graph
        successors = list(dependency_graph.successors(finished_task))
        for succ in successors:
            # Check if all predecessors are complete in the global dependency graph
            predecessors = list(dependency_graph.predecessors(succ))
            if all(predecessor in completed_tasks for predecessor in predecessors):
                # Enqueue the task to the task queue of this subgraph (device)
                subgraph_of_succ = partition_dict[succ]
                print(
                    f"succ {succ} belongs to {subgraph_of_succ} while the current graph is {partition_dict[finished_task]}")
                # cannot use "if subgraph_of_succ" since subgraph id can be 0
                if subgraph_of_succ is not None:
                    # Enqueue the task to the task queue of the correct subgraph (device)
                    device_queues[subgraph_of_succ].append(succ)

    # It is an SCHEDULING problem within each device.
    device_queues = initialize_queues(subgraph_dict, comp_graph)
    total_items = sum(len(queue) for queue in device_queues.values())
    print("len of the init: ", total_items, 'The init device_queues is ', device_queues)

    # Initialize the set to track completed tasks
    completed_tasks = set()

    # This list will store all the constraints that we batch before optimization
    last_job_dict = {subgraph_id: None for subgraph_id in subgraph_dict.keys()}
    # Process each subgraph independently
    while any(queue for queue in device_queues.values()):
        for subgraph_id, queue in device_queues.items():
            if queue:
                # Get the next task to execute for this subgraph
                task = queue.popleft()

                # check if this task get completed
                if task in completed_tasks:
                    raise ValueError("this is a repeated task")

                # check if all dependency satisfy
                for predecessor in nx.ancestors(comp_graph, task):
                    if predecessor not in completed_tasks:
                        raise ValueError(f"{task} 's dependency {predecessor} not satisfied")

                # Ensure the task starts after its ready time
                model.addConstr(start[task] >= ready[task],
                                name=f"start_after_ready_{task}_on_subgraph_{subgraph_id}")

                # Ensure that the task starts after the previous task finishes within the same subgraph
                if last_job_dict[subgraph_id] is not None:
                    model.addConstr(start[task] >= finish[last_job_dict[subgraph_id]],
                                    name=f"start_after_prev_finish_{task}_on_subgraph_{subgraph_id}")

                # Track the finish time of the current task
                last_job_dict[subgraph_id] = task
                print("the current subgraph is", subgraph_id, "the new last job is ", last_job_dict[subgraph_id])

                # Track task completion
                completed_tasks.add(task)

                # Update the queue based on the completion of the task
                update_queue(device_queues, task, comp_graph, completed_tasks, partition_dict)

    # Get the collection of nodes that are in the graph but not in completed_tasks
    all_nodes = set(comp_graph.nodes())
    remaining_nodes = all_nodes - completed_tasks
    assert len(remaining_nodes) == 0, f"the remaining nodes {remaining_nodes} but all nodes should be scheduled"


def optimal_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph, subgraph_dict, edge_cut_list):
    for source_op_ID, dest_op_ID in comp_graph.getEdgeIDs():
        model.addConstr(finish[source_op_ID] <= start[dest_op_ID])
    M = 100000
    order = {}
    for subgraph in subgraph_dict.values():
        non_connected_pairs = find_non_connected_pairs(subgraph)
        for op_a, op_b in non_connected_pairs:
            order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"order_{op_a}_{op_b}")
            model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]), name=f"NoOverlap1_{op_a}_{op_b}")
            model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")

    # Add constraint to ensure each device can only send or receive from one link at a time, communication scheduling
    # Only edges in the edge_cut_list will bring communication cost
    order_link = {}
    for communication_a, communication_b in combinations(edge_cut_list, 2):
        order_link[communication_a, communication_b] = model.addVar(vtype=GRB.BINARY)
        model.addConstr(comm_start[communication_b] >= comm_end[communication_a] - M * (1 - order_link[communication_a, communication_b]))
        model.addConstr(comm_start[communication_a] >= comm_end[communication_b] - M * order_link[communication_a, communication_b])