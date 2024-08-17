from collections import deque
from enum import Enum
from itertools import combinations
from queue import PriorityQueue

import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import find_non_connected_pairs, CompGraph


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


def FIFO_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph, subgraph_dict: dict,
                    edge_cut_list: list, partition_dict: dict):
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
                if subgraph_of_succ != partition_dict[finished_task]:
                    print(f"succ {succ} belongs to {subgraph_of_succ} while the current graph is {partition_dict[finished_task]}")
                # cannot use "if subgraph_of_succ" since subgraph id can be 0
                if subgraph_of_succ is not None:
                    # Enqueue the task to the task queue of the correct subgraph (device)
                    device_queues[subgraph_of_succ].append(succ)

    ready = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="ready")  # ready[node_id] represent the ready time of this node, simulating Queue

    for node in comp_graph.nodes():
        for predecessor in comp_graph.predecessors(node):
            model.addConstr(ready[node] >= finish[predecessor], name=f"fifo_{predecessor}_to_{node}")

    # It is an SCHEDULING problem within each device.
    device_queues = initialize_queues(subgraph_dict, comp_graph)
    total_items = sum(len(queue) for queue in device_queues.values())
    print("len of the init: ", total_items, 'The init device_queues is ', device_queues)

    # Initialize the set to track completed tasks
    completed_tasks = set()

    # This list will store all the constraints that we batch before optimization
    last_job_dict = {subgraph_id: None for subgraph_id in subgraph_dict.keys()}
    last_communication_dict = {subgraph_id: None for subgraph_id in subgraph_dict.keys()}
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
                # Operator scheduling within device
                if last_job_dict[subgraph_id] is not None:
                    model.addConstr(start[task] >= finish[last_job_dict[subgraph_id]],
                                    name=f"start_after_prev_finish_{task}_on_subgraph_{subgraph_id}")

                # Communication Scheduling. One device can only send or receive from up to one link at the same time
                for predecessor in comp_graph.predecessors(task):
                    if (predecessor, task) in edge_cut_list:
                        if last_communication_dict[subgraph_id] is not None:
                            model.addConstr(comm_start[predecessor, task] >= comm_end[last_communication_dict[subgraph_id]])
                        source_subgraph = partition_dict[predecessor]
                        last_communication_dict[source_subgraph] = (predecessor, task)
                        last_communication_dict[subgraph_id] = (predecessor, task)

                # Track the finish time of the current task
                last_job_dict[subgraph_id] = task

                # Track task completion
                completed_tasks.add(task)

                # Update the queue based on the completion of the task
                update_queue(device_queues, task, comp_graph, completed_tasks, partition_dict)

    # Get the collection of nodes that are in the graph but not in completed_tasks
    all_nodes = set(comp_graph.nodes())
    remaining_nodes = all_nodes - completed_tasks
    assert len(remaining_nodes) == 0, f"the remaining nodes {remaining_nodes} but all nodes should be scheduled"


def priority_queue_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                              subgraph_dict: dict,
                              edge_cut_list: list, partition_dict: dict):
    def initialize_queues(subgraph_dict, dependency_graph) -> dict[any, PriorityQueue]:
        # Initialize a queue for each subgraph (device)
        device_queues = {subgraph_id: PriorityQueue() for subgraph_id, subgraph in subgraph_dict.items()}

        # Initialize with tasks that have no predecessors in the global graph
        for subgraph_id, subgraph in subgraph_dict.items():
            for operator_id in subgraph.nodes():
                # Check if the node has no predecessors in the global dependency graph
                global_predecessors = list(dependency_graph.predecessors(operator_id))

                # If the node has no predecessors in the global graph, it can be added to the queue
                if not global_predecessors:
                    # Add to the appropriate subgraph's queue
                    device_queues[subgraph_id].put((1, operator_id))

        return device_queues

    def update_queue(device_queues: dict[any, PriorityQueue], finished_task, dependency_graph, completed_tasks,
                     partition_dict):
        # Check all successors of the finished task in the global dependency graph
        successors = list(dependency_graph.successors(finished_task))
        for succ in successors:
            # Check if all predecessors are complete in the global dependency graph
            predecessors = list(dependency_graph.predecessors(succ))
            if all(predecessor in completed_tasks for predecessor in predecessors):
                # Enqueue the task to the task queue of this subgraph (device)
                subgraph_of_succ = partition_dict[succ]
                if subgraph_of_succ != partition_dict[finished_task]:
                    print(
                        f"succ {succ} belongs to {subgraph_of_succ} while the current graph is {partition_dict[finished_task]}")
                # cannot use "if subgraph_of_succ" since subgraph id can be 0
                if subgraph_of_succ is not None:
                    # Enqueue the task to the task queue of the correct subgraph (device)
                    device_queues[subgraph_of_succ].put((1, succ))

    ready = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="ready")  # ready[node_id] represent the ready time of this node, simulating Queue

    for node in comp_graph.nodes():
        for predecessor in comp_graph.predecessors(node):
            model.addConstr(ready[node] >= finish[predecessor], name=f"fifo_{predecessor}_to_{node}")

    # It is an SCHEDULING problem within each device.
    device_queues = initialize_queues(subgraph_dict, comp_graph)
    total_items = sum(queue.qsize() for queue in device_queues.values())
    print("len of the init: ", total_items, 'The init device_queues is ', device_queues)

    # Initialize the set to track completed tasks
    completed_tasks = set()

    # This list will store all the constraints that we batch before optimization
    last_job_dict = {subgraph_id: None for subgraph_id in subgraph_dict.keys()}
    last_communication_dict = {subgraph_id: None for subgraph_id in subgraph_dict.keys()}
    # Process each subgraph independently
    while any(not pqueue.empty() for pqueue in device_queues.values()):
        for subgraph_id, queue in device_queues.items():
            if not queue.empty():
                # Get the next task to execute for this subgraph
                _, task = queue.get()

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
                # Operator scheduling within device
                if last_job_dict[subgraph_id] is not None:
                    model.addConstr(start[task] >= finish[last_job_dict[subgraph_id]],
                                    name=f"start_after_prev_finish_{task}_on_subgraph_{subgraph_id}")

                # Communication Scheduling. One device can only send or receive from up to one link at the same time
                for predecessor in comp_graph.predecessors(task):
                    if (predecessor, task) in edge_cut_list:
                        if last_communication_dict[subgraph_id] is not None:
                            model.addConstr(comm_start[predecessor, task] >= comm_end[last_communication_dict[subgraph_id]])
                        source_subgraph = partition_dict[predecessor]
                        last_communication_dict[source_subgraph] = (predecessor, task)
                        last_communication_dict[subgraph_id] = (predecessor, task)

                # Track the finish time of the current task
                last_job_dict[subgraph_id] = task

                # Track task completion
                completed_tasks.add(task)

                # Update the queue based on the completion of the task
                update_queue(device_queues, task, comp_graph, completed_tasks, partition_dict)

    # Get the collection of nodes that are in the graph but not in completed_tasks
    all_nodes = set(comp_graph.nodes())
    remaining_nodes = all_nodes - completed_tasks
    assert len(remaining_nodes) == 0, f"the remaining nodes {remaining_nodes} but all nodes should be scheduled"


class SchedulingAlgorithm(Enum):
    FIFO = "FIFO"
    OPTIMIZED = "OPTIMIZED"


def execute_scheduling_function(sch_fun_type: str, model: Model, **kwargs):
    if sch_fun_type == SchedulingAlgorithm.FIFO.value:
        # Explicitly select the arguments required by FIFO_scheduling
        fifo_kwargs = {
            key: kwargs[key] for key in
            ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph', 'subgraph_dict', 'edge_cut_list',
             'partition_dict'] if key in kwargs
        }
        return FIFO_scheduling(model, **fifo_kwargs)

    elif sch_fun_type == SchedulingAlgorithm.OPTIMIZED.value:
        # Explicitly select the arguments required by optimal_scheduling
        optimized_kwargs = {
            key: kwargs[key] for key in
            ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph', 'subgraph_dict', 'edge_cut_list'] if
            key in kwargs
        }
        return optimal_scheduling(model, **optimized_kwargs)

    else:
        raise ValueError(f"Unknown scheduling algorithm: {sch_fun_type}")
