from queue import PriorityQueue

import networkx as nx
from gurobipy import Model

from optimizer.model.graph import CompGraph


def priority_queue_min_comp_cost(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                                 device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict):

    def initialize_queues(subgraph_dict, dependency_graph) -> dict[any, PriorityQueue]:
        # Initialize a queue for each device
        device_queue_dict = {device: PriorityQueue() for device, subgraph in subgraph_dict.items()}

        # Initialize with tasks that have no predecessors in the global graph
        for device, subgraph in subgraph_dict.items():
            for operator_id in subgraph.nodes():
                # Check if the node has no predecessors in the global dependency graph
                global_predecessors = list(dependency_graph.predecessors(operator_id))

                # If the node has no predecessors in the global graph, it can be added to the queue
                if not global_predecessors:
                    # Add to the appropriate subgraph's queue
                    device_queue_dict[device].put((comp_graph.getOperatorCompCostByDevice(operator_id, device), operator_id))

        return device_queue_dict

    def update_queue(device_queue_dict: dict[any, PriorityQueue], finished_task, dependency_graph, completed_tasks,
                     partition_dict):
        # Check all successors of the finished task in the global dependency graph
        successors = list(dependency_graph.successors(finished_task))
        for successor in successors:
            # Check if all predecessors are complete in the global dependency graph
            predecessors = list(dependency_graph.predecessors(successor))
            if all(predecessor in completed_tasks for predecessor in predecessors):
                # Enqueue the task to the task queue of this subgraph (device)
                device_of_successor = partition_dict[successor]
                # cannot use "if subgraph_of_succ" since subgraph id can be 0
                if device_of_successor is not None:
                    # Enqueue the task to the task queue of the correct subgraph (device)
                    device_queue_dict[device_of_successor].put((comp_graph.getOperatorCompCostByDevice(successor, device_of_successor), successor))

    # It is an SCHEDULING problem within each device.
    device_queues = initialize_queues(device_subgraph_mapping, comp_graph)
    total_items = sum(queue.qsize() for queue in device_queues.values())
    print("len of the init: ", total_items, 'The init device_queues is ', device_queues)

    # Initialize the set to track completed tasks
    completed_tasks = set()

    # This list will store all the constraints that we batch before optimization
    last_job_dict = {subgraph_id: None for subgraph_id in device_subgraph_mapping.keys()}
    last_communication_dict = {subgraph_id: None for subgraph_id in device_subgraph_mapping.keys()}
    # Process each subgraph independently
    while any(not pqueue.empty() for pqueue in device_queues.values()):
        for device_id, queue in device_queues.items():
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

                # Ensure that the task starts after the previous task finishes within the same subgraph
                # Operator scheduling within device
                if last_job_dict[device_id] is not None:
                    model.addConstr(start[task] >= finish[last_job_dict[device_id]], name=f"start_after_prev_finish_{task}_on_subgraph_{device_id}")

                # Communication scheduling. One device can only send or receive from up to one link at the same time
                for predecessor in comp_graph.predecessors(task):
                    if (predecessor, task) in edge_cut_list:
                        if last_communication_dict[device_id] is not None:
                            model.addConstr(comm_start[predecessor, task] >= comm_end[last_communication_dict[device_id]])
                        source_device = operator_device_mapping[predecessor]
                        last_communication_dict[source_device] = (predecessor, task)
                        last_communication_dict[device_id] = (predecessor, task)

                # Track the finish time of the current task
                last_job_dict[device_id] = task

                # Track task completion
                completed_tasks.add(task)

                # Update the queue based on the completion of the task
                update_queue(device_queues, task, comp_graph, completed_tasks, operator_device_mapping)

    # Get the collection of nodes that are in the graph but not in completed_tasks
    all_nodes = set(comp_graph.nodes())
    remaining_nodes = all_nodes - completed_tasks
    assert len(remaining_nodes) == 0, f"the remaining nodes {remaining_nodes} but all nodes should be scheduled"
