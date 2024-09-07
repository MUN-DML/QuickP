from collections import deque

import networkx as nx
from gurobipy import Model

from optimizer.model.graph import CompGraph


def FIFO_scheduling(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                    device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict):
    def initialize_queues(subgraph_dict, dependency_graph):
        # Initialize a queue for each subgraph (device)
        device_queue_dict = {device: deque() for device, subgraph in subgraph_dict.items()}

        # Initialize with tasks that have no predecessors in the global graph
        for subgraph_id, subgraph in subgraph_dict.items():
            for operator_id in subgraph.nodes():
                # Check if the node has no predecessors in the global dependency graph
                global_predecessors = list(dependency_graph.predecessors(operator_id))

                # If the node has no predecessors in the global graph, it can be added to the queue
                if not global_predecessors:
                    # Add to the appropriate subgraph's queue
                    device_queue_dict[subgraph_id].append(operator_id)

        return device_queue_dict

    # After an operator (task) on this device has finished
    def handle_operator_completion(device_communication_queue, current_op, comp_graph, operator_device_mapping, edge_cut_list):
        # Check if there are any communications after this task is finished
        for successor in comp_graph.successors(current_op):
            # Check if the communication is cross-device
            if (current_op, successor) in edge_cut_list:
                source_device = operator_device_mapping[current_op]
                target_device = operator_device_mapping[successor]
                assert source_device != target_device

                # Append the communication to the source device's queue
                device_communication_queue[source_device].append((current_op, successor))
                print(f"Queued communication from {current_op} to {successor} on {source_device}")

    def update_queue(device_queues, finished_task, dependency_graph, completed_tasks, partition_dict, current_device_communication_order):
        # Check all successors of the finished task in the global dependency graph
        successors = list(dependency_graph.successors(finished_task))
        for succ in successors:
            # Check if all predecessors are complete in the global dependency graph
            predecessors = list(dependency_graph.predecessors(succ))
            # Ensure all cross-device communications for this task are completed
            # Check if all predecessor tasks are completed
            if not all(pred in completed_tasks for pred in predecessors):
                continue  # Skip this successor, as not all predecessors are done

            # Check if all required cross-device communications are complete
            for pred in predecessors:
                if (pred, succ) in edge_cut_list:
                    source_device = operator_device_mapping[pred]
                    target_device = operator_device_mapping[succ]

                    # Cross-device communication: check if it has completed
                    if source_device != target_device:
                        # Communication from pred to succ must be completed
                        if (pred, succ) in communication_queue_by_device[source_device] or last_communication_dict[
                            source_device] != (pred, succ):
                            continue  # Skip this successor if communication isn't done

            # If all predecessors and communications are complete, enqueue the successor task
            succ_device = operator_device_mapping[succ]
            device_queues[succ_device].append(succ)

            if all(predecessor in completed_tasks for predecessor in predecessors) and all_communications_done:
                # Enqueue the task to the task queue of this subgraph (device)
                subgraph_of_succ = partition_dict[succ]
                # cannot use "if subgraph_of_succ" since subgraph id can be 0
                if subgraph_of_succ is not None:
                    # Enqueue the task to the task queue of the correct subgraph (device)
                    device_queues[subgraph_of_succ].append(succ)

    # It is an SCHEDULING problem within each device.
    device_queues = initialize_queues(device_subgraph_mapping, comp_graph)
    total_items = sum(len(queue) for queue in device_queues.values())
    print("len of the init: ", total_items, 'The init device_queues is ', device_queues)

    # Initialize the set to track completed tasks
    completed_tasks = set()

    # This list will store all the constraints that we batch before optimization
    last_job_dict = {subgraph_id: None for subgraph_id in device_subgraph_mapping.keys()}
    last_communication_dict = {subgraph_id: None for subgraph_id in device_subgraph_mapping.keys()}
    communication_queue_by_device = {device: deque() for device in device_subgraph_mapping.keys()}
    device_communication_order = {device: [] for device in device_subgraph_mapping.keys()}

    # Process each subgraph independently
    while any(queue for queue in device_queues.values()):
        for current_device, queue in device_queues.items():
            if queue:
                # Get the next task to execute for this subgraph
                current_op = queue.popleft()

                # check if this task get completed
                if current_op in completed_tasks:
                    raise ValueError("this is a repeated task")

                # check if all dependency satisfy
                for predecessor in nx.ancestors(comp_graph, current_op):
                    if predecessor not in completed_tasks:
                        raise ValueError(f"{current_op} 's dependency {predecessor} not satisfied")

                # Ensure that the task starts after the previous task finishes within the same subgraph
                # Operator scheduling within device
                if last_job_dict[current_device] is not None:
                    model.addConstr(start[current_op] >= finish[last_job_dict[current_device]], name=f"start_after_prev_finish_{current_op}_on_subgraph_{current_device}")

                # Communication scheduling. One device can only send to up to one link at the same time
                for predecessor in comp_graph.predecessors(current_op):
                    # in edge_cut_list => there exists a cross-device communication
                    if (predecessor, current_op) in edge_cut_list:
                        source_device = operator_device_mapping[predecessor]
                        assert source_device != current_device
                        communication_queue_by_device[source_device].append((predecessor, current_op))

                        if last_communication_dict[source_device] is not None:
                            model.addConstr(comm_start[predecessor, current_op] >= comm_end[last_communication_dict[source_device]])
                        last_communication_dict[source_device] = (predecessor, current_op)
                        device_communication_order[source_device].append((predecessor, current_op))

                # Track the finish time of the current task
                last_job_dict[current_device] = current_op

                # Track task completion
                completed_tasks.add(current_op)

                # Update the queue based on the completion of the task
                update_queue(device_queues, current_op, comp_graph, completed_tasks, operator_device_mapping)

    # Check all nodes are scheduled
    all_nodes = set(comp_graph.nodes())
    remaining_nodes = all_nodes - completed_tasks
    assert len(remaining_nodes) == 0, f"the remaining nodes {remaining_nodes} but all nodes should be scheduled"

    # Check all communications are scheduled
    for (predecessor, successor) in edge_cut_list:
        source_device = operator_device_mapping[predecessor]
        target_device = operator_device_mapping[successor]
        assert source_device != target_device
        assert (predecessor, successor) in device_communication_order[source_device]
