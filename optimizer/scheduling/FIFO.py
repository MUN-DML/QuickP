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

    def process_communication_queue(source_device, model, comm_start, comm_end, completed_communications_by_device,
                                    pending_comm_queue_by_device):
        # Process communication in FIFO order for the source device
        if len(pending_comm_queue_by_device[source_device]) > 0:
            # Dequeue the first communication (FIFO)
            predecessor, current_op = pending_comm_queue_by_device[source_device].popleft()

            # Add Gurobi constraints or simulate the start and end times for the communication
            if last_communication_dict[source_device] is not None:
                model.addConstr(comm_start[predecessor, current_op] >= comm_end[last_communication_dict[source_device]])

            # Mark this communication as completed
            last_communication_dict[source_device] = (predecessor, current_op)
            completed_communications_by_device[source_device].add((predecessor, current_op))

            # Execute or simulate communication start and end times as necessary
            print(f"Completed communication from {predecessor} to {current_op} on {source_device}")

    def update_queue(device_op_queues, current_task, dependency_graph, finished_tasks, op_d_mapping, pending_comm_queue_by_device, completed_comm_by_device):
        # Track task completion
        finished_tasks.add(current_task)

        # Check all successors of the finished task in the global dependency graph
        successors = list(dependency_graph.successors(current_task))
        for succ in successors:
            # Check if all predecessors are complete in the global dependency graph
            predecessors = list(dependency_graph.predecessors(succ))

            # adding pending communications to each device
            if (current_task, successor) in edge_cut_list:
                source_device = operator_device_mapping[current_op]
                target_device = operator_device_mapping[successor]
                assert source_device != target_device

                # Append the communication to the source device's queue
                pending_comm_queue_by_device[source_device].append((current_task, successor))
                print(f"Queued communication from {current_task} to {successor} on {source_device}")

            # adding pending operators to each device
            # Check if all predecessor tasks are completed
            if not all(pred in finished_tasks for pred in predecessors):
                continue  # Skip this successor, as not all predecessors are done
            # Check if all required cross-device communications are complete
            for pred in predecessors:
                if (pred, succ) in edge_cut_list:
                    source_device = op_d_mapping[pred]
                    target_device = op_d_mapping[succ]

                    # Cross-device communication: check if it has completed
                    if source_device != target_device:
                        # Communication from pred to succ must be completed
                        if ((pred, succ) in pending_comm_queue_by_device[source_device]
                                or (pred, succ) not in completed_comm_by_device[source_device]):
                            continue  # Skip this successor if communication isn't done

            # If all predecessors and communications are complete, enqueue the successor task
            succ_device = op_d_mapping[succ]
            device_op_queues[succ_device].append(succ)

    # It is an SCHEDULING problem within each device.
    device_queues = initialize_queues(device_subgraph_mapping, comp_graph)
    total_items = sum(len(queue) for queue in device_queues.values())
    print("len of the init: ", total_items, 'The init device_queues is ', device_queues)

    # Initialize the set to track completed tasks
    completed_tasks = set()

    # operator execution
    last_job_dict = {subgraph_id: None for subgraph_id in device_subgraph_mapping.keys()}
    device_node_order = {device: [] for device in device_subgraph_mapping.keys()}

    # Communication
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

                # check if all cross communication done
                for predecessor in nx.ancestors(comp_graph, current_op):
                    if (predecessor, current_op) in edge_cut_list:
                        source_device = operator_device_mapping[predecessor]
                        assert (predecessor, current_op) in device_communication_order[source_device]

                # Ensure that the task starts after the previous task finishes within the same subgraph
                # Operator scheduling within device
                if last_job_dict[current_device] is not None:
                    model.addConstr(start[current_op] >= finish[last_job_dict[current_device]], name=f"start_after_prev_finish_{current_op}_on_subgraph_{current_device}")

                # Track the finish time of the current task
                last_job_dict[current_device] = current_op

                # Update the queue based on the completion of the task
                update_queue(device_queues, current_op, comp_graph, completed_tasks, operator_device_mapping, communication_queue_by_device, device_communication_order)
                # Communication execution: Process communications independently in parallel

        process_communication_queue(model, comm_start, comm_end, device_communication_order, communication_queue_by_device)

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
