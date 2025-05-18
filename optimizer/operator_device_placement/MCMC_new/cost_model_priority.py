import heapq

import networkx as nx

from optimizer.model.graph import CompGraph


class Task:
    def __init__(self, name, device, compute_cost, priority):
        self.name = name
        self.device = device
        self.compute_cost = compute_cost
        self.priority = priority
        self.state = 'NOTREADY'
        self.ready_time = 0
        self.start_time = None
        self.end_time = None
        self.predecessors = set()
        self.successors = set()

    # Max heap; a task with the least priority will be at the end of the queue
    def __lt__(self, other):
        return self.priority > other.priority

def evaluate_mcmc_priority(G: CompGraph, device_ops_mapping, compute_costs, comm_costs, op_priority_map):
    devices = list(device_ops_mapping.keys())

    # Build Task objects and assign devices
    tasks = {}
    for op in compute_costs:
        for d, ops in device_ops_mapping.items():
            if op in ops:
                tasks[op] = Task(op, d, compute_costs[op], op_priority_map[op])
                break

    # Build edges (predecessors and successors)
    # add references not creating no tasks objs
    for u, v in G.getEdgeIDs():
        tasks[u].successors.add(tasks[v])
        tasks[v].predecessors.add(tasks[u])

    # Track per-device end time
    device_last_end_time = {d: 0 for d in devices}

    # Initialize ready queue
    '''
    ready_queue is sorted by ready_time of each task
    '''
    ready_queue = []
    for d in devices:
        for op in device_ops_mapping[d]:
            task = tasks[op]
            if not task.predecessors:
                task.state = 'READY'
                heapq.heappush(ready_queue, task)

    # Main simulation loop
    while ready_queue:
        task = heapq.heappop(ready_queue)
        device = task.device

        # Execute task
        task.start_time = max(task.ready_time, device_last_end_time[device])
        task.end_time = task.start_time + task.compute_cost
        task.state = 'COMPLETE'
        device_last_end_time[device] = task.end_time


        # Check all successors
        for succ in task.successors:
            if succ.state != 'NOTREADY':
                raise ValueError('Not topology order')

            if all(pred.state == 'COMPLETE' for pred in succ.predecessors):
                succ.ready_time = max(
                    pred.end_time + comm_costs.get((pred.name, succ.name))
                    if pred.device != succ.device else pred.end_time
                    for pred in succ.predecessors
                )
                succ.state = 'READY'
                heapq.heappush(ready_queue, succ)

    # Return total simulation time
    latency = max(task.end_time for task in tasks.values())
    return latency