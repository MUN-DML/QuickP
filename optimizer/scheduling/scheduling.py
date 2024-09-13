import itertools
from enum import Enum

import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import find_non_connected_pairs, CompGraph, is_not_connected
from optimizer.scheduling.FIFO import FIFO_scheduling
from optimizer.scheduling.optimal import optimal_scheduling
from optimizer.scheduling.near_optimal_scheduling_simple import near_optimal_scheduling
from optimizer.scheduling.near_optimal_scheduling_with_sampling import near_optimal_scheduling_with_sampling
from optimizer.scheduling.priority_heteroG import priority_queue_max_rank_heteroG
from optimizer.scheduling.priority_min_comp_cost import priority_queue_min_comp_cost


def add_topo_order_constraints(model, original_topo_list, x, device_ids, finish, start):
    M = 1000000
    # Iterate over topologically sorted nodes
    for a, b in itertools.combinations(original_topo_list, 2):
        # For each consecutive pair of operators, add a constraint for each device
        for device_id in device_ids:
            # Ensure the correct order for each potential device assignment
            # This constraint will only apply if both a and b are assigned to the same device
            model.addConstr(finish[a] <= start[b] + M * (2 - x[a, device_id] - x[b, device_id]),
                            name=f"bigM_topo_order_{a}_{b}_on_device_{device_id}")


def FIFO_scheduling_solver(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                           device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict):
    M = 1000000
    order = {}
    ready = model.addVars(comp_graph.getOperatorIDs(), vtype=GRB.CONTINUOUS, lb=0.0,
                          name="ready")  # ready[node_id] represent the ready time of this node, simulating Queue

    for node in comp_graph.nodes():
        for predecessor in comp_graph.predecessors(node):
            model.addConstr(ready[node] >= finish[predecessor], name=f"fifo_{predecessor}_to_{node}")

    for subgraph in device_subgraph_mapping.values():
        non_connected_pairs = find_non_connected_pairs(subgraph)
        for op_a, op_b in non_connected_pairs:
            order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"Order_{op_a}_{op_b}")
            # Constraint 1: If task_i is ready before or at the same time as task_j (order[task_i, task_j] = 1)
            model.addConstr(ready[op_a] - ready[op_b] <= M * (1 - order[op_a, op_b]),
                            name=f"BigM_Constraint1_{op_a}_{op_b}")

            # Constraint 2: If task_j is ready before task_i (order[task_i, task_j] = 0)
            model.addConstr(ready[op_b] - ready[op_a] <= M * order[op_a, op_b], name=f"BigM_Constraint2_{op_a}_{op_b}")

            model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]), name=f"NoOverlap1_{op_a}_{op_b}")
            model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")

    for device, subgraph in device_subgraph_mapping.items():
        outgoings = [edge for edge in edge_cut_list if edge[0] in subgraph]
        topo_order = list(nx.topological_sort(subgraph))
        # Create a mapping of nodes to their topological order position
        topo_order_map = {node: index for index, node in enumerate(topo_order)}
        # Sort the edges based on the topological order of the source nodes
        sorted_outgoings = sorted(outgoings, key=lambda edge: topo_order_map[edge[0]])
        for comm1, comm2 in zip(sorted_outgoings, sorted_outgoings[1:]):
            source_node_1 = comm1[0]
            source_node_2 = comm2[0]
            # in this case, these two nodes does not have dependency, implement FCFS policy
            if is_not_connected(subgraph, source_node_1, source_node_2):
                order_1_first = model.addVar(vtype=GRB.BINARY, name=f"order_{source_node_1}_first_{source_node_2}")
                # Enforce the order based on the finish times using Big M
                # If finish[source_node_1] <= finish[source_node_2], set order_1_first = 1
                model.addConstr(finish[source_node_1] - finish[source_node_2] <= M * (1 - order_1_first),
                                name=f"order_decision_1_{source_node_1}_{source_node_2}")

                # If finish[source_node_2] < finish[source_node_1], set order_1_first = 0
                model.addConstr(finish[source_node_2] - finish[source_node_1] <= M * order_1_first,
                                name=f"order_decision_2_{source_node_1}_{source_node_2}")

                # If order_1_first == 1, communication 1 finishes before communication 2 starts
                model.addConstr(comm_start[comm2] >= comm_end[comm1] - M * (1 - order_1_first),
                                name=f"FCFS_comm1_first_{source_node_1}_{source_node_2}")

                # If order_1_first == 0, communication 2 finishes before communication 1 starts
                model.addConstr(comm_start[comm1] >= comm_end[comm2] - M * order_1_first,
                                name=f"FCFS_comm2_first_{source_node_1}_{source_node_2}")
            # in this case, a must be b's preceding node
            else:
                assert nx.has_path(subgraph, source_node_1, source_node_2)
                model.addConstr(comm_end[comm1] <= comm_start[comm2])


class SchedulingAlgorithm(Enum):
    FIFO = "FIFO"
    OPTIMIZED = "OPTIMIZED"
    PRIORITY_MIN_COMP = "PRIORITY_MIN_COMP"
    NEAR_OPTIMAL = "NEAR_OPTIMAL"
    FIFO_SOLVER = "FIFO_SOLVER",
    PRIORITY_HETEROG = "PRIORITY_HETEROG"
    SAMPLING_NEAR_OPTIMAL = "SAMPLING_NEAR_OPTIMAL"


def execute_scheduling_function(sch_fun_type: str, model: Model, **kwargs):
    # Define the required arguments for each scheduling algorithm
    required_args = {
        SchedulingAlgorithm.FIFO.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                         'device_subgraph_mapping', 'edge_cut_list', 'operator_device_mapping'],
        SchedulingAlgorithm.OPTIMIZED.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                              'device_subgraph_mapping', 'edge_cut_list'],
        SchedulingAlgorithm.PRIORITY_MIN_COMP.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                                      'device_subgraph_mapping', 'edge_cut_list',
                                                      'operator_device_mapping'],
        SchedulingAlgorithm.FIFO_SOLVER.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                                'device_subgraph_mapping', 'edge_cut_list', 'operator_device_mapping'],
        SchedulingAlgorithm.PRIORITY_HETEROG.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                                     'device_subgraph_mapping', 'edge_cut_list',
                                                     'operator_device_mapping'],
        SchedulingAlgorithm.NEAR_OPTIMAL.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                                 'device_subgraph_mapping', 'edge_cut_list', 'operator_device_mapping',
                                                 'threshold'],
        SchedulingAlgorithm.SAMPLING_NEAR_OPTIMAL.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                                         'device_subgraph_mapping', 'edge_cut_list',
                                                         'operator_device_mapping', 'rho', 'sampling_function'],
    }

    if sch_fun_type not in required_args:
        raise ValueError(f"Unknown scheduling algorithm: {sch_fun_type}")

    # Check if all required arguments are provided
    missing_args = [arg for arg in required_args[sch_fun_type] if arg not in kwargs]
    if missing_args:
        raise ValueError(f"Missing arguments for {sch_fun_type}: {', '.join(missing_args)}")

    # Select the appropriate arguments for the scheduling function
    selected_kwargs = {key: kwargs[key] for key in required_args[sch_fun_type]}

    # Dynamically dispatch to the appropriate scheduling function
    if sch_fun_type == SchedulingAlgorithm.FIFO.value:
        return FIFO_scheduling(model, **selected_kwargs)
    elif sch_fun_type == SchedulingAlgorithm.OPTIMIZED.value:
        return optimal_scheduling(model, **selected_kwargs)
    elif sch_fun_type == SchedulingAlgorithm.PRIORITY_MIN_COMP.value:
        return priority_queue_min_comp_cost(model, **selected_kwargs)
    elif sch_fun_type == SchedulingAlgorithm.FIFO_SOLVER.value:
        return FIFO_scheduling_solver(model, **selected_kwargs)
    elif sch_fun_type == SchedulingAlgorithm.PRIORITY_HETEROG.value:
        return priority_queue_max_rank_heteroG(model, **selected_kwargs)
    elif sch_fun_type == SchedulingAlgorithm.NEAR_OPTIMAL.value:
        return near_optimal_scheduling(model, **selected_kwargs)
    elif sch_fun_type == SchedulingAlgorithm.SAMPLING_NEAR_OPTIMAL.value:
        return near_optimal_scheduling_with_sampling(model, **selected_kwargs)
