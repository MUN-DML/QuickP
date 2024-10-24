import itertools
from enum import Enum

import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import find_non_connected_pairs, CompGraph, is_not_connected
from optimizer.scheduling.FIFO import FIFO_scheduling
from optimizer.scheduling.multi_stage_list_schedule import three_stage_list_schedule
from optimizer.scheduling.optimal import optimal_scheduling
from optimizer.scheduling.priority_heteroG import priority_queue_max_rank_heteroG
from optimizer.scheduling.priority_min_comp_cost import priority_queue_min_comp_cost


def add_topo_order_constraints_with_grouper(model, graph: CompGraph, x, device_ids, finish, start, group_ops_mapping: dict, M):
    op_group_mapping = graph.create_op_group_id_mapping()
    topological_order_mapping = {node: index for index, node in enumerate(list(nx.topological_sort(graph)))}
    non_reachable_pairs = find_non_connected_pairs(graph)
    ungrouped_non_reachable_pairs = [(a,b) for (a,b) in non_reachable_pairs if a not in op_group_mapping or b not in op_group_mapping]
    print('sisi', len(non_reachable_pairs), len(ungrouped_non_reachable_pairs))

    for group_id, group in group_ops_mapping.items():
        group = sorted(group, key=lambda node: topological_order_mapping[node])
        for a, b in zip(group, group[1:]):
            model.addConstr(finish[a] <= start[b])

    # Iterate over topologically sorted nodes
    for a, b in ungrouped_non_reachable_pairs:
        # For each consecutive pair of operators, add a constraint for each device
        for device_id in device_ids:
            # Ensure the correct order for each potential device assignment
            # This constraint will only apply if both a and b are assigned to the same device
            model.addConstr(finish[a] <= start[b] + M * (2 - x[a, device_id] - x[b, device_id]),
                            name=f"bigM_topo_order_{a}_{b}_on_device_{device_id}")

def add_topo_order_constraint(model, graph, x, device_ids, finish, start, M):
    # Iterate over topologically sorted nodes
    non_reachable_pairs = find_non_connected_pairs(graph)
    print('numero de op', len(non_reachable_pairs))
    for a, b in non_reachable_pairs:
        # For each consecutive pair of operators, add a constraint for each device
        for device_id in device_ids:
            # Ensure the correct order for each potential device assignment
            # This constraint will only apply if both a and b are assigned to the same device
            model.addConstr(finish[a] <= start[b] + M * (2 - x[a, device_id] - x[b, device_id]),
                            name=f"bigM_topo_order_{a}_{b}_on_device_{device_id}")


class SchedulingAlgorithm(Enum):
    FIFO = "FIFO"
    OPTIMIZED = "OPTIMIZED"
    PRIORITY_MIN_COMP = "PRIORITY_MIN_COMP"
    PRIORITY_HETEROG = "PRIORITY_HETEROG"
    THREE_STAGE = "THREE_STAGE"


def execute_scheduling_function(sch_fun_type: str, model: Model, **kwargs):
    # Define the required arguments for each scheduling algorithm
    required_args = {
        SchedulingAlgorithm.FIFO.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                         'device_subgraph_mapping', 'edge_cut_list', 'operator_device_mapping'],
        SchedulingAlgorithm.OPTIMIZED.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                              'device_subgraph_mapping', 'edge_cut_list', 'M'],
        SchedulingAlgorithm.PRIORITY_MIN_COMP.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                                      'device_subgraph_mapping', 'edge_cut_list',
                                                      'operator_device_mapping'],
        SchedulingAlgorithm.PRIORITY_HETEROG.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                                     'device_subgraph_mapping', 'edge_cut_list',
                                                     'operator_device_mapping'],
        SchedulingAlgorithm.THREE_STAGE.value: ['start', 'finish', 'comm_start', 'comm_end', 'comp_graph',
                                                'device_subgraph_mapping', 'edge_cut_list',
                                                'operator_device_mapping', 'computing_cost_dict',
                                                'communication_cost_dict']
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
    elif sch_fun_type == SchedulingAlgorithm.PRIORITY_HETEROG.value:
        return priority_queue_max_rank_heteroG(model, **selected_kwargs)
    elif sch_fun_type == SchedulingAlgorithm.THREE_STAGE.value:
        return three_stage_list_schedule(model, **selected_kwargs)
