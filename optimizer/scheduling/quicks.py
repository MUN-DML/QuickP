import networkx as nx
from gurobipy import Model, GRB

from optimizer.model.graph import CompGraph
from optimizer.scheduling.scheduling_util import three_stage_split_subgraph, split_three_stage_subgraph


def quickS():
    pass


def optimize_dependent_relied_part(model: Model, start, finish, comm_start, comm_end, comp_graph: CompGraph,
                                   device_subgraph_mapping: dict, edge_cut_list: list, operator_device_mapping: dict,
                                   rho, sampling_function):
    M = 1000000
    topological_order = list(nx.topological_sort(comp_graph))
    topological_order_mapping = {node: index for index, node in enumerate(topological_order)}
    order = {}

    stage_two_finish = model.addVars(device_subgraph_mapping.keys(), vtype=GRB.CONTINUOUS, lb=0.0,
                                     name="non_isolated_part_finish")

    # form new device non-isolated part mapping
    # split into isolated and non-isolated part
    for device, subgraph in device_subgraph_mapping.items():
        # Simply the search space by
        stage_one, stage_two, stage_three = split_three_stage_subgraph(
            subgraph, operator_device_mapping, edge_cut_list)

        # we only care about
        # independent relied part => random topo sort, this stage should always be first, we don't care know
        stage_one_sorted = sorted(list(stage_one.nodes), key=lambda node: topological_order_mapping[node])
        for a, b in zip(stage_one_sorted, stage_one_sorted[1:]):
            model.addConstr(finish[a] <= start[b])
        # stage_three => non-exporting part, FIFO is okay, we don't care
        stage_three_sorted = sorted(list(stage_one.nodes), key=lambda node: topological_order_mapping[node])
        for a, b in zip(stage_three_sorted, stage_three_sorted[1:]):
            model.addConstr(finish[a] <= start[b])

        # nonoverlapping between stage 1 and 2
        for stage_two_node in stage_two:
            model.addConstr(stage_two_finish[device] >= finish[stage_two_node])
            model.addConstr(start[stage_two_node] >= finish[stage_one_sorted[-1]])
        # # nonoverlapping between stage 2 and 3
        model.addConstr(start[stage_three_sorted[0]] >= stage_two_finish[device])


        stage_two_pairs = find_non_connected_pairs(stage_two)
        for op_a, op_b in stage_two_pairs:
            order[op_a, op_b] = model.addVar(vtype=GRB.BINARY, name=f"order_{op_a}_{op_b}")
            model.addConstr(start[op_b] >= finish[op_a] - M * (1 - order[op_a, op_b]), name=f"NoOverlap1_{op_a}_{op_b}")
            model.addConstr(start[op_a] >= finish[op_b] - M * order[op_a, op_b], name=f"NoOverlap2_{op_a}_{op_b}")

