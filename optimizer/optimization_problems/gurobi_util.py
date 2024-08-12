from gurobipy import *

from py_util import compare_2d_list

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.computing_graph.computing_graph import get_computation_graph
from optimizer.computing_graph.op_graph_util import get_proper_optimizer
from optimizer.model.graph import DeviceGraph, CompGraph, has_more_than_one_component, keep_largest_component, \
    visualize_graph
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


def init_computing_and_device_graph(num_device, filename: str, hetero_adjust_rate, if_clean_extra_operator=False,
                                    model_type=TFModelEnum.SMALL):
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
    comp_graph.clean_marginal_operators()
    visualize_graph(comp_graph, show_node_labels=False)
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
            tensor_size = comp_graph.getOperatorOutputInBit(source_op_ID)
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
