from collections import defaultdict

from DNN_model_tf.tf_model_enum import TFModelEnum


def show_quick_p_result(model, x, start, finish, comp_cost_map, model_type: TFModelEnum, comp_graph, deviceTopo):

    def get_operator_device_mapping_through_x(x):
        mapping = {}
        for (operator_id, device_id), var in x.items():
            # Check if the variable has a value of 1 (operator is assigned to device)
            if var.X > 0.5:  # Since the variable is binary, this checks if it is assigned
                mapping[operator_id] = device_id
        return mapping

    operator_device_placement = get_operator_device_mapping_through_x(x)
    assignment_dict = defaultdict(list)
    device_total_cost_map = defaultdict(int)
    device_util_rate_map = defaultdict(int)
    # populate assignment_dict
    for op, device in operator_device_placement.items():
        # Assignment: {device: [(op1, start[op1], finish[op1]), (...)]}
        assignment_dict[device].append((op, start[op].X, finish[op].X))
    # Sort operators by their start times for each device
    for device, ops in assignment_dict.items():
        assignment_dict[device] = sorted(ops, key=lambda x: x[1])

    # Print operator placement
    for device, op_info_tuples in assignment_dict.items():
        sum_comp = 0
        print(f"Device: {device}")
        for op_tuple in op_info_tuples:
            op = op_tuple[0]
            comp_cost = comp_cost_map[op]
            sum_comp += comp_cost
            if comp_cost == 0:
                continue
            print(f"  Operator: {op_tuple[0]}, Start: {op_tuple[1]}, Finish: {op_tuple[2]}, Comp Cost: {comp_cost}")
        device_utility_rate = sum_comp / model.ObjVal
        device_total_cost_map[device] = sum_comp
        device_util_rate_map[device] = device_utility_rate

    print('Expected Training time = ', model.ObjVal, 's', sep='')
    print("Device Utility Rate:", device_util_rate_map)
    print("total_computing_time_per_device:", device_total_cost_map)
    print('The Placement Searching Runtime = ', "%.2f" % model.Runtime, 's', sep='')
    print('Expected Training time = ', model.ObjVal, 's', sep='')
    print(f"This is the near-optimal solution of such configuration: \n"
          f"model type: {model_type} \n"
          f"number of operators: {comp_graph.number_of_nodes()} \n"
          f"number of devices: {deviceTopo.number_of_nodes()} \n"
          f"The environment is homogenous")