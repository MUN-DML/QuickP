from collections import defaultdict


def show_quick_p_result(model, operator_device_placement, start, finish, comp_cost_map):

    assignment_dict = defaultdict(list)
    # populate result['Assignment']
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

