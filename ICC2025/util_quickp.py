from collections import defaultdict

from DNN_model_tf.tf_model_enum import TFModelEnum
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt

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


def get_proper_alpha(comp_graph, deviceTopo, if_visualize=True):
    any_d = deviceTopo.getDeviceIDs()[0]
    homo_op_cost_dict = comp_graph.getOpCompCostMapByDevice(any_d)

    # getting data of the histogram
    count, bins_count = np.histogram(list(homo_op_cost_dict.values()), bins=500)
    bins_count = bins_count[:-1]  # Remove the last bin edge to match cdf length

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

    # Define a typical smooth, always-increasing function: logistic curve

    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    # Fit the data to the logistic curve

    popt, _ = curve_fit(logistic, bins_count, cdf, maxfev=10000)

    # Generate smooth curve using the fitted parameters

    x_smooth = np.linspace(bins_count.min(), bins_count.max(), 500)

    y_smooth = logistic(x_smooth, *popt)

    # Get the alpha computation cost
    slope_smooth = (popt[1] * popt[0] * np.exp(-popt[1] * (x_smooth - popt[2]))) / (

            (1 + np.exp(-popt[1] * (x_smooth - popt[2]))) ** 2

    )

    # Calculate max and min slopes

    max_slope = np.max(slope_smooth)

    min_slope = np.min(slope_smooth)

    # Calculate the midpoint of max and min slopes

    mid_slope = (max_slope + min_slope) / 3.2

    # Find the point where the slope is closest to the midpoint slope

    closest_mid_index = np.argmin(np.abs(slope_smooth - mid_slope))

    x_midpoint = x_smooth[closest_mid_index]

    y_midpoint = y_smooth[closest_mid_index]

    if if_visualize:
        # Plotting the smooth CDF
        plt.figure(figsize=(10, 6))
        plt.plot(bins_count, cdf, 'o', label='Original Data', alpha=0.6)
        plt.plot(x_smooth, y_smooth, '-', label='Smooth Increasing Curve (Logistic Fit)', linewidth=2)
        plt.scatter(

            x_midpoint,

            y_midpoint,

            color='purple',

            label=f'Point with Slope ~ Midpoint\n(x={x_midpoint:.2f}, y={y_midpoint:.2f}, slope={mid_slope:.5f})',

        )
        plt.legend(loc='best')
        plt.show()

    return x_midpoint