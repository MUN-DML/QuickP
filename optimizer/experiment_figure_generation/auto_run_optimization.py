import itertools
import os
import sys
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction
from optimizer.optimization_problems.simulator import simulate
from optimizer.scheduling.proposed_scheduling_revised import SamplingFunction

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum
from optimizer.operator_device_placement.metis.weight_functions import EdgeWeightFunction, NodeWeightFunction


def populate_training_time_list():

    result_matrix = {}

    flexible_setting = {
        "rho": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "sampling_function": [SamplingFunction.PROBABILISTIC_SAMPLING, SamplingFunction.RANDOM, SamplingFunction.HEAVY_HITTER]
    }

    fix_setting = {
        "number_of_devices": 4,
        "model_type": TFModelEnum.SMALL,
        "placement": 'METIS',
        "scheduling_function": "NEAR_OPTIMAL_REVISED",
        "node_weight_function": NodeWeightFunction.AVE_COMP_COST,
        "edge_weight_function": EdgeWeightFunction.SOURCE_OUTPUT_TENSOR,
        "weight_norm_function": WeightNormalizationFunction.MIN_MAX,

    }

    result_matrix = {}  # Use a regular dictionary since we want scalar values

    # Extract the keys and values from flexible_setting
    flexible_keys, flexible_values = zip(*flexible_setting.items())  # Unpack keys and their corresponding values

    # Iterate over all combinations of flexible settings using itertools.product
    for flexible_combination in itertools.product(*flexible_values):
        # Create a dictionary from the flexible combination
        current_flexible_setting = dict(zip(flexible_keys, flexible_combination))

        # Combine fixed settings and flexible settings
        current_setting = {**fix_setting, **current_flexible_setting}

        # Execute the simulation
        expected_training = simulate(**current_setting)

        # If no result is returned, set default value of 0
        if not expected_training:
            expected_training = 0

        # Create a tuple key based on the flexible setting values
        adjustment_key = tuple(current_flexible_setting.values())
        result_matrix[adjustment_key] = expected_training

    return result_matrix


def generate_curve(result_dict, xlabel="rho", ylabel="Latency (s)",
                   title="Latency vs rho for Different Sampling Functions", fig_size=(10, 6)):
    """
    Generates curves to show latency for different sampling functions over different rho values.

    :param result_dict: Dictionary with tuple keys representing (rho, sampling_function) and scalar values for performance.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param title: Title of the graph.
    :param fig_size: Tuple representing the figure size.
    """

    # Prepare to group data by sampling_function
    grouped_data = {}

    for (rho, sample_function), latency in result_dict.items():
        if sample_function not in grouped_data:
            grouped_data[sample_function] = {'rho': [], 'latency': []}
        grouped_data[sample_function]['rho'].append(rho)
        grouped_data[sample_function]['latency'].append(latency)

    # Sort data by rho for each sample_function to ensure curves are plotted correctly
    for sample_function in grouped_data:
        rho_latency_pairs = sorted(zip(grouped_data[sample_function]['rho'], grouped_data[sample_function]['latency']))
        grouped_data[sample_function]['rho'], grouped_data[sample_function]['latency'] = zip(*rho_latency_pairs)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot a curve for each sampling function
    for sample_function, data in grouped_data.items():
        ax.plot(data['rho'], data['latency'], marker='o', label=sample_function.name)

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add legend to differentiate between sample functions
    ax.legend(title="Sampling Function")

    # Show the plot
    plt.show()


if __name__ == '__main__':
    data_dict = populate_training_time_list()
    generate_curve(data_dict)
