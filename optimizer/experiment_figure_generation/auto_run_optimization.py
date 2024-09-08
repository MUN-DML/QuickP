import itertools
import os
import sys
from collections import defaultdict

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
        "rho": [0.25, 0.5, 0.75, 1.0],
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

    result_matrix = defaultdict(list)  # Use defaultdict to simplify appending

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

        # Create a key for the result based on the current flexible setting
        adjustment_key = '_'.join([f"{k}_{str(v)}" for k, v in current_flexible_setting.items()])
        result_matrix[adjustment_key].append(expected_training)

    return result_matrix


def generate_graph(result_dict):
    print(result_dict)
    import matplotlib.pyplot as plt
    import numpy as np
    x_labels = result_dict.pop("ratios")
    x = np.arange(len(x_labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = np.linspace(-1.5, 1.5, len(result_dict)) * width
    # Plotting the bars
    for i, (adjustment_type, performance) in enumerate(result_dict.items()):
        ax.bar(x + offsets[i], performance, width, label=adjustment_type)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('ratios')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time Comparison with Different Adjustments')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    data_dict = populate_training_time_list()
    print(data_dict)
