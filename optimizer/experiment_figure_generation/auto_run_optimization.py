import os
import sys

from optimizer.operator_device_placement.metis.subgraph_util import WeightNormalizationFunction
from optimizer.optimization_problems.simulator import simulate

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum
from optimizer.operator_device_placement.metis.weight_functions import EdgeWeightFunction, NodeWeightFunction


def populate_training_time_list():

    result_matrix = {}

    flexible_setting = {
        "rho": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

    }

    fix_setting = {
        "number_of_device": 4,
        "model_type": TFModelEnum.SMALL,
        "scheduling_function": "NEAR_OPTIMAL_REVISED",
        "node_weight_function": NodeWeightFunction.AVE_COMP_COST,
        "edge_weight_function": EdgeWeightFunction.SOURCE_OUTPUT_TENSOR,
        "weight_norm_function": WeightNormalizationFunction.MIN_MAX,

    }

    for adjustment_type, setting_dict in data_matrix.items():
        result_matrix.setdefault(adjustment_type, [])
        for ratio in ratio_list:
            entire_setting = {**setting_dict, "adjustment_ratio": ratio}

            expected_training = simulate(**fix_setting, )
            if not expected_training:
                expected_training = 0

            result_matrix[adjustment_type].append(expected_training)

    result_matrix["ratios"] = ratio_list
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
    generate_graph(data_dict)
