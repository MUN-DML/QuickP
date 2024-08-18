import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.experiment_figure_generation.optimization_enum import OptimizationProblem
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum
from optimizer.operator_device_placement.metis.weight_functions import EdgeWeightFunction


def run_optimization_command(problem_type: OptimizationProblem,
                             adjustment_type: dict,
                             if_weight_norm: bool,
                             model_type: TFModelEnum,
                             edge_weight_function: EdgeWeightFunction,
                             number_of_devices=2
                             ):
    if problem_type == OptimizationProblem.BASELINE:
        # Call optimize_baseline
        return problem_type(number_of_devices=number_of_devices, model_type=model_type, if_weight_norm=if_weight_norm)
    elif problem_type == OptimizationProblem.GRAPH_PARTITION:
        # Call optimize_after_graph_partition
        return problem_type(number_of_devices=number_of_devices, model_type=model_type, if_weight_norm=if_weight_norm,
                            edge_weight_function=edge_weight_function, adjust_matrix=adjustment_type)
    else:
        raise ValueError("Invalid optimization problem type")


def populate_training_time_list(increment=0.05, min_value=0.0, max_value=1.0):
    ratio_list = [round(x, 10) for x in [min_value + i * increment for i in range(int((max_value - min_value) / increment) + 1)]]

    data_matrix = {
        "no_adjustment": {"node_enable": False, "edge_enable": False},
        "edge_adjustment": {"node_enable": False, "edge_enable": True},
        "node_adjustment": {"node_enable": True, "edge_enable": False},
        "both_adjustment": {"node_enable": True, "edge_enable": True}
    }
    result_matrix = {}

    fix_setting = {
        "problem_type": OptimizationProblem.GRAPH_PARTITION,
        "edge_weight_function": EdgeWeightFunction.MOCK_COMMUNICATION_COST_WITH_COMP,
        "model_type": TFModelEnum.VGG,
        "if_weight_norm": False
    }

    for adjustment_type, setting_dict in data_matrix.items():
        result_matrix.setdefault(adjustment_type, [])
        for ratio in ratio_list:
            entire_setting = {**setting_dict, "adjustment_ratio": ratio}
            expected_training = run_optimization_command(**fix_setting,
                                                         adjustment_type=entire_setting)
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
