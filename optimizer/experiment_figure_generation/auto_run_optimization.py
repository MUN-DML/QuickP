import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.experiment_figure_generation.optimization_enum import OptimizationProblem
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum
from optimizer.graph_partitioner.weight_functions import EdgeWeightFunction


def run_optimization_command(problem_type: OptimizationProblem,
                             adjustment_type: dict,
                             number_of_devices=2,
                             model_type: TFModelEnum = TFModelEnum.SMALL,
                             edge_weight_function: EdgeWeightFunction = EdgeWeightFunction.SOURCE_OUTPUT_TENSOR):
    if problem_type == OptimizationProblem.BASELINE:
        # Call optimize_baseline
        return problem_type(number_of_devices=number_of_devices, model_type=model_type)
    elif problem_type == OptimizationProblem.GRAPH_PARTITION:
        # Call optimize_after_graph_partition
        return problem_type(number_of_devices=number_of_devices, model_type=model_type,
                     edge_weight_function=edge_weight_function, adjust_matrix=adjustment_type)
    else:
        raise ValueError("Invalid optimization problem type")


def populate_training_time_list(increment=0.5):
    ratio_list = [round(x, 10) for x in [0 + i * increment for i in range(int(1 / increment) + 1)]]

    data_matrix = {
        "no_adjustment": {"node_enable": False, "edge_enable": False},
        "edge_adjustment": {"node_enable": False, "edge_enable": True},
        "node_adjustment": {"node_enable": True, "edge_enable": False},
        "both_adjustment": {"node_enable": True, "edge_enable": True}
    }
    result_matrix = {}

    for adjustment_type, setting_dict in data_matrix.items():
        result_matrix.setdefault(adjustment_type, [])
        for ratio in ratio_list:
            entire_setting = {**setting_dict, "adjustment_ratio": ratio}
            expected_training = run_optimization_command(problem_type=OptimizationProblem.GRAPH_PARTITION,
                                                         adjustment_type=entire_setting)
            result_matrix[adjustment_type].append(expected_training)

    data_matrix["ratios"] = ratio_list
    return data_matrix


def generate_graph(data_dict):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(data_dict["ratios"]))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the bars
    bars1 = ax.bar(x - 1.5 * width, data_dict["no_adjustment"], width, label='No Adjustment')
    bars2 = ax.bar(x - 0.5 * width, data_dict["edge_adjustment"], width, label='Edge Weight Adjustment')
    bars3 = ax.bar(x + 0.5 * width, data_dict["node_adjustment"], width, label='Node Weight Adjustment')
    bars4 = ax.bar(x + 1.5 * width, data_dict["both_adjustment"], width, label='Both Adjustments')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('ratios')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time Comparison with Different Adjustments')
    ax.set_xticks(x)
    ax.set_xticklabels(data_dict["ratios"])
    ax.legend()

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    data_dict = populate_training_time_list()
    generate_graph(data_dict)
