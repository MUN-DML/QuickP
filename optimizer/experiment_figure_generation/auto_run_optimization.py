import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.experiment_figure_generation.optimization_enum import OptimizationProblem
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum
from optimizer.graph_partitioner.weight_functions import EdgeWeightFunction


def populate_parameter_list(increment=0.05):
    ratio_list = [round(x, 10) for x in [0 + i * increment for i in range(int(1 / increment) + 1)]]
    data_dict = {
        "models": ratio_list,
        "no_adjustment": [],
        "edge_adjustment": [],
        "node_adjustment": [],
        "both_adjustment": []
    }
    for ratio in ratio_list:
        pass

    return data_dict


def run_optimization_command(problem_type: OptimizationProblem,
                             number_of_devices=2,
                             model_type: TFModelEnum = TFModelEnum.SMALL,
                             edge_weight_function: EdgeWeightFunction = EdgeWeightFunction.SOURCE_OUTPUT_TENSOR):
    if problem_type == OptimizationProblem.BASELINE:
        # Call optimize_baseline
        problem_type(number_of_devices=number_of_devices, model_type=model_type)
    elif problem_type == OptimizationProblem.GRAPH_PARTITION:
        # Call optimize_after_graph_partition
        problem_type(number_of_devices=number_of_devices, model_type=model_type,
                     edge_weight_function=edge_weight_function)
    else:
        raise ValueError("Invalid optimization problem type")


def generate_graph(data_dict):
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample data
    fix_rate_ratio = [0.1, 0.2, 0.3]
    training_times_no_adjustment = [10, 20, 30]
    training_times_edge_adjustment = [15, 25, 35]
    training_times_node_adjustment = [12, 22, 32]
    training_times_both_adjustment = [14, 24, 34]

    x = np.arange(len(data_dict["ratios"]))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the bars
    bars1 = ax.bar(x - 1.5 * width, data_dict["no_adjustment"], width, label='No Adjustment')
    bars2 = ax.bar(x - 0.5 * width, data_dict["edge_adjustment"], width, label='Edge Weight Adjustment')
    bars3 = ax.bar(x + 0.5 * width, data_dict["node_adjustment"], width, label='Node Weight Adjustment')
    bars4 = ax.bar(x + 1.5 * width, data_dict["both_adjustment"], width, label='Both Adjustments')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Models')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Time Comparison with Different Adjustments')
    ax.set_xticks(x)
    ax.set_xticklabels(data_dict["ratios"])
    ax.legend()

    fig.tight_layout()

    plt.show()


populate_parameter_list()
