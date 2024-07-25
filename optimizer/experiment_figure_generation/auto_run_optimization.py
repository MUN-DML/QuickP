import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.experiment_figure_generation.optimization_enum import OptimizationProblem
from optimizer.experiment_figure_generation.tf_model_enum import TFModelEnum


def populate_parameter_list():
    pass


def run_optimization_command(problem_type: OptimizationProblem, model_type: TFModelEnum, parameter_list):
    pass


def generate_graph():
    pass
