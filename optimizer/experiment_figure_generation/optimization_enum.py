from enum import Enum

from optimizer.main_simulator.after_graph_partition_hetero import optimize_after_graph_partition
from optimizer.operator_device_placement.optimal.optimal_placement import get_optimize_placement


class OptimizationProblem(Enum):
    BASELINE = get_optimize_placement
    GRAPH_PARTITION = optimize_after_graph_partition
