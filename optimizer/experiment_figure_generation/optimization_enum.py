from enum import Enum

from optimizer.optimization_problems.after_graph_partition import optimize_after_graph_partition
from optimizer.optimization_problems.baseline import optimize_baseline


class OptimizationProblem(Enum):
    BASELINE = optimize_baseline
    GRAPH_PARTITION = optimize_after_graph_partition
