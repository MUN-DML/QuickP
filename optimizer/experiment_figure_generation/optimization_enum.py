from enum import Enum


class OptimizationProblem(Enum):
    BASELINE = {'filename': 'after_graph_partition.py'}
    GRAPH_PARTITION = {'filename': 'after_graph_partition.py'}
