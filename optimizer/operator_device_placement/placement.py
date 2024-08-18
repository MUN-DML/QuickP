from enum import Enum

from optimizer.operator_device_placement.optimal.optimal_placement import get_optimize_placement


class PlacementGenerator(Enum):
    METIS = "METIS"
    OPTIMIZED = "OPTIMIZED"

def get_placement_y_edge_cut(placement_fun_type: str):
    if placement_fun_type == PlacementGenerator.METIS.value:
        pass
    elif placement_fun_type == PlacementGenerator.OPTIMIZED.value:
        pass
