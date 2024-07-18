# Define an enum for partitioning types
from enum import Enum


class PartitionType(Enum):
    HOMO = 1
    HETERO = 2


# Function to get imbalance settings based on partition type
def get_imbalance_settings(partition_type: PartitionType, ratio_list=None):
    if partition_type == PartitionType.HOMO:
        return [1.1]  # Homogeneous cutting: Allow a small imbalance
    elif partition_type == PartitionType.HETERO:
        if ratio_list is not None and all(r > 1 for r in ratio_list):
            return ratio_list  # Use the provided ratio list for heterogeneous cutting
        else:
            raise ValueError("For HETERO partition type, ratio_list must be provided and all values must be greater than 1")
    else:
        raise ValueError("Invalid partition type")
