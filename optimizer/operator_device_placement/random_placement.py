import random

from optimizer.model.graph import CompGraph, DeviceGraph


def get_random_device_placement(comp_graph: CompGraph, deviceTopo: DeviceGraph, M):
    operator_device_mapping = {}
    for node in comp_graph.nodes:
        random_device = random.choice(deviceTopo.getDeviceIDs())
        operator_device_mapping[node] = random_device
