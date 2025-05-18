from optimizer.model.graph import CompGraph, DeviceGraph


def get_mcmc_init_device_placement(comp_graph: CompGraph, deviceTopo: DeviceGraph, M):
    operator_device_mapping = {}
    for node in comp_graph.nodes:
        random_device = deviceTopo.getDeviceIDs()[0]
        operator_device_mapping[node] = random_device
    return operator_device_mapping