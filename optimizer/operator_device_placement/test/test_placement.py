from optimizer.computing_graph.profiled_computation_graph_json.test_comp_graph import get_test_graph


def get_test_device_placement(comp_graph, device_topo):
    operator_device_map = {}

    # Iterate over all nodes in the graph
    for node in comp_graph.nodes():
        if node.startswith('d0'):
            operator_device_map[node] = 'mock_device_0'
        elif node.startswith('d1'):
            operator_device_map[node] = 'mock_device_1'
        elif node.startswith('d2'):
            operator_device_map[node] = 'mock_device_2'
        else:
            raise ValueError("WRONG OPERATOR DEVICE PLACEMENT")

    return operator_device_map

'''
g = get_test_graph()
a = get_test_device_placement(g, 1)
print(a)
'''