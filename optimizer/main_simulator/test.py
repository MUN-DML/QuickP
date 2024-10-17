import networkx as nx
from matplotlib import pyplot as plt

from optimizer.model.graph import CompGraph, visualize_graph
'''
{'adam/AssignAddVariableOp_1', 'adam/Sub_3/ReadVariableOp/resource', 'adam/Mul_2', 'adam/Sub_3/ReadVariableOp', 'adam/Sub_3'} get merged into  36c7112048199869c4f331ddba674666
no cycle detect
{'adam/Sub_7', 'adam/Mul_6', 'adam/Sub_7/ReadVariableOp', 'adam/AssignAddVariableOp_3', 'adam/Sub_7/ReadVariableOp/resource'} get merged into  e360f640835bd88b1ba19bd84b6bcceb
no cycle detect
{'adam/Sub_2/ReadVariableOp/resource', 'adam/AssignAddVariableOp', 'adam/Mul_1', 'adam/Sub_2/ReadVariableOp', 'adam/Sub_2'} get merged into  1643564623a819264cc27464d29fcab7
no cycle detect
{'adam/Sub_6', 'adam/Mul_5', 'adam/Sub_6/ReadVariableOp', 'adam/Sub_6/ReadVariableOp/resource', 'adam/AssignAddVariableOp_2'} get merged into  3991d50d7c9c17caa4c54fc598a93084
no cycle detect
{'sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits', 'gradient_tape/sequential_1/dense_1/MatMul/MatMul', 'sequential_1/dense_1/MatMul', 'sequential_1/flatten_1/Reshape', 'gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul', 'sequential_1/dense_1/Add'} get merged into  9db76bc43a4eb7bb8cbec630ffa84038
no cycle detect
{'sparse_categorical_crossentropy/Cast_2', 'gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv_1', 'gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv_2'} get merged into  e7441e53728c05ca94084e6fca6fa257
[('9db76bc43a4eb7bb8cbec630ffa84038', 'sparse_categorical_crossentropy/Sum', 'forward'), ('sparse_categorical_crossentropy/Sum', 'gradient_tape/sparse_categorical_crossentropy/truediv/Neg', 'forward'), ('gradient_tape/sparse_categorical_crossentropy/truediv/Neg', 'e7441e53728c05ca94084e6fca6fa257', 'forward'), ('e7441e53728c05ca94084e6fca6fa257', 'gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv', 'forward'), ('gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv', 'gradient_tape/sparse_categorical_crossentropy/Reshape', 'forward'), ('gradient_tape/sparse_categorical_crossentropy/Reshape', 'gradient_tape/sparse_categorical_crossentropy/Tile', 'forward'), ('gradient_tape/sparse_categorical_crossentropy/Tile', 'gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims', 'forward'), ('gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims', '9db76bc43a4eb7bb8cbec630ffa84038', 'forward')]
'''
comp_graph = CompGraph.load_from_file("graph_without_cycle_fix.json")
sub = comp_graph.subgraph(['9db76bc43a4eb7bb8cbec630ffa84038', 'sparse_categorical_crossentropy/Sum', 'gradient_tape/sparse_categorical_crossentropy/truediv/Neg',
                           'sparse_categorical_crossentropy/Cast_2', 'gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv_1', 'gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv_2',
                           'gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv', 'gradient_tape/sparse_categorical_crossentropy/Reshape', 'gradient_tape/sparse_categorical_crossentropy/Tile',
                           'gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims', '9db76bc43a4eb7bb8cbec630ffa84038'])

red_nodes = ['sparse_categorical_crossentropy/Cast_2',
             'gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv_1',
             'gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv_2']

# Define the rest of the nodes in blue
blue_nodes = ['9db76bc43a4eb7bb8cbec630ffa84038',
              'sparse_categorical_crossentropy/Sum',
              'gradient_tape/sparse_categorical_crossentropy/truediv/Neg',
              'gradient_tape/sparse_categorical_crossentropy/truediv/RealDiv',
              'gradient_tape/sparse_categorical_crossentropy/Reshape',
              'gradient_tape/sparse_categorical_crossentropy/Tile',
              'gradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/ExpandDims']

pos = nx.spring_layout(sub)  # Spring layout positions the nodes nicely

# Draw nodes
nx.draw_networkx_nodes(sub, pos, nodelist=red_nodes, node_color='red', node_size=500)
nx.draw_networkx_nodes(sub, pos, nodelist=blue_nodes, node_color='blue', node_size=500)

# Draw edges
nx.draw_networkx_edges(sub, pos)



# Display the graph
plt.title("Subgraph with Selected Nodes in Red and Others in Blue")
plt.show()

sub.visualize_multipath_component_in_wcc()