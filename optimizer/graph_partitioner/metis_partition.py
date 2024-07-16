import os
import sys

import networkx as nx
from matplotlib import pyplot as plt
from networkx import DiGraph

from optimizer.graph_partitioner.subgraph_util import expand_subgraph
from py_util import tensor_shape_to_bits

# https://metis.readthedocs.io/en/latest/
# http://glaros.dtc.umn.edu/gkhome/metis/metis/download
'''
sudo apt-get install cmake, sudo apt-get install build-essential
gunzip metis-5.1.0.tar.gz
tar -xvf metis-5.1.0.tar
cd metis-5.1.0
read Install.txt
gcc --version / which gcc
make config shared=1 => Build files have been written to: /home/hola/Downloads/metis-5.1.0/build/Linux-x86_64
make install
in .bashrc, export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH; export METIS_DLL=/usr/local/lib/libmetis.so
'''
os.environ['METIS_DLL'] = '/usr/local/lib/libmetis.so'
import metis

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)
from optimizer.model.graph import CompGraph


def metis_partition(graph: CompGraph, num_partitions=3):
    # Assign weight to each node
    # Step 2: Calculate the node weights based on the `comp_cost` attribute
    for node in graph.nodes:
        total_cost = graph.getOperatorCompCostSum(node)
        graph.nodes[node]['node_weight'] = total_cost
    for edge in graph.edges:
        source_op, dest_op = edge
        shape, dtype = graph.getOperatorOutputSizeAndType(source_op)
        graph.edges[edge]['edge_weight'] = tensor_shape_to_bits(shape, dtype=dtype)

    graph.graph['node_weight_attr'] = 'node_weight'
    graph.graph['edge_weight_attr'] = 'edge_weight'

    # Convert the DiGraph to an undirected graph for partitioning
    G_undirected = graph.to_undirected()

    metis_graph = metis.networkx_to_metis(G_undirected)

    # Perform graph partitioning using METIS
    '''
    Returns a 2-tuple (objval, parts), where parts is a list of partition indices corresponding 
    and objval is the value of the objective function that was minimized (either the edge cuts or the total volume).
    '''
    (edgecuts, parts) = metis.part_graph(metis_graph, nparts=num_partitions)

    # Assign partition labels to the original DiGraph nodes {node_id: placement_index}
    partition_dict = {node: part for node, part in zip(graph.nodes(), parts)}
    nx.set_node_attributes(graph, partition_dict, 'partition')
    # Count the number of nodes in each partition
    # Count the number of nodes and sum of weights in each partition
    partition_counts = {i: 0 for i in range(num_partitions)}
    partition_weights = {i: 0 for i in range(num_partitions)}
    for node, part in zip(graph.nodes(), parts):
        partition_counts[part] += 1
        partition_weights[part] += graph.nodes[node]['node_weight']
    print(partition_counts)
    print(partition_weights)

    # Print the partition labels for each node
    # for node, data in graph.nodes(data=True):
    #     print(f"Node {node}: Partition {data['partition']}")

    # Visualize the partitioned graph
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=False, node_color=[colors[partition_dict[node]] for node in graph.nodes()],
            edge_color='gray', node_size=200, font_size=16)
    plt.title(f'Graph Partitioning into {num_partitions} Parts using METIS', size=20)
    plt.show()

    # return the placement dict
    return partition_dict


def construct_sub_graph(digraph: DiGraph, placement: dict[str, int]) -> list[DiGraph]:
    subgraph_dict = {}
    for operator, placement_index in placement.items():
        if placement_index not in subgraph_dict:
            subgraph_dict[placement_index] = nx.DiGraph()
        else:
            expand_subgraph(subgraph_dict[placement_index], operator, digraph)
    return list(subgraph_dict.values())
