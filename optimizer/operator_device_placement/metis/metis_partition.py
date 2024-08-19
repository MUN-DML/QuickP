import os
import sys
from typing import Any

import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
from optimizer.operator_device_placement.metis.subgraph_util import identify_edges_cut, WeightNormalizationFunction
from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction


def metis_partition(graph: CompGraph, num_partitions: int , node_weight_function: NodeWeightFunction,
                    edge_weight_function: EdgeWeightFunction,
                    visualization=False, adjust_matrix=None,
                    weight_normalize: WeightNormalizationFunction = None,
                    sub_graph_weight_sum_ratio: list = None) -> tuple[dict[Any, Any], list[tuple], int]:
    def visualize_graph_partitioned(weight_graph: CompGraph, partition_result: dict):
        # Visualize the partitioned graph
        nx.set_node_attributes(weight_graph, partition_result, 'partition')

        # Generate a color map with enough distinct colors
        unique_partitions = set(partition_result.values())
        num_partitions = len(unique_partitions)
        colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

        if num_partitions > len(colors):
            raise ValueError(f"Number of partitions ({num_partitions}) exceeds available colors ({len(colors)}).")

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(weight_graph)
        node_colors = [colors[partition_result[node] % len(colors)] for node in weight_graph.nodes()]

        nx.draw(weight_graph, pos, with_labels=False, node_color=node_colors, edge_color='gray', node_size=200,
                font_size=16)
        plt.title(f'Graph Partitioning into {num_partitions} Parts using METIS', size=20)
        plt.show()

    # Convert the DiGraph to an undirected graph for partitioning
    G_undirected = graph.to_undirected()

    metis_graph = metis.networkx_to_metis(G_undirected)

    # Perform graph partitioning using METIS
    edgecuts, parts = metis.part_graph(metis_graph, nparts=num_partitions, tpwgts=sub_graph_weight_sum_ratio)
    # Assign partition labels to the original DiGraph nodes {node_id: placement_index}
    partition_dict = {node: part for node, part in zip(graph.nodes(), parts)}

    # verify whether the sum_of_cut_weight == edgecuts
    cut_edge_list, sum_of_cut_weight = identify_edges_cut(graph, partition_dict)
    assert sum_of_cut_weight == edgecuts
    if visualization:
        # Visualize the partitioned graph
        visualize_graph_partitioned(graph, partition_dict)

    # return the placement dict, list of edges cut and the weighted graph itself
    return partition_dict, cut_edge_list, edgecuts
