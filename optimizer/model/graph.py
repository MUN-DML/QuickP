import json
import os
import random
from typing import Union
import tensorflow as tf

from matplotlib import pyplot as plt
from networkx import DiGraph, draw_networkx_labels, spring_layout, draw, draw_networkx_edge_labels, node_link_graph, \
    node_link_data, number_weakly_connected_components, has_path, NetworkXError, topological_sort, algorithms

from py_util import convert_data_size, convert_time


class CompGraph(DiGraph):
    @staticmethod
    def from_json(json_data):
        """
        Static method to convert JSON data to a CompGraph object.

        Parameters:
        json_data (str): JSON data as a string.

        Returns:
        CompGraph: A CompGraph object.
        """
        data = json.loads(json_data)
        # Convert tensor shapes from lists back to TensorShape objects
        for node in data['nodes']:
            if 'output_size' in node:
                node['output_size'] = tf.TensorShape(node['output_size'])
            if 'output_type' in node:
                node['output_type'] = tf.dtypes.as_dtype(node['output_type'])
        graph = node_link_graph(data, directed=True)
        return CompGraph(graph)

    @staticmethod
    def load_from_file(file_path):
        """
        Static method to load a CompGraph object from a JSON file.

        Parameters:
        file_path (str): The file path to load the JSON data from.

        Returns:
        CompGraph: A CompGraph object, or None if the file does not exist.
        """
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return None

        with open(file_path, 'r') as f:
            json_data = f.read()
        return CompGraph.from_json(json_data)

    def to_json(self):
        """
        Instance method to convert the current CompGraph object to JSON data.

        Returns:
        str: JSON data as a string.
        """
        data = node_link_data(self)
        # Convert TensorShape objects to lists for JSON serialization
        for node in data['nodes']:
            if 'output_size' in node and isinstance(node['output_size'], tf.TensorShape):
                node['output_size'] = node['output_size'].as_list()
            if 'output_type' in node and isinstance(node['output_type'], tf.DType):
                node['output_type'] = node['output_type'].name
        return json.dumps(data)

    def save_to_file(self, file_path):
        """
        Instance method to save the current CompGraph object to a JSON file.

        Parameters:
        file_path (str): The file path to save the JSON data.
        """
        json_data = self.to_json()
        with open(file_path, 'w') as f:
            f.write(json_data)

    def generata_random_cost(self, device_number):
        if len(self.getOperatorIDs()) == 0:
            raise ValueError("need to profile the real DNN first")
        for node in self.getOperatorObjs():
            assert node["comp_cost"] is not None
            existing_real_device = list(node["comp_cost"].keys())
            for i in range(device_number):
                device_name = f"mock_device_{i}"
                base = node["comp_cost"].values()

                base_num = sum(base) / len(base)
                adjustment_range = 0.05 * base_num

                # Generate a random adjustment within the range [-5%, 5%]
                adjustment = random.uniform(-adjustment_range, adjustment_range)

                # Apply the adjustment to the number
                adjusted_number = base_num + adjustment

                node["comp_cost"][device_name] = adjusted_number

            # Delete keys after iteration
            for key in existing_real_device:
                del node["comp_cost"][key]

    def add_new_node(self, operator_id, op_type, output_size: tf.TensorShape, output_type):
        super().add_node(node_for_adding=operator_id, mem=0, op_type=op_type, comp_cost={}, output_size=output_size,
                         output_type=output_type)

    def add_new_edge(self, source_id, dest_id):
        super().add_edge(u_of_edge=source_id, v_of_edge=dest_id)

    def getOperator(self, node_id):
        return self.nodes[node_id]

    def getConnection(self, source_id, dest_id):
        return self.edges[source_id, dest_id]

    def getOperatorOutputSizeAndType(self, node_id):
        return self.nodes[node_id]["output_size"], self.nodes[node_id]["output_type"]

    def getOperatorCompCostByDevice(self, node_id, device_id):
        if self.nodes[node_id] is None:
            raise ValueError("node {0} does not exist".format(node_id))
        if self.nodes[node_id]["comp_cost"] is None:
            raise ValueError("no comp_cost found")
        if device_id not in self.nodes[node_id]["comp_cost"].keys():
            raise ValueError(f"no {device_id} found")
        return self.nodes[node_id]["comp_cost"][device_id]

    def getOperatorCompCostSum(self, node_id):
        if node_id not in self.nodes:
            raise ValueError("node {0} does not exist".format(node_id))
        return int(sum(self.nodes[node_id]['comp_cost'].values()))

    def getAllOperators(self):
        return list(self.nodes(data=True))

    def getOperatorItems(self):
        return self.nodes.items()

    def getOperatorIDs(self):
        return list(self.nodes.keys())

    def getOperatorObjs(self) -> list[dict]:
        return list(self.nodes.values())

    def getAllEdges(self):
        return list(self.edges(data=True))

    def getEdgeItems(self):
        return self.edges.items()

    def getEdgeIDs(self) -> list[tuple[any, any]]:
        return list(self.edges.keys())

    def getEdgeObjs(self) -> list[dict]:
        return list(self.edges.values())

    def __str__(self):
        nodes_str = "\n".join(
            [f"Operator ID: {node_id}, Attributes: {attrs}" for node_id, attrs in self.nodes(data=True)])
        edges_str = "\n".join(
            [f"Edge from {src} to {dest}, Attributes: {attrs}" for src, dest, attrs in self.edges(data=True)])
        return f"CompGraph with {self.number_of_nodes()} operators and {self.number_of_edges()} edges.\n" \
               f"Operators:\n{nodes_str}\n\n" \
               f"Edges:\n{edges_str}"


# Undirected Graph
class DeviceGraph(DiGraph):

    def generata_fat_tree_topo(self, device_number, intra_node_band, inter_node_band, max_num_device_per_node):
        # num_nodes mean the number of servers where each server might have multiple devices
        num_nodes = device_number // max_num_device_per_node
        if device_number % max_num_device_per_node != 0:
            raise ValueError("device_number % max_num_device_per_node should be == 0")

        # Create nodes and make all devices within each node fully connected
        for node_index in range(num_nodes):
            start_device_id = node_index * max_num_device_per_node
            end_device_id = (node_index + 1) * max_num_device_per_node

            # Add intra-node edges
            for device_id in range(start_device_id, end_device_id):
                device_id_name = f"mock_device_{device_id}"
                self.add_new_node(device_id_name, 10000000000)
                for other_device_id in range(start_device_id, end_device_id):
                    if device_id != other_device_id:
                        other_device_id_name = f"mock_device_{other_device_id}"
                        self.add_new_edge(device_id_name, other_device_id_name, intra_node_band)

        # Add inter-node edges
        for node_index in range(num_nodes):
            current_node_start_device_id = node_index * max_num_device_per_node
            current_node_end_device_id = (node_index + 1) * max_num_device_per_node
            for current_device_id in range(current_node_start_device_id, current_node_end_device_id):
                device_id_name = f"mock_device_{current_device_id}"
                current_device_id_list = list(range(current_node_start_device_id, current_node_end_device_id))
                other_device_id_list = [element for element in list(range(device_number)) if
                                        element not in current_device_id_list]
                for other_device_id in other_device_id_list:
                    other_device_id_name = f"mock_device_{other_device_id}"
                    self.add_new_edge(device_id_name, other_device_id_name, inter_node_band)

    def is_fully_connected_bidirectional(self):
        """
        Check if every node in the directed graph is fully connected with all other nodes bidirectionally.

        :param digraph: A NetworkX directed graph (DiGraph)
        :return: True if every node is fully connected bidirectionally, False otherwise.
        """
        nodes = list(self.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u = nodes[i]
                v = nodes[j]
                if not (self.has_edge(u, v) and self.has_edge(v, u)):
                    return False
        return True

    def add_new_node(self, device_id, capacity):
        super().add_node(node_for_adding=device_id, memory_capacity=capacity)

    def add_new_fully_connected_node(self, device_id, capacity):
        super().add_node(node_for_adding=device_id, memory_capacity=capacity)
        for existing_node_id in self.getDeviceIDs():
            if existing_node_id != device_id:  # Avoid self-loop
                self.add_new_edge(device_id, existing_node_id, None)
                self.add_new_edge(existing_node_id, device_id, None)

    def add_new_edge(self, source_id, dest_id, bandwidth):
        super().add_edge(u_of_edge=source_id, v_of_edge=dest_id, bandwidth=bandwidth)

    def getDevice(self, node_id):
        return self.nodes[node_id]

    def getDeviceMaxMem(self, node_id):
        return self.nodes[node_id]['memory_capacity']

    def getConnection(self, source_id, dest_id):
        return self.edges[source_id, dest_id]

    def get_link_bandwidth(self, source_id, dest_id):
        return self.getConnection(source_id, dest_id)["bandwidth"]

    def update_link_bandwidth(self, source_id, dest_id, bandwidth):
        link = self.getConnection(source_id, dest_id)
        link["bandwidth"] = bandwidth

    def getAllDevices(self):
        return list(self.nodes(data=True))

    def getDeviceItems(self):
        return self.nodes.items()

    def getDeviceIDs(self) -> list[any]:
        return list(self.nodes.keys())

    def getDeviceObjs(self) -> list[dict]:
        return list(self.nodes.values())

    def getAllEdges(self):
        return list(self.edges(data=True))

    def getEdgeIDs(self) -> list[tuple[any, any]]:
        return list(self.edges.keys())

    def getEdgeObjs(self) -> list[dict]:
        return list(self.edges.values())

    def calUnitCommCostInUS(self, source_id, dest_id):

        # the source_id and dest_id are integers. Need to remap to the real device ip
        if source_id == dest_id:
            return 0
        speed = convert_data_size(self.get_link_bandwidth(source_id, dest_id), 'GB', 'bit')
        return convert_time(1 / speed, 's', 'us')

    def check_all_link_bandwidth(self):
        # sample edge is (1, 2, {'bandwidth': None})
        for edge in self.edges.data():
            if not edge[2]["bandwidth"]:
                raise ValueError(f"Bandwidth from {edge[0]} to {edge[1]} is not valid")

    def __str__(self):
        return ""


def visualize_graph(graph: DiGraph, show_node_labels=True, show_edge_labels=True):
    pos = spring_layout(graph, seed=500)  # Seed for reproducible layout
    draw(graph, pos, with_labels=False, node_size=10, font_size=8)
    if show_node_labels:
        # Create a dictionary with node labels including their attributes
        node_labels = {node: f"{node}\n" + '\n'.join([f"{key}: {value}" for key, value in graph.nodes[node].items()])
                       for node in graph.nodes()}
        draw_networkx_labels(graph, pos, node_labels, font_size=8)
        # Create a dictionary with edge labels including their attributes
    if show_edge_labels:
        edge_labels = {(u, v): '\n'.join([f"{key}: {value}" for key, value in data.items()]) for u, v, data in
                       graph.edges(data=True)}
        draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.show()


def combine_graphs(GList: [DiGraph]) -> Union[DiGraph, DeviceGraph, CompGraph]:
    # Create a new directed graph to combine G1 and G2
    if any(isinstance(graph, DeviceGraph) for graph in GList):
        G_combined = DeviceGraph()
    elif any(isinstance(graph, CompGraph) for graph in GList):
        G_combined = CompGraph()
    else:
        G_combined = DiGraph()

    # Add all nodes and edges from G1 and G2 to G_combined
    for graph in GList:
        for node, data in graph.nodes(data=True):
            G_combined.add_node(node, **data)
        for u, v, data in graph.edges(data=True):
            G_combined.add_edge(u, v, **data)

    # Connect every node in G1 to every node in G2
    for i in range(len(GList)):
        for j in range(len(GList)):
            if i != j:
                for node_i in GList[i].nodes():
                    for node_j in GList[j].nodes():
                        G_combined.add_edge(node_i, node_j)
    return G_combined


def has_more_than_one_component(digraph):
    # Calculate the number of weakly connected components
    num_components = number_weakly_connected_components(digraph)

    if num_components > 1:
        return True
    else:
        return False


def keep_largest_component(digraph):
    import networkx as nx
    # Find all weakly connected components
    weakly_connected_components = list(nx.weakly_connected_components(digraph))

    if len(weakly_connected_components) <= 1:
        print("The graph already has one or no weakly connected components.")
        return digraph

    # Find the largest component
    largest_component = max(weakly_connected_components, key=len)

    # Create a subgraph containing only the largest component
    largest_component_subgraph = digraph.subgraph(largest_component).copy()

    return largest_component_subgraph


def determine_node_order(graph, node1, node2):
    """
    Determines if node1 is prior or later than node2 in the topologically sorted order.

    Parameters:
    graph (nx.DiGraph): The directed graph.
    node1: The first node to check.
    node2: The second node to check.

    Returns:
    int: 1 if node1 is prior to node2,
         2 if node1 is later than node2,
         "not found" if either of the nodes is not in the graph.
    """
    try:
        # Perform topological sort
        sorted_nodes = list(topological_sort(graph))

        # Get the positions of the nodes
        pos_node1 = sorted_nodes.index(node1)
        pos_node2 = sorted_nodes.index(node2)

        # if node1 is ancestor to node2
        if pos_node1 < pos_node2:
            return 1
        # if node1 is successor of node2
        else:
            return 2
    except NetworkXError as e:
        # Handle case where the graph is not a DAG
        print(f"Error: {e}")
        return None
    except ValueError:
        # Handle case where node1 or node2 is not in the graph
        return None


def is_subgraph(sub_g: DiGraph, original_g: DiGraph) -> bool:
    gm = algorithms.isomorphism.DiGraphMatcher(original_g, sub_g)
    # Returns True if a subgraph of G1 is isomorphic to G2.
    return gm.subgraph_is_isomorphic()
