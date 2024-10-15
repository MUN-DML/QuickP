import json
import os
import random
from collections import defaultdict
from itertools import combinations
from typing import Union

import networkx as nx

from matplotlib import pyplot as plt
from networkx import DiGraph, draw_networkx_labels, spring_layout, draw, draw_networkx_edge_labels, node_link_graph, \
    node_link_data, number_weakly_connected_components, algorithms

from optimizer.operator_device_placement.metis.weight_functions import NodeWeightFunction, EdgeWeightFunction
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

    def generata_random_cost(self, device_number: int, adjustment_percent: int = None):
        if len(self.getOperatorIDs()) == 0:
            raise ValueError("need to profile the real DNN first")

        # Identify the original real device keys before adding mock devices
        original_real_devices = None
        for node in self.getOperatorObjs():
            original_real_devices = [key for key in node["comp_cost"].keys() if "mock_device_" not in key]
            break  # We only need to do this once, as the keys are the same for all nodes

        # Loop over each mock device
        for i in range(device_number):
            device_name = f"mock_device_{i}"

            # Select a random adjustment percent within [-adjustment_percent, adjustment_percent]
            if adjustment_percent:
                random_adjustment_percent = random.uniform(-adjustment_percent, adjustment_percent)
                # Calculate the adjustment factor
                adjustment_factor = 1 + random_adjustment_percent / 100
            else:
                adjustment_factor = 1

            # Apply the adjustment factor to all operators for this device
            for node in self.getOperatorObjs():
                assert node["comp_cost"] is not None

                # Calculate base using only the original real device data
                base = [node["comp_cost"][key] for key in original_real_devices]
                base_num = sum(base) / len(base)

                # Apply the adjustment factor to the base number
                adjusted_number = base_num * adjustment_factor
                node["comp_cost"][device_name] = adjusted_number

        # After processing all devices, remove the real devices' costs
        for node in self.getOperatorObjs():
            existing_real_device = list(node["comp_cost"].keys())
            for key in existing_real_device:
                if "mock_device_" not in key:  # Ensure only real devices are removed
                    del node["comp_cost"][key]

    def get_node_weight_function(self, node_weight_function: NodeWeightFunction):
        functions = {
            NodeWeightFunction.SUM_COMP_COST: self.getOperatorCompCostSum,
            NodeWeightFunction.AVE_COMP_COST: self.getOperatorCompCostAve,
        }
        return functions[node_weight_function]

    def get_edge_weight_function(self, edge_weight_function: EdgeWeightFunction):
        functions = {
            EdgeWeightFunction.MOCK_COMMUNICATION_COST: self.getOperatorMockCommCostInUS,
            EdgeWeightFunction.SOURCE_OUTPUT_TENSOR: self.getEdgeTensorSize,
        }
        return functions[edge_weight_function]

    def add_new_node(self, operator_id, op_type, memory=None, comp_cost_map=None):
        memory = memory or 0  # Default to 0 if memory is None or a falsy value
        comp_cost_map = comp_cost_map or {}
        super().add_node(node_for_adding=operator_id, mem=memory, op_type=op_type, comp_cost=comp_cost_map)

    def add_new_edge(self, source_id, dest_id, tensor_size_in_bit):
        super().add_edge(u_of_edge=source_id, v_of_edge=dest_id, tensor_size_in_bit=tensor_size_in_bit)

    def getOperator(self, node_id):
        return self.nodes[node_id]

    def getConnection(self, source_id, dest_id):
        return self.edges[source_id, dest_id]

    def getEdgeTensorSize(self, source_id, dest_id):
        return self.edges[source_id, dest_id]['tensor_size_in_bit']

    def getOperatorMockCommCostInUS(self, source_id, dest_id, mock_band_in_GB_per_second=20):
        output_size = self.getEdgeTensorSize(source_id, dest_id)
        result = convert_time(output_size / convert_data_size(mock_band_in_GB_per_second, 'GB', 'bit'), 's', 'us')
        return result

    def getOperatorCompCostByDevice(self, node_id, device_id):
        if self.nodes[node_id] is None:
            raise ValueError("node {0} does not exist".format(node_id))
        if self.nodes[node_id]["comp_cost"] is None:
            raise ValueError("no comp_cost found")
        if device_id not in self.nodes[node_id]["comp_cost"].keys():
            raise ValueError(f"no {device_id} found")
        return self.nodes[node_id]["comp_cost"][device_id]

    def getMemorySize(self, node_id):
        if self.nodes[node_id] is None:
            raise ValueError("node {0} does not exist".format(node_id))
        if self.nodes[node_id]["mem"] is None:
            raise ValueError("no mem found")
        return self.nodes[node_id]["mem"]

    def setMemorySize(self, node_id, mem):
        if self.nodes[node_id] is None:
            raise ValueError("node {0} does not exist".format(node_id))
        self.nodes[node_id]["mem"] = mem

    def getCompCostMapByOp(self, node_id):
        if self.nodes[node_id] is None:
            raise ValueError("node {0} does not exist".format(node_id))
        if self.nodes[node_id]["comp_cost"] is None:
            raise ValueError("no comp_cost found")
        return self.nodes[node_id]["comp_cost"]

    def set_node_computing_cost_map(self, node_id, comp_cost_map: dict):
        self.nodes[node_id]["comp_cost"] = comp_cost_map

    def getOpCompCostMapByDevice(self, device_id):
        comp_cost_map = {}
        for node_id in self.nodes:
            # Check if node exists and has computing cost for the specified device
            if self.nodes[node_id] is None:
                raise ValueError(f"node {node_id} does not exist")
            if "comp_cost" not in self.nodes[node_id] or self.nodes[node_id]["comp_cost"] is None:
                raise ValueError(f"no comp_cost found for node {node_id}")
            if device_id not in self.nodes[node_id]["comp_cost"].keys():
                raise ValueError(f"device {device_id} not found for node {node_id}")
            # Add the computing cost for the current node to the result dictionary
            comp_cost_map[node_id] = self.nodes[node_id]["comp_cost"][device_id]
        return comp_cost_map

    def getOperatorCompCostSum(self, node_id):
        if node_id not in self.nodes:
            raise ValueError("node {0} does not exist".format(node_id))
        return int(sum(self.nodes[node_id]['comp_cost'].values()))

    def getOperatorCompCostAve(self, node_id):
        if node_id not in self.nodes:
            raise ValueError("node {0} does not exist".format(node_id))
        return int(sum(self.nodes[node_id]['comp_cost'].values()) / len(self.nodes[node_id]['comp_cost']))

    def get_colocation_group(self, node_id):
        if node_id not in self.nodes:
            raise ValueError("node {0} does not exist".format(node_id))
        return self.nodes[node_id]["colocation_group"]

    def set_colocation_group(self, node_id, colocation_group):
        if node_id not in self.nodes:
            raise ValueError("node {0} does not exist".format(node_id))
        self.nodes[node_id]["colocation_group"] = [colocation_group]

    def update_colocation_group(self, node_id, colocation_group):
        if node_id not in self.nodes:
            raise ValueError("node {0} does not exist".format(node_id))
        # for not labelled node
        if "colocation_group" not in self.nodes[node_id]:
            self.nodes[node_id]["colocation_group"] = [colocation_group]
        # for already labelled node
        elif isinstance(self.get_colocation_group(node_id), list) and len(self.get_colocation_group(node_id)) >= 1:
            self.nodes[node_id]["colocation_group"].append(colocation_group)
        else:
            raise ValueError("colocation_group is not a list or len == 0")

    def create_colocation_group_to_ops_map(self) -> dict[any, list[str]]:
        """Generate a dict that maps a colocation group to its op id list."""
        colocation_group_map = defaultdict(list)

        for op_id, op_data in self.nodes(data=True):
            # Check if the node has a 'colocation_group' attribute
            group_list = op_data.get('colocation_group')
            # every node should have colocation group
            if group_list is None or not group_list:
                continue
            if len(group_list) > 1:
                # this function should only be called after the group merge
                raise ValueError(f'colocation group {op_id} has multiple colocation_groups')
            group_id = group_list[0]
            colocation_group_map[group_id].append(op_id)
        # {'':list(op_graph.nodes)[0:40], '1': list(op_graph.nodes)[41:80], '2': list(op_graph.nodes)[80:121]}
        # {'':list(op_graph.nodes)[0:600], '1': list(op_graph.nodes)[601:1200], '2': list(op_graph.nodes)[1201:1600]}
        return dict(colocation_group_map)

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

    def clean_marginal_operators(self):
        nodes_to_remove = [node for node in self.nodes if
                           (self.in_degree(node) + self.out_degree(node)) == 0
                           and self.getOperatorCompCostSum(node) == 0]
        print("removed nodes:", nodes_to_remove)
        self.remove_nodes_from(nodes_to_remove)

    def get_comp_cost_sum_ratio(self, number_of_device: int):
        device_sums = defaultdict(float)

        for node_id in self.nodes():
            comp_cost = self.nodes[node_id]['comp_cost']
            for device, cost in comp_cost.items():
                device_sums[device] += cost

        return dict(list(device_sums.items())[:number_of_device])

    def is_edge_mergable(self, source, target):
        if not self.has_edge(source, target):
            raise ValueError(f"Edge {source}, {target} does not exist")
        # merging 𝑢, 𝑣 is acyclic if and only if (𝑢, 𝑣) is the only path from 𝑢 to 𝑣 on G.
        # If u's out degree or v's in degree is 1, there can only exist one path
        if self.out_degree(source) == 1 or self.in_degree(target) == 1:
            return True
        #  the size of a minimum cut set is equal to the maximum number of disjoint paths that can be found between any pair of vertices.
        # paths = list(nx.node_disjoint_paths(computation_graph, source, target))
        min_cut_size = len(nx.minimum_edge_cut(self, source, target))
        return min_cut_size <= 1

    def __str__(self):
        nodes_str = "\n".join(
            [f"Operator ID: {node_id}, Attributes: {attrs}" for node_id, attrs in self.nodes(data=True)])
        edges_str = "\n".join(
            [f"Edge from {src} to {dest}, Attributes: {attrs}" for src, dest, attrs in self.edges(data=True)])
        return f"CompGraph with {self.number_of_nodes()} operators and {self.number_of_edges()} edges.\n" \
               f"Operators:\n{nodes_str}\n\n" \
               f"Edges:\n{edges_str}"


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

    def get_fastest_link(self) -> tuple[str, str, dict]:
        fastest_edge = max(self.edges.data(), key=lambda edge: edge[2].get('bandwidth', 0))
        return fastest_edge

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


def get_weekly_connected_components(G):
    # Get weakly connected components
    weakly_connected_components = list(nx.weakly_connected_components(G))


def keep_largest_component(digraph):
    # Find all weakly connected components
    weakly_connected_components = list(nx.weakly_connected_components(digraph))

    if len(weakly_connected_components) <= 1:
        print("The graph already has one or no weakly connected components.")
        return digraph

    # Find the largest component
    largest_component = max(weakly_connected_components, key=len)

    # Identify nodes to be deleted (those not in the largest component)
    nodes_to_delete = set(digraph.nodes()) - set(largest_component)
    print(f"Nodes to be deleted: {nodes_to_delete}")

    # Create a subgraph containing only the largest component
    largest_component_subgraph = digraph.subgraph(largest_component).copy()

    return largest_component_subgraph


def determine_node_order(topo_positions, node1, node2):
    """
    Determines if node1 is prior or later than node2 in the topologically sorted order.

    Parameters:
    topo_positions (dict): A dictionary mapping nodes to their topological positions.
    node1: The first node to check.
    node2: The second node to check.

    Returns:
    int: 1 if node1 is prior to node2,
         2 if node1 is later than node2,
         None if either of the nodes is not in the graph.
    """
    try:
        pos_node1 = topo_positions[node1]
        pos_node2 = topo_positions[node2]
        # print(pos_node1, pos_node2)
        # if node1 is ancestor to node2
        if pos_node1 < pos_node2:
            return 1
        # if node1 is successor of node2
        else:
            return 2
    except KeyError:
        # Handle case where node1 or node2 is not in the graph
        return None


def topo_order_until_node(dag, target_node):
    assert nx.is_directed_acyclic_graph(dag)
    topo_sorted_gen = nx.topological_sort(dag)
    for node in topo_sorted_gen:
        yield node
        if node == target_node:
            break
    else:
        raise ValueError(f"Node {target_node} not found in the DAG")


def is_subgraph(sub_g: DiGraph, original_g: DiGraph) -> bool:
    gm = algorithms.isomorphism.DiGraphMatcher(original_g, sub_g)
    # Returns True if a subgraph of G1 is isomorphic to G2.
    return gm.subgraph_is_isomorphic()


# We need to identify all maximal sets of nodes where all nodes in each set are mutually non-reachable
def find_non_connected_pairs(G):
    # Compute the transitive closure of the graph
    TC = nx.transitive_closure(G)

    # Generate all unique pairs of nodes
    all_nodes = list(nx.topological_sort(G))
    pairs = combinations(all_nodes, 2)

    # Filter pairs to find non-connected pairs
    non_connected_pairs = [
        (node1, node2) for node1, node2 in pairs
        if not TC.has_edge(node1, node2) and not TC.has_edge(node2, node1)
    ]

    return non_connected_pairs


# Function to check if two nodes are not connected
def is_not_connected(G, node_a, node_b):
    # Check if there is no path from node_a to node_b and vice versa
    return not nx.has_path(G, node_a, node_b) and not nx.has_path(G, node_b, node_a)


# Modify the function to store unique tensor sizes in a set
def operator_unique_tensor_sizes(graph: CompGraph):
    tensor_size_dict = {}

    for node in graph.getOperatorIDs():
        # Get all outgoing edges for the node
        outgoing_edges = graph.out_edges(node, data=True)
        tensor_sizes = {data['tensor_size_in_bit'] for _, _, data in outgoing_edges}  # Use a set to ensure uniqueness

        # If the node has outgoing edges, store the unique tensor sizes
        if tensor_sizes:
            tensor_size_dict[node] = tensor_sizes

    return tensor_size_dict
