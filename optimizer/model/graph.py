import networkx
import random

from matplotlib import pyplot as plt
from networkx import DiGraph, Graph, gnp_random_graph, spring_layout, draw


class CompGraph(DiGraph):
    def random_rebuild(self, operator_num):
        G = gnp_random_graph(operator_num, 0.5, directed=True)
        for i in G.nodes():
            self.add_new_node(operator_id=i, op_type="not specified")
        for (u, v) in G.edges():
            if u < v:
                self.add_new_edge(u, v)

    def add_new_node(self, operator_id, op_type, output_size=0):
        super().add_node(node_for_adding=operator_id, mem=0, op_type=op_type, comp_cost={}, output_size=output_size)

    def add_new_edge(self, source_id, dest_id):
        super().add_edge(u_of_edge=source_id, v_of_edge=dest_id)

    def getOperator(self, node_id):
        return self.nodes[node_id]

    def getConnection(self, source_id, dest_id):
        return self.edges[source_id, dest_id]

    def getAllOperators(self):
        return list(self.nodes(data=True))

    def getOperatorIDs(self):
        return list(self.nodes.keys())

    def getOperatorObjs(self):
        return list(self.nodes.values())

    def getAllEdges(self):
        return list(self.edges(data=True))

    def getEdgeIDs(self):
        return list(self.edges.keys())

    def getEdgeObjs(self):
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

    def add_new_node(self, device_id, capacity):
        super().add_node(node_for_adding=device_id, memory_capacity=capacity)
        for existing_node_id in self.getDeviceIDs():
            if existing_node_id != device_id:  # Avoid self-loop
                self.add_new_edge(device_id, existing_node_id, None)
                self.add_new_edge(existing_node_id, device_id, None)

    def add_new_edge(self, source_id, dest_id, bandwidth):
        super().add_edge(u_of_edge=source_id, v_of_edge=dest_id, bandwidth=bandwidth)

    def getDevice(self, node_id):
        return self.nodes[node_id]

    def getConnection(self, source_id, dest_id):
        return self.edges[source_id, dest_id]

    def update_link_bandwidth(self, source_id, dest_id, bandwidth):
        link = self.getConnection(source_id, dest_id)
        link["bandwidth"] = bandwidth

    def getAllDevices(self):
        return list(self.nodes(data=True))

    def getDeviceIDs(self):
        return list(self.nodes.keys())

    def getDeviceObjs(self):
        return list(self.nodes.values())

    def getAllEdges(self):
        return list(self.edges(data=True))

    def getEdgeIDs(self):
        return list(self.edges.keys())

    def getEdgeObjs(self):
        return list(self.edges.values())

    def calculateCommunicationCost(self, tensor_size, source_id, dest_id):
        speed = self.getConnection(source_id, dest_id)["bandwidth"]
        return tensor_size / speed

    def check_all_link_bandwidth(self):
        # sample edge is (1, 2, {'bandwidth': None})
        for edge in self.edges.data():
            if not edge[2]["bandwidth"]:
                raise ValueError(f"Bandwidth from {edge[0]} to {edge[1]} is not valid")

    def __str__(self):
        return ""


def visualize_graph(graph, show_labels=True):
    pos = spring_layout(graph, seed=225)  # Seed for reproducible layout
    draw(graph, pos, with_labels=show_labels, node_size=20)
    plt.show()


def combine_graphs(GList: [DiGraph]) -> DiGraph:
    # Create a new directed graph to combine G1 and G2
    G_combined = DiGraph()

    # Add all nodes and edges from G1 and G2 to G_combined
    for graph in GList:
        G_combined.add_nodes_from(graph.nodes())
        G_combined.add_edges_from(graph.edges())

    # Connect every node in G1 to every node in G2
    for i in range(len(GList)):
        for j in range(len(GList)):
            if i != j:
                for node_i in GList[i].nodes():
                    for node_j in GList[j].nodes():
                        G_combined.add_edge(node_i, node_j)

    return G_combined
