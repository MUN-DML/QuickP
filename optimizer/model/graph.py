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

    def add_new_nodes_from(self, operator_list):
        super().add_edges_from(ebunch_to_add=operator_list)

    def add_new_edge(self, source_id, dest_id):
        super().add_edge(u_of_edge=source_id, v_of_edge=dest_id)

    def add_new_edges_from(self, edge_tuple_list):
        super().add_edges_from(ebunch_to_add=edge_tuple_list)

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
class DeviceGraph(Graph):

    def random_rebuild(self, device_num):
        for i in range(device_num):
            self.add_new_node(device_id=i, comp_sp=random.randint(50, 200), capacity=random.randint(200, 400))
        # generate edge tuple list
        tuple_list = []
        for i in range(device_num - 1):
            for j in range(i + 1, device_num):
                tuple_list.append((i, j))
        for obj in tuple_list:
            self.add_new_edge(obj[0], obj[1], random.randint(50, 100))

    def add_new_node(self, device_id, comp_sp, capacity):
        super().add_node(node_for_adding=device_id, computing_speed=comp_sp,
                         memory_capacity=capacity)

    def add_new_nodes_from(self, device_list, comp_sp, capacity):
        super().add_edges_from(ebunch_to_add=device_list, computing_speed=comp_sp, memory_capacity=capacity)

    def add_new_edge(self, source_id, dest_id, com_sp):
        super().add_edge(u_of_edge=source_id, v_of_edge=dest_id, communication_speed=com_sp)

    def add_new_edges_from(self, edge_tuple_list, com_cost):
        super().add_edges_from(ebunch_to_add=edge_tuple_list, communication_cost=com_cost)

    def getDevice(self, node_id):
        return self.nodes[node_id]

    def getConnection(self, source_id, dest_id):
        return self.edges[source_id, dest_id]

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

    def calculateCommunicationCost(self, tensor_size, path_list):
        cost = 0
        for i in range(len(path_list) - 1):
            speed = self.getConnection(path_list[i], path_list[i + 1])["communication_speed"]
            cost += tensor_size / speed
        return cost

    def __str__(self):
        return ""


def visualize_graph(graph, show_labels=True):
    pos = spring_layout(graph, seed=225)  # Seed for reproducible layout
    draw(graph, pos, with_labels=show_labels, node_size=20)
    plt.show()
