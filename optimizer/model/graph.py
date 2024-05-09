import networkx
import random

from matplotlib import pyplot as plt
from networkx import DiGraph, Graph, gnp_random_graph, spring_layout, draw


class CompGraph(DiGraph):
    def random_rebuild(self, operator_num):
        G = gnp_random_graph(operator_num, 0.5, directed=True)
        for i in G.nodes():
            self.add_new_node(operator_id=i, mem=random.randint(50, 200), op_type="not specified")
        for (u, v) in G.edges():
            if u < v:
                self.add_new_edge(u, v)

    def add_new_node(self, operator_id, mem, op_type):
        super().add_node(node_for_adding=operator_id, mem=mem, op_type=op_type)

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


class CompCostMatrix:
    def __init__(self, operator_ids, device_ids):
        self.cost_matrix = {}
        for operator_id in operator_ids:
            for device_id in device_ids:
                # self.cost_matrix[operator_id, device_id] means the computing cost of this operator on this device
                self.cost_matrix[operator_id, device_id] = random.randint(50, 300)


def visualize_graph(graph, show_labels=True):
    pos = spring_layout(graph, seed=225)  # Seed for reproducible layout
    draw(graph, pos, with_labels=show_labels)
    plt.show()
