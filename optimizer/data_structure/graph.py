import networkx as nx
import random


class CompGraph(nx.DiGraph):
    def random_rebuild(self, operator_num):
        operator_list = list(range(operator_num))
        for i in range(operator_num):
            self.add_new_node(i, random.randint(50, 200), random.randint(3, 60), "not specified")
        # generate edge tuple list
        tuple_list = []
        for i in range(operator_num - 1):
            for j in range(i + 1, operator_num):
                tuple_list.append((i, j))
        for obj in tuple_list:
            self.add_new_edge(obj[0], obj[1])

    def add_new_node(self, operator_id, size, comp_cost, op_type):
        nx.DiGraph.add_node(self, node_for_adding=operator_id, size=size, computing_cost=comp_cost, op_type=op_type)

    def add_new_nodes_from(self, operator_list):
        nx.DiGraph.add_edges_from(self, ebunch_to_add=operator_list)

    def add_new_edge(self, source_id, dest_id):
        nx.DiGraph.add_edge(self, u_of_edge=source_id, v_of_edge=dest_id)

    def add_new_edges_from(self, edge_tuple_list):
        nx.DiGraph.add_edges_from(self, ebunch_to_add=edge_tuple_list)

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
class DeviceGraph(nx.Graph):

    def random_rebuild(self, device_num):
        for i in range(device_num):
            self.add_new_node(i, random.randint(50, 200), random.randint(200, 400))
        # generate edge tuple list
        tuple_list = []
        for i in range(device_num - 1):
            for j in range(i + 1, device_num):
                tuple_list.append((i, j))
        for obj in tuple_list:
            self.add_new_edge(obj[0], obj[1], random.randint(50, 100))

    def add_new_node(self, device_id, comp_sp, capacity):
        nx.Graph.add_node(self, node_for_adding=device_id, computing_speed=comp_sp,
                          memory_capacity=capacity)

    def add_new_nodes_from(self, device_list, comp_sp, capacity):
        nx.Graph.add_edges_from(self, ebunch_to_add=device_list, computing_speed=comp_sp, memory_capacity=capacity)

    def add_new_edge(self, source_id, dest_id, com_sp):
        nx.Graph.add_edge(self, u_of_edge=source_id, v_of_edge=dest_id, communication_speed=com_sp)

    def add_new_edges_from(self, edge_tuple_list, com_cost):
        nx.Graph.add_edges_from(self, ebunch_to_add=edge_tuple_list, communication_cost=com_cost)

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

    def __str__(self):
        return ""


class CompCostMatrix:
    def __init__(self, operator_ids, device_ids):
        self.cost_matrix = {}
        for operator_id in operator_ids:
            for device_id in device_ids:
                self.cost_matrix[operator_id, device_id] = random.randint(50, 300)
