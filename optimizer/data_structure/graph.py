import networkx as nx
import random


class Node:
    def __init__(self, op_id=0, name="operator", size=0, op_type="not specified", comp_cost=0):
        self.id = op_id
        self.name = name
        self.size = size
        self.type = op_type
        self.compCost = comp_cost

    def set_comp_cost(self, comp_cost):
        self.compCost = comp_cost

    def __str__(self):
        return f"NodeID: {self.id}, Node Name: {self.name}, Node Type: {self.type}, Memory: {self.size}"


class Edge:
    def __init__(self, name="", sourceID=0, targetID=0, cost=0):
        self.id = (sourceID, targetID)
        self.name = name
        self.sourceID = sourceID
        self.destID = targetID
        self.communicationCost = cost

    def __str__(self):
        return f"EdgeID: {self.id}, Edge Name: {self.name}, sourceNodeID Type: {self.sourceID}, destinationNodeID: {self.destID}"


class DAG:
    def __init__(self, name):
        self.name = name
        # {id1: Node1, id2: Node2}
        self.__nodes = {}
        # {(source_id, dest_id): Edge1}
        self.__edges = {}

    def add_node(self, node_id, node_type, size, comp_cost):
        self.__nodes[node_id] = Node(node_id, node_type, size, comp_cost)

    def add_edge(self, source_id, dest_id, com_cost):
        self.__edges[source_id, dest_id] = Edge(source_id, dest_id, com_cost)

    def getNodes(self):
        return self.__nodes

    def getEdges(self):
        return self.__edges

    def __str__(self):
        return ""


# Undirected Graph
class DeviceGraph(nx.Graph):

    def random_rebuild(self, device_num):
        device_list = list(range(device_num))
        for i in range(device_num):
            self.add_new_node(i, 200, random.randint(3, 60))
        # generate edge tuple list
        tuple_list = []
        for i in range(device_num - 1):
            for j in range(i + 1, device_num):
                tuple_list.append((i, j))
        for obj in tuple_list:
            self.add_new_edge(obj[0], obj[1], random.randint(50, 100))

    def add_new_node(self, device_id, comp_sp, capacity):
        nx.Graph.add_node(self, node_for_adding=device_id, source_id=device_id, computing_speed=comp_sp,
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

    def getAllEdges(self):
        return list(self.edges(data=True))

    def getEdgeIDs(self):
        return list(self.edges.keys())

    def __str__(self):
        return ""
