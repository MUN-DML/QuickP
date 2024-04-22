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

class Tree:
    def __init__(self, name):
        self.name = name
        # {id1: Node1, id2: Node2}
        self.__nodes = {}
        self.__edges = {}

    def add_node(self, node_id, node_type, size, comp_cost):
        self.__nodes[node_id] = Node(node_id, node_type, size, comp_cost)

    def add_edge(self, edge_id, source_id, dest_id, com_cost):
        self.__edges[edge_id] = Node(edge_id, source_id, dest_id, com_cost)

    def getNodes(self):
        return self.__nodes

    def getEdges(self):
        return self.__edges

    def __str__(self):
        return ""

node = Node()
print(node)
