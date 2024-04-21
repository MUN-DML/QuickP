class Node:
    def __init__(self, op_id=0, name="operator", size=0, op_type="not specified"):
        self.id = op_id
        self.name = name
        self.size = size
        self.type = op_type
        self.compCost = 0

    def __str__(self):
        return f"NodeID: {self.id}, Node Name: {self.name}, Node Type: {self.type}, Memory: {self.size}"


class Edge:
    def __init__(self, edge_id=0, name="operator", sourceID=0, targetID=0, cost=0):
        self.id = edge_id
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

    def getNodes(self):
        return self.__nodes

    def getEdges(self):
        return self.__edges

    def __str__(self):
        return ""


node = Node()
print(node)
