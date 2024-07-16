import networkx as nx

from optimizer.graph_partitioner.subgraph_util import expand_subgraph, creates_cycle
from optimizer.model.graph import is_subgraph


def split_DAG_min_inter_subgraph_edges(G, M):
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("DAG is not a directed acyclic graph")
    # Step 1: Preprocessing
    in_degree = {node: 0 for node in G.nodes()}
    out_degree = {node: 0 for node in G.nodes()}

    for u, v in G.edges():
        in_degree[v] += 1
        out_degree[u] += 1

    degree = {node: in_degree[node] + out_degree[node] for node in G.nodes()}
    sorted_nodes = sorted(degree, key=degree.get, reverse=True)

    # Step 2: Initialization
    subgraphs = [nx.DiGraph() for _ in range(M)]
    load_balance = [0] * M
    inter_subgraph_edges = 0

    # Step 3: Node Assignment
    for node in sorted_nodes:
        min_edges = float('inf')
        best_subgraph_index = -1

        # find the subgraph that results in the fewest additional inter-subgraph edges after adding this node with edges
        for current_subgraph_index in range(M):
            if not creates_cycle(subgraphs[current_subgraph_index], node, G):
                # computes the number of inter-subgraph edges that would result from adding the current node to subgraph i.
                current_edges = compute_inter_subgraph_edges(node, subgraphs[current_subgraph_index], G, subgraphs)
                if current_edges < min_edges or (
                        current_edges == min_edges and load_balance[current_subgraph_index] < load_balance[
                    best_subgraph_index]):
                    min_edges = current_edges
                    best_subgraph_index = current_subgraph_index

        # Improved: Handle case when no best subgraph is found
        if best_subgraph_index == -1:
            best_subgraph_index = load_balance.index(min(load_balance))

        expand_subgraph(subgraphs[best_subgraph_index], node, G)

        load_balance[best_subgraph_index] += 1

    assert len(subgraphs) == M

    for subgraph in subgraphs:
        assert is_subgraph(subgraph, G)
        assert nx.is_directed_acyclic_graph(subgraph)

    return subgraphs


# calculate the number of edges that would cross between different subgraphs if a node were added to a particular subgraph
def compute_inter_subgraph_edges(node, subgraph, G, all_subgraphs):
    edges = 0
    for neighbor in G.neighbors(node):
        if neighbor not in subgraph and find_subgraph(neighbor, all_subgraphs) != find_subgraph(node, all_subgraphs):
            edges += 1
    for neighbor in G.predecessors(node):
        if neighbor not in subgraph and find_subgraph(neighbor, all_subgraphs) != find_subgraph(node, all_subgraphs):
            edges += 1
    return edges


# indicate which subgraph the node is in
def find_subgraph(node, subgraphs):
    for i, subgraph in enumerate(subgraphs):
        if node in subgraph:
            return i
    return -1
