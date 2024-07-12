import networkx as nx

from optimizer.model.graph import visualize_graph, is_subgraph


def split_DAG_min_inter_subgraph_edges(G, M):
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
                        current_edges == min_edges and load_balance[current_subgraph_index] < load_balance[best_subgraph_index]):
                    min_edges = current_edges
                    best_subgraph_index = current_subgraph_index

        # Improved: Handle case when no best subgraph is found
        if best_subgraph_index == -1:
            best_subgraph_index = load_balance.index(min(load_balance))

        subgraphs[best_subgraph_index].add_node(node, **G.nodes[node])
        for neighbor in G.neighbors(node):
            if neighbor in subgraphs[best_subgraph_index]:
                subgraphs[best_subgraph_index].add_edge(node, neighbor, **G[node][neighbor])
        for neighbor in G.predecessors(node):
            if neighbor in subgraphs[best_subgraph_index]:
                subgraphs[best_subgraph_index].add_edge(neighbor, node, **G[neighbor][node])

        load_balance[best_subgraph_index] += 1

    assert len(subgraphs) == M

    for subgraph in subgraphs:
        assert is_subgraph(subgraph, G)

    return subgraphs


# expand the subgraph by one node and make it still a subgraph of original_graph
def expand_subgraph(sub_graph, node, original_graph):
    if node not in original_graph.nodes:
        raise ValueError("node {} not in the original graph".format(node))
    # Add the node to the subgraph
    sub_graph.add_node(node)

    # Add edges from the original graph to the subgraph
    for predecessor in original_graph.predecessors(node):
        if predecessor in original_graph.nodes and not sub_graph.has_edge(predecessor, node):
            sub_graph.add_edge(predecessor, node, **original_graph.get_edge_data(predecessor, node) or {})

    for successor in original_graph.successors(node):
        # successor must be in the original G to prevent bring extra nodes and edges
        if successor in original_graph.nodes and not sub_graph.has_edge(node, successor):
            sub_graph.add_edge(node, successor, **original_graph.get_edge_data(node, successor) or {})


def creates_cycle(subgraph, node, G):
    # Make the update on the copied graph
    subgraph_copy = subgraph.copy()

    expand_subgraph(subgraph_copy, node, G)

    # Check for cycles
    has_cycle = not nx.is_directed_acyclic_graph(subgraph_copy)

    return has_cycle


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


def find_subgraph(node, subgraphs):
    for i, subgraph in enumerate(subgraphs):
        if node in subgraph:
            return i
    return -1
