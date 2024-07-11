import networkx as nx


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
        best_subgraph = -1

        for i in range(M):
            if not creates_cycle(subgraphs[i], node, G):
                current_edges = compute_inter_subgraph_edges(node, subgraphs[i], G, subgraphs)
                if current_edges < min_edges or (
                        current_edges == min_edges and load_balance[i] < load_balance[best_subgraph]):
                    min_edges = current_edges
                    best_subgraph = i

        subgraphs[best_subgraph].add_node(node, **G.nodes[node])
        for neighbor in G.neighbors(node):
            if neighbor in subgraphs[best_subgraph]:
                subgraphs[best_subgraph].add_edge(node, neighbor, **G[node][neighbor])
        for neighbor in G.predecessors(node):
            if neighbor in subgraphs[best_subgraph]:
                subgraphs[best_subgraph].add_edge(neighbor, node, **G[neighbor][node])

        load_balance[best_subgraph] += 1

    # Optional: refine partitions to reduce inter-subgraph edges
    refine_partitions(subgraphs, G, M)

    return subgraphs, inter_subgraph_edges


def creates_cycle(subgraph, node, G):
    # Add the node temporarily to the subgraph
    subgraph.add_node(node)

    # Add edges from the original graph to the subgraph
    for neighbor in G.neighbors(node):
        if neighbor in subgraph:
            subgraph.add_edge(node, neighbor, **G[node][neighbor])
    for neighbor in G.predecessors(node):
        if neighbor in subgraph:
            subgraph.add_edge(neighbor, node, **G[neighbor][node])

    # Check for cycles
    has_cycle = not nx.is_directed_acyclic_graph(subgraph)

    # Remove the node and its edges to restore the subgraph
    subgraph.remove_node(node)

    return has_cycle


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


def refine_partitions(subgraphs, G, M):
    # Local optimization to swap nodes between subgraphs to reduce inter-subgraph edges
    pass