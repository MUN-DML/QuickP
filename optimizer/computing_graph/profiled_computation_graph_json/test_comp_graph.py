import networkx as nx
import matplotlib.pyplot as plt

from optimizer.model.graph import CompGraph


def get_test_graph():
    # Step 1: Create a directed graph (DAG) using NetworkX
    G = CompGraph()

    # Step 3: Add edges (dependencies within each device)

    # Edges within Device 1
    G.add_edges_from([
        ('d0_1', 'd0_2'),  # d1_1 -> d1_2
        ('d0_2', 'd0_3'),  # d1_2 -> d1_3
        ('d0_3', 'd0_4'),   # d1_3 -> d1_4
        ('d0_1', 'd0_5'),
        ('d0_5', 'd0_6'),
        ('d0_4', 'd0_7')
    ])

    # Edges within Device 2
    G.add_edges_from([
        ('d1_1', 'd1_2'),  # d2_1 -> d2_2
        ('d1_2', 'd1_3'),  # d2_2 -> d2_3
        ('d1_3', 'd1_4'),   # d2_3 -> d2_4
        ('d1_1', 'd1_5'),
        ('d1_5', 'd1_6'),
        ('d1_4', 'd1_7'),
        ('d1_4', 'd1_8'),
        ('d1_8', 'd1_9')
    ])

    # Edges within Device 3
    G.add_edges_from([
        ('d2_1', 'd2_2'),  # d3_1 -> d3_2
        ('d2_2', 'd2_3'),  # d3_2 -> d3_3
        ('d2_3', 'd2_4')   # d3_3 -> d3_4
    ])

    # Step 4: Add cross-device dependencies (edges between devices) in a way that avoids cycles

    # From Device 1 to Device 2
    G.add_edges_from([
        ('d0_2', 'd1_2'),  # d1_2 -> d2_1
        ('d0_4', 'd1_3')   # d1_4 -> d2_3
    ])

    # From Device 2 to Device 3
    G.add_edges_from([
        ('d1_2', 'd2_4'),  # d2_2 -> d3_1
        ('d1_4', 'd2_4')   # d2_4 -> d3_3
    ])

    # Step 5: Assign the "comp_cost" attribute = 5 to every node
    comp_cost_dict = {"mock_device_0": 5.0, "mock_device_1": 5.0, "mock_device_2": 5.0}
    for node in G.nodes():
        G.nodes[node]['comp_cost'] = comp_cost_dict
    for edge in G.edges():
        G.edges[edge]['tensor_size_in_bit'] = 1600000

    assert nx.is_directed_acyclic_graph(G)
    '''
    # Define node colors based on device
    node_colors = []
    for node in G.nodes():
        if node.startswith('d1'):
            node_colors.append('lightblue')  # Device 1 nodes
        elif node.startswith('d2'):
            node_colors.append('lightgreen')  # Device 2 nodes
        else:
            node_colors.append('lightcoral')  # Device 3 nodes

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Spring layout for visualization

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=10, font_weight='bold', arrowsize=20)
    plt.title('Corrected Directed Acyclic Graph (DAG) with 3 Devices')
    plt.show()
    '''
    return G
'''
g = get_test_graph()
print(g.nodes(data=True))
print(g.edges(data=True))
'''


def get_test_graph_co_location():
    # Step 1: Create a directed graph (DAG) using NetworkX
    G = CompGraph()

    # Step 3: Add edges (dependencies within each device)

    # Edges within Device 1
    G.add_edges_from([
        ('a', 'b'),  # d1_1 -> d1_2
        ('a', 'g'),  # d1_2 -> d1_3
        ('f', 'g'),   # d1_3 -> d1_4
        ('c', 'b'),
        ('c', 'd'),
        ('d', 'e')
    ])

    # Step 5: Assign the "comp_cost" attribute = 5 to every node
    comp_cost_dict = {"mock_device_0": 5.0, "mock_device_1": 5.0, "mock_device_2": 5.0}
    for node in G.nodes():
        G.nodes[node]['comp_cost'] = comp_cost_dict
    for edge in G.edges():
        G.edges[edge]['tensor_size_in_bit'] = 1000

    assert nx.is_directed_acyclic_graph(G)

    return G