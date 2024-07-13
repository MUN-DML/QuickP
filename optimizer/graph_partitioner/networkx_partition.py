import networkx as nx
from matplotlib import pyplot as plt
from networkx import DiGraph
from networkx.algorithms.community import kernighan_lin_bisection


def networkx_partition(G: DiGraph):
    G_undirected = G.to_undirected()
    # Perform graph partitioning using the Kernighan-Lin algorithm
    partition = kernighan_lin_bisection(G_undirected)

    # Assign partition labels to the original DiGraph nodes
    partition_dict = {node: (0 if node in partition[0] else 1) for node in G.nodes()}
    nx.set_node_attributes(G, partition_dict, 'partition')

    # Print the partition labels for each node
    for node, data in G.nodes(data=True):
        print(f"Node {node}: Partition {data['partition']}")

    # Visualize the partitioned graph
    colors = ['lightblue', 'lightgreen']
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=[colors[partition_dict[node]] for node in G.nodes()],
            edge_color='gray', node_size=2000, font_size=16)
    plt.title('Graph Partitioning using Kernighan-Lin Algorithm', size=20)
    plt.show()
