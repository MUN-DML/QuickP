import numpy as np

import numpy as np


def topo_level_v2(comp, prec_list, size_node):
    """
    Compute the topological level of each node in the graph.

    Parameters:
    - comp: numpy array indicating the component type and processing time.
    - prec_list: list of predecessor nodes for each node.
    - size_node: total number of nodes.

    Returns:
    - level_ops: numpy array where level_ops[i] is the topological level of node i.
    """
    seq = []
    level_ops = np.zeros(size_node, dtype=int)

    # Identify the head nodes (those without predecessors)
    for i in range(size_node):
        if comp[i, 0] > 0 and not prec_list[i][0]:  # head node has no predecessors
            seq.append(i)
            level_ops[i] = 1  # root level

    marked = np.zeros(size_node, dtype=int)
    marked[seq] = 1

    level_count = 2
    while seq:
        temp_seq = seq
        seq = []

        for i in range(size_node):
            if comp[i, 0] > 0 and marked[i] == 0:
                # Remove processed nodes from the predecessor list
                prec_list[i][0] = [prec for prec in prec_list[i][0] if prec not in temp_seq]

                # If no more predecessors, mark the node and assign the level
                if not prec_list[i][0]:
                    marked[i] = 1
                    level_ops[i] = level_count
                    seq.append(i)

        level_count += 1

    return level_ops