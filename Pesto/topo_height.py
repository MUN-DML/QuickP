import numpy as np


def topo_level_v2(comp, prec_list, size_node):
    seq = []
    level_ops = np.zeros(size_node, dtype=int)  # Equivalent to zeros(size_node, 1) in MATLAB

    # Identifying root-level nodes (Level 1)
    for i in range(size_node):
        if comp[i, 0] > 0 and len(prec_list[i]) == 0:  # Head node condition
            seq.append(i)
            level_ops[i] = 1

    marked = np.zeros(size_node, dtype=int)  # Track processed nodes (equivalent to marked in MATLAB)

    # Corrected: Mark each node in seq (which is a list) individually
    for idx in seq:
        marked[idx] = 1  # Mark the root nodes

    level_count = 2  # Start level counting from 2 (as level 1 is assigned to root nodes)

    while seq:
        temp_seq = seq
        seq = []

        for i in range(size_node):
            if comp[i, 0] > 0 and marked[i] == 0:  # If node is valid and not yet marked
                # Update the prerequisite list by removing nodes from the current sequence
                prec_list[i] = [p for p in prec_list[i] if p not in temp_seq]

                # If the node has no remaining prerequisites, mark it for the next level
                if len(prec_list[i]) == 0:
                    marked[i] = 1
                    level_ops[i] = level_count
                    seq.append(i)

        level_count += 1

    return level_ops


# Example usage
comp = np.array([[1], [1], [1], [1]])  # A sample component matrix with 4 nodes
prec_list = [[], [0], [1], [2]]  # Dependencies (e.g., node 1 depends on node 0, node 2 depends on node 1, etc.)
size_node = 4

level_ops = topo_level_v2(comp, prec_list, size_node)
print(level_ops)