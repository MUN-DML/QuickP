import numpy as np

import numpy as np


def ops_merge_type(super_n, comp, prec_list, succ_list, attach_list, parent, child):
    # Attach GPU to kernel
    succ_pos = np.where(np.array(succ_list[parent][0]) == child)[0][0]  # Find position of child in succ_list[parent]

    # Update attach_list and super_n
    attach_list[parent] = [parent] + attach_list[parent] + attach_list[child]  # Merge attach lists
    super_n[child] = parent
    super_n[attach_list[child]] = parent
    attach_list[child] = []

    # Update comp (merge child into parent)
    comp[parent][1] += comp[child][1]

    # Change type to child's type if not type 2
    if comp[parent][0] != 2:
        comp[parent][0] = comp[child][0]

    # Remove child from the comp matrix
    comp[child] = [0, 0]

    # Cut connection from parent to child
    del succ_list[parent][0][succ_pos]
    del succ_list[parent][1][succ_pos]

    # Add parent's grandchild to succ_list[parent]
    if succ_list[child][0]:
        grandchildren = succ_list[child][0]
        for i, grandchild in enumerate(grandchildren):
            if grandchild in succ_list[parent][0]:  # Already in parent's succ list
                idx = succ_list[parent][0].index(grandchild)
                succ_list[parent][1][idx] += succ_list[child][1][i]
            else:  # Add grandchild to parent's succ list
                succ_list[parent][0].append(grandchild)
                succ_list[parent][1].append(succ_list[child][1][i])

        # Update predecessor list for grandchildren
        for grandchild in succ_list[child][0]:
            pred_idx = prec_list[grandchild][0].index(child)
            if parent in prec_list[grandchild][0]:  # Already a predecessor
                idx = prec_list[grandchild][0].index(parent)
                prec_list[grandchild][1][idx] += prec_list[grandchild][1][pred_idx]
                del prec_list[grandchild][0][pred_idx]
                del prec_list[grandchild][1][pred_idx]
            else:  # Replace child with parent
                prec_list[grandchild][0][pred_idx] = parent

    # Add predecessors of child to parent
    if len(prec_list[child][0]) > 1:
        predecessors = prec_list[child][0]
        for pred in predecessors:
            if pred != parent:
                if pred in prec_list[parent][0]:  # Already in parent's predecessors
                    idx = prec_list[parent][0].index(pred)
                    prec_list[parent][1][idx] += prec_list[child][1][predecessors.index(pred)]
                    scp_idx = succ_list[pred][0].index(parent)
                    succ_list[pred][1][scp_idx] += prec_list[child][1][predecessors.index(pred)]
                else:  # Add new predecessor
                    prec_list[parent][0].append(pred)
                    prec_list[parent][1].append(prec_list[child][1][predecessors.index(pred)])
                    succ_list[pred][0].append(parent)
                    succ_list[pred][1].append(prec_list[child][1][predecessors.index(pred)])

                # Remove child from successor list of pred
                if child in succ_list[pred][0]:
                    child_idx = succ_list[pred][0].index(child)
                    del succ_list[pred][0][child_idx]
                    del succ_list[pred][1][child_idx]

    # Clear the child's predecessor and successor lists
    prec_list[child][0] = []
    prec_list[child][1] = []
    succ_list[child][0] = []
    succ_list[child][1] = []

    return super_n, comp, prec_list, succ_list, attach_list