from collections import defaultdict
from typing import Dict, List

import networkx as nx
from networkx.classes import DiGraph


def create_colocation_group_to_ops_map(op_graph: DiGraph) -> Dict[any, List[str]]:
    """Generate a dict that maps a colocation group to its op id list."""
    colocation_group_map = defaultdict(list)

    for op_id, op_data in op_graph.nodes(data=True):
        # Check if the node has a 'colocation_group' attribute
        group = op_data.get('colocation_group')
        if group is not None:
            colocation_group_map[group].append(op_id)

    return dict(colocation_group_map)


def get_op_group_map(groups_op_mapping: Dict[any, List[any]]) -> Dict[str, any]:
    op_group_map = {}

    for group, ops in groups_op_mapping.items():
        for op in ops:
            op_group_map[op] = group

    return op_group_map


