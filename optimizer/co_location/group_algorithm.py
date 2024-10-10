from networkx.classes import DiGraph

from optimizer.co_location.grouper_util import sort_by_critical_score, bfs_with_colocation
from optimizer.model.graph import CompGraph, DeviceGraph


# group with nodes with small computing cost but a large communication cost if on different devices
def quickcut_group(computing_graph: CompGraph, device_topo: DeviceGraph):
    computing_cost_dict = computing_graph.getOpCompCostMapByDevice(device_topo.getDeviceIDs()[0])
    node_order = sort_by_critical_score(computing_graph, computing_cost_dict)
    for node in node_order:
        if 'colocation_group' in computing_graph.nodes[node]:
            continue
        bfs_with_colocation(computing_graph, device_topo, node, computing_cost_dict)


class ColocationGroupMap():
    """Data structure that maps group to a set of groups that are co-located."""

    def __init__(self):
        self._map = {}

    def colocate(self, group1, group2):
        prev_group1 = self._map.get(group1, set([group1]))
        prev_group2 = self._map.get(group2, set([group2]))
        new_group = prev_group1 | prev_group2

        # update existing groups
        for group in new_group:
            self._map[group] = new_group

    def __getitem__(self, op):
        return self._map[op]

    def __len__(self):
        return len(self._map)

    def items(self):
        return self._map.items()


def _update_colocation_group(op_graph, colocation_group_map):
    """Updates colocation groups of operators in op_graph with new mapping."""
    # pick the shortest group name among groups as new group name
    group_dict = {group: min(group_set, key=_len_and_str)
                  for group, group_set in colocation_group_map.items()}

    # print merged groups
    reverse_mapping = {}
    for prev_name, new_name in group_dict.items():
        if new_name in reverse_mapping:
            reverse_mapping[new_name].append(prev_name)
        else:
            reverse_mapping[new_name] = [prev_name]
    for new_name, prev_names in reverse_mapping.items():
        _LOGGER.debug('Change group: %s -> %s', sorted(prev_names), new_name)

    # update colocation group
    for _, op_data in op_graph.nodes.items():
        if isinstance(op_data['colocation_group'], list):
            new_group = None
            for colocation_group in op_data['colocation_group']:
                ret = group_dict.get(colocation_group, colocation_group)
                if new_group is None:
                    new_group = ret
                else:
                    assert new_group == ret, 'node=%s, cur=%s, new=%s' % (
                        op_data['name'], new_group, ret)
        else:
            prev_group_name = op_data['colocation_group']
            new_group = group_dict.get(prev_group_name, prev_group_name)

        op_data['colocation_group'] = new_group


def _run_colocation_step(op_graph, ignore_control_edges):
    """Check whether there are operators that can be co-located.

    When the output of an operator is consumed only by another operator,
    assign the same colocation group for them

    Returns:
      True if there are opeartors that can be co-located.
      False, otherwise.
    """
    colocation_candidates = ColocationGroupMap()

    for op_id, op_data in op_graph.nodes.items():
        # TODO: should we consider tensor-wise? not operator wise?
        out_edges = list(op_graph.out_edges(op_id))
        if len(out_edges) != 1:
            continue

        next_op_id = out_edges[0][1]

        # pass control edges because this does not have data transfer
        edge_data = op_graph.get_edge_data(op_id, next_op_id)
        if ignore_control_edges and edge_data["is_control"]:
            continue

        next_op_data = op_graph.nodes[next_op_id]

        op_group = op_data['colocation_group']
        next_op_group = next_op_data['colocation_group']

        if op_group != next_op_group:
            # these two can be colocated
            _LOGGER.debug('Possible colocation ops. %s[%s] -> %s[%s]',
                          op_data['name'],
                          op_group,
                          next_op_data['name'],
                          next_op_group)
            colocation_candidates.colocate(op_group, next_op_group)

    if len(colocation_candidates) > 0:
        _update_colocation_group(op_graph, colocation_candidates)
        return True

    return False

