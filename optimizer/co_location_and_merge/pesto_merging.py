import numpy as np
import copy
import networkx as nx
import random

from DNN_model_tf.tf_model_enum import TFModelEnum
from optimizer.main_simulator.gurobi_util import init_computing_and_device_graph
from optimizer.model.graph import visualize_graph, CompGraph


# Compute the level of each node in the dag
def topo_level_v2(prec_list, node_size):
    seq = []
    level_ops = np.zeros(node_size, dtype=int)

    # find the head node
    for m in range(node_size):
        if not prec_list[m][0]:
            seq.append(m)
            level_ops[m] = 1

    marked = np.zeros(node_size, dtype=int)
    marked[seq] = 1
    level_count = 2
    while seq:
        temp_seq = [x + 1 for x in seq]
        seq = []
        for m in range(node_size):
            if marked[m] == 0:
                prec_list[m][0] = [x for x in prec_list[m][0] if x not in temp_seq]
                if not prec_list[m][0]:
                    marked[m] = 1
                    level_ops[m] = level_count
                    seq.append(m)
        level_count += 1
    return level_ops


# Pesto merge
def get_pesto_merging_result(original_dag, para_config):
    # Initial variable definitions from the DAG
    comp = np.array([original_dag.nodes[every_node]['weight'] for every_node in original_dag.nodes])
    ori_node_size = len(original_dag.nodes)
    print("ori_node_size: ", ori_node_size)
    print("----------------------------------------------")
    new_node_size = ori_node_size
    new_dag = copy.deepcopy(original_dag)
    merge_result = []
    loop = 0
    for dag_node1 in original_dag.nodes:
        print("nei")
        print(original_dag[dag_node1])
        break

    while new_node_size > para_config[0]:
        prec_list = [[[], []] for _ in range(ori_node_size)]
        succ_list = [[[], []] for _ in range(ori_node_size)]
        for dag_node in new_dag.nodes:
            predecessors = list(new_dag.predecessors(dag_node))
            if predecessors:
                prec_list[dag_node - 1][0] = [pred for pred in predecessors]
                prec_list[dag_node - 1][1] = [new_dag[pred][dag_node]['weight'] for pred in predecessors]
            successors = list(new_dag.successors(dag_node))
            if successors:
                succ_list[dag_node - 1][0] = [succ for succ in successors]
                succ_list[dag_node - 1][1] = [new_dag[dag_node][succ]['weight'] for succ in successors]

        loop_result = regular_merging(comp, ori_node_size, prec_list, succ_list, para_config[1])
        merge_result.extend(loop_result.tolist())
        # print("prec", prec_list)
        # print("succ", succ_list)
        # print("loop_result", loop_result)
        new_dag = merge_nodes_in_dag(new_dag, loop_result)
        new_node_size = len(new_dag.nodes)
        comp.fill(0)
        for new_node in new_dag.nodes:
            comp[new_node - 1] = new_dag.nodes[new_node]['weight']
        loop += 1
        print("loop: ", loop, "edge_merge: ", len(loop_result), "edge_total_merge: ",
              ori_node_size - new_node_size, "edge_remain: ", len(new_dag.edges), "nodes_remain: ", new_node_size)
        print("----------------------------------------------")
        if len(loop_result) == 0:
            break

    visualize_graph(new_dag, False, False)
    # print("result:", merge_result)
    # return merge_result
    return new_dag.number_of_nodes()


# Decide which nodes need to be merged in each iteration
def regular_merging(comp, node_size, prec_list, succ_list, max_weight):
    prec_list_copy = copy.deepcopy(prec_list)
    level_ops = topo_level_v2(prec_list_copy, node_size)
    marked = np.zeros(node_size, dtype=int)
    edge_store = np.zeros((500000, 3), dtype=object)
    edge_ct = 0

    # Generating all edges
    for j in range(node_size):
        suc_length = len(succ_list[j][0])
        if suc_length > 0:
            edge_store[edge_ct:edge_ct + suc_length, 0] = j + 1
            edge_store[edge_ct:edge_ct + suc_length, 1] = succ_list[j][0]
            edge_store[edge_ct:edge_ct + suc_length, 2] = succ_list[j][1]
            edge_ct += suc_length
    edge_store = edge_store[:edge_ct, :]

    # Sort based on size
    edge_store = edge_store[edge_store[:, 2].argsort()[::-1]]

    total_batch = np.zeros((5000, 2), dtype=int)
    row_ct = 0

    # Merge if level_ops(successor) == level_ops(predecessor) + 1
    for j in range(edge_ct):
        cur_row = j
        father = edge_store[cur_row, 0] - 1
        son = edge_store[cur_row, 1] - 1
        # print("father", father + 1, "son", son + 1)
        if marked[father] == 1 or marked[son] == 1:
            continue
        ops_sort = sorted([int(level_ops[k - 1]) for k in succ_list[father][0]])
        if (len(prec_list[son][0]) == 1 or len(succ_list[father][0]) == 1 or
            level_ops[son] == level_ops[father] + 1 or
            level_ops[son] == ops_sort[0]) and comp[father] + comp[son] < max_weight:
            marked[father] = 1
            marked[son] = 1
            total_batch[row_ct, :] = [father + 1, son + 1]
            row_ct += 1
            pre_set = prec_list[son][0]
            # print("add father", father + 1, "son", son + 1)
            for parent in pre_set:
                if level_ops[parent] == level_ops[father]:
                    marked[parent] = 1

    filtered_data = total_batch[~np.all(total_batch == 0, axis=1)]
    return filtered_data


# Returns the new dag after merging
def merge_nodes_in_dag(old_dag, merge_list):
    new_dag = copy.deepcopy(old_dag)

    for node_u, node_v in merge_list:
        if node_u > node_v:
            node_u, node_v = node_v, node_u

        new_node_weight = new_dag.nodes[node_u].get('weight', 0) + new_dag.nodes[node_v].get('weight', 0)
        new_dag.nodes[node_u]['weight'] = new_node_weight

        for predecessor in list(new_dag.predecessors(node_v)):
            if predecessor != node_u:
                weight_pre = new_dag[predecessor][node_v]['weight']
                new_dag.add_edge(predecessor, node_u, weight=weight_pre)

        for successor in list(new_dag.successors(node_v)):
            if successor != node_u:
                weight_succ = new_dag[node_v][successor]['weight']
                new_dag.add_edge(node_u, successor, weight=weight_succ)

        new_dag.remove_node(node_v)

    return new_dag

def get_dag(i):
    if i==1:
        deviceTopo, comp_graph = init_computing_and_device_graph(8, None, model_type=TFModelEnum.ALEXNET)
        visualize_graph(comp_graph, False, False)
        name_to_id = {name: i + 1 for i, name in enumerate(comp_graph.nodes)}
        comp_graph: CompGraph = nx.relabel_nodes(comp_graph, name_to_id)
        for node in comp_graph.nodes:
            comp_graph.nodes[node]['weight'] = comp_graph.getOperatorCompCostByDevice(node, 'mock_device_0')
        for u, v, data in comp_graph.edges(data=True):
            data['weight'] = data['tensor_size_in_bit']
        return comp_graph
    if i == 2:
        dag = nx.DiGraph()
        dag.add_weighted_edges_from(
            [(1, 6, 99), (1, 3, 99), (1, 4, 99), (2, 5, 99), (3, 8, 99), (4, 10, 99), (6, 7, 99),
             (6, 9, 99), (7, 9, 99), (7, 8, 99), (9, 10, 99)])
        node_weights = [99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
        for node, weight in enumerate(node_weights, start=1): dag.add_node(
            node, weight=weight)
        return dag
    if i==3:
        dag = nx.DiGraph()
        num_nodes = 1600
        for i in range(num_nodes):
            dag.add_node(i + 1, weight=99)
        for _ in range(3000):
            u = random.randint(1, num_nodes - 1)
            v = random.randint(u + 1, num_nodes)
            dag.add_edge(u, v, weight=99)
        return dag


if __name__ == "__main__":
    # Create a DAG for testing purposes

    dag=get_dag(1)
    # [0]:maximum node number, [1]:maximum weight after merge
    config = [300, 1500000]
    result = get_pesto_merging_result(dag, config)

    # dag1 = merge_nodes_in_dag(dag, [[1, 4], [7, 8]])
    # for node in dag1.nodes():
    #     pre = list(dag1.predecessors(node))
    #     suc = list(dag1.successors(node))
    #     print(f"Node {node}:")
    #     print(f" Predecessors: {pre}")
    #     print(f" Successors: {suc}")
