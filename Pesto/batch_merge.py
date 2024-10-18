import os
import json
import time
import numpy as np

from Pesto.op_merge_type import ops_merge_type
from Pesto.topo_height import topo_level_v2

strType = ['nmt-2lyr-1024-PCIE']
num_models = len(strType)
pci = 1
timelmt = 200

# Parameters
para_config = np.array([700, 12000, 80])
size_config = len(para_config)

for fileseq in range(1):
    for cur_size in range(size_config):

        print(f'Model {strType[fileseq]} starts at {time.strftime("%H:%M:%S.%f")[:-3]}')
        num_c2g_links = 2
        num_gpus = 2

        # Path and loading logic
        if pci:
            speed_mat = np.load('speed_mat_pinned.npy')
            strPath = 'D:\\GPU_Scheduling\\logs\\pcie'
            my_directory = 'D:\\GPU_Scheduling\\results\\pci\\coarsening\\'
        else:
            comm_nvlink = np.load('comm_nvlink02_pinned.npy')
            strPath = 'D:\\GPU_Scheduling\\logs\\nvlink_02'
            my_directory = 'D:\\GPU_Scheduling\\results\\NVLink_02\\Coarsening\\'

        strName = ['edge-newk_cpu_ops_ids.json', 'edge-newk_gpu_ops_ids.json',
                   'edge-updated_execk_graph-final_ids.json', 'edge-newk_gpukernel_ops_ids.json',
                   'colocs.json', 'memory.json']

        total_expected_ops = 120000
        map_ops = np.zeros((total_expected_ops, 1))

        coalloc = 1
        print('Reading json log...')
        start_time = time.time()

        # Loading JSON logs
        cpu_struct = json.load(open(os.path.join(strPath, strType[fileseq], strName[0])))
        gpu_struct = json.load(open(os.path.join(strPath, strType[fileseq], strName[1])))
        graph_struct = json.load(open(os.path.join(strPath, strType[fileseq], strName[2])))
        kernel_struct = json.load(open(os.path.join(strPath, strType[fileseq], strName[3])))

        coalloc_filepath = os.path.join(strPath, strType[fileseq], strName[4])
        if os.path.isfile(coalloc_filepath):
            coalloc_struct = json.load(open(coalloc_filepath))
        else:
            coalloc = 0

        memory_struct = json.load(open(os.path.join(strPath, strType[fileseq], strName[5])))

        # Process ops ids and graph information
        names_cpu = list(cpu_struct.keys())
        cpu_total_ops = len(names_cpu)
        names_kernel = list(kernel_struct.keys())
        kernel_total_ops = len(names_kernel)
        names_gpu = list(gpu_struct.keys())
        gpu_total_ops = len(names_gpu)

        size_node = cpu_total_ops + gpu_total_ops + kernel_total_ops
        comp = np.zeros((size_node, 2))
        origin_order = np.zeros(size_node)

        # Processing the nodes
        for i in range(cpu_total_ops):
            temp_loc = int(names_cpu[i][1:])
            origin_order[i] = temp_loc
            comp[temp_loc, 0] = 1  # CPU type
            comp[temp_loc, 1] = cpu_struct[names_cpu[i]]['median']

        for i in range(kernel_total_ops):
            temp_loc = int(names_kernel[i][1:])
            origin_order[cpu_total_ops + i] = temp_loc
            comp[temp_loc, 0] = 4  # Kernel type, initially
            comp[temp_loc, 1] = kernel_struct[names_kernel[i]]['median']

        for i in range(gpu_total_ops):
            temp_loc = int(names_gpu[i][1:])
            origin_order[cpu_total_ops + kernel_total_ops + i] = temp_loc
            comp[temp_loc, 0] = 2  # GPU type
            comp[temp_loc, 1] = gpu_struct[names_gpu[i]]['median']

        elapsed = time.time() - start_time
        print(f'Vertex information: {elapsed} seconds')

        # Processing the graph structure
        attach_list = [None] * 60000
        names_graph = list(graph_struct.keys())

        prec_list = [[None, None] for _ in range(size_node)]
        succ_list = [[None, None] for _ in range(size_node)]
        super_n = list(range(size_node))
        paring = np.zeros(size_node)

        for name in names_graph:
            temp_loc = int(name[1:])
            current_node = name
            prece_nodes = list(graph_struct[current_node].keys())

            if prece_nodes:
                indeg = len(prece_nodes)
                for prece_entry in range(indeg):
                    prece_id = int(prece_nodes[prece_entry][1:])
                    temp_val = graph_struct[current_node][prece_nodes[prece_entry]]

                    # GPU precedence
                    if comp[prece_id, 0] == 2:  # GPU type
                        paring[prece_id] = temp_loc
                        comp[prece_id, 0] = 3  # Kernel with GPU exec
                        comp[prece_id, 1] = 1  # Reduced compute time

                    # Successor and predecessor lists
                    if succ_list[prece_id][0] is None:
                        succ_list[prece_id][0] = []
                    succ_list[prece_id][0].append(temp_loc)

                    if prec_list[temp_loc][0] is None:
                        prec_list[temp_loc][0] = []
                    prec_list[temp_loc][0].append(prece_id)

                    if temp_val == 0:
                        succ_list[prece_id][1].append(0)
                        prec_list[temp_loc][1].append(0)
                    else:
                        succ_list[prece_id][1].append(temp_val['size'][0])
                        prec_list[temp_loc][1].append(temp_val['size'][0])

        # Merging kernels and GPU executions
        for i in range(size_node):
            gpu_id = paring[i]
            if gpu_id > 0:
                super_n, comp, prec_list, succ_list, attach_list = ops_merge_type(super_n, comp, prec_list, succ_list,
                                                                                  attach_list, i, gpu_id)

        elapsed = time.time() - start_time
        print(f'Kernel GPU merging: {elapsed} seconds')

import numpy as np

# Assuming comp, prec_list, succ_list, attach_list, and para_config are already defined
while np.count_nonzero(comp[:, 0]) > para_config[0, cur_size]:
    good_val = np.count_nonzero(comp[:, 0])
    level_ops = topo_level_v2(comp, prec_list, size_node)  # Assuming topo_level_v2 is implemented
    marked = np.zeros(size_node)

    edge_store = np.zeros((500000, 3))
    edge_ct = 0

    # Generating all edges
    for i in range(size_node):
        if comp[i, 0] > 0:  # If component type is valid
            suc_length = len(succ_list[i][0]) if succ_list[i][0] is not None else 0
            if suc_length > 0:
                edge_store[edge_ct:edge_ct + suc_length, 0] = i
                edge_store[edge_ct:edge_ct + suc_length, 1] = succ_list[i][0]
                edge_store[edge_ct:edge_ct + suc_length, 2] = succ_list[i][1]
                edge_ct += suc_length

    edge_store = edge_store[:edge_ct, :]  # Trim unused rows

    # Sort based on size (column 3)
    edge_store = edge_store[edge_store[:, 2].argsort()[::-1]]

    total_batch = np.zeros((5000, 2))
    row_ct = 0

    # Merging based on level_ops(successor) = level_ops(predecessor) + 1
    for i in range(edge_ct):
        cur_row = edge_store[i]
        father = int(cur_row[0])
        son = int(cur_row[1])

        if marked[father] == 1 or marked[son] == 1 or (comp[father, 0] == 1 and comp[son, 0] > 1) or (comp[son, 0] == 1 and comp[father, 0] > 1):
            continue

        ops_sort = sorted([level_ops[s] for s in succ_list[father][0] if s is not None])

        if (len(prec_list[son][0]) == 1 or len(succ_list[father][0]) == 1 or level_ops[son] == level_ops[father] + 1 or level_ops[son] < ops_sort[1]) \
                and comp[father, 1] + comp[son, 1] < para_config[1, cur_size] and len(attach_list[father]) < para_config[2, cur_size]:
            marked[father] = 1
            marked[son] = 1
            total_batch[row_ct, :] = [father, son]
            row_ct += 1

            pre_set = prec_list[son][0]
            for parent in pre_set:
                if level_ops[parent] == level_ops[father]:
                    marked[parent] = 1

    # If no valid merges were found, increase size limits and retry
    if total_batch[0, 0] == 0:
        para_config[1, cur_size] += 1000
        para_config[2, cur_size] += 1
        continue
    else:
        for z in range(row_ct):
            father = int(total_batch[z, 0])
            son = int(total_batch[z, 1])

            # Merge the operations using a custom merge function (to be implemented)
            super_n, comp, prec_list, succ_list, attach_list = ops_merge_type(super_n, comp, prec_list, succ_list, attach_list, father, son)

# Output final para_config
print(para_config)
print(f'Coarsening graph has {np.count_nonzero(comp[:, 0])} ops')