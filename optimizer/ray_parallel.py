# ray_parallel.py
import json
from enum import Enum

import torch
import tensorflow as tf
import os
import sys
import warnings
import ray

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.ssh_parallel import execute_command_on_server

warnings.filterwarnings("ignore")


def check_cluster():
    nodes = ray.nodes()
    for node in nodes:
        print(f"Node {node['NodeID']} with IP {node['NodeManagerAddress']}")
        print(f" - Resources: {node['Resources']}")
        print(f" - Alive: {node['Alive']}")


# ray stop; ray status
# head node: ray start --head --node-ip-address=192.168.0.66 --port=6379 --dashboard-port=8265 --num-gpus=1
# worker node: ray start --address='192.168.0.66:6379'
# ray job submit --working-dir /home/hola/Desktop/DNN -- python3 ray_parallel.py
# If project not on all servers: RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python3 ray_parallel.py
# If project already on all servers: RAY_ADDRESS='http://192.168.0.66:6379' ray job submit -- python3 ray_parallel.py
class TaskType(Enum):
    INTRA_NODE = 1
    INTER_NODE = 2


@ray.remote
def run_parallel_task(task_type, target_ip=None, local_hostname=None, target_name=None):
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    sys.path.append(project_root)

    if task_type == TaskType.INTRA_NODE:
        # Import and run intra-node task
        from optimizer.device_topo.intra_node_topo_parallel import get_intra_node_topo
        return get_intra_node_topo()
    elif task_type == TaskType.INTER_NODE:
        # Import and run inter-node task
        from optimizer.device_topo.intel_node_topo_parallel import get_intel_node_topo
        return get_intel_node_topo(target_ip, from_node=local_hostname, to_node=target_name)
    else:
        raise ValueError("Invalid task type")


if __name__ == "__main__":
    execute_command_on_server({"ip": "192.168.0.66", "username": "hola", "password": "1314520"}, "ray start --head --node-ip-address=192.168.0.66 --port=6379 --dashboard-port=8265 --num-gpus=1", timeout=30)
    execute_command_on_server({"ip": "192.168.0.6", "username": "hola", "password": "1314520"}, "ray start --address=192.168.0.66:6379", timeout=30)
    # Initialize Ray
    ray.init(_node_ip_address='192.168.0.6')
    check_cluster()
    intra_future = run_parallel_task.remote(TaskType.INTRA_NODE)
    intra_result = ray.get(intra_future)
    print("Intra-node task result:", intra_result)
    for result in intra_result:
        if result is not None:
            print("Result:", json.dumps({"bandwidths": result[0], "devices": result[1]}, indent=2))
        else:
            print("Result: None")
