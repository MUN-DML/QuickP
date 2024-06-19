# ray_parallel.py
import torch
import tensorflow as tf
import os
import sys
import warnings
import ray

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
# RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- python3 ray_parallel.py
# RAY_ADDRESS='http://127.0.0.1:8265' ray job submit -- python3 ray_parallel.py
@ray.remote
def remote_setup_and_run():
    # The following two lines must be present
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    sys.path.append(project_root)

    # Import the function
    from optimizer.device_topo.intra_node_topo_parallel import get_intra_node_topo
    return get_intra_node_topo()


if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='auto')
    check_cluster()
    # Run the function in parallel on the Ray cluster
    print(ray.get(remote_setup_and_run.remote()))
