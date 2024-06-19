# ray_parallel.py
import torch
import tensorflow as tf
import os
import sys
import warnings
import ray
import json
import shutil

warnings.filterwarnings("ignore")


# ray stop
# ray start --head --node-ip-address=192.168.0.66 --port=6379 --dashboard-port=8265 --num-cpus=4 --num-gpus=1
# ray start --address='192.168.0.66:6379'
# ray job submit --working-dir /home/hola/Desktop/DNN -- python3 ray_parallel.py
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

    # Run the function in parallel on the Ray cluster
    print(ray.get(remote_setup_and_run.remote()))
