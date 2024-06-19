# ray_parallel.py

import os
import sys
import warnings
import ray
import json
import shutil

warnings.filterwarnings("ignore")

# Ensure the parent directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
from optimizer.device_topo.intra_node_topo_parallel import get_intra_node_topo


# Helper function to upload a directory to all nodes
def upload_directory_to_nodes(local_directory, remote_directory):
    shutil.make_archive(local_directory, 'zip', local_directory)
    remote_path = os.path.join("/tmp", os.path.basename(local_directory) + ".zip")
    ray.put(open(f"{local_directory}.zip", "rb").read(), remote_path)


@ray.remote
def remote_setup_and_run():
    # Run the function
    return get_intra_node_topo()


if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='auto')

    # Run the function in parallel on the Ray cluster
    futures = [remote_setup_and_run.remote() for _ in range(len(ray.nodes()))]
    results = ray.get(futures)

    # Print the results
    for result in results:
        print("Result:", json.dumps({"bandwidths": result[0], "devices": result[1]}, indent=2))
