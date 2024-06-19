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


@ray.remote
def remote_setup_and_run():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    sys.path.append(project_root)

    # Import the function
    try:
        from optimizer.device_topo.intra_node_topo_parallel import get_intra_node_topo
    except ImportError as e:
        print(f"Import error: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    # Initialize Ray
    ray.init(address='auto')

    # Run the function in parallel on the Ray cluster
    print(ray.get(remote_setup_and_run.remote()))

