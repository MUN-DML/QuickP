# dask_parallel.py

import os
import sys
import warnings
from dask.distributed import Client, LocalCluster
import dask
import json

warnings.filterwarnings("ignore")

# Ensure the parent directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Import the function from intra_node_topo_parallel.py
from optimizer.device_topo.intra_node_topo_parallel import get_intra_node_topo

if __name__ == "__main__":
    # Set up a local Dask cluster
    cluster = LocalCluster()
    client = Client(cluster)

    # Wrap the function with Dask delayed
    task = dask.delayed(get_intra_node_topo)()

    # Compute the result
    future = client.compute(task)
    result = future.result()

    # Print the result
    print("Result:", json.dumps({"bandwidths": result[0], "devices": result[1]}, indent=2))
