# dask_parallel.py

import os
import sys
import warnings
from dask.distributed import Client, SSHCluster
import dask
import json

from distributed import UploadDirectory

warnings.filterwarnings("ignore")

# Ensure the parent directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
# Import the function from intra_node_topo_parallel.py
from optimizer.device_topo.intra_node_topo_parallel import get_intra_node_topo


if __name__ == "__main__":
    # Set up a local Dask cluster
    # Define the list of hostnames or IP addresses
    hosts = ["localhost"]  # Replace with your actual hosts

    # Define connection options
    connect_options = {
        "known_hosts": None,  # Automatically accept host keys
        "username": "root",  # Replace with your SSH username
        "password": "1314520",  # Use this if you are using password authentication
        # "client_keys": ["path/to/private_key"],  # Replace with the path to your SSH private key
    }

    # Define options for workers
    worker_options = {
        "nthreads": 2,
        "nprocs": 1,
        "memory_limit": "2GB"
    }

    # Define options for the scheduler
    scheduler_options = {
        "port": 8786,
        "dashboard_address": ":8787"
    }
    cluster = SSHCluster(
        hosts=hosts,
        connect_options=connect_options,
        worker_options=worker_options,
        scheduler_options=scheduler_options
    )

    client = Client(cluster)
    client.run(set_pythonpath_on_worker)

    # Upload the necessary files to all workers
    files_to_upload = [
        'optimizer/model/graph.py',
        'optimizer/device_topo/intra_node_util.py',
        'optimizer/device_topo/intra_node_topo_parallel.py'
    ]
    for file in files_to_upload:
        client.upload_file(os.path.join(project_root, file))

    client.register_plugin(UploadDirectory(os.path.join(project_root, "optimizer")))
    # Wrap the function with Dask delayed
    task = dask.delayed(get_intra_node_topo)()

    # Compute the result
    future = client.compute(task)
    result = future.result()

    # Print the result
    print("Result:", json.dumps({"bandwidths": result[0], "devices": result[1]}, indent=2))
