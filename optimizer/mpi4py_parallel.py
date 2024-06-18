import os
import sys
import warnings
# sudo apt-get install libopenmpi-dev
# pip install mpi4py
from mpi4py import MPI
import json

warnings.filterwarnings("ignore")

# Ensure the parent directory is in the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

# Import the function from intra_node_topo_parallel.py
from optimizer.device_topo.intra_node_topo_parallel import get_intra_node_topo


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running on {size} nodes")

    # Execute the function on each node
    try:
        bandwidths, devices = get_intra_node_topo()
        print(f"Process {rank} got bandwidths: {bandwidths}, devices: {devices}")
    except Exception as e:
        print(f"Error on process {rank}: {e}", file=sys.stderr)
        bandwidths, devices = None, None

    # Gather results from all nodes
    all_bandwidths = comm.gather(bandwidths, root=0)
    all_devices = comm.gather(devices, root=0)

    result = {"bandwidths": all_bandwidths, "devices": all_devices}
    print("Result:", json.dumps(result, indent=2))


# mpirun -np 1 --allow-run-as-root --host localhost python3 mpi4py_parallel.py
if __name__ == "__main__":
    main()
