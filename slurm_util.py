import json
import os
import subprocess
from enum import Enum

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_idle_nodes():
    try:
        result = subprocess.run(['sinfo', '-h', '-o', '%N', '-t', 'idle'], check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        idle_nodes = result.stdout.decode().strip().split(',')
        return idle_nodes
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running sinfo: {e.stderr.decode()}")
        return []


# scontrol show node hola-Precision-3660
# scontrol show node hola-Legion-T7-34IAZ7
def get_node_ip(node):
    try:
        result = subprocess.run(['scontrol', 'show', 'node', node], check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        node_info = result.stdout.decode()
        for line in node_info.split('\n'):
            if 'NodeAddr=' in line:
                ip_address = line.split('NodeAddr=')[1].split()[0]
                return ip_address
        return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running scontrol for node {node}: {e.stderr.decode()}")
        return None


def get_server_ips():
    idle_nodes = get_idle_nodes()
    if not idle_nodes:
        print("No idle nodes found or an error occurred.")
        return
    # ['hola-Legion-T7-34IAZ7,hola-Precision-3660']
    node_ips = {}
    for node in idle_nodes:
        ip_address = get_node_ip(node)
        if ip_address:
            node_ips[node] = ip_address
    return node_ips


def get_slurm_available_nodes():
    try:
        # Run sinfo command to get the number of idle nodes
        result = subprocess.run(['sinfo', '--noheader', '--states=idle', '--format=%D'],
                                stdout=subprocess.PIPE, text=True, check=True)

        # Parse the output to get the total number of available nodes
        available_nodes = sum(int(x) for x in result.stdout.split())

        return available_nodes
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running sinfo: {e}")
        return 0


# Define an enumeration
class SLURM_RUN_CONF(Enum):
    INTRA_NODE = {"path": 'optimizer/device_topo/intra_node_topo_parallel.py', "time": '00:30', "mem": '2000'}
    INTER_NODE = {"path": 'optimizer/device_topo/intel_node_topo_parallel.py', "time": '00:30', "mem": '2000'}
    COMPUTING_COST = {"path": 'optimizer/computing_graph/computing_cost_parallel.py', "time": "1:30", "mem": '3G'}

    def __init__(self, value):
        if not isinstance(value, dict):
            raise ValueError(f"Value of {self.name} must be a dictionary")
        if 'path' not in value or 'time' not in value or 'mem' not in value:
            raise ValueError(f"Value of {self.name} must contain 'path, mem, and time' keys")
        if not isinstance(value['path'], str) or not isinstance(value['mem'], str) or not isinstance(value['time'],
                                                                                                     str):
            raise ValueError(f"The 'path, mem, and time' values of {self.name} must be strings")


def run_srun_command(num_nodes: int, command_type: SLURM_RUN_CONF):
    global script_dir
    path = os.path.join(script_dir, command_type.value['path'])
    time = command_type.value['time']
    mem = command_type.value['mem']
    command = [
        'srun',
        '--job-name=Bandwidth_parallel',
        f'--time={time}',
        f'--gpus={num_nodes}',
        '--gpus-per-node=1',
        f'--nodes={num_nodes}',
        '--ntasks-per-node=1',
        '--cpus-per-task=12',
        f'--mem={mem}',
        'python3', f'{path}'
    ]
    if command_type == SLURM_RUN_CONF.INTER_NODE:
        # # Serialize the dictionary to a JSON string
        command.extend(['--dict', json.dumps(get_server_ips())])
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.stdout:
            return result.stdout.decode()
        else:
            print(f"Error: {result.stderr.decode()}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the srun command: {e}")
        return None
