import subprocess


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
