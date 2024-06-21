from optimizer.ssh_parallel import execute_parallel, ParallelCommandType

host_ip_mapping = execute_parallel(ParallelCommandType.IP_ADD_MAPPING)
