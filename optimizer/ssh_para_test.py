from optimizer.ssh_parallel import execute_parallel, SLURM_RUN_CONF

print(execute_parallel(SLURM_RUN_CONF.INTRA_NODE))
