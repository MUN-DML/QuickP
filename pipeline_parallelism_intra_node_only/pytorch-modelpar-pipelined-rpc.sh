#!/bin/bash
#SBATCH --nodes 1  # can only be one since Pipe in pytorch does not support multi nodes
#SBATCH --gres=gpu:1 # request 1 GPU per node
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=8G      
#SBATCH --time=0:10:00
#SBATCH --output=%N-%j.out

# This is needed to initialize pytorch's RPC module, required for the Pipe class which we'll use for Pipeline Parallelism
export MASTER_ADDR=192.168.0.66 #Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_PORT=3456
export TORCH_NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.

echo "r$SLURM_NODEID master: $SLURM_SUBMIT_DIR"
echo "r$SLURM_NODEID Launching python script"

echo "SLURM_NTASKS_PER_NODE: $SLURM_NTASKS_PER_NODE"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "starting training..."
srun python3.10 pytorch-modelpar-pipelined-rpc.py --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))