#!/bin/bash
#SBATCH --nodes=1             # minimum node number is 2
#SBATCH --gpus=1              # specifies the total number of GPUs required for an entire job across all nodes
#SBATCH --gres=gpu:1          # Generic resources required per node. We only have one GPU per node
#SBATCH --ntasks-per-node=1   # The number of tasks (GPUs) used in each node.
#SBATCH --ntasks=1            # total number of tasks (GPUs) used in this job across all nodes
#SBATCH --cpus-per-task=12    # accelerate data-loader workers to load data in parallel.
#SBATCH --mem=8G
#SBATCH --time=0-03:00
#SBATCH --output=%N-%j.out

export MASTER_ADDR=192.168.0.66 #Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_PORT=3456
export WORLD_SIZE=1

srun python3.10 UtilTest.py