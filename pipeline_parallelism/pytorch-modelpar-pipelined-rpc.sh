#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1 # request 2 GPUs
#SBATCH --tasks-per-node=1 
#SBATCH --cpus-per-task=1 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance
#SBATCH --mem=8G      
#SBATCH --time=0:10:00
#SBATCH --output=%N-%j.out

# This is needed to initialize pytorch's RPC module, required for the Pipe class which we'll use for Pipeline Parallelism
export MASTER_ADDR=192.168.0.66 #Store the master node’s IP address in the MASTER_ADDR environment variable.
export MASTER_PORT=34567
 
echo "starting training..."
time python3.10 pytorch-modelpar-pipelined-rpc.py --init_method tcp://$MASTER_ADDR:3456 --world_size $((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))