#!/bin/bash
#SBATCH --job-name=your-job-name
#SBATCH --time=0-03:00

### e.g. request 2 nodes with 1 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:1        #GPU number in each node; should equal to ntasks-per-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12    #accelerate Pytorch data loader
#SBATCH --mem=8gb
#SBATCH --output=%N-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus per node * num_nodes
export MASTER_PORT=3456
export WORLD_SIZE=2
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
export MASTER_ADDR=192.168.0.66 #Store the master node’s IP address in the MASTER_ADDR environment variable.
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}

echo "MASTER_ADDR="$MASTER_ADDR

### the command to run
srun python3.10 main.py --epochs 2