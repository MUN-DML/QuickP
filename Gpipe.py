import argparse
import datetime
import os
import time

import torchvision
from torchvision import transforms

from PyUtil import getStdModelForCifar10, getStdCifar10DataLoader, saveModelState, retrieve_existing_model, testPYModel

# This guide can only be run with the torch backend. must write when using both keras and pytorch
# sudo apt install python3-packaging
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
import numpy as np
from torchgpipe import GPipe
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.nn.functional as F

# Store argument values
parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
# always 1 in our platform
parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
parser.add_argument('--max_epochs', type=int, default=2, help='')
'''
Change the following accordingly
'''
parser.add_argument('--num_workers', type=int, default=2, help='')
parser.add_argument('--init_method', default='tcp://192.168.0.66:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
args = parser.parse_args()
# Update world size
args.world_size = args.gpus * args.num_workers

model = getStdModelForCifar10()
model.cuda()

# balance's length is equal to the number of computing nodes
# model layers and sum of balance have the same length
# balance determines the number of layers in each partition
# devices specify the GPU number on each device
# chunks means the number of micro-batches
model = GPipe(model, balance=[4, 4], devices=[1, 1], chunks=8)
# a Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss().cuda()
batch_size = 64
in_device = model.devices[0]
out_device = model.devices[-1]

# Creation of Distributed Data Parallel obj requires that torch.distributed (dist.init_process_group) to be initialized
# Backend includes mpi, gloo(CPU), nccl(GPU), and ucc. https://pytorch.org/docs/stable/distributed.html
# rank is the GPU index
# dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
# Wrap the model
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])
# train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size)
'''
If Use keras dataset instead of torchvision 
https://keras.io/guides/writing_a_custom_training_loop_in_torch/ 
'''
train_dataloader = getStdCifar10DataLoader(batch_size, 1)
# start training
epochs = 3
epoch_start = time.time()
for epoch in range(epochs):
    for step, (inputs, targets) in enumerate(train_dataloader):
        # Gpipe is also model para. Input and output layers are not on the same devices
        inputs = inputs.to(in_device, non_blocking=True)
        targets = targets.to(out_device, non_blocking=True)

        # Forward pass
        output = model(inputs)
        loss = loss_fn(output, targets)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Optimizer variable updates
        optimizer.step()

        elapse_time = datetime.timedelta(seconds=time.time() - epoch_start)
        # 'From Node ID {}'.format(int(os.environ.get("SLURM_NODEID")))
        print('From Node ID {}'.format(int(os.environ.get("SLURM_NODEID"))),
              f"Seen so far: {(step + 1) * batch_size} samples", "Training time {}".format(elapse_time))
saveModelState(model, modelName="cao")
model = retrieve_existing_model(GPipe(getStdModelForCifar10(), balance=[8], chunks=8), "cao")
test_dataloader = getStdCifar10DataLoader(batch_size, 1, train=False)
testPYModel(model, test_dataloader)
