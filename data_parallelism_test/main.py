import datetime
import os
import builtins
import argparse
import sys
import time

import torch
import numpy as np
import random
import torch.distributed as dist
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn as nn

sys.path.append("../")
from PyUtil import getStdModelForCifar10, getArgs


# https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904

def main(args):
    nodeID = int(os.environ.get("SLURM_NODEID"))
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                            world_size=args.world_size, rank=args.rank)
    ### model ###
    model = getStdModelForCifar10()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
            model_without_ddp = model.module
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    ### optimizer, criterion ###
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().cuda()

    ### data ###
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True, sampler=train_sampler, drop_last=True)

    val_dataset = dataset_train = CIFAR10(root='../data', train=False, download=True, transform=transform_train)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True, sampler=val_sampler, drop_last=True)

    torch.backends.cudnn.benchmark = True

    ### main loop ###
    for epoch in range(0, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # adjust lr if needed #

        train_one_epoch(train_loader, model, criterion, optimizer, epoch, nodeID)
        # if args.rank == 0:  # only val and save on master node
        #    validate(val_loader, model, criterion, epoch, args)
        # save checkpoint if needed #


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, nodeID):
    pass
    # only one gpu is visible here, so you can send cpu data to gpu by
    # input_data = input_data.cuda() as normal
    train_loss = 0
    correct = 0
    total = 0

    epoch_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()

        inputs = inputs.cuda()
        targets = targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100 * correct / total

        batch_time = time.time() - start

        elapse_time = time.time() - epoch_start
        elapse_time = datetime.timedelta(seconds=elapse_time)
        print("From Node: {}, Training time {}, epoch {}, steps {}".format(nodeID, elapse_time, epoch, batch_idx))


'''
def validate(val_loader, model, criterion, epoch, args):
    pass
'''

if __name__ == '__main__':
    args = getArgs()
    main(args)
