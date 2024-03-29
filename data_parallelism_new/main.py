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
from VGGParaCifar import vgg16, vgg11

beginning_time = None
ending_time = None


# https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904

def main(args):
    nodeID = int(os.environ.get("SLURM_NODEID"))

    ### model ###
    model = vgg11()
    # model = getStdModelForCifar10()

    ### init group
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                                world_size=args.world_size, rank=args.rank)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model.features = torch.nn.parallel.DistributedDataParallel(model.features, device_ids=[args.gpu])
        else:
            model.cuda()
            model.features = torch.nn.parallel.DistributedDataParallel(model.features)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    ### optimizer, criterion ###
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    ### only one GPU per node, so we can directly use cuda() instead of .to()
    criterion = nn.CrossEntropyLoss().cuda()

    ### data ###
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True, sampler=train_sampler, drop_last=True)

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
    print('From Rank: {}, starting time{}, ending time {}, taking time{}'.format(args.rank, beginning_time, ending_time,
                                                                                 ending_time.timestamp() - beginning_time.timestamp()))


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, nodeID):
    global beginning_time, ending_time
    # only one gpu is visible here, so you can send cpu data to gpu by
    # input_data = input_data.cuda() as normal
    train_loss = 0
    correct = 0
    total = 0
    epoch_start = datetime.datetime.now()
    if epoch == 0:
        beginning_time = epoch_start

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

        elapse_time = datetime.datetime.now().timestamp() - epoch_start.timestamp()
        elapse_time = datetime.timedelta(seconds=elapse_time)
        if batch_idx % 30 == 0:
            print("From Node: {}, Training time {}, epoch {}, steps {}".format(nodeID, elapse_time, epoch, batch_idx))

    if epoch == (args.epochs - 1):
        ending_time = datetime.datetime.now()


'''
def validate(val_loader, model, criterion, epoch, args):
    pass
'''

if __name__ == '__main__':
    args = getArgs()
    main(args)
