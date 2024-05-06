import datetime
import os
import sys
import time

import torch
import numpy as np
import random
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.nn as nn

sys.path.append("../")
from py_util import getStdModelForCifar10, getArgs
from data_parallelism_new.sampler import UnevenDistributedSampler
from vgg import vgg16, vgg11
from resnet import ResNet18
from alexnet import AlexNet

beginning_time = None
ending_time = None
computing_time = 0
device = torch.device("cuda:0")


# https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904

def main(args):
    nodeID = int(os.environ.get("SLURM_NODEID"))

    # model
    # model = vgg11()
    model = getStdModelForCifar10()
    # model = ResNet18()
    # model = AlexNet(10)

    # init group
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.init_method,
                                world_size=args.world_size, rank=args.rank)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.

        model = model.to(device)
        if args.gpu is not None:
            # torch.cuda.set_device(args.gpu)
            # model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            # model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # optimizer, criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # only one GPU per node, so we can directly use cuda() instead of .to()
    criterion = nn.CrossEntropyLoss()

    # data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=args.world_size, rank=args.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    torch.backends.cudnn.benchmark = True

    # main loop
    for epoch in range(0, args.epochs):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # adjust lr if needed #

        train_one_epoch(train_loader, model, criterion, optimizer, epoch, nodeID)
        # if args.rank == 0:  # only val and save on master node
        #    validate(val_loader, model, criterion, epoch, args)
        # save checkpoint if needed #
    total_time = datetime.timedelta(seconds=ending_time.timestamp() - beginning_time.timestamp())
    c_time = datetime.timedelta(seconds=computing_time)
    print('From Rank: {}, starting time{}, ending time {}, taking time{}, computing time{}'.format(args.rank,
                                                                                                   beginning_time,
                                                                                                   ending_time,
                                                                                                   total_time, c_time))
    # Tear down the process group
    dist.destroy_process_group()


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, nodeID):
    global beginning_time, ending_time, computing_time
    # only one gpu is visible here, so you can send cpu data to gpu by
    # input_data = input_data.cuda() as normal
    train_loss = 0
    correct = 0
    total = 0
    epoch_start = datetime.datetime.now()
    if epoch == 0:
        beginning_time = epoch_start

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = datetime.datetime.now().timestamp()

        inputs = inputs.to(device)
        targets = targets.to(device)

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

        computing_time += datetime.datetime.now().timestamp() - start

        if batch_idx % 25 == 0:
            print("From Node: {}, epoch {}, steps {}".format(nodeID, epoch, batch_idx))

    if epoch == (args.epochs - 1):
        ending_time = datetime.datetime.now()


'''
def validate(val_loader, model, criterion, epoch, args):
    pass
'''

if __name__ == '__main__':
    args = getArgs()
    main(args)
