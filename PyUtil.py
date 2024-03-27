import argparse
import os

import torchvision
from torchvision import transforms

# This guide can only be run with the torch backend. must write when using both keras and pytorch
# sudo apt install python3-packaging
os.environ["KERAS_BACKEND"] = "torch"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# our module must be nn.Sequential as GPipe will automatically split the module into partitions with consecutive layers
# The previous layer's out_channels should match the next layer's in_channels
# https://stackoverflow.com/questions/68606661/what-is-difference-between-nn-module-and-nn-sequential;
# using nn-module or nn-sequential
def getStdModelForCifar10():
    return nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Linear(120, 84),
        # the out_features number of the last layer should match the class number
        nn.Linear(84, 10))


# Data loading code for CiFar10
def getStdCifar10DataLoader(batch_size, num_workers=1, train=True):
    """
    If Use keras dataset instead of torchvision
    https://keras.io/guides/writing_a_custom_training_loop_in_torch/
    """
    # Data loading code for CiFar10
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=train, transform=transform_train, download=True)
    # sampler=train_sampler; if sampler is defined, set the shuffle to false
    return torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                       pin_memory=True, num_workers=num_workers)


def saveModelState(model, modelName):
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "./model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = "./model/{}".format(modelName)
    torch.save(model.state_dict(), filepath)


def testPYModel(model, test_loader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for (inputs, targets) in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def getArgs():
    parser = argparse.ArgumentParser(description='cifar10 classification models, single node model parallelism test')
    parser.add_argument('--lr', default=0.1, help='')
    parser.add_argument('--batch_size', type=int, default=512, help='')
    parser.add_argument('--max_epochs', type=int, default=2, help='')
    parser.add_argument('--num_workers', type=int, default=2, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--init_method', default='tcp://192.168.0.66:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', '192.168.0.66'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '3456'))
    args = parser.parse_args()
    print('World Size:', args.world_size)
    return args


# set the env variables of rank and world_size
def setup(rank, world_size):
    os.environ["RANK"] = rank
    os.environ["WORLD_SIZE"] = world_size


def retrieve_existing_model(obj, modelName):
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "./model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = "./model/{}".format(modelName)
    obj.load_state_dict(torch.load(filepath))
    obj.eval()
    return obj
