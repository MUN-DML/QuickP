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
def getStdCifar10DataLoader(batch_size, num_workers=1, train=False):
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


def testPYModel(model, test_loader):
    return model.eval()
