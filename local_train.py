import torch

from py_util import getStdCifar10DataLoader, train_one_epoch, getStdModelForCifar10
from resnet import ResNet18
from vgg import vgg11
import torch.nn as nn


def main():
    train_loader = getStdCifar10DataLoader()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = getStdModelForCifar10().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_one_epoch(model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, device=device)


if __name__ == '__main__':
    main()
