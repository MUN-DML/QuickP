import torch
import datetime
from py_util import getStdCifar10DataLoader, train, getStdModelForCifar10
from resnet import ResNet18
from vgg import vgg11
import torch.nn as nn


def main():
    train_loader = getStdCifar10DataLoader(batch_size=200)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = vgg11().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    train(epoch=1, model=model, train_loader=train_loader, optimizer=optimizer, criterion=criterion, device=device, profile_one_itr=True)


if __name__ == '__main__':
    main()
