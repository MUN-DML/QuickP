import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.pipeline.sync import Pipe

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from util.PyUtil import getArgs

'''
parser = argparse.ArgumentParser(description='cifar10 classification models, single node model parallelism test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=512, help='')
parser.add_argument('--num_workers', type=int, default=0, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--init_method', default='tcp://192.168.0.66:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
args = parser.parse_args()
print('World Size:', args.world_size)
'''


def main():
    args = getArgs()

    # Convolutional + pooling part of the model
    class ConvPart(nn.Module):

        def __init__(self):
            super(ConvPart, self).__init__()

            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)

            return x

    # Dense feedforward part of the model
    class MLPPart(nn.Module):

        def __init__(self):
            super(MLPPart, self).__init__()

            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)

            return x

    # by default
    torch.distributed.rpc.init_rpc('worker', rank=0,
                                   world_size=args.world_size)  # initializing RPC is required by Pipe we use below

    part1 = ConvPart().to('cuda:0')  # Load part1 on the first GPU
    part2 = MLPPart().to('cuda:1')  # Load part2 on the second GPU

    net = nn.Sequential(part1, part2)  # Pipe requires all modules be wrapped with nn.Sequential()
    '''https://pytorch.org/docs/stable/pipeline.html. Pipe only supports intra-node pipelining currently, but will be 
    expanded to support inter-node pipelining in the future. The forward function returns an RRef to allow for 
    inter-node pipelining in the future, where the output might be on a remote host. For intra-node pipelining you 
    can use local_value() to retrieve the output locally.'''
    net = Pipe(net, chunks=32)  # Wrap with Pipe to perform Pipeline Parallelism

    criterion = nn.CrossEntropyLoss().to('cuda:1')  # Load the loss function on the last GPU
    optimizer = optim.SGD(net.parameters(), lr=args.lr)

    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_train = CIFAR10(root='../data', train=True, download=False, transform=transform_train)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers)

    perf = []

    total_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()

        inputs = inputs.to('cuda:0')
        targets = targets.to('cuda:1')

        # Models wrapped with Pipe() return a RRef object. Since the example is single node, all values are local to the node and we can grab them
        outputs = net(inputs).local_value()
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

        batch_time = time.time() - start

        images_per_sec = args.batch_size / batch_time

        perf.append(images_per_sec)

    total_time = time.time() - total_start


if __name__ == '__main__':
    main()
