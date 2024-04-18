import torchvision
from torchviz import make_dot
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pyutil import getStdModelForCifar10

# network
net = getStdModelForCifar10()

# Input to the model
transform_train = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)

img, label = train_dataset[0]
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
label = torch.tensor(label).reshape(1)


def train_and_output_photo(model, image, output):
    # Move the model to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)
    output = output.to(device)

    ### optimizer, criterion ###
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
    ### only one GPU per node, so we can directly use cuda() instead of .to()
    criterion = nn.CrossEntropyLoss().to(device)

    # forward propagation
    prediction = model(image)
    print(prediction)
    print(output)
    # define loss
    loss = criterion(prediction, output)

    # backward pass
    loss.backward()

    # gradient descent
    optimizer.step()

    # Computational Graph
    make_dot(prediction,
             params=dict(model.named_parameters()))


train_and_output_photo(net, img, label)
