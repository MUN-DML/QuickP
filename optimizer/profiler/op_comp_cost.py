import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision
import torchviz
import torchvision.transforms as transforms
from pyutil import getStdModelForCifar10

# https://pytorch.org/tutorials/beginner/profiler.html
# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Create an instance of the model
model = getStdModelForCifar10()

# Create a sample input tensor (batch_size=1, input_size=784)
# Input to the model
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)

img, label = train_dataset[0]
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
label = torch.tensor(label).reshape(1)

# Capture the forward pass operations
with torch.autograd.profiler.profile() as prof:
    out = model(img)

# Run forward pass with profiling enabled
with torch.autograd.profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof2:
    with torch.autograd.profiler.record_function("forward_pass"):
        output = model(img)
    loss = output.sum()
    with torch.autograd.profiler.record_function("backward_pass"):
        loss.backward()

# Print the computation time of each operator
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# Print the computation time of each operator
print(prof2.key_averages().table(sort_by="self_cpu_time_total"))
# Visualize the computation graph (optional, if you have torchviz installed)
# torchviz.make_dot(output, params=dict(model.named_parameters())).render("computation_graph_forward", format="png")