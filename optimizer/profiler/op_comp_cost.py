import torch
import torch.nn as nn
import torch.autograd as autograd
import torchvision
import torchviz
import torchvision.transforms as transforms
from pyutil import getStdModelForCifar10

# https://pytorch.org/tutorials/beginner/profiler.html
# https://discuss.pytorch.org/t/cuda-memory-profiling/182065/2
# Create an instance of the model
model = getStdModelForCifar10().cuda()

# Input to the model
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)

img, label = train_dataset[0]
# if there is only one GPU. cuda() is the easiest way to do to(device0)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2])).cuda()
label = torch.tensor(label).reshape(1).cuda()

# Warm up
model(img)

# Run forward pass with profiling enabled
with torch.autograd.profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True, with_stack=True) as prof2:
    # aggregate performance metrics of all operations in the sub-task will show up under forward_pass label
    with torch.autograd.profiler.record_function("forward_pass"):
        output = model(img)
    loss = output.sum()
    with torch.autograd.profiler.record_function("backward_pass"):
        loss.backward()

# Print the computation time of each operator
print(prof2.key_averages().table(sort_by="self_cpu_time_total"))

# Visualize the computation graph (optional, if you have torchviz installed)
# torchviz.make_dot(output, params=dict(model.named_parameters())).render("computation_graph_forward", format="png")