import torch
import torchvision
from torch.autograd.profiler import profile
from torchvision import transforms
from VGGParaCifar import vgg11
from onnx2torch import convert
from py_util import getStdModelForCifar10

# Input to the model
transform_train = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)

img, label = train_dataset[0]
# if there is only one GPU. cuda() is the easiest way to do to(device0)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
label = torch.tensor(label).reshape(1)

# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert("example.onnx").cuda()
img = img.cuda()
label = label.cuda()

# Run forward pass with profiling enabled
with torch.autograd.profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True, with_stack=True) as prof:
    # aggregate performance metrics of all operations in the sub-task will show up under forward_pass label
    with torch.autograd.profiler.record_function("forward_pass"):
        output = torch_model_1(img)
    loss = output.sum()
    with torch.autograd.profiler.record_function("backward_pass"):
        loss.backward()
# Print profiling results
print(prof.key_averages().table(sort_by="self_cpu_time_total"))