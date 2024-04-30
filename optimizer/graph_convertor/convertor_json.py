import torch
import torchvision
from torchvision import transforms
from onnx2json import convert
from vgg import vgg11
from py_util import getStdModelForCifar10

'''
torch.onnx.export(model,                                # model being run
                  torch.randn(1, 28, 28).to(device),    # model input (or a tuple for multiple inputs)
                  "fashion_mnist_model.onnx",           # where to save the model (can be a file or file-like object)
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'])            # the model's output names
'''

# network
net = getStdModelForCifar10()

# Input to the model
transform_train = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)

x, _ = train_dataset[0]
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))

# Export the model
torch.onnx.export(net,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  "example.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=9,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['X'],  # the model's input names
                  output_names=['Y']  # the model's output names
                  )

help(convert)
onnx_json = convert(
  input_onnx_file_path="example.onnx",
  output_json_path="example.json",
  json_indent=2,
)