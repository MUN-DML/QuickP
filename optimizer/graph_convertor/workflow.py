import numpy as np
import onnxruntime as ort
import torch
import torchvision
from torchvision import transforms
from onnx2json import convert

from optimizer.graph_convertor.onnx_util import model_to_onnx, onnx_to_dict, to_json, generate_prof_json
from vgg import vgg11
from py_util import getStdModelForCifar10, getStdCifar10DataLoader

'''
torch.onnx.export(model,                                # model being run
                  torch.randn(1, 28, 28).to(device),    # model input (or a tuple for multiple inputs)
                  "fashion_mnist_model.onnx",           # where to save the model (can be a file or file-like object)
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'])            # the model's output names
'''

# network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = vgg11().to(device)

# Input to the model
batch = 200
data_loader = getStdCifar10DataLoader(batch_size=batch)
for i, data in enumerate(data_loader):
    inputs, labels = data
    sample_inputs = inputs.to(device)

path = "example.onnx"

model_to_onnx(net, sample_inputs, path)
graph_dict = onnx_to_dict(path)
to_json(graph_dict, "onnx_graph.json")

generate_prof_json("example.onnx", data_loader, batch)
