import numpy as np
import onnxruntime as ort
import torch
import torchvision
from torchvision import transforms
from onnx2json import convert

from optimizer.graph_convertor.onnx_util import model_to_onnx, onnx_to_dict, to_json
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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = vgg11().to(device)

# Input to the model
transform_train = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)

x, _ = train_dataset[0]
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2])).to(device)
path = "example.onnx"

model_to_onnx(net, x, path)
graph_dict = onnx_to_dict(path)
to_json(graph_dict, "onnx_graph.json")

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
print(ort.get_available_providers())
sess_profile = ort.InferenceSession("example.onnx", sess_options=sess_options,
                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

input_name = sess_profile.get_inputs()[0].name
# put x from GPU to CPU
x = x.cpu().numpy()
sess_profile.run(None, {input_name: x})
profile_file = sess_profile.end_profiling()
print(profile_file)