import torch

from optimizer.graph_convertor.onnx_util import model_to_onnx, onnx_to_dict, to_json, generate_prof_json, \
    load_prof_result
from vgg import vgg11
from py_util import getStdCifar10DataLoader

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
# operator graph
graph_dict = onnx_to_dict(path)
# to_json(graph_dict, "onnx_graph.json")

profile_path = generate_prof_json("example.onnx", data_loader, batch, 4, 3)
# profiling result
profile_result = load_prof_result(profile_path)
to_json(profile_result, "filtered_result.json")

