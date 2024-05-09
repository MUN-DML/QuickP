import networkx as nx
import torch

from optimizer.graph_convertor.onnx_util import model_to_onnx, onnx_to_dict, to_json, generate_prof_json, \
    load_prof_result, get_comp_graph, update_graph_with_prof
from optimizer.model.graph import visualize_graph
from vgg import vgg11
from py_util import getStdCifar10DataLoader

"""
https://github.com/microsoft/onnxruntime-training-examples/blob/master/on_device_training/desktop/python/mnist.ipynb
Check the README.md file for both forward and backward passes
"""


def model_2_comp_graph(model, device):
    # Input to the model
    batch = 200
    data_loader = getStdCifar10DataLoader(batch_size=batch)
    for i, data in enumerate(data_loader):
        inputs, labels = data
        sample_inputs = inputs.to(device)

    path = "example.onnx"

    model_to_onnx(model, sample_inputs, path)
    # operator graph
    graph_dict = onnx_to_dict(path)
    # to_json(graph_dict, "onnx_graph.json")
    graph = get_comp_graph(graph_dict)
    if not nx.is_directed_acyclic_graph(graph):
        raise "comp_graph is not directed acyclic"
    # visualize_graph(graph, show_labels=False)

    profile_path = generate_prof_json("example.onnx", data_loader, batch, 4, 3)
    # profiling result
    profile_result = load_prof_result(profile_path)
    # to_json(profile_result, "group_result.json")

    # update the xnetwork graph with profiling result
    update_graph_with_prof(graph, profile_result)
    return graph


# network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = vgg11().to(device)
graph = model_2_comp_graph(net, device)
to_json(graph.getAllOperators(), "final_prof.json")
