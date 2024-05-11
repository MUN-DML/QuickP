import torch

from optimizer.graph_convertor.onnx_util import to_json
from DNN_model_ptytorch.vgg import vgg11

model = vgg11().cuda()
example_input = torch.rand(1, 3, 32, 32).cuda()  # Adjust size according to the model
traced_model = torch.jit.trace(model, example_input)

nodes = {}
edges = []

for node in traced_model.graph.nodes():
    if node.kind() != 'prim::Constant':
        node_id = node.output().debugName()  # Unique identifier for the node
        nodes[node_id] = {
            "name": node.scopeName(),  # Human-readable name if available, or uses scope name
            "id": node_id,
            "type": node.kind(),
            "inputs": [inp.debugName() for inp in node.inputs()],
            "outputs": [out.debugName() for out in node.outputs()]
        }
        for inp in node.inputs():
            if inp.node().kind() != 'prim::Constant':
                edges.append({"from": inp.debugName(), "to": node_id})

graph_dict = {"nodes": nodes, "edges": edges}

to_json(graph_dict, "jit_graph.json")
