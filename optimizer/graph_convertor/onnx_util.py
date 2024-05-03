import json

import onnx
import torch


def model_to_onnx(model, input, path="example.onnx"):
    torch.onnx.export(model,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['X'],  # the model's input names
                      output_names=['Y']  # the model's output names
                      )


def onnx_to_graph(input_path="example.onnx", output_path="model_graph.json"):
    onnx_model = onnx.load(input_path)

    # Extract the graph
    graph = onnx_model.graph

    nodes = {}
    edges = []

    for node in graph.node:
        nodes[node.name] = {
            "name": node.name,
            "type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output)
        }
        for inp in node.input:
            edges.append({"from": inp, "to": node.name})

    graph_dict = {"nodes": nodes, "edges": edges}

    json_data = json.dumps(graph_dict, indent=4)  # Converts dictionary to JSON string with pretty printing
    # If you want to print the JSON data to check its content
    # To save the JSON data to a file
    with open(output_path, 'w') as json_file:
        json_file.write(json_data)
