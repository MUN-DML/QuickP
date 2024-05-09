import json
from collections import defaultdict

import numpy as np
import onnx
# pip install onnxruntime-gpu for cuda 11
# pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ for cuda12
import onnxruntime as ort
import torch
import nvtx

from optimizer.model.graph import CompGraph
from py_util import get_local_device_name


def model_to_onnx(model, input, path="example.onnx"):
    """
    torch.onnx.export(model,                                # model being run
                      torch.randn(1, 28, 28).to(device),    # model input (or a tuple for multiple inputs)
                      "fashion_mnist_model.onnx",           # where to save the model (can be a file or file-like object)
                      input_names = ['input'],              # the model's input names
                      output_names = ['output'])            # the model's output names
    """
    torch.onnx.export(model,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=17,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['X'],  # the model's input names
                      output_names=['Y']  # the model's output names
                      )


def onnx_to_dict(input_path="example.onnx"):
    onnx_model = onnx.load(input_path)
    onnx.checker.check_model(onnx_model)
    # Extract the graph
    graph = onnx_model.graph

    nodes = []
    edges = []

    for node in graph.node:
        nodes.append({
            "name": node.name,
            "type": node.op_type
        })
        for inp in node.input:
            edges.append({"from": inp, "to": node.name})
        for out in node.output:
            edges.append({"from": node.name, "to": out})
    return {"nodes": nodes, "edges": edges}


def to_json(graph_dict, output_path):
    json_data = json.dumps(graph_dict, indent=4)  # Converts dictionary to JSON string with pretty printing
    # If you want to print the JSON data to check its content
    # To save the JSON data to a file
    with open(output_path, 'w') as json_file:
        json_file.write(json_data)


# https://github.com/microsoft/onnxruntime/issues/20398
# https://github.com/microsoft/onnxruntime/issues/7212
# http://www.xavierdupre.fr/app/mlprodict/helpsphinx/notebooks/onnx_profile_ort.html a better example
def generate_prof_json(onnx_path, data_loader, batch_size, warm_up_end_step, num_prof_iter):
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    print(ort.get_available_providers())
    # https://onnxruntime.ai/docs/api/python/api_summary.html
    sess_profile = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = sess_profile.get_inputs()[0].name
    label_name = sess_profile.get_outputs()[0].name

    for i, (input_data, targets) in enumerate(data_loader):
        # X is numpy array on cpu. Put input to GPU
        X_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_data.numpy(), 'cuda', 0)
        Y_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type([batch_size, 10], np.int32, 'cuda', 0)

        io_binding = sess_profile.io_binding()
        # binds an input tensor to a GPU memory buffer
        io_binding.bind_input(name=input_name, device_type=X_ortvalue.device_name(), device_id=0,
                              element_type=np.float32,
                              shape=X_ortvalue.shape(), buffer_ptr=X_ortvalue.data_ptr())
        io_binding.bind_output(
            name=label_name,
            device_type=Y_ortvalue.device_name(),
            device_id=0,
            element_type=np.float32,
            shape=Y_ortvalue.shape(),
            buffer_ptr=Y_ortvalue.data_ptr())
        # put input tensor from GPU to CPU.
        if i < warm_up_end_step:
            with nvtx.annotate("warmup"):
                sess_profile.run_with_iobinding(io_binding)
        elif i < warm_up_end_step + num_prof_iter:
            with nvtx.annotate("profile part"):
                sess_profile.run_with_iobinding(io_binding)
        else:
            break
    return sess_profile.end_profiling()


def load_prof_result(prof_json_path: str, warm_up_end_step=3):
    def remove_after_first_underscore(s):
        return s.split('_', 1)[0]

    def default_entry():
        return {"name": None, "time": [], "mem": None}

    with open(prof_json_path, "r") as f:
        result = json.load(f)
        data_filtered = [{"name": remove_after_first_underscore(item["name"]), "dur": item["dur"],
                          "mem": int(item["args"]["output_size"]) + int(item["args"]["parameter_size"])}
                         for item in result
                         if item["dur"] != 0 and item["cat"] != "Session"]

        # Create a default dict where each value is a list
        grouped_data = defaultdict(default_entry)

        # Iterate over each dictionary in the list
        for item in data_filtered:
            # Append the dictionary to the list of its corresponding name
            if grouped_data[item['name']]["name"] is None:
                grouped_data[item['name']]["name"] = item.get("name")
            if grouped_data[item['name']]["mem"] is None:
                grouped_data[item['name']]["mem"] = item.get("mem")
            grouped_data[item['name']]["time"].append(item.get("dur"))
        # Skip warm up
        for (key, value) in grouped_data.items():
            grouped_data[key]["time"] = value["time"][warm_up_end_step:]

        # check each value has the same length
        len_list = [len(node_list) for node_list in grouped_data.values()]
        if not all(length == len_list[0] for length in len_list):
            raise ValueError("operators show different stage numbers")

        # Average the comp cost
        for (key, value) in grouped_data.items():
            grouped_data[key]["time"] = sum(value["time"]) / len(value["time"])

        return grouped_data


def get_comp_graph(dict):
    graph = CompGraph()
    for node in dict["nodes"]:
        graph.add_new_node(operator_id=node["name"], mem=0, op_type=node["type"])
    for edge in dict["edges"]:
        graph.add_new_edge(source_id=edge["from"], dest_id=edge["to"])
    return graph


def update_graph_with_prof(op_graph: CompGraph, group_dict):
    device_name = get_local_device_name()
    for node_id in op_graph.getOperatorIDs():
        if node_id in group_dict.keys():
            op_graph.nodes[node_id]["mem"] = group_dict[node_id]["mem"]
            if "comp_cost" not in op_graph.nodes[node_id]:
                op_graph.nodes[node_id]["comp_cost"] = {}
            op_graph.nodes[node_id]["comp_cost"][device_name] = group_dict[node_id]["time"]

