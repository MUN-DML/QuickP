import json

import onnx
# pip install onnxruntime-gpu for cuda 11
# pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/ for cuda12
import onnxruntime as ort
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


def onnx_to_dict(input_path="example.onnx"):
    onnx_model = onnx.load(input_path)
    onnx.checker.check_model(onnx_model)
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

    return {"nodes": nodes, "edges": edges}


def to_json(graph_dict, output_path):
    json_data = json.dumps(graph_dict, indent=4)  # Converts dictionary to JSON string with pretty printing
    # If you want to print the JSON data to check its content
    # To save the JSON data to a file
    with open(output_path, 'w') as json_file:
        json_file.write(json_data)


# https://github.com/microsoft/onnxruntime/issues/20398
# http://www.xavierdupre.fr/app/mlprodict/helpsphinx/notebooks/onnx_profile_ort.html a better example
def generate_prof_json(onnx_path, data_loader):
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    print(ort.get_available_providers())
    sess_profile = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    input_name = sess_profile.get_inputs()[0].name

    for i, (input_data, _) in enumerate(data_loader):
        if i == 5:
            break
        # put input tensor from GPU to CPU.
        sess_profile.run(None, {input_name: input_data.cpu().numpy()})
    return sess_profile.end_profiling()


def load_prof_result():
    pass
