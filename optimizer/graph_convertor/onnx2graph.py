import json

import onnx

# Load the ONNX model
onnx_model = onnx.load("example.onnx")

# Extract the graph
graph = onnx_model.graph

nodes = {}
edges = []

for node in graph.node:
    node_id = node.output[0] if node.output else node.name
    nodes[node_id] = {
        "name": node.name,
        "id": node_id,
        "type": node.op_type,
        "inputs": list(node.input),
        "outputs": list(node.output)
    }
    for inp in node.input:
        edges.append({"from": inp, "to": node_id})

graph_dict = {"nodes": nodes, "edges": edges}

json_data = json.dumps(graph_dict, indent=4)  # Converts dictionary to JSON string with pretty printing
# If you want to print the JSON data to check its content
print(json_data)
# To save the JSON data to a file
with open('model_graph.json', 'w') as json_file:
    json_file.write(json_data)