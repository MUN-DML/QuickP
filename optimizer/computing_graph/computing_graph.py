import socket

import keras
# it is weird that on my server, have to import torch to activate tensorflow
import tensorflow as tf
from keras import Sequential
from networkx import is_directed_acyclic_graph
from transformers import TFGPT2LMHeadModel

from optimizer.computing_graph.tool import Conf_TB, CONF
from optimizer.model.graph import CompGraph
from optimizer.computing_graph.op_graph_util import compile_model, train_loss, train_accuracy, parse_to_comp_graph, \
    process_op_df, update_graph_with_prof, profile_train, get_cifar_data_loader, parse_tensorboard, \
    find_specific_pb_file, process_mem_dict, get_gpt_data_loader


def get_computation_graph(model: tf.keras.Model, optimizer=keras.optimizers.Adam(3e-4),
                          loss_fn=keras.losses.SparseCategoricalCrossentropy(), batch_size=200) -> CompGraph:
    compile_model(model, optimizer, loss_fn)

    # tf.function is a decorator that tells TensorFlow to create a graph from the Python function
    # https://www.tensorflow.org/guide/function
    # https://www.tensorflow.org/tensorboard/get_started
    @tf.function
    def training_step(train_x, train_y, attention_mask=None):
        # https://www.tensorflow.org/guide/autodiff
        with tf.GradientTape() as tape:
            # Forward pass
            if isinstance(model, TFGPT2LMHeadModel):
                if attention_mask is None:
                    raise ValueError("attention_mask cannot be None")
                outputs = model(train_x, attention_mask=attention_mask, labels=train_y)
                loss = outputs.loss
            else:
                predictions = model(train_x, training=True)
                loss = loss_fn(train_y, predictions)
                loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss.update_state(loss)
        train_accuracy.update_state(train_y, predictions)
        return loss

    if isinstance(model, keras.Sequential):
        inputs_spec = [
            tf.TensorSpec(shape=[batch_size, 32, 32, 3], dtype=tf.float32, name="input"),
            tf.TensorSpec(shape=[batch_size, 1], dtype=tf.uint8, name="target")
        ]
    elif isinstance(model, TFGPT2LMHeadModel):
        inputs_spec = [
            tf.TensorSpec(shape=[batch_size, 128], dtype=tf.int32, name="input_ids"),
            tf.TensorSpec(shape=[batch_size, 128], dtype=tf.int32, name="labels"),
            tf.TensorSpec(shape=[batch_size, 128], dtype=tf.int32, name="attention_mask")
        ]
    else:
        raise ValueError("Unsupported model type")

    concrete_function = training_step.get_concrete_function(*inputs_spec)

    graph = parse_to_comp_graph(concrete_function)
    data_loader_func = get_cifar_data_loader if isinstance(model, keras.Sequential) else get_gpt_data_loader
    parent_directory = profile_train(concrete_function, data_loader_func(batch_size, True), num_prof_step=20)
    plane_pb_file = find_specific_pb_file(parent_directory, "xplane.pb")
    dataframe = parse_tensorboard(plane_pb_file, Conf_TB(CONF.OP))
    mem_data = parse_tensorboard(plane_pb_file, Conf_TB(CONF.MEM))
    op_dict = process_op_df(dataframe)
    mem_dict = process_mem_dict(mem_data)
    update_graph_with_prof(graph, op_dict, mem_dict, socket.gethostname())
    if not is_directed_acyclic_graph(graph):
        raise "comp_graph is not directed acyclic"
    return graph
