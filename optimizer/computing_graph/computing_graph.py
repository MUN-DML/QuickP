import socket

import keras
import numpy as np
# it is weird that on my server, have to import torch to activate tensorflow
import tensorflow as tf
from keras import Sequential
from networkx import is_directed_acyclic_graph
from transformers import TFGPT2LMHeadModel

from optimizer.computing_graph.concrete_function_constarint import get_input_spec
from optimizer.computing_graph.tool import Conf_TB, CONF
from optimizer.model.graph import CompGraph
from optimizer.computing_graph.op_graph_util import compile_model, train_loss, train_accuracy, parse_to_comp_graph, \
    process_op_df, update_graph_with_prof, profile_train, get_cifar_data_loader, parse_tensorboard, \
    find_specific_pb_file, process_mem_dict, get_llm_data_loader


def get_computation_graph(model: keras.Model, optimizer=keras.optimizers.Adam(3e-4),
                          loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                          max_len=128) -> CompGraph:
    # from_logits=False since all of our model has softmax activation that outputs probabilities

    compile_model(model, optimizer, loss_fn)

    batch_size = 512 if isinstance(model, keras.Sequential) else 16

    # tf.function is a decorator that tells TensorFlow to create a graph from the Python function
    # https://www.tensorflow.org/guide/function
    # https://www.tensorflow.org/tensorboard/get_started
    @tf.function
    def training_step(train_x: tf.Tensor, train_y: tf.Tensor):
        # https://www.tensorflow.org/guide/autodiff
        with tf.GradientTape() as tape:
            predictions = model(train_x, training=True)
            loss = loss_fn(train_y, predictions)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss.update_state(loss)
        train_accuracy.update_state(train_y, predictions)
        return loss

    @tf.function
    def training_step_llm(token_ids: tf.Tensor, segment_ids, padding_mask, train_y: tf.Tensor):
        # https://www.tensorflow.org/guide/autodiff
        with tf.GradientTape() as tape:
            # Forward pass
            model_inputs = {
                "token_ids": token_ids,
                "segment_ids": segment_ids,
            }

            # Add padding_mask only if it is provided
            if padding_mask is not None:
                model_inputs["padding_mask"] = padding_mask
            outputs = model(model_inputs)
            loss = loss_fn(train_y, outputs)
            predictions = outputs
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss.update_state(loss)
        train_accuracy.update_state(train_y, predictions)
        return loss

    inputs_spec = get_input_spec(model, batch_size, max_len)
    #for index, (text, label) in enumerate(get_gpt_data_loader(1).take(1)):
    training_fun = training_step if isinstance(model, keras.Sequential) else training_step_llm
    concrete_function = training_fun.get_concrete_function(*inputs_spec)

    graph = parse_to_comp_graph(concrete_function)
    data_loader_func = get_cifar_data_loader if isinstance(model, keras.Sequential) else get_llm_data_loader
    is_llm = False if isinstance(model, keras.Sequential) else True
    parent_directory = profile_train(concrete_function, data_loader_func(batch_size, True), num_prof_step=50, is_llm=is_llm, max_length=max_len)
    plane_pb_file = find_specific_pb_file(parent_directory, "xplane.pb")
    dataframe = parse_tensorboard(plane_pb_file, Conf_TB(CONF.OP))
    mem_data = parse_tensorboard(plane_pb_file, Conf_TB(CONF.MEM))
    op_dict = process_op_df(dataframe)
    mem_dict = process_mem_dict(mem_data)
    update_graph_with_prof(graph, op_dict, mem_dict, socket.gethostname())
    if not is_directed_acyclic_graph(graph):
        raise "comp_graph is not directed acyclic"
    return graph
