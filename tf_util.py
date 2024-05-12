import datetime

import keras
import torch
import numpy as np
from keras import Sequential
from keras.src.datasets import cifar10
from keras.src.utils import to_categorical
import tensorflow as tf
import networkx as nx
from tensorflow import data as tf_data
from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.model.graph import visualize_graph, CompGraph


def getCifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


# GPU training: https://www.tensorflow.org/guide/gpu
def train_model(model: Sequential, x_train, y_train, x_test, y_test, call_back_list, batch_size=200):
    print(model.summary())
    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=1, batch_size=batch_size, shuffle=True,
              callbacks=call_back_list)


def compile_model(model: Sequential, optimizer=keras.optimizers.Adam(3e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy()):
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])


def testExistModel(model: Sequential, x_test, y_test, test_num):
    for i in range(test_num):
        image = np.expand_dims(x_test[i], axis=0)
        prediction = model.predict(image)[0]
        real = y_test[i]
        index_max_pre = np.argmax(prediction)
        index_max_real = np.argmax(real)
        if index_max_pre == index_max_real:
            print("match")
        else:
            print("not match")


# https://github.com/eval-submissions/HeteroG/blob/heterog/profiler.py tf profiling example
def profile_train(concrete_function, inputs, targets):
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)
    log_dir = "logs/profile/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Start the profiler
    tf.profiler.experimental.start(log_dir, options=options)
    for _ in range(5):
        concrete_function(inputs, targets)
    tf.profiler.experimental.stop()


def parse_to_comp_graph(concrete_function):
    graph = concrete_function.graph

    # Create a directed graph
    G = CompGraph()

    # Add nodes and edges to the graph
    for op in graph.get_operations():
        # Each node is an operation in the TensorFlow graph
        G.add_new_node(op.name, 0, op_type=op.type)
        for input_tensor in op.inputs:
            # Create an edge from input operation to the current operation
            G.add_new_edge(input_tensor.op.name, op.name)
    if not nx.is_directed_acyclic_graph(G):
        raise "comp_graph is not directed acyclic"
    visualize_graph(G, show_labels=False)


def get_comp_graph(model: Sequential, optimizer=keras.optimizers.Adam(3e-4),
                   loss_fn=keras.losses.SparseCategoricalCrossentropy()):
    compile_model(model, optimizer, loss_fn)

    # tf.function is a decorator that tells TensorFlow to create a graph from the Python function
    # https://www.tensorflow.org/guide/function
    @tf.function
    def training_step(x, y):
        # https://www.tensorflow.org/guide/autodiff
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    inputs = tf.TensorSpec(shape=[200, 32, 32, 3], dtype=tf.float32, name="input")
    targets = tf.TensorSpec(shape=[None], dtype=tf.int32, name="target")
    # to obtain a concrete function from a tf.function
    concrete_function = training_step.get_concrete_function(inputs, targets)
    parse_to_comp_graph(concrete_function)
    # profile_train(concrete_function, inputs, targets)


get_comp_graph(VGG16_tf())
