import json
import os
from datetime import datetime

import keras
import torch
import numpy as np
from keras import Sequential
from keras.src.datasets import cifar10
from keras.src.utils import to_categorical
import tensorflow as tf
import networkx as nx
from tensorboard.backend.event_processing import event_accumulator, plugin_event_multiplexer
from tensorboard.data import provider
from tensorboard.plugins.base_plugin import TBContext
from tensorboard_plugin_profile.profile_plugin import ProfilePlugin
from tensorboard_plugin_profile.protobuf import trace_events_pb2
from tensorflow import data as tf_data
from tensorflow.python.eager.polymorphic_function.concrete_function import ConcreteFunction

from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.model.graph import visualize_graph, CompGraph

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')


def getCifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # make the y value for each single image to be a array of length 10
    # y_train = to_categorical(y_train, 10)
    # y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def get_cifar_data_loader(batch_size=200, train=True):
    (x_train, y_train), (x_test, y_test) = getCifar()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.shuffle(50000).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)
    if train:
        return train_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)
    else:
        return test_dataset.cache().prefetch(tf.data.experimental.AUTOTUNE)


# GPU training: https://www.tensorflow.org/guide/gpu
def train_model(model: Sequential, x_train, y_train, x_test, y_test, call_back_list, batch_size=200):
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


'''
Command to trigger tensorboard: python3 -m tensorboard.main --logdir=logs
'''


# https://github.com/eval-submissions/HeteroG/blob/heterog/profiler.py tf profiling example
# https://github.com/tensorflow/profiler/issues/24
# https://www.tensorflow.org/guide/intro_to_modules
def profile_train(concrete_function: ConcreteFunction, dataloader: tf.data.Dataset, num_warmup_step=2,
                  num_prof_step=200):
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    # Start the profiler, cannot set the parameter profiler=True
    tf.summary.trace_on(graph=True)
    step = 0
    for (x_train, y_train) in dataloader:
        # warmup steps
        if step < num_warmup_step:
            concrete_function(x_train, y_train)
            # Call only one tf.function when tracing, so export after 1 iteration
            if step == 0:
                with train_summary_writer.as_default():
                    # TensorFlow Summary Trace API to log autographed functions for visualization in TensorBoard.
                    # https://www.tensorflow.org/tensorboard/graphs
                    # profiling will end trace_export
                    tf.summary.trace_export(
                        name="my_func_trace",
                        step=step,
                        profiler_outdir=log_dir)
        # Profiling steps
        elif step < num_warmup_step + num_prof_step:
            if step == num_warmup_step:
                tf.profiler.experimental.start(log_dir, options=options)
            concrete_function(x_train, y_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=step)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=step)
        # after profiling
        else:
            tf.profiler.experimental.stop()
            break
        step += 1
    return log_dir


def parse_to_comp_graph(concrete_function: ConcreteFunction):
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


def parse_tensorboard(path):
    def get_log():
        event_acc = event_accumulator.EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()
        print("fuck", tags)
        # Extract scalar data
        tensors = {}
        for tag in tags['tensors']:
            tensors[tag] = event_acc.Tensors(tag)

        # Convert tensor data to JSON
        def tensor_to_json(tensor_event):
            tensor_proto = tensor_event.tensor_proto
            tensor_dict = {
                'step': tensor_event.step,
                'wall_time': tensor_event.wall_time,
                'tensor': {
                    'dtype': tensor_proto.dtype,
                    'tensor_shape': [dim.size for dim in tensor_proto.tensor_shape.dim],
                    'tensor_content': tensor_proto.tensor_content.hex()
                }
            }
            return tensor_dict

        tensors_json = {tag: [tensor_to_json(t) for t in tensors[tag]] for tag in tensors}
        print(tensors_json)

    '''
    # Initialize the Event Multiplexer
    multiplexer = plugin_event_multiplexer.EventMultiplexer({
        'run1': path
    })

    # Load the event files
    multiplexer.Reload()

    data_provider = provider.DataProvider()
    context = TBContext(logdir=path, multiplexer=multiplexer, data_provider=data_provider)
    plugin = ProfilePlugin(context)
    profiles = plugin.profiles()
    # Load the profile data
    for profile in profiles:
        print(f"Profile: {profile}")
    '''


def work_flow(model: Sequential, optimizer=keras.optimizers.Adam(3e-4),
              loss_fn=keras.losses.SparseCategoricalCrossentropy(), batch_size=200):
    compile_model(model, optimizer, loss_fn)

    # tf.function is a decorator that tells TensorFlow to create a graph from the Python function
    # https://www.tensorflow.org/guide/function
    # https://www.tensorflow.org/tensorboard/get_started
    @tf.function
    def training_step(train_x, train_y):
        # https://www.tensorflow.org/guide/autodiff
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = model(train_x, training=True)
            loss = loss_fn(train_y, predictions)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_loss.update_state(loss)
        train_accuracy.update_state(train_y, predictions)
        return loss

    # tf.TensorSpec constrain the type of inputs accepted by a tf.function
    # shape=[200, 32, 32, 3], 200 is batch size, 32x32x3 is the size for each image
    inputs_constraint = tf.TensorSpec(shape=[batch_size, 32, 32, 3], dtype=tf.float32, name="input")
    targets_constraint = tf.TensorSpec(shape=[batch_size, 1], dtype=tf.uint8, name="target")
    # to obtain a concrete function from a tf.function.
    # ConcreteFunctions can be executed just like PolymorphicFunctions,
    # but their input is restricted to the types to which they're specialized.
    concrete_function = training_step.get_concrete_function(inputs_constraint, targets_constraint)
    # parse_to_comp_graph(concrete_function)

    path = profile_train(concrete_function, get_cifar_data_loader(batch_size, True))


work_flow(VGG16_tf())
