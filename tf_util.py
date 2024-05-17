import json
import os
from collections import defaultdict
from datetime import datetime
import pandas as pd
import keras
import torch
import numpy as np
from keras import Sequential
from keras.src.datasets import cifar10
from keras.src.utils import to_categorical
import tensorflow as tf
import networkx as nx
import tensorboard_plugin_profile.convert.raw_to_tool_data as rttd
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
    def augment_images(image, label):
        # Data augmentation transformations
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)  # Random brightness
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image, label

    (x_train, y_train), (x_test, y_test) = getCifar()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    if train:
        return train_dataset.shuffle(50000).map(augment_images).batch(batch_size).cache().prefetch(
            tf.data.experimental.AUTOTUNE)
    else:
        return test_dataset.batch(batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)


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

    for index, (x_train, y_train) in enumerate(dataloader):
        # warmup steps
        if index < num_warmup_step:
            concrete_function(x_train, y_train)
            # Call only one tf.function when tracing, so export after 1 iteration
            if index == 0:
                with train_summary_writer.as_default():
                    # TensorFlow Summary Trace API to log autographed functions for visualization in TensorBoard.
                    # https://www.tensorflow.org/tensorboard/graphs
                    # profiling will end trace_export
                    tf.summary.trace_export(
                        name="my_func_trace",
                        step=index,
                        profiler_outdir=log_dir)
        # Profiling steps
        elif index < num_warmup_step + num_prof_step:
            if index == num_warmup_step:
                tf.profiler.experimental.start(log_dir, options=options)
            concrete_function(x_train, y_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=index)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=index)
        # after profiling
        else:
            tf.profiler.experimental.stop()
            break

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


def csv_to_op_prof(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    # from csv to dataframe
    df = pd.read_csv(path, usecols=['Operation', 'Avg. self-time (us)'])
    # from dataframe to dict
    data = {row['Operation']: row['Avg. self-time (us)'] for index, row in df.head(200).iterrows()}
    print(data)


def parse_tensorboard(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError

    def default_entry():
        return {"param": {'tqx': ''}, "output_path": ''}

    # Create a default dict where each value is a list
    grouped_conf = defaultdict(default_entry)
    # Simpler input format
    simple_conf_dict = {
        'framework_op_stats^': ('out:csv', 'op_profile.csv'),
        'memory_profile^': ('', 'mem_profile.json')
    }
    # Update the grouped_data dict with the values from simple_conf_dict
    for key, (tqx, output_path) in simple_conf_dict.items():
        grouped_conf[key]['param']['tqx'] = tqx
        grouped_conf[key]['output_path'] = output_path
    grouped_conf = dict(grouped_conf)

    def process_pb(tool_name, params, o_path):
        # Process and convert the input file
        print("\033[32mImport TensorFlow...\033[0m")
        print("\033[32mXSpace to Tool Data...\033[0m")
        # https://github.com/tensorflow/profiler/blob/85dcfd10656d623330b11c3bbb8afed6418ec533/plugin/tensorboard_plugin_profile/convert/raw_to_tool_data.py
        tv = rttd.xspace_to_tool_data([input_path], tool_name, params)

        if isinstance(tv, tuple):
            tv = tv[0]
        # Write the processed data to the output file
        print("\033[32mWriting file...\033[0m")
        with open(o_path, "w") as f:
            f.write(tv)
        print("\033[32mDone!\033[0m")

    for (tool, value) in grouped_conf.items():
        process_pb(tool, value["param"], value['output_path'])


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


plane_pb_file = 'logs/20240515-214906/plugins/profile/2024_05_15_21_49_20/hola-Legion-T7-34IAZ7.xplane.pb'
# parse_tensorboard(plane_pb_file)
csv_to_op_prof('op_profile.csv')
