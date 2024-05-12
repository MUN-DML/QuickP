import keras
import torch
import numpy as np
from keras import Sequential
from keras.src.datasets import cifar10
from keras.src.utils import to_categorical
import tensorflow as tf

from DNN_model_tf.vgg_tf import VGG16_tf
from optimizer.model.graph import visualize_graph


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
                  loss=keras.losses.BinaryCrossentropy(from_logits=True)):
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")])


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


def get_comp_graph():
    # Create a dummy optimizer and loss for demonstration
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model = VGG16_tf()

    # tf.function is a decorator that tells TensorFlow to create a graph from the Python function
    @tf.function
    def training_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # to obtain a concrete function from a tf.function
    concrete_function = training_step.get_concrete_function(
        tf.TensorSpec(shape=[200, 32, 32, 3], dtype=tf.float32, name="input"),
        tf.TensorSpec(shape=[None], dtype=tf.int32, name="target")
    )

    graph = concrete_function.graph

    import networkx as nx

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes and edges to the graph
    for op in graph.get_operations():
        # Each node is an operation in the TensorFlow graph
        G.add_node(op.name, op_type=op.type)
        for input_tensor in op.inputs:
            # Create an edge from input operation to the current operation
            G.add_edge(input_tensor.op.name, op.name)
    print(G.nodes)


get_comp_graph()
