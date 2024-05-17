import os
from datetime import datetime

import keras
# it is weird that on my server, have to import torch to activate tensorflow
import torch
import tensorflow as tf
from keras import Sequential
from pathlib import Path
from DNN_model_tf.vgg_tf import VGG16_tf
from tf_util import train_model, getCifar, compile_model, train_loss, train_accuracy, parse_to_comp_graph, \
    csv_to_op_prof, update_graph_with_prof, profile_train, get_cifar_data_loader, parse_tensorboard, \
    find_specific_pb_file

'''
model = VGG16_tf()
compile_model(model)

# Create a TensorBoard callback
# profiling: https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
# python3 -m tensorboard.main --logdir=~/my/training/dir
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 # a range of batches
                                                 profile_batch='50,55')

(x_train, y_train), (x_test, y_test) = getCifar()
train_model(model, x_train, y_train, x_test, y_test, [tboard_callback], batch_size=200)
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
    graph = parse_to_comp_graph(concrete_function)
    # plane_pb_file = 'logs/20240515-214906/plugins/profile/2024_05_15_21_49_20/hola-Legion-T7-34IAZ7.xplane.pb'
    parent_directory = profile_train(concrete_function, get_cifar_data_loader(batch_size, True))
    plane_pb_file = find_specific_pb_file(parent_directory, "xplane.pb")
    path = parse_tensorboard(plane_pb_file)
    prof_data = csv_to_op_prof('op_profile.csv')
    print(update_graph_with_prof(graph, prof_data))


model = VGG16_tf()
work_flow(model)
