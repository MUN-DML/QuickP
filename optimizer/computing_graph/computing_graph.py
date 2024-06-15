import keras
# it is weird that on my server, have to import torch to activate tensorflow
import tensorflow as tf
from keras import Sequential
from networkx import is_directed_acyclic_graph

from optimizer.computing_graph.tool import Conf_TB, CONF
from optimizer.model.graph import CompGraph
from optimizer.computing_graph.op_graph_util import compile_model, train_loss, train_accuracy, parse_to_comp_graph, \
    process_op_df, update_graph_with_prof, profile_train, get_cifar_data_loader, parse_tensorboard, \
    find_specific_pb_file, process_mem_dict


def get_computation_graph(model: Sequential, optimizer=keras.optimizers.Adam(3e-4),
                          loss_fn=keras.losses.SparseCategoricalCrossentropy(), batch_size=200) -> CompGraph:
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
    parent_directory = profile_train(concrete_function, get_cifar_data_loader(batch_size, True), num_prof_step=20)
    plane_pb_file = find_specific_pb_file(parent_directory, "xplane.pb")
    dataframe = parse_tensorboard(plane_pb_file, Conf_TB(CONF.OP))
    mem_data = parse_tensorboard(plane_pb_file, Conf_TB(CONF.MEM))
    op_dict = process_op_df(dataframe)
    mem_dict = process_mem_dict(mem_data)
    update_graph_with_prof(graph, op_dict, mem_dict)
    if not is_directed_acyclic_graph(graph):
        raise "comp_graph is not directed acyclic"
    return graph
