import json
import os
from datetime import datetime
from io import StringIO

import pandas as pd
import keras
import torch
import numpy as np
from keras import Sequential
from keras.src.datasets import cifar10
import tensorflow as tf
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import networkx as nx
# pip install -q --upgrade keras-nlp
import keras_nlp
# pip install tensorboard-plugin-profile
import tensorboard_plugin_profile.convert.raw_to_tool_data as rttd
from pathlib import Path
from pandas import DataFrame
from tensorflow.python.eager.polymorphic_function.concrete_function import ConcreteFunction
from tensorflow.python.framework.dtypes import DType

from optimizer.computing_graph.tool import Conf_TB, CONF
from optimizer.model.graph import visualize_graph, CompGraph

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')


def getCifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return (x_train, y_train), (x_test, y_test)

def getGpt_data_loader():
    #https://keras.io/examples/generative/text_generation_gpt/
    BATCH_SIZE = 64
    MIN_STRING_LEN = 512  # Strings shorter than this will be discarded
    SEQ_LEN = 128  # Length of training sequences, in tokens

    # Model
    EMBED_DIM = 256
    FEED_FORWARD_DIM = 128
    NUM_HEADS = 3
    NUM_LAYERS = 2
    VOCAB_SIZE = 5000  # Limits parameters in model.

    # Training
    EPOCHS = 5

    # Inference
    NUM_TOKENS_TO_GENERATE = 80
    keras.utils.get_file(
        origin="https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip",
        extract=True,
    )
    dir = os.path.expanduser("~/.keras/datasets/simplebooks/")

    # Load simplebooks-92 train set and filter out short lines.
    raw_train_ds = (
        tf_data.TextLineDataset(dir + "simplebooks-92-raw/train.txt")
        .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
        .batch(BATCH_SIZE)
        .shuffle(buffer_size=256)
    )

    # Load simplebooks-92 validation set and filter out short lines.
    raw_val_ds = (
        tf_data.TextLineDataset(dir + "simplebooks-92-raw/valid.txt")
        .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
        .batch(BATCH_SIZE)
    )

    """
    ## Train the tokenizer

    We train the tokenizer from the training dataset for a vocabulary size of `VOCAB_SIZE`,
    which is a tuned hyperparameter. We want to limit the vocabulary as much as possible, as
    we will see later on
    that it has a large effect on the number of model parameters. We also don't want to include
    *too few* vocabulary terms, or there would be too many out-of-vocabulary (OOV) sub-words. In
    addition, three tokens are reserved in the vocabulary:

    - `"[PAD]"` for padding sequences to `SEQ_LEN`. This token has index 0 in both
    `reserved_tokens` and `vocab`, since `WordPieceTokenizer` (and other layers) consider
    `0`/`vocab[0]` as the default padding.
    - `"[UNK]"` for OOV sub-words, which should match the default `oov_token="[UNK]"` in
    `WordPieceTokenizer`.
    - `"[BOS]"` stands for beginning of sentence, but here technically it is a token
    representing the beginning of each line of training data.
    """

    # Train tokenizer vocabulary
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
        raw_train_ds,
        vocabulary_size=VOCAB_SIZE,
        lowercase=True,
        reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
    )

    """
    ## Load tokenizer

    We use the vocabulary data to initialize
    `keras_nlp.tokenizers.WordPieceTokenizer`. WordPieceTokenizer is an efficient
    implementation of the WordPiece algorithm used by BERT and other models. It will strip,
    lower-case and do other irreversible preprocessing operations.
    """

    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocab,
        sequence_length=SEQ_LEN,
        lowercase=True,
    )

    """
    ## Tokenize data

    We preprocess the dataset by tokenizing and splitting it into `features` and `labels`.
    """

    # packer adds a start token
    start_packer = keras_nlp.layers.StartEndPacker(
        sequence_length=SEQ_LEN,
        start_value=tokenizer.token_to_id("[BOS]"),
    )

    def preprocess(inputs):
        outputs = tokenizer(inputs)
        features = start_packer(outputs)
        labels = outputs
        return features, labels

    # Tokenize and split into train and label sequences.
    train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
        tf_data.AUTOTUNE
    )
    val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(
        tf_data.AUTOTUNE
    )


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
    num_batches = 10
    if train:
        return (train_dataset.shuffle(50000).map(augment_images).batch(batch_size).take(num_batches).cache().repeat()
                .prefetch(tf.data.experimental.AUTOTUNE))
    else:
        return test_dataset.batch(batch_size).take(num_batches).cache().repeat().prefetch(tf.data.experimental.AUTOTUNE)


# GPU training: https://www.tensorflow.org/guide/gpu
def train_model(model: Sequential, x_train, y_train, x_test, y_test, call_back_list, batch_size=200):
    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=1, batch_size=batch_size, shuffle=True,
              callbacks=call_back_list)


def compile_model(model: Sequential, optimizer=keras.optimizers.Adam(),
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


# Command to trigger tensorboard: python3 -m tensorboard.main --logdir=logs
# https://github.com/tensorflow/profiler/issues/24
# https://www.tensorflow.org/guide/intro_to_modules
def profile_train(concrete_function: ConcreteFunction, dataloader: tf.data.Dataset, num_warmup_step=2,
                  num_prof_step=200):
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(log_dir)
    # Start the profiler, cannot make the parameter profiler True
    tf.summary.trace_on(graph=True, profiler=False)

    for index, (x_train, y_train) in enumerate(dataloader):
        # warmup steps
        if index < num_warmup_step:
            concrete_function(x_train, y_train)
            # Call only one trace_export when tracing, so export after 1 iteration
            if index == 0:
                with train_summary_writer.as_default():
                    # TensorFlow Summary Trace API to log autographed functions for visualization in TensorBoard.
                    # https://www.tensorflow.org/tensorboard/graphs
                    # profiling will end trace_export
                    tf.summary.trace_export(
                        name="my_func_trace",
                        step=0,
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
        shape: tf.TensorShape = op.outputs[0].shape if op.outputs else tf.TensorShape([])
        dtype: DType = op.outputs[0].dtype if op.outputs else tf.float32
        # Each node is an operation in the TensorFlow graph
        G.add_new_node(op.name, op_type=op.type, output_size=shape, output_type=dtype)
        for input_tensor in op.inputs:
            # Create an edge from input operation to the current operation
            G.add_new_edge(input_tensor.op.name, op.name)
    if not nx.is_directed_acyclic_graph(G):
        raise "comp_graph is not directed acyclic"
    # visualize_graph(G, show_labels=False)
    return G


def process_op_df(df: DataFrame) -> dict:
    df = df[df['Host/device'] == 'Device']
    # from csv to dataframe
    df = df[['Operation', 'Avg. self-time (us)']]
    # from dataframe to dict
    return {row['Operation']: row['Avg. self-time (us)'] for index, row in df.iterrows()}


def process_mem_dict(mem_data: dict) -> dict:
    new_dict = {}
    if "memoryProfilePerAllocator" not in mem_data:
        raise ValueError("input dict does not contain memory profile per-allocator")
    mem_data = mem_data["memoryProfilePerAllocator"]
    if "GPU_0_bfc" not in mem_data:
        raise ValueError("input dict does not contain GPU_0_bfc")
    mem_data = mem_data["GPU_0_bfc"]
    # gpu_host_bfc, GPU_0_bfc, mklcpu
    if "memoryProfileSnapshots" not in mem_data:
        raise ValueError("input dict does not contain memoryProfileSnapshots")
    # ['memoryProfileSnapshots', 'profileSummary', 'activeAllocations', 'specialAllocations', 'sampledTimelineSnapshots']
    mem_data = mem_data["memoryProfileSnapshots"]
    if type(mem_data) is not list:
        raise ValueError("should be a list")
    # extract "memoryActivity" from each obj in the list
    mem_data = [item['activityMetadata'] for item in mem_data]
    if type(mem_data) is not list:
        raise ValueError("should be a list")
    for dt in mem_data:
        if type(dt) is not dict:
            raise ValueError("each value should be a dict")
        if "tfOpName" not in dt or "allocationBytes" not in dt:
            raise ValueError("input dict does not contain tfOpName")
        new_dict[dt["tfOpName"]] = dt["allocationBytes"]
    return new_dict


def update_graph_with_prof(graph, prof_dict, mem_dict, device_host_name):

    for node_id in graph.getOperatorIDs():
        operator_dict = graph.getOperator(node_id)

        # Initialize keys if they do not exist
        operator_dict.setdefault("comp_cost", {})
        operator_dict.setdefault("mem", 0)

        # Update computation cost if available
        if node_id in prof_dict:
            operator_dict["comp_cost"][device_host_name] = int(prof_dict[node_id])
        else:
            operator_dict["comp_cost"][device_host_name] = 0
        # Update memory cost if available
        if node_id in mem_dict:
            operator_dict["mem"] = int(mem_dict[node_id])


def parse_tensorboard(input_path, conf: Conf_TB):
    if not os.path.exists(input_path):
        raise FileNotFoundError

    def process_pb(tool_name, params):
        # Process and convert the input file
        print("\033[32mImport TensorFlow...\033[0m")
        print("\033[32mXSpace to Tool Data...\033[0m")
        # https://github.com/tensorflow/profiler/blob/85dcfd10656d623330b11c3bbb8afed6418ec533/plugin/tensorboard_plugin_profile/convert/raw_to_tool_data.py
        tv = rttd.xspace_to_tool_data([input_path], tool_name, params)
        if isinstance(tv, tuple):
            tv = tv[0]
        if conf.get_tool_type() == CONF.OP:
            data_io = StringIO(tv)
            df = pd.read_csv(data_io)
            return df
        elif conf.get_tool_type() == CONF.MEM:
            # Decode bytes to string
            json_str = tv.decode('utf-8')
            # Convert JSON string to dictionary
            data_dict = json.loads(json_str)
            return data_dict
        else:
            raise ValueError("tool type not supported yet")

    return process_pb(conf.tool, conf.params)


def write(data, o_path):
    # Write the processed data to the output file
    print("\033[32mWriting file...\033[0m")
    with open(o_path, "w") as f:
        f.write(data)
    print("\033[32mDone!\033[0m")


def find_specific_pb_file(parent_dir, file_suffix):
    parent_path = Path(parent_dir)
    for file in parent_path.rglob(f'*{file_suffix}'):
        return str(file)
    return None


def distribute_profile_train(concrete_function: ConcreteFunction, dataloader: tf.data.Dataset, num_warmup_step=2,
                             num_prof_step=200):
    # https://www.tensorflow.org/api_docs/python/tf/profiler/experimental/client/trace
    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    options = tf.profiler.experimental.ProfilerOptions(host_tracer_level=3,
                                                       python_tracer_level=1,
                                                       device_tracer_level=1)

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
                tf.profiler.experimental.server.start(6009)
                tf.profiler.experimental.client.trace(
                    'grpc://192.168.0.66:6009',
                    log_dir,
                    5000,
                    options=options)
            concrete_function(x_train, y_train)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=index)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=index)

    return log_dir
