from enum import Enum

import keras
import tensorflow as tf
from transformers import TFGPT2LMHeadModel


class InputSpec(Enum):
    CIFAR10 = lambda batch_size: [
        tf.TensorSpec(shape=[batch_size, 32, 32, 3], dtype=tf.float32, name="input"),
        tf.TensorSpec(shape=[batch_size, 1], dtype=tf.uint8, name="target")
    ]
    IMDB = lambda batch_size, max_len: [
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="token_ids"),
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="segment_ids"),
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.bool, name="padding_mask"),
        tf.TensorSpec(shape=[batch_size, 2], dtype=tf.int64, name="labels"), # 2 class for Imdb-review
    ]


def get_input_spec(model, batch_size, max_len=None) -> list[tf.TensorSpec]:
    if isinstance(model, keras.Sequential):
        return InputSpec.CIFAR10(batch_size)
    else:
        if max_len is None:
            raise ValueError("max_len must be provided for IMDB model")
        return InputSpec.IMDB(batch_size, max_len)
