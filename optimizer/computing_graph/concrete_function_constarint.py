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
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="input_ids"),
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="labels"),
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="attention_mask")
    ]


def get_input_spec(model, batch_size, max_len=None) -> list[tf.TensorSpec]:
    if isinstance(model, keras.Sequential):
        return InputSpec.CIFAR10(batch_size)
    elif isinstance(model, TFGPT2LMHeadModel):
        if max_len is None:
            raise ValueError("max_len must be provided for IMDB model")
        return InputSpec.IMDB(batch_size, max_len)
    else:
        raise ValueError("Unsupported model type")
