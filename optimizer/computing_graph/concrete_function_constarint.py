from enum import Enum

import keras
import tensorflow as tf
from keras_nlp.src.models.bert.bert_text_classifier import BertTextClassifier
from keras_nlp.src.models.f_net.f_net_text_classifier import FNetTextClassifier


class InputSpec(Enum):
    CIFAR10 = lambda batch_size: [
        tf.TensorSpec(shape=[batch_size, 32, 32, 3], dtype=tf.float32, name="input"),
        tf.TensorSpec(shape=[batch_size, 1], dtype=tf.uint8, name="target")
    ]
    IMDB_BERT = lambda batch_size, max_len: [
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="token_ids"),
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="segment_ids"),
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.bool, name="padding_mask"),
        tf.TensorSpec(shape=[batch_size, ], dtype=tf.int64, name="labels"), # 2 class for Imdb-review
    ]
    IMDB_FNET = lambda batch_size, max_len: [
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="token_ids"),
        tf.TensorSpec(shape=[batch_size, max_len], dtype=tf.int32, name="segment_ids"),
        tf.TensorSpec(shape=[batch_size, ], dtype=tf.int64, name="labels"),  # 2 class for Imdb-review
    ]


def get_input_spec(model, batch_size, max_len=None) -> list[tf.TensorSpec]:
    if isinstance(model, keras.Sequential):
        return InputSpec.CIFAR10(batch_size)
    else:
        if max_len is None:
            raise ValueError("max_len must be provided for IMDB model")
        elif isinstance(model, BertTextClassifier):
            return InputSpec.IMDB_BERT(batch_size, max_len)
        elif isinstance(model, FNetTextClassifier):
            return InputSpec.IMDB_FNET(batch_size, max_len)
        else:
            raise ValueError("llm model must be either BertTextClassifier or FNetTextClassifier")
