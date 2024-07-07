import time
import torch
# pip install -q --upgrade keras-nlp
import keras_nlp
import tensorflow as tf
from typing import Tuple

from keras_nlp.src.models import GPT2Tokenizer, GPT2CausalLM


def get_pre_trained_gpt2_model() -> Tuple[GPT2CausalLM, GPT2Tokenizer]:
    # To speed up training and generation, we use preprocessor of length 128
    # instead of full length 1024.
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )
    return gpt2_lm, preprocessor.tokenizer

'''
gpt2_model, tokenizer = get_pre_trained_gpt2_model()


# Wrap the model call in a tf.function to create a computation graph
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 128], dtype=tf.int32)])
def gpt2_computation_graph(input_ids):
    return gpt2_model(input_ids)


# Get the vocabulary size
vocab_size = len(tokenizer.get_vocabulary())
print("Vocabulary Size:", vocab_size)

# Example input tensor
example_input = tf.random.uniform(shape=[1, 128], dtype=tf.int32, minval=0, maxval=vocab_size)

# Run the computation graph
output = gpt2_computation_graph(example_input)
print(output)
'''