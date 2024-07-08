# pip install transformers
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf


def get_openai_gpt2_model_and_tokenizer():
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer


def test_openai_gpt2_model():
    model, tokenizer = get_openai_gpt2_model_and_tokenizer()
    # {'input_ids': <tf.Tensor: shape=(1, 6), dtype=int32, numpy=array([[15496,    11,   616,  3290,   318, 13779]], dtype=int32)>,
    # 'attention_mask': <tf.Tensor: shape=(1, 6), dtype=int32, numpy=array([[1, 1, 1, 1, 1, 1]], dtype=int32)>}
    inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    print(inputs)
    outputs = model(inputs)
    logits = outputs[0]
    print(logits)
