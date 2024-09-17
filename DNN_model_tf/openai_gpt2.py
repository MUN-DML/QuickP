# pip install transformers
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf


def get_openai_gpt2_model():
    return TFGPT2LMHeadModel.from_pretrained('gpt2')


def get_openai_gpt2_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def test_openai_gpt2_model():
    model, tokenizer = get_openai_gpt2_model(), get_openai_gpt2_tokenizer()
    print(model.summary())
    '''
    # {'input_ids': <tf.Tensor: shape=(1, 6), dtype=int32, numpy=array([[15496,    11,   616,  3290,   318, 13779]], dtype=int32)>,
    # 'attention_mask': <tf.Tensor: shape=(1, 6), dtype=int32, numpy=array([[1, 1, 1, 1, 1, 1]], dtype=int32)>}
    inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    '''
    # {'input_ids': <tf.Tensor: shape=(1, 128), dtype=int32, numpy=array([[15496,    11,   616,  3290,   318, 13779, 50256,
    # 'attention_mask': <tf.Tensor: shape=(1, 128), dtype=int32, numpy=array([[1, 1, 1, 1, 1, 1, 0, 0, 0
    inputs = tokenizer(["Hello, my dog is cute"], padding="max_length", truncation=True, max_length=128,
                       return_tensors="tf")
    outputs = model(inputs)
    logits = outputs[0]
    print(logits)
