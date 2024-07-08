# pip install transformers
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf


def get_openai_gpt2_model():
    model = TFGPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    outputs = model(inputs)
    logits = outputs[0]