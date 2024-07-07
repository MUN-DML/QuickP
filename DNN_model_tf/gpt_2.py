import time
import torch
# pip install -q --upgrade keras-nlp
import keras_nlp


def get_pre_trained_gpt2_model():
    # To speed up training and generation, we use preprocessor of length 128
    # instead of full length 1024.
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )
    start = time.time()

    output = gpt2_lm.generate("My trip to Yosemite was", max_length=200)
    print("\nGPT-2 output:")
    print(output)

    end = time.time()
    print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")


get_pre_trained_gpt2_model()
