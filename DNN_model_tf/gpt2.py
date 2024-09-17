import keras
import keras_nlp


def get_keras_gpt2_tokenizer():
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )
    return preprocessor

def get_keras_gpt2_model():
    preprocessor = get_keras_gpt2_tokenizer()
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )
    return gpt2_lm
