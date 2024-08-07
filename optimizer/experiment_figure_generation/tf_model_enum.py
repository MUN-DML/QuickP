from enum import Enum

from DNN_model_tf.alexnet import alexnet
from DNN_model_tf.openai_gpt2 import get_openai_gpt2_model
from DNN_model_tf.small import small_tf
from DNN_model_tf.vgg_tf import VGG16_tf


class TFModelEnum(Enum):
    VGG = VGG16_tf
    SMALL = small_tf
    GPT2 = get_openai_gpt2_model
    ALEXNET = alexnet
