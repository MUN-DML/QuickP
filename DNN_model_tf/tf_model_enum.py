from enum import Enum

from DNN_model_tf.alexnet import alexnet
from DNN_model_tf.bert import bert
from DNN_model_tf.fnet import fnet
from DNN_model_tf.small import small_tf
from DNN_model_tf.vgg_tf import VGG16_tf


class TFModelEnum(Enum):
    VGG = VGG16_tf
    SMALL = small_tf
    ALEXNET = alexnet
    BERT = bert
    FNET = fnet
    TEST = "TEST"
