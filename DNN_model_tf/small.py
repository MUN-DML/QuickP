from zipfile import ZipFile

import keras
import os
from keras.src.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from keras.src.models import Sequential
from keras.src.layers import Dropout

from tensorflow.keras import datasets, layers, models
'''
https://github.com/shanghai1997/DNN/commit/fdcd1e013208fd019088c8edf09413ef85ce1222
https://keras.io/examples/vision/image_classification_from_scratch/
'''


def small_tf():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(32, 32, 3)))  # Flatten the input images to 1D
    model.add(layers.Dense(10, activation='softmax'))  # Output layer with 10 neurons for 10 classes

    return model