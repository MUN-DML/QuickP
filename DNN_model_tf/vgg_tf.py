from zipfile import ZipFile

import keras
import os
from keras.src.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from keras.src.datasets import cifar10
from keras.src.layers import Dropout

'''
https://github.com/shanghai1997/DNN/commit/fdcd1e013208fd019088c8edf09413ef85ce1222
https://keras.io/examples/vision/image_classification_from_scratch/
'''


def initModelForCifar10():
    model = Sequential()
    # ignore the input layer
    # input_shape is the single image size of the suitable dataset. For Cifar_10 => input_shape=(32, 32, 3)
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    # Three fully connected layer
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(units=4096, activation="relu"))
    model.add(Dropout(0.5))

    # the last fully connected layer is a classifier, 10 class for Cifar_10
    model.add(Dense(units=10, activation="softmax"))

    model.compile(optimizer=keras.optimizers.Adam(3e-4), loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")])
    return model


