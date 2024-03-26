from zipfile import ZipFile

import keras
import os
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.src.datasets import cifar10
from keras.src.layers import Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


def checkCatDog():
    path_cat_dog = 'kagglecatsanddogs_5340'
    path_cat_dog_zip = 'kagglecatsanddogs_5340.zip'

    if os.path.exists(path_cat_dog):
        print("the imagenet is here")
        return True

    if os.path.exists(path_cat_dog_zip):
        print("need to extract the file")
        with ZipFile(path_cat_dog_zip, 'r') as zip_ref:
            zip_ref.extractall()
        return True

    # file is not here, download that
    print("need to download the file")
    return False


def getCatDogDF():

    # method 1 but the test file in imageNet is not labelled
    trdata = ImageDataGenerator()
    train_ds = trdata.flow_from_directory(directory="tiny-imagenet-200/train", target_size=(224, 224))
    tsdata = ImageDataGenerator()
    val_ds = tsdata.flow_from_directory(directory="tiny-imagenet-200/test/images", target_size=(224, 224))
    return train_ds, val_ds


def initModelForCifar10():
    model = Sequential()
    # ignore the input layer
    # For imageNet model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    # fully connected layer
    model.add(Dense(units=512, activation="relu"))
    model.add(Dropout(0.2))
    # imageNet has a class of 200, so model.add(Dense(units=200, activation="softmax"))
    model.add(Dense(units=10, activation="softmax"))
    opt = Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")])
    return model


def data_augmentation(images):
    data_augmentation_layers = [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
    ]
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


def getCifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


class MyVGG16:
    def __init__(self, name):
        self.name = name
        self.model = initModelForCifar10()
        self.model.summary()

    def train(self):
        (x_train, y_train), (x_test, y_test) = getCifar()
        print(self.model.summary())
        checkpoint = ModelCheckpoint("vgg16.keras")
        self.model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), validation_split=0.4,
                       epochs=2, callbacks=[checkpoint], shuffle=True)


#myVGG16 = MyVGG16("")
#myVGG16.train()
