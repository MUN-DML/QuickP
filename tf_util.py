from keras import Sequential
from keras.src.datasets import cifar10
from keras.src.utils import to_categorical


def getCifar():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def train(model: Sequential, x_train, y_train, x_test, y_test):
    print(model.summary())
    model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=1, batch_size=200, shuffle=True)

