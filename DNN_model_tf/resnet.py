import tensorflow as tf
from keras import layers, models, datasets

def residual_block(x, filters, stride=1):
    shortcut = x

    # First convolutional layer of the block
    x = layers.Conv2D(filters, kernel_size=(3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second convolutional layer of the block
    x = layers.Conv2D(filters, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Adjust the shortcut to match dimensions if necessary
    if stride != 1:
        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add the shortcut to the output
    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)

    return x


# Build ResNet-18 for CIFAR-10
def build_resnet18(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Initial convolutional layer
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks with increasing number of filters
    x = residual_block(x, 64, stride=1)
    x = residual_block(x, 64, stride=1)

    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)

    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)

    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)

    # Global Average Pooling and output layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs, outputs)
    return model
