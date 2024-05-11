from datetime import datetime

import tensorflow as tf

from DNN_model_tf.vgg_tf import VGG16_tf
from tf_util import train_model, getCifar, compile_model

model = VGG16_tf()
compile_model(model)

print(tf.config.list_physical_devices('GPU'))
print(tf.config.list_physical_devices('CPU'))
print(tf.test.gpu_device_name())

# Create a TensorBoard callback
# profiling: https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 histogram_freq=1,
                                                 # a range of batches
                                                 profile_batch='50,51')

(x_train, y_train), (x_test, y_test) = getCifar()
train_model(model, x_train, y_train, x_test, y_test, [tboard_callback], batch_size=200)
