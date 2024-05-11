from datetime import datetime

import tensorflow as tf

from tf_util import train_model

model = tf.keras.applications.VGG16()
print(model.summary())
optimizer = tf.keras.optimizers.Adam()

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 histogram_freq=1,
                                                 profile_batch='500,520')

train_model()
