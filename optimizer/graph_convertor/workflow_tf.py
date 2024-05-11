import tensorflow as tf

model = tf.keras.applications.VGG16()
optimizer = tf.keras.optimizers.Adam()

# Convert to a TensorFlow function that includes both forward and backward passes
@tf.function
def train_step(input_data, target_data):
    with tf.GradientTape() as tape:
        predictions = model(input_data, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(target_data, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
