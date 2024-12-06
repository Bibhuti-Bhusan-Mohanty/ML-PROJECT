import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('Model\keras_model.h5')

# Re-save it in a compatible format
model.save('Model\new_model.h5')
