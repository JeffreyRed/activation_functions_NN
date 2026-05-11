import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# GELU Example Commonly Used in Transformers

model = Sequential([
    Dense(256, activation=tf.nn.gelu, input_shape=(768,)),
    Dense(256, activation=tf.nn.gelu),
    Dense(10, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.summary()