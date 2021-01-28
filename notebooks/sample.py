import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from vit_keras import vit
from keras.datasets import mnist

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
base_model = vit.vit_l32(
activation=None,
pretrained=False,
include_top=False,
pretrained_top=False,
)
model = tf.keras.Sequential([
	base_model,
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(5, activation='softmax', dtype='float32')
])
model.compile(
optimizer="adam",
loss=tf.keras.losses.SparseCategoricalCrossentropy(),
metrics=['categorical_accuracy'])
model.fit(x=x_train, y=y_train, batch_size=32, epochs=20, shuffle=True, validation_data=(x_test, y_test))̥
