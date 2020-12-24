import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, \
	BatchNormalization

with strategy.scope():
	img_adjust_layer = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input,
	                                          input_shape=[*IMAGE_SIZE, 3])
	base_model = tf.keras.applications.EfficientNetB5(include_top=False, weights='imagenet')
	base_model.trainable = False

	model = tf.keras.Sequential([
		tf.keras.layers.experimental.preprocessing.Normalization(),
		img_adjust_layer,
		base_model,
		LeakyReLU(),
		BatchNormalization(),
		Flatten(),
		LeakyReLU(),
		Dense(512),
		LeakyReLU(),
		Dense(256),
		LeakyReLU(),

		Dense(128),
		LeakyReLU(),

		Dense(64),
		LeakyReLU(),

		Dense(32),
		LeakyReLU(),

		Dense(16),
		LeakyReLU(),

		Dense(8),

		Dense(5, activation="softmax")

	])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
		loss='sparse_categorical_crossentropy',
		metrics=['sparse_categorical_accuracy'])
