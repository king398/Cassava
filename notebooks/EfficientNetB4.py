import tensorflow as tf

with strategy.scope():
	img_adjust_layer = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input,
	                                          input_shape=[*IMAGE_SIZE, 3])
	base_model = tf.keras.applications.EfficientNetB4(include_top=False)
	base_model.trainable = True

	model = tf.keras.Sequential([
		tf.keras.layers.BatchNormalization(renorm=True),
		img_adjust_layer,
		base_model,
		tf.keras.layers.GlobalAveragePooling2D(),
		tf.keras.layers.Dense(8, activation='relu'),
		# tf.keras.layers.BatchNormalization(renorm=True),
		tf.keras.layers.Dense(len(CLASSES), activation='softmax')
	])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler, epsilon=0.001),
		loss='sparse_categorical_crossentropy',
		metrics=['sparse_categorical_accuracy'])
