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

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

early = EarlyStopping(monitor='val_accuracy',
                      mode='min',
                      patience=5)
STEPS_PER_EPOCH = 17118 // BATCH_SIZE
VALID_STEPS = 4279 // BATCH_SIZE

history = model.fit(train_dataset,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=10,
                    validation_data=valid_dataset,
                    validation_steps=VALID_STEPS, callbacks=[early])
