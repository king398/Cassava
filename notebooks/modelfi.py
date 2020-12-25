import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_policy(policy)

with strategy.scope():
	base_model = tf.keras.applications.ResNet152(include_top=False)
	base_model.trainable = True

	model = tf.keras.Sequential([
		tf.keras.layers.BatchNormalization(),
		base_model,
		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.GlobalAveragePooling2D(),
		tf.keras.layers.Dense(256),
		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dense(128),
		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dropout(0.15),
		tf.keras.layers.Dense(64),
		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dense(32),
		tf.keras.layers.Dropout(0.15),

		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dense(16),

		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dense(8),
		tf.keras.layers.LeakyReLU(),
		tf.keras.layers.Dense(len(CLASSES), activation='softmax')
	])

	model.compile(
		optimizer=tf.keras.optimizers.Adam(lr=0.001, epsilon=0.001),
		loss='sparse_categorical_crossentropy',
		metrics=['sparse_categorical_accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

early = EarlyStopping(monitor='val_loss',
                      mode='min',
                      patience=5)
STEPS_PER_EPOCH = 17118 // BATCH_SIZE
VALID_STEPS = 4279 // BATCH_SIZE

history = model.fit(train_dataset,
                    steps_per_epoch=STEPS_PER_EPOCH,

                    epochs=25,
                    validation_data=valid_dataset,
                    validation_steps=VALID_STEPS, callbacks=[early])
